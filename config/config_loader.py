"""配置加载器：读取 config/settings.yaml，替换环境变量占位符，并做结构校验。

设计要点：
    1. 环境变量占位符支持 ``${VAR}`` 与 ``${VAR:-默认值}`` 两种写法，递归作用于
       整个 YAML 树（dict / list / str 全覆盖）。
    2. 使用 Pydantic 模型做"启动即失败"（fail-fast）校验：配置写错时在进程启动
       阶段就抛 ``ConfigError``，而不是等到索引或检索时才发现。
    3. 提供进程级单例 ``get_config()``，避免各模块重复解析 YAML。

用法::

    from config.config_loader import get_config

    cfg = get_config()
    print(cfg.vector_store.host, cfg.embedding.model)

命令行自检::

    python -m config.config_loader            # 打印关键配置摘要
    python -m config.config_loader --validate  # 仅校验配置是否合法
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

__all__ = [
    "AppConfig",
    "ConfigError",
    "EmbeddingConfig",
    "LLMConfig",
    "MCPServerConfig",
    "MCPToolConfig",
    "VectorStoreConfig",
    "get_config",
    "get_mcp_server",
    "get_mcp_servers",
    "load_settings",
    "reload_config",
    "resolve_path",
    "PROJECT_ROOT",
]

# ---------------------------------------------------------------------------
# 路径常量：PROJECT_ROOT 为项目根目录（本文件位于 <root>/config/config_loader.py）
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"

# ${VAR} 或 ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """配置缺失、格式非法或校验失败时抛出。"""


def _load_dotenv(dotenv_path: Path) -> None:
    """加载 .env 到 os.environ（已存在的真实环境变量优先，不会被覆盖）。

    Args:
        dotenv_path: .env 文件路径；不存在时静默跳过。

    Returns:
        None
    """
    if not dotenv_path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - dotenv 为可选依赖
        # 未安装 python-dotenv 时做一个最小实现，避免因为可选依赖导致启动失败
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _substitute_env(value: Any) -> Any:
    """递归替换配置树中的 ``${VAR}`` / ``${VAR:-default}`` 占位符。

    Args:
        value: 任意 YAML 节点（dict / list / str / 标量）。

    Returns:
        替换后的同构数据结构；标量原样返回。
    """
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_PATTERN.sub(_replace, value)
    return value


def _coerce_scalar(value: Any) -> Any:
    """把环境变量替换后残留的数字字符串还原为 int/float/bool。

    YAML 中 ``port: ${QDRANT_PORT:-6333}`` 会被替换成字符串 ``"6333"``，
    Pydantic 可以自动转换，但为了严格性与可读性，这里主动归一化一次。

    Args:
        value: 待归一化的值。

    Returns:
        归一化后的值（int / float / bool / 原值）。
    """
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    if re.fullmatch(r"-?\d+\.\d+", value.strip()):
        return float(value.strip())
    return value


def _normalize_tree(value: Any) -> Any:
    """对整棵配置树做标量归一化。

    Args:
        value: 任意 YAML 节点。

    Returns:
        归一化后的同构数据结构。
    """
    if isinstance(value, dict):
        return {k: _normalize_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_tree(v) for v in value]
    return _coerce_scalar(value)


# ---------------------------------------------------------------------------
# Pydantic 配置模型
# ---------------------------------------------------------------------------
class ProjectConfig(BaseModel):
    """项目级元信息。"""

    name: str = "enterprise-rag-mcp"
    version: str = "0.1.0"
    env: Literal["dev", "staging", "prod"] = "dev"
    timezone: str = "Asia/Shanghai"


class HNSWConfig(BaseModel):
    """Qdrant HNSW 索引参数。"""

    m: int = Field(default=16, ge=4, le=64)
    ef_construct: int = Field(default=128, ge=16, le=512)


class VectorStoreConfig(BaseModel):
    """向量库（Qdrant）连接与集合参数。"""

    provider: Literal["qdrant"] = "qdrant"
    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    prefer_grpc: bool = False
    api_key: str = ""
    timeout: int = Field(default=30, ge=1)
    collection: str = "enterprise_docs"
    distance: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    payload_indexes: list[str] = Field(default_factory=list)

    @property
    def url(self) -> str:
        """返回 Qdrant REST 基础地址，例如 ``http://localhost:6333``。"""
        return f"http://{self.host}:{self.port}"


class FallbackEmbeddingConfig(BaseModel):
    """本地嵌入降级备选模型。"""

    model: str = "BAAI/bge-base-zh-v1.5"
    dimension: int = Field(default=768, ge=1)


class EmbeddingConfig(BaseModel):
    """嵌入模型配置（可插拔 provider）。

    Attributes:
        provider: ``local`` 走本地 sentence-transformers；``openai_compatible``
            走远端 ``/v1/embeddings``（DeepSeek 官方开放 embedding 接口后使用）。
        dimension: 向量维度，必须与 Qdrant 集合维度一致。
    """

    provider: Literal["local", "openai_compatible"] = "local"
    model: str = "BAAI/bge-m3"
    dimension: int = Field(default=1024, ge=1)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=512)
    max_length: int = Field(default=8192, ge=64)
    device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    cache_dir: str = "./.cache/embedding"
    base_url: str = ""
    api_key: str = ""
    fallback: FallbackEmbeddingConfig = Field(default_factory=FallbackEmbeddingConfig)

    @model_validator(mode="after")
    def _check_remote_fields(self) -> EmbeddingConfig:
        """当 provider 为远端时必须提供 base_url 与 api_key。"""
        if self.provider == "openai_compatible":
            missing = [
                name
                for name, val in (("base_url", self.base_url), ("api_key", self.api_key))
                if not val
            ]
            if missing:
                raise ValueError(
                    "embedding.provider=openai_compatible 时必须配置: " + ", ".join(missing)
                )
        return self


class LLMConfig(BaseModel):
    """LLM（对话/编排模型）配置。"""

    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    # 2026-10 起官方只接受 deepseek-flash / deepseek-v4-pro；
    # 旧名 deepseek-chat 仅作兼容别名且已公告停用。
    model: str = "deepseek-flash"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=2, ge=0, le=5)
    tool_call_retry: int = Field(default=1, ge=0, le=3)
    # 分类类任务（意图识别/参数抽取）是否关闭思考模式。
    # V4 默认开启思考，短 max_tokens 下 content 会为空（实测），
    # 而分类任务不需要推理，关闭后 output token 从 72 降到 1。
    classification_thinking: bool = False

    @property
    def is_configured(self) -> bool:
        """是否已配置 API Key（未配置时 Agent 走离线降级路径）。"""
        return bool(self.api_key)


class IngestionConfig(BaseModel):
    """文档摄取与分块参数。"""

    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=1024)
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".md", ".markdown", ".txt"]
    )
    index_batch_size: int = Field(default=100, ge=1, le=1000)
    docs_dir: str = "./docs"
    state_file: str = "./.state/ingestion_state.json"

    @model_validator(mode="after")
    def _check_overlap(self) -> IngestionConfig:
        """overlap 必须严格小于 chunk_size，否则分块会死循环。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"ingestion.chunk_overlap ({self.chunk_overlap}) 必须小于 "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class MCPToolConfig(BaseModel):
    """单个 MCP 工具的开关与描述。"""

    name: str
    enabled: bool = True
    description: str = ""


class MCPServerConfig(BaseModel):
    """单个 MCP server 的注册信息。"""

    name: str
    enabled: bool = True
    description: str = ""
    transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    command: str = "python"
    args: list[str] = Field(default_factory=list)
    url: str = ""
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    tools: list[MCPToolConfig] = Field(default_factory=list)

    def enabled_tool_names(self) -> list[str]:
        """返回该 server 下所有被启用的工具名。"""
        return [tool.name for tool in self.tools if tool.enabled]

    def get_tool(self, tool_name: str) -> MCPToolConfig | None:
        """按名字查找工具配置，未注册时返回 ``None``。"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None


class MCPRegistryConfig(BaseModel):
    """MCP 注册表配置。"""

    protocol_version: str = "2024-11-05"
    client_name: str = "enterprise-rag-mcp-client"
    client_version: str = "0.1.0"
    call_timeout: int = Field(default=30, ge=1)
    servers: list[MCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_names(self) -> MCPRegistryConfig:
        """server 名与同一 server 内的 tool 名都不允许重复。"""
        seen: set[str] = set()
        for server in self.servers:
            if server.name in seen:
                raise ValueError(f"mcp.servers 中存在重复的 server 名: {server.name}")
            seen.add(server.name)
            tool_names = [tool.name for tool in server.tools]
            if len(tool_names) != len(set(tool_names)):
                raise ValueError(f"server '{server.name}' 中存在重复的 tool 名")
        return self


class SessionConfig(BaseModel):
    """会话存储配置。"""

    db_path: str = "./.data/sessions.db"
    max_turns: int = Field(default=10, ge=1, le=100)


class APIConfig(BaseModel):
    """FastAPI 服务配置。"""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    sse_heartbeat_seconds: int = Field(default=15, ge=1)
    session: SessionConfig = Field(default_factory=SessionConfig)


class ObservabilityConfig(BaseModel):
    """日志与可观测性配置。"""

    log_dir: str = "./logs"
    structured_log_file: str = "./logs/tool_calls.jsonl"
    request_log_file: str = "./logs/requests.jsonl"
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_prompts: bool = False


class EvalConfig(BaseModel):
    """评测配置。"""

    test_cases_file: str = "./eval/test_cases.json"
    report_file: str = "./eval/eval_report.md"
    concurrency: int = Field(default=2, ge=1, le=16)
    metrics: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    """应用总配置（对应 settings.yaml 的根节点）。"""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    mcp: MCPRegistryConfig = Field(default_factory=MCPRegistryConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @field_validator("vector_store")
    @classmethod
    def _check_dimension(cls, value: VectorStoreConfig) -> VectorStoreConfig:
        """占位校验：集合维度与嵌入维度的强一致在 embedding 校验器中统一处理。"""
        return value


def load_settings(settings_path: str | Path | None = None) -> AppConfig:
    """加载并校验配置文件。

    Args:
        settings_path: 配置文件路径；为 ``None`` 时使用 ``config/settings.yaml``。

    Returns:
        校验通过的 :class:`AppConfig` 实例。

    Raises:
        ConfigError: 文件不存在、YAML 语法错误或结构校验失败。
    """
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    if not path.is_file():
        raise ConfigError(
            f"配置文件不存在: {path}\n"
            f"请确认已在项目根目录执行，或通过 --settings 指定正确路径。"
        )

    _load_dotenv(PROJECT_ROOT / ".env")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - 权限/编码等系统级异常
        raise ConfigError(f"配置文件读取失败: {path} -> {exc}") from exc

    try:
        raw_data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 语法错误: {path} -> {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError(f"配置根节点必须是映射(mapping)，实际为 {type(raw_data).__name__}")

    resolved = _normalize_tree(_substitute_env(raw_data))

    # MCP server 的 env 段会被用作子进程环境变量，必须是 str -> str。
    # 但 _normalize_tree 会把 "6333" 归一化成 int 6333，这里强制转回字符串。
    for server in resolved.get("mcp", {}).get("servers", []) or []:
        env_block = server.get("env")
        if isinstance(env_block, dict):
            server["env"] = {str(k): str(v) for k, v in env_block.items()}

    try:
        config = AppConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigError(f"配置校验失败:\n{exc}") from exc

    if config.embedding.dimension != _expected_dimension_hint(config):
        raise ConfigError(
            "embedding.dimension 与 embedding.model 的常见输出维度不一致："
            f"model={config.embedding.model}, dimension={config.embedding.dimension}, "
            f"期望={_expected_dimension_hint(config)}。"
            "请修正 settings.yaml（维度与 Qdrant 集合维度必须一致）。"
        )

    return config


# 常见嵌入模型的输出维度表，用于配置期自检（不在表内的模型跳过校验）
_KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-small-zh-v1.5": 512,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _expected_dimension_hint(config: AppConfig) -> int:
    """返回模型已知维度；未知模型时返回配置值本身（等价于跳过校验）。"""
    return _KNOWN_DIMENSIONS.get(config.embedding.model, config.embedding.dimension)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """获取进程级单例配置（首次调用时加载）。

    Returns:
        :class:`AppConfig` 实例；同一进程内多次调用返回同一对象。

    Raises:
        ConfigError: 配置文件缺失或非法。
    """
    return load_settings()


def reload_config() -> AppConfig:
    """清除缓存并重新加载配置（用于测试或热更新）。

    Returns:
        重新加载后的 :class:`AppConfig` 实例。
    """
    get_config.cache_clear()
    return get_config()


def get_mcp_servers(enabled_only: bool = True) -> list[MCPServerConfig]:
    """返回 MCP server 列表。

    Args:
        enabled_only: 为 ``True`` 时仅返回 ``enabled: true`` 的 server。

    Returns:
        server 配置列表（保持 settings.yaml 中的声明顺序）。
    """
    servers = get_config().mcp.servers
    return [s for s in servers if s.enabled] if enabled_only else list(servers)


def get_mcp_server(server_name: str) -> MCPServerConfig:
    """按名字获取单个 MCP server 配置。

    Args:
        server_name: server 名称，如 ``knowledge_base``。

    Returns:
        匹配的 server 配置。

    Raises:
        ConfigError: 该 server 未在 settings.yaml 中注册。
    """
    for server in get_config().mcp.servers:
        if server.name == server_name:
            return server
    registered = [s.name for s in get_config().mcp.servers]
    raise ConfigError(f"MCP server 未注册: {server_name}；已注册: {registered}")


def resolve_path(relative_path: str | Path) -> Path:
    """把配置中的相对路径解析为项目根目录下的绝对路径。

    Args:
        relative_path: 配置里书写的路径（相对或绝对）。

    Returns:
        绝对路径；若传入已是绝对路径则原样返回。
    """
    path = Path(relative_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _summarize(config: AppConfig) -> str:
    """生成便于人眼检查的配置摘要文本。"""
    lines = [
        f"项目           : {config.project.name} v{config.project.version} (env={config.project.env})",
        f"向量库         : qdrant {config.vector_store.host}:{config.vector_store.port} "
        f"collection={config.vector_store.collection} distance={config.vector_store.distance}",
        f"嵌入模型       : provider={config.embedding.provider} model={config.embedding.model} "
        f"dim={config.embedding.dimension} device={config.embedding.device}",
        f"LLM            : {config.llm.provider}/{config.llm.model} "
        f"api_key={'已配置' if config.llm.is_configured else '未配置(读取 DEEPSEEK_API_KEY)'}",
        f"分块           : chunk_size={config.ingestion.chunk_size} "
        f"overlap={config.ingestion.chunk_overlap} batch={config.ingestion.index_batch_size}",
        f"服务           : http://{config.api.host}:{config.api.port} "
        f"session_max_turns={config.api.session.max_turns}",
        "MCP servers    :",
    ]
    for server in config.mcp.servers:
        flag = "on " if server.enabled else "off"
        tools = ", ".join(
            f"{t.name}{'' if t.enabled else '(disabled)'}" for t in server.tools
        ) or "(无工具声明)"
        lines.append(f"  [{flag}] {server.name} via {server.transport} -> {tools}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：校验并打印配置摘要。

    Args:
        argv: 命令行参数（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码：0 成功，2 配置非法。
    """
    parser = argparse.ArgumentParser(description="校验并打印 settings.yaml 配置摘要")
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS_PATH),
        help="配置文件路径（默认 config/settings.yaml）",
    )
    parser.add_argument("--validate", action="store_true", help="仅校验，不打印摘要")
    args = parser.parse_args(argv)

    try:
        config = load_settings(args.settings)
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    if not args.validate:
        print(_summarize(config))
    print("[OK] 配置校验通过")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
