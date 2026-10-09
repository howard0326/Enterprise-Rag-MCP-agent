"""LLM 客户端：调用 DeepSeek（OpenAI 兼容协议），支持 JSON 模式与流式输出。

关键工程点：
    1. **JSON 模式的健壮解析**：用于意图识别。即使开了 ``response_format=json_object``，
       模型偶尔仍会包上 markdown 代码块或前后加解释文字，因此解析走"剥离围栏 →
       截取首个平衡 JSON 对象 → 失败则抛错"，绝不静默返回空结构（那会让意图识别
       悄悄退化，评测时查不出来）。
    2. **无 Key 时明确报错而不是假装成功**：``NoLLMError`` 让上层走规则式降级路径。
       阶段4 的 Agent 与阶段5 的评测需要在没有 Key 的情况下依然可运行，
       但**降级必须是显式的**（日志里能看出来），否则评测数字会被污染。
    3. **连接复用**：``httpx.AsyncClient`` 常驻，避免每次请求重建 TLS 连接
       （阶段2 已实测：本机重建连接代价约 2 秒）。
    4. **超时与重试**：网络抖动重试 ``max_retries`` 次；4xx（如鉴权失败）不重试，
       直接抛出 —— 重试改变不了 Key 错误。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from config.config_loader import LLMConfig, get_config

__all__ = [
    "LLMError",
    "NoLLMError",
    "LLMResponse",
    "LLMClient",
    "get_llm_client",
    "extract_json",
]

logger = logging.getLogger("backend.llm")


class LLMError(RuntimeError):
    """LLM 调用失败（网络、协议、鉴权等）。"""


class NoLLMError(LLMError):
    """未配置 API Key —— 上层应走规则式降级路径。"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出中稳健地抽取 JSON 对象。

    处理顺序：直接解析 → 剥离 markdown 围栏 → 扫描首个括号平衡的对象。

    Args:
        text: 模型返回的原始文本。

    Returns:
        解析出的字典。

    Raises:
        LLMError: 三种方式都失败。
    """
    if not text or not text.strip():
        raise LLMError("模型返回为空，无法解析 JSON")

    candidates: list[str] = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    # 扫描首个括号平衡的 JSON 对象（跳过字符串内的括号）
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : idx + 1])
                start = -1

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMError(f"无法从模型输出中解析出 JSON 对象: {text[:200]!r}")


@dataclass
class LLMResponse:
    """一次 LLM 调用的结果。

    Attributes:
        content: 回复正文。
        prompt_tokens: 输入 token 数（用于阶段5 的可观测性统计）。
        completion_tokens: 输出 token 数。
        model: 实际使用的模型名。
        latency_ms: 端到端耗时。
        finish_reason: 结束原因。
        raw: 原始响应（排障用；不写入日志以免泄漏 prompt）。
    """

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 总数。"""
        return self.prompt_tokens + self.completion_tokens

    def to_metrics(self) -> dict[str, Any]:
        """转成可写入结构化日志的指标字典。"""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 2),
        }


class LLMClient:
    """DeepSeek（OpenAI 兼容）异步客户端。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        """初始化客户端（不建立连接）。

        Args:
            config: LLM 配置；``None`` 时读 ``config/settings.yaml``。
        """
        self.config = config or get_config().llm
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        """是否已配置 API Key。"""
        return bool(self.config.api_key)

    async def _get_client(self) -> Any:
        """懒加载 httpx 异步客户端（长连接复用）。

        Returns:
            ``httpx.AsyncClient`` 实例。

        Raises:
            NoLLMError: 未配置 API Key。
        """
        if not self.available:
            raise NoLLMError(
                "未配置 DEEPSEEK_API_KEY（.env）。请填写后重试，"
                "或在调用方使用规则式降级路径。"
            )
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            )
        return self._client

    def _payload(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_mode: bool,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """构造请求体。

        Args:
            messages: OpenAI 格式消息数组。
            json_mode: 是否要求 JSON 输出。
            temperature: 覆盖默认温度。
            max_tokens: 覆盖默认最大输出长度。
            stream: 是否流式。

        Returns:
            请求体字典。
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发起一次对话补全（非流式）。

        Args:
            messages: 消息数组。
            json_mode: 是否要求 JSON 输出。
            temperature: 温度覆盖。
            max_tokens: 最大输出 token 覆盖。

        Returns:
            :class:`LLMResponse`。

        Raises:
            NoLLMError: 未配置 Key。
            LLMError: 网络错误、鉴权失败或响应格式异常。
        """
        import asyncio
        import time

        import httpx

        client = await self._get_client()
        body = self._payload(
            messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await client.post("/chat/completions", json=body)
                if 400 <= resp.status_code < 500:
                    # 鉴权/参数错误重试无意义，直接暴露
                    raise LLMError(
                        f"LLM 请求被拒绝（HTTP {resp.status_code}）: {resp.text[:300]}"
                    )
                resp.raise_for_status()
                data = resp.json()
                break
            except LLMError:
                raise
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise LLMError(
                        f"LLM 调用失败（已重试 {attempt} 次）: {type(exc).__name__}: {exc}"
                    ) from exc
                await asyncio.sleep(0.5 * (2**attempt))
        else:  # pragma: no cover - 循环必然 break 或 raise
            raise LLMError(f"LLM 调用失败: {last_error}")

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            usage = data.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 响应格式异常: {str(data)[:300]}") from exc

        return LLMResponse(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            model=str(data.get("model", self.config.model)),
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=str(choice.get("finish_reason", "")),
            raw=data,
        )

    async def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式对话补全，逐段产出文本。

        Args:
            messages: 消息数组。
            temperature: 温度覆盖。
            max_tokens: 最大输出 token 覆盖。

        Yields:
            文本增量（delta content）。

        Raises:
            NoLLMError: 未配置 Key。
            LLMError: 网络错误或响应格式异常。
        """
        import httpx

        client = await self._get_client()
        body = self._payload(
            messages, json_mode=False, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", errors="ignore")
                    raise LLMError(f"LLM 流式请求失败（HTTP {resp.status_code}）: {text[:300]}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    choices = parsed.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except LLMError:
            raise
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 流式调用失败: {type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        """关闭底层连接（幂等）。"""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


_llm_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """获取进程级 LLM 客户端单例。

    Returns:
        :class:`LLMClient` 实例。
    """
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton
