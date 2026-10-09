"""开机/重启后的环境自检：一条命令确认 RAG + MCP 全链路是否可用。

检查项：
    1. Python 与虚拟环境
    2. Qdrant 服务与集合数据（关键：Qdrant 是常驻进程，关机后必须重启）
    3. 嵌入模型本地缓存
    4. SQLite 业务库与工单
    5. ``.env`` 中的 DEEPSEEK_API_KEY（阶段4 的 LLM 依赖）
    6. 结构化日志文件

运行::

    .venv\\Scripts\\python.exe scripts\\healthcheck.py

退出码：0 全部就绪；1 有阻塞项（会在末尾给出修复命令）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"

_blockers: list[str] = []
_warnings: list[str] = []


def report(status: str, label: str, detail: str = "") -> None:
    """打印一行检查结果并记录阻塞项。

    Args:
        status: ``OK`` / ``WARN`` / ``FAIL`` 之一。
        label: 检查项名称。
        detail: 附加说明。
    """
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    if status == FAIL:
        _blockers.append(label)
    elif status == WARN:
        _warnings.append(label)


def check_python() -> None:
    """检查 Python 版本与虚拟环境。"""
    print("\n1) Python 与虚拟环境")
    version = sys.version_info
    if version >= (3, 10):
        report(OK, f"解释器 Python {version.major}.{version.minor}.{version.micro}", sys.executable)
    else:
        report(FAIL, "Python 版本过低（需 >= 3.10）", f"当前 {version.major}.{version.minor}")

    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.is_file():
        report(OK, "虚拟环境存在", str(venv))
    else:
        report(FAIL, "虚拟环境缺失", "执行: python -m venv .venv && "
                                    ".venv\\Scripts\\python.exe -m pip install -r requirements.txt")

    for module, hint in (
        ("fastapi", "pip install -r requirements.txt"),
        ("qdrant_client", "pip install -r requirements.txt"),
        ("mcp", "pip install -r requirements.txt"),
        ("sentence_transformers", "pip install -r requirements.txt"),
        ("langgraph", "pip install -r requirements.txt"),
    ):
        try:
            __import__(module)
            report(OK, f"依赖 {module}")
        except ImportError:
            report(FAIL, f"依赖 {module} 缺失", hint)


def check_qdrant() -> None:
    """检查 Qdrant 服务、集合与已索引数据。"""
    print("\n2) Qdrant 向量库（关机后需要重启）")
    try:
        import httpx
    except ImportError:
        report(FAIL, "httpx 缺失，无法探测 Qdrant")
        return

    try:
        resp = httpx.get("http://127.0.0.1:6333/collections", timeout=6)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        report(
            FAIL,
            "Qdrant 未运行",
            f"{type(exc).__name__}；启动命令见 RESUME.md 第 2 步",
        )
        return

    names = [c["name"] for c in resp.json().get("result", {}).get("collections", [])]
    report(OK, "Qdrant 服务响应正常", f"集合: {names}")

    if "enterprise_docs" in names and "enterprise_docs_meta" in names:
        report(OK, "知识库集合已建立")
    else:
        report(FAIL, "知识库集合缺失", "执行: .venv\\Scripts\\python.exe -m rag_pipeline."
                                      "ingestion_cli --path ./docs --collection enterprise_docs")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="127.0.0.1", port=6333, timeout=10)
        points = 0
        docs = 0
        if "enterprise_docs" in names:
            points = client.count("enterprise_docs", exact=True).count
        if "enterprise_docs_meta" in names:
            docs = client.count("enterprise_docs_meta", exact=True).count
        if points > 0 and docs > 0:
            report(OK, "已索引数据完好", f"{docs} 个文档 / {points} 个片段")
        else:
            report(
                FAIL,
                "索引数据为空",
                "执行: .venv\\Scripts\\python.exe -m rag_pipeline.ingestion_cli --path ./docs",
            )
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "读取集合统计失败", f"{type(exc).__name__}: {exc}")


def check_model_cache() -> None:
    """检查 bge-m3 本地模型缓存。"""
    print("\n3) 嵌入模型缓存")
    local_dir = ROOT / ".cache" / "models" / "BAAI--bge-m3"
    if (local_dir / "config.json").is_file():
        size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
        report(OK, "bge-m3 本地缓存存在", f"{size / 1e9:.2f} GB @ {local_dir}")
    elif (ROOT / ".cache" / "huggingface" / "hub").is_dir():
        report(WARN, "本地目录缓存缺失，但有 HF 镜像缓存", "首次加载会自动重建本地目录")
    else:
        report(
            WARN,
            "模型缓存缺失",
            "首次运行会自动下载（约 2.3 GB），需设置 HF_ENDPOINT=https://hf-mirror.com",
        )


def check_business_db() -> None:
    """检查 SQLite 业务库。"""
    print("\n4) 业务数据库（SQLite）")
    db_path = ROOT / ".data" / "business.db"
    if not db_path.is_file():
        report(WARN, "业务库尚未创建", "首次调用 create_ticket/query_orders 时会自动建库")
        return
    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        tickets = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        conn.close()
        report(OK, "业务库可读写", f"订单 {orders} 条 / 工单 {tickets} 条")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "业务库读取失败", f"{type(exc).__name__}: {exc}")


def check_env() -> None:
    """检查 .env 与 DEEPSEEK_API_KEY。"""
    print("\n5) 环境变量（阶段4 的 LLM 依赖）")
    env_file = ROOT / ".env"
    if env_file.is_file():
        report(OK, ".env 存在")
    else:
        report(WARN, ".env 缺失", "执行: Copy-Item .env.example .env")
        return

    # 手动解析，避免依赖 python-dotenv 是否安装
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip().strip("'\"")

    key = values.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    if key and key.startswith("sk-") and "xxxx" not in key:
        report(OK, "DEEPSEEK_API_KEY 已配置", f"{key[:6]}…{key[-4:]}（长度 {len(key)}）")
    else:
        report(
            FAIL,
            "DEEPSEEK_API_KEY 未配置",
            "阶段4 的 Agent 与阶段5 的评测需要它；否则只能走规则式降级路径",
        )


def check_logs() -> None:
    """检查结构化日志。"""
    print("\n6) 结构化日志")
    log_file = ROOT / "logs" / "tool_calls.jsonl"
    if log_file.is_file():
        lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        last = ""
        if lines:
            try:
                rec = json.loads(lines[-1])
                last = f"最后一条 {rec.get('server')}.{rec.get('tool')} ({rec.get('timestamp')})"
            except json.JSONDecodeError:
                last = "最后一条解析失败（文件可能被并发写入）"
        report(OK, f"工具调用日志 {len(lines)} 条", last)
    else:
        report(WARN, "日志文件不存在", "运行 python -m mcp_servers.test_mcp 后会生成")


def main() -> int:
    """执行全部检查并打印结论。"""
    print("=" * 72)
    print("企业级 RAG + MCP 智能助手 —— 环境自检")
    print("=" * 72)
    check_python()
    check_qdrant()
    check_model_cache()
    check_business_db()
    check_env()
    check_logs()

    print("\n" + "=" * 72)
    if _blockers:
        print(f"结论：有 {len(_blockers)} 个阻塞项，需先修复：")
        for item in _blockers:
            print(f"  - {item}")
    else:
        print("结论：全部就绪 ✅ 可以继续阶段4")
    if _warnings:
        print(f"\n警告 {len(_warnings)} 项（不阻塞）：")
        for item in _warnings:
            print(f"  - {item}")
    print("=" * 72)
    return 1 if _blockers else 0


if __name__ == "__main__":
    sys.exit(main())
