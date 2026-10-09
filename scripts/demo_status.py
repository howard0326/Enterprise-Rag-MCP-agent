"""演示用状态面板：一屏展示 LLM 状态、MCP 工具目录、索引数据与会话数。

用法：.venv\\Scripts\\python.exe scripts\\demo_status.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"


def line(char: str = "=", n: int = 68) -> None:
    """打印分隔线。"""
    print(char * n)


def main() -> int:
    """输出演示状态面板。"""
    line()
    print(" 企业级 RAG + MCP 智能助手 —— 运行状态")
    line()

    try:
        health = httpx.get(f"{BASE}/health", timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] 后端不可达: {type(exc).__name__}")
        print("       请先执行: powershell -ExecutionPolicy Bypass -File scripts\\start_demo.ps1")
        return 1

    llm = health["llm"]
    print(f"\n【1】总体状态 : {health['status']}")
    print(f"     LLM      : {llm['model']}")
    print(f"     意图识别  : 走 {llm['intent_source']} 路径"
          f"（{'模型分类' if llm['intent_source'] == 'llm' else '规则兜底'}）")
    print(f"     会话数    : {health['sessions']}")

    print("\n【2】MCP 工具服务器")
    for server in health["servers"]:
        mark = "OK  " if server["ok"] else "FAIL"
        tools = ", ".join(server["tools"]) if server["tools"] else "-"
        print(f"     [{mark}] {server['server']:16s} {server['latency_ms']:>7.1f}ms  {tools}")

    print("\n【3】工具 JSON Schema（模型选择工具的依据）")
    try:
        catalog = httpx.get(f"{BASE}/tools", timeout=30).json()
        for name, meta in sorted(catalog["tools"].items()):
            schema = meta.get("input_schema") or {}
            required = schema.get("required", [])
            props = list((schema.get("properties") or {}).keys())
            print(f"     {name}")
            print(f"        server   : {meta['server']}")
            print(f"        required : {required}")
            print(f"        props    : {props}")
    except Exception as exc:  # noqa: BLE001
        print(f"     读取失败: {exc}")

    print("\n【4】知识库索引")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="127.0.0.1", port=6333, timeout=10)
        names = [c.name for c in client.get_collections().collections]
        print(f"     集合     : {names}")
        if "enterprise_docs" in names:
            print(f"     片段数   : {client.count('enterprise_docs', exact=True).count}")
        if "enterprise_docs_meta" in names:
            print(f"     文档数   : {client.count('enterprise_docs_meta', exact=True).count}")
    except Exception as exc:  # noqa: BLE001
        print(f"     读取失败: {type(exc).__name__}: {exc}")

    print("\n【5】业务数据库")
    try:
        import sqlite3

        conn = sqlite3.connect(str(ROOT / ".data" / "business.db"))
        orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        tickets = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        rows = conn.execute(
            "SELECT order_id, customer_id, status FROM orders ORDER BY order_id LIMIT 3"
        ).fetchall()
        conn.close()
        print(f"     订单     : {orders} 条")
        print(f"     工单     : {tickets} 条")
        print(f"     示例     : {[f'{r[0]}/{r[1]}/{r[2]}' for r in rows]}")
    except Exception as exc:  # noqa: BLE001
        print(f"     读取失败: {exc}")

    line()
    print(" 演示页面：")
    print("   Qdrant 控制台 : http://127.0.0.1:6333/dashboard")
    print("   接口文档      : http://127.0.0.1:8000/docs")
    print("   健康检查      : http://127.0.0.1:8000/health")
    line()
    return 0


if __name__ == "__main__":
    sys.exit(main())
