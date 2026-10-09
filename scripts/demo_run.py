"""演示脚本：依次跑四类问题，逐步展示 Agent 的决策过程与工具调用。

用法：
    .venv\\Scripts\\python.exe scripts\\demo_run.py              # 全部 6 个场景
    .venv\\Scripts\\python.exe scripts\\demo_run.py --scenario 3  # 只跑第 3 个

演示节奏（建议照着讲）：
    1 纯知识问答  → 展示 RAG 检索 + 来源引用
    2 纯数据查询  → 展示 MCP 工具调用
    3 混合问题    → 展示多工具编排（这是核心亮点）
    4 操作请求    → 展示写操作（真实建单）
    5 追问边界    → 展示"信息不足时如实说明"而非编造
    6 多轮对话    → 展示会话上下文
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"
SESSION = "demo-live"

SCENARIOS: list[dict[str, str]] = [
    {
        "title": "纯知识问答 —— RAG 检索 + 来源引用",
        "say": "先问一个制度类问题。注意它会调用知识库检索工具，并给出出处。",
        "message": "退货政策是什么？",
    },
    {
        "title": "纯数据查询 —— MCP 工具调用",
        "say": "这次只查业务数据，不走知识库。注意工具换成了 query_orders。",
        "message": "查一下 CUST-1001 的所有订单。",
    },
    {
        "title": "混合问题 —— 多工具编排（核心亮点）",
        "say": "这个问题的关键在于：它需要【先查订单事实，再查政策条款】，"
               "两个工具都要调。注意模型的回答方式——它没有直接下结论，"
               "而是列出事实、引用条款、然后说明还缺什么信息。",
        "message": "订单 ORD-001 符合退货政策吗？",
    },
    {
        "title": "操作请求 —— 写操作，真实建单",
        "say": "注意这会产生真实数据。它会建单并返回工单号。"
               "同时它还会先查订单拿上下文，让工单描述更准确。",
        "message": "帮我给订单 ORD-001 创建一个高优先级工单，退款一直没到账。",
    },
    {
        "title": "边界：信息不足时如实说明",
        "say": "故意问一个没有足够信息就能回答的问题。"
               "关键看它【会不会编】——正确行为是追问而不是瞎答。",
        "message": "我那个订单能退吗？",
    },
    {
        "title": "多轮对话 —— 会话上下文",
        "say": "接着上一轮继续问，看它能否利用上下文。",
        "message": "那它的状态是什么？",
    },
]


def section(n: int, total: int, title: str) -> None:
    """打印场景标题。"""
    print("\n" + "=" * 72)
    print(f"  场景 {n}/{total}：{title}")
    print("=" * 72)


def show(response: dict, elapsed: float, verbose: bool = True) -> None:
    """打印一次对话的结构化结果。"""
    intents = ", ".join(response.get("intents", []))
    print(f"\n  ▸ 意图识别   : {intents}（走 {response.get('intent_source')} 路径）")

    plan = response.get("plan", [])
    if plan:
        print("  ▸ 执行计划   :")
        for step in plan:
            tool = step.get("tool") or step.get("action")
            print(f"      {step['action']:10s} → {tool:18s} {step.get('reason', '')}")

    tools = response.get("tool_calls", [])
    if tools:
        print("  ▸ 工具调用   :")
        for t in tools:
            mark = "成功" if t.get("ok") else f"失败({t.get('code')})"
            attempts = int(t.get("attempts") or 1)
            # attempts 是"总尝试次数"：1 = 首次成功，2 = 失败后重试过一次
            retry_note = "首次成功" if attempts <= 1 else f"重试 {attempts - 1} 次后成功"
            print(
                f"      {t['tool']:20s} {mark:16s} "
                f"{t.get('latency_ms', 0):>7.1f}ms  {retry_note}"
            )

    sources = response.get("sources", [])
    if sources:
        print("  ▸ 引用来源   :")
        for s in sources:
            page = f" 第{s['page']}页" if s.get("page") else ""
            print(f"      《{s['source']}》{page}  相关度 {s.get('score')}")

    degraded = response.get("degraded")
    if degraded:
        print(f"  ▸ 降级标记   : ⚠️ {'；'.join(response.get('degraded_reasons', [])[:1])}")

    metrics = response.get("metrics", {})
    print(
        f"  ▸ 耗时/token : {elapsed:.2f}s   "
        f"prompt={metrics.get('prompt_tokens')} completion={metrics.get('completion_tokens')}"
    )

    if verbose:
        print("\n  ┌─ 回答 " + "─" * 60)
        for ln in str(response.get("response", "")).splitlines():
            print(f"  │ {ln}")
        print("  └" + "─" * 66)


def main() -> int:
    """跑演示场景。"""
    parser = argparse.ArgumentParser(description="RAG + MCP 助手演示")
    parser.add_argument("--scenario", type=int, default=0, help="只跑第 N 个场景（1 起）")
    parser.add_argument("--session", default=SESSION, help="会话 ID")
    parser.add_argument("--wait", action="store_true", help="每个场景后等待回车")
    args = parser.parse_args()

    try:
        health = httpx.get(f"{BASE}/health", timeout=20).json()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 后端不可达（{type(exc).__name__}）")
        print("       请先执行: powershell -ExecutionPolicy Bypass -File scripts\\start_demo.ps1")
        return 1

    print("=" * 72)
    print(" 企业级 RAG + MCP 智能助手 —— 现场演示")
    print(f" 服务状态: {health['status']}   LLM: {health['llm']['model']}   "
          f"意图识别: {health['llm']['intent_source']}")
    print("=" * 72)

    targets = (
        [SCENARIOS[args.scenario - 1]]
        if 1 <= args.scenario <= len(SCENARIOS)
        else SCENARIOS
    )
    offset = args.scenario - 1 if args.scenario else 0

    for idx, scenario in enumerate(targets, 1 + offset):
        section(idx, len(SCENARIOS), scenario["title"])
        print(f"\n  【讲解】{scenario['say']}")
        print(f"\n  【提问】{scenario['message']}")
        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{BASE}/chat",
                json={"message": scenario["message"], "session_id": args.session},
                timeout=120,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [FAIL] {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        show(payload, elapsed)

        if args.wait and idx < len(SCENARIOS):
            input("\n  >>> 按回车继续下一个场景…")

    print("\n" + "=" * 72)
    print(" 演示结束。可继续在 http://127.0.0.1:8000/docs 手动试其他问题。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
