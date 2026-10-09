"""MCP 工具服务器端到端测试。

测试内容（全部走真实 stdio 协议，不是进程内直调）：
    1. 注册表装载 settings.yaml 的 server 配置并逐个握手；
    2. ``list_tools`` 聚合工具目录，校验工具名与 JSON Schema 关键字段；
    3. 对 4 个工具各调用一次，校验统一信封 ``{"ok", "data", "error"}`` 与数据字段；
    4. 负路径：不存在的 doc_id、非法日期、被禁用的工具 —— 校验错误码；
    5. 结构化日志落盘校验（``logs/tool_calls.jsonl``）。

运行::

    python -m mcp_servers.test_mcp

退出码：0 全部通过，1 存在失败用例。

沙箱注意：stdio 传输依赖命名管道，受限环境（如 DSH workspace-write）下
``CreateFile`` 会以 WinError 5 被拒，需在普通终端或提权后运行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_servers.registry import (  # noqa: E402
    MCPRegistryError,
    MCPServerRegistry,
    ToolCallResult,
    ToolDisabledError,
    ToolNotAvailableError,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """断言并打印结果。

    Args:
        label: 用例描述。
        condition: 断言条件。
        detail: 附加信息（失败时展示）。
    """
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label} {detail}")
        _failures.append(label)


def check_fields(label: str, payload: Any, fields: list[str]) -> None:
    """校验字典包含指定字段。

    Args:
        label: 用例描述。
        payload: 待校验对象。
        fields: 必需字段名列表。
    """
    if not isinstance(payload, dict):
        check(f"{label}（应为 dict）", False, f"实际类型 {type(payload).__name__}")
        return
    missing = [f for f in fields if f not in payload]
    check(label, not missing, f"缺失字段 {missing}")


def section(title: str) -> None:
    """打印分节标题。"""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(result: ToolCallResult, limit: int = 110) -> None:
    """打印一次调用的摘要。

    Args:
        result: 调用结果。
        limit: 输出截断长度。
    """
    payload = json.dumps(result.data, ensure_ascii=False) if result.data is not None else None
    text = payload if payload is not None else (result.error or "")
    print(f"    {INFO} {result.server}.{result.tool} ok={result.ok} "
          f"code={result.code} {result.latency_ms:.0f}ms")
    print(f"        -> {text[:limit]}")


def run_suite(registry: MCPServerRegistry) -> int:
    """执行全部测试并返回退出码（注册表由调用方负责创建与关闭）。

    Args:
        registry: 已初始化的注册表实例。

    Returns:
        0 全部通过，1 存在失败用例。
    """
    section("1) 注册表装载与握手")
    print(f"  {INFO} 已启用 server: {registry.enabled_server_names()}")
    print(f"  {INFO} 已禁用 server: {sorted(registry.disabled_servers)}")

    health = registry.health_check()
    for entry in health:
        status = PASS if entry["ok"] else FAIL
        print(
            f"  {status} {entry['server']:16s} tools={entry['tools']} "
            f"({entry['latency_ms']:.0f}ms) {entry['error'] or ''}"
        )
        if not entry["ok"]:
            _failures.append(f"handshake:{entry['server']}")

    section("2) 工具目录聚合")
    catalog = registry.list_tools()
    print(f"  {INFO} 工具总数: {len(catalog)} -> {sorted(catalog)}")
    expected_tools = {"search_documents", "get_document_metadata", "query_orders", "create_ticket"}
    check("工具目录包含 4 个预期工具", set(catalog) == expected_tools,
          f"实际 {sorted(catalog)}")
    for name, meta in sorted(catalog.items()):
        schema = meta.get("input_schema") or {}
        props = sorted((schema.get("properties") or {}).keys())
        required = schema.get("required") or []
        print(f"    {name:22s} server={meta['server']:15s} required={required} props={props}")

    section("3) 工具调用（每个工具一次）")
    kb = registry.call_tool("knowledge_base", "search_documents",
                            {"query": "退货政策是什么", "top_k": 3})
    show(kb)
    check("search_documents 调用成功", kb.ok, kb.error or "")
    if kb.ok:
        hits = kb.data
        check("search_documents 返回非空列表", isinstance(hits, list) and len(hits) > 0)
        if isinstance(hits, list) and hits:
            check_fields(
                "命中字段完整（content/source/score/page/doc_id）",
                hits[0],
                ["content", "source", "score", "page", "doc_id"],
            )
            check("score 在 [0,1] 区间",
                  all(0.0 <= float(h.get("score", -1)) <= 1.0 for h in hits),
                  f"实际 {[h.get('score') for h in hits]}")
            check("score 降序", [h["score"] for h in hits] == sorted(
                (h["score"] for h in hits), reverse=True))
            print(f"    {INFO} Top1 来源={hits[0]['source']} 分数={hits[0]['score']}")

    meta_id = (kb.data[0]["doc_id"] if kb.ok and kb.data else "")
    md = registry.call_tool("knowledge_base", "get_document_metadata", {"doc_id": meta_id})
    show(md)
    check("get_document_metadata 调用成功", md.ok, md.error or "")
    if md.ok:
        check_fields(
            "元数据字段完整",
            md.data,
            ["doc_id", "title", "source", "content_type", "page_count", "chunk_count",
             "created_at", "file_hash"],
        )

    orders = registry.call_tool("business_db", "query_orders",
                                {"customer_id": "CUST-1001",
                                 "date_range": {"start": "2025-03-01", "end": "2025-03-31"}})
    show(orders)
    check("query_orders 调用成功", orders.ok, orders.error or "")
    if orders.ok:
        rows = orders.data
        check("query_orders 返回非空列表", isinstance(rows, list) and len(rows) > 0)
        if isinstance(rows, list) and rows:
            check_fields(
                "订单字段完整",
                rows[0],
                ["order_id", "customer_id", "status", "amount", "created_at", "items"],
            )
            ids = [r["order_id"] for r in rows]
            # 3 月区间应含 ORD-001(03-02)/ORD-002(03-20)，排除 ORD-007(01-12)
            check("日期范围过滤生效（3 月区间不含 ORD-007）",
                  "ORD-007" not in ids, f"实际 {ids}")
            check("包含主线订单 ORD-001", "ORD-001" in ids, f"实际 {ids}")
            check("返回条数与 count 字段一致",
                  len(rows) == int(orders.raw.get("count", -1)),
                  f"len={len(rows)} count={orders.raw.get('count')}")
            print(f"    {INFO} 3 月命中订单: {ids}")
            # 不限时间时应拿到 3 条（ORD-001/002/007）
            all_orders = registry.call_tool("business_db", "query_orders",
                                            {"customer_id": "CUST-1001"})
            all_ids = [r["order_id"] for r in all_orders.data] if all_orders.ok else []
            check("不限日期时返回该客户全部 3 条订单",
                  sorted(all_ids) == ["ORD-001", "ORD-002", "ORD-007"], f"实际 {all_ids}")

    ticket = registry.call_tool("business_db", "create_ticket",
                                {"title": "订单 ORD-001 退款未到账",
                                 "description": "客户反馈 2025-03-05 签收后申请退货，质检已通过但退款 5 天未到账。",
                                 "priority": "high"})
    show(ticket)
    check("create_ticket 调用成功", ticket.ok, ticket.error or "")
    if ticket.ok:
        check_fields("工单字段完整", ticket.data,
                     ["ticket_id", "title", "priority", "status", "created_at"])
        check("工单号为 TKT- 前缀", str(ticket.data.get("ticket_id", "")).startswith("TKT-"),
              str(ticket.data.get("ticket_id")))
        check("工单状态为 open", ticket.data.get("status") == "open")
        print(f"    {INFO} ticket_id = {ticket.data.get('ticket_id')}")

    section("4) 负路径与错误信封")
    bad_doc = registry.call_tool("knowledge_base", "get_document_metadata",
                                 {"doc_id": "0000000000000000"})
    show(bad_doc)
    check("不存在的 doc_id 返回 NOT_FOUND", (not bad_doc.ok) and bad_doc.code == "NOT_FOUND",
          f"code={bad_doc.code}")

    bad_date = registry.call_tool("business_db", "query_orders",
                                  {"customer_id": "CUST-1001",
                                   "date_range": {"start": "2025/13/45"}})
    show(bad_date)
    check("非法日期返回 INVALID_ARGUMENT",
          (not bad_date.ok) and bad_date.code == "INVALID_ARGUMENT", f"code={bad_date.code}")

    # 日期范围的宽松解析：字符串区间与只写年月
    loose = registry.call_tool("business_db", "query_orders",
                               {"customer_id": "CUST-1001", "date_range": "2025-03-01~2025-03-31"})
    show(loose)
    loose_ids = [r["order_id"] for r in loose.data] if loose.ok else []
    check("字符串区间 '2025-03-01~2025-03-31' 解析正确",
          loose.ok and sorted(loose_ids) == ["ORD-001", "ORD-002"],
          f"ok={loose.ok} ids={loose_ids}")
    month = registry.call_tool("business_db", "query_orders",
                               {"customer_id": "CUST-1001", "date_range": "2025-01"})
    month_ids = [r["order_id"] for r in month.data] if month.ok else []
    check("只写年月 '2025-01' 解析为整月", month.ok and month_ids == ["ORD-007"],
          f"ok={month.ok} ids={month_ids}")

    cn_priority = registry.call_tool("business_db", "create_ticket",
                                     {"title": "工单中文优先级测试", "description": "验证中文优先级归一化",
                                      "priority": "高优先级"})
    show(cn_priority)
    check("中文优先级 '高优先级' 归一化为 high",
          cn_priority.ok and cn_priority.data.get("priority") == "high",
          f"ok={cn_priority.ok} priority={cn_priority.data.get('priority') if cn_priority.ok else None}")

    bad_priority = registry.call_tool("business_db", "create_ticket",
                                      {"title": "测试工单", "description": "测试描述", "priority": "super"})
    show(bad_priority)
    check("非法优先级返回 INVALID_ARGUMENT",
          (not bad_priority.ok) and bad_priority.code == "INVALID_ARGUMENT",
          f"code={bad_priority.code}")

    empty_query = registry.call_tool("knowledge_base", "search_documents", {"query": ""})
    show(empty_query)
    check("空 query 返回 INVALID_ARGUMENT",
          (not empty_query.ok) and empty_query.code == "INVALID_ARGUMENT",
          f"code={empty_query.code}")

    try:
        registry.call_tool("not_exist_server", "any_tool", {})
        check("未注册 server 抛 ServerNotRegisteredError", False, "未抛异常")
    except MCPRegistryError as exc:
        check("未注册 server 抛 ServerNotRegisteredError",
              "未注册" in str(exc) or "ServerNotRegistered" in type(exc).__name__, str(exc))

    # 未暴露的工具属于"调用方用错了工具名"，属于编程错误而非业务错误，
    # 因此设计上抛 ToolNotAvailableError（让阶段4 的 Agent 立刻发现目录不一致），
    # 而不是返回业务错误信封。
    try:
        registry.call_tool("business_db", "drop_table", {})
        check("未暴露工具抛 ToolNotAvailableError", False, "未抛异常")
    except ToolNotAvailableError as exc:
        check("未暴露工具抛 ToolNotAvailableError", "未暴露" in str(exc), str(exc))

    section("5) 工具启用/禁用开关")
    from config.config_loader import get_config

    cfg = get_config()
    first_server = cfg.mcp.servers[0]
    first_tool = first_server.tools[0]
    check(
        f"配置读取到 {first_server.name}.{first_tool.name} 为启用",
        registry.is_tool_enabled(first_server.name, first_tool.name),
    )
    check("未注册 server 的工具开关判定为禁用",
          not registry.is_tool_enabled("not_exist_server", "any"))
    print(f"  {INFO} 禁用开关生效路径：registry.call_tool 会在建立会话前直接拒绝")

    section("6) 结构化日志落盘")
    log_path = ROOT / "logs" / "tool_calls.jsonl"
    check("日志文件已生成", log_path.is_file(), str(log_path))
    if log_path.is_file():
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        check("日志含本次调用记录", len(lines) >= 5, f"仅 {len(lines)} 行")
        last = json.loads(lines[-1])
        check_fields("日志记录字段完整", last,
                     ["timestamp", "server", "tool", "args", "result", "latency_ms"])
        print(f"  {INFO} 最后一条: {json.dumps(last, ensure_ascii=False)[:150]}")

    section("汇总")
    print(f"  失败用例数: {len(_failures)}")
    for item in _failures:
        print(f"    {FAIL} {item}")
    if not _failures:
        print("\n  全部测试通过 ✅")
        return 0
    print("\n  存在失败用例 ❌")
    return 1


if __name__ == "__main__":
    _registry = MCPServerRegistry()
    try:
        _code = run_suite(_registry)
    finally:
        _registry.shutdown()
    sys.exit(_code)
