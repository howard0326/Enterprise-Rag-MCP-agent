"""business_db MCP server —— 业务数据库工具服务（SQLite）。

暴露的工具：
    - query_orders(customer_id, date_range, top_k)   按客户与日期范围查询订单
    - create_ticket(title, description, priority)    创建工单并返回 ticket_id

安全与工程约束（每条都有明确理由）：
    1. **全参数化查询**：所有 SQL 只用 ``?`` 占位符，禁止任何字符串拼接。
       ``date_range`` 的日期先经 ``datetime.date.fromisoformat`` 解析再入库比较，
       从类型层面堵死注入面（即使有人绕过 Pydantic 校验也无效）。
    2. **异常必须收敛在工具内部**：mcp 2.x 会把工具内未捕获的异常转成
       ``UnexpectedToolError`` 抛给客户端，而客户端 task group 会因此整体失败
       （阶段1 已实测）。因此每个工具都必须 try/except 并返回
       ``{"ok": false, "error": ..., "code": ...}`` 信封。
    3. **写操作走事务**：``create_ticket`` 在事务中提交，失败回滚并返回错误信封，
       绝不留下半写状态。
    4. **stdio 传输下 stdout 只允许 JSON-RPC 报文**：日志一律走 stderr。

数据库：
    默认 ``./.data/business.db``（由 settings.yaml 的 mcp.servers[].env.BIZ_DB_PATH
    经 registry 注入）。首次启动自动建表并写入演示数据；``BIZ_DB_PATH=":memory:"``
    可切换为内存库（测试用，进程退出即销毁）。

手动自测::

    python -m mcp_servers.business_db.server
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import threading
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# MCP SDK 版本兼容层
#   mcp 1.x : from mcp.server.fastmcp import FastMCP
#   mcp 2.x : from mcp.server.mcpserver import MCPServer   （FastMCP 已改名）
# 两版的 add_tool / run(transport="stdio") 签名一致，仅构造函数参数名由
# instructions 改为 description，因此这里做成兼容导入，不锁死 SDK 大版本。
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 取决于运行环境安装的 SDK 版本
    from mcp.server.mcpserver import MCPServer as _ServerClass  # type: ignore[attr-defined]

    _SERVER_KW_DESC = "description"
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as _ServerClass  # type: ignore[no-redef]

    _SERVER_KW_DESC = "instructions"

SERVER_NAME = "business_db"
SERVER_VERSION = "0.1.0"
DEFAULT_DB_PATH = "./.data/business.db"

__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "CreateTicketParams",
    "DateRange",
    "QueryOrdersParams",
    "TicketPriority",
    "BusinessDB",
    "get_database",
    "query_orders",
    "create_ticket",
    "build_server",
    "main",
]

logger = logging.getLogger(f"mcp.{SERVER_NAME}")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# 演示数据：覆盖 6 种订单状态，便于阶段5 出题（含 ORD-001 / CUST-1001 这条主线）
_DEMO_ORDERS: list[tuple[str, str, str, float, str, str, str | None, list[dict[str, Any]]]] = [
    (
        "ORD-001", "CUST-1001", "delivered", 299.00, "CNY", "2025-03-02T10:15:00",
        "2025-03-05T14:20:00",
        [{"sku": "SKU-A1", "name": "人体工学椅", "qty": 1, "price": 299.00}],
    ),
    (
        "ORD-002", "CUST-1001", "shipped", 1580.00, "CNY", "2025-03-20T09:00:00", None,
        [{"sku": "SKU-B7", "name": "显示器 27 寸", "qty": 2, "price": 790.00}],
    ),
    (
        "ORD-003", "CUST-1002", "refunded", 68.50, "CNY", "2025-02-11T16:40:00",
        "2025-02-14T11:05:00",
        [{"sku": "SKU-C3", "name": "键盘腕托", "qty": 1, "price": 68.50}],
    ),
    (
        "ORD-004", "CUST-1002", "pending", 4200.00, "CNY", "2025-03-28T20:30:00", None,
        [{"sku": "SKU-D9", "name": "升降桌", "qty": 1, "price": 4200.00}],
    ),
    (
        "ORD-005", "CUST-1003", "paid", 129.90, "CNY", "2025-03-15T08:20:00", None,
        [{"sku": "SKU-E2", "name": "台灯", "qty": 1, "price": 129.90}],
    ),
    (
        "ORD-006", "CUST-1003", "cancelled", 899.00, "CNY", "2025-01-30T13:10:00", None,
        [{"sku": "SKU-F5", "name": "空气净化器", "qty": 1, "price": 899.00}],
    ),
    (
        "ORD-007", "CUST-1001", "refunded", 259.00, "CNY", "2025-01-12T11:00:00",
        "2025-01-15T09:30:00",
        [{"sku": "SKU-G8", "name": "收纳柜", "qty": 1, "price": 259.00}],
    ),
]

_TICKET_SEQ: dict[str, int] = {}


class TicketPriority(str, Enum):
    """工单优先级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class DateRange(BaseModel):
    """订单查询的日期范围（闭区间，按订单创建日期过滤）。"""

    start: str | None = Field(
        default=None,
        description="起始日期，格式 YYYY-MM-DD（含当天）。不传表示不限下界。",
    )
    end: str | None = Field(
        default=None,
        description="结束日期，格式 YYYY-MM-DD（含当天）。不传表示不限上界。",
    )

    @field_validator("start", "end")
    @classmethod
    def _validate_iso_date(cls, value: str | None) -> str | None:
        """校验日期格式，非法格式直接拒绝（避免脏参数进入 SQL 层）。

        Args:
            value: 待校验的日期字符串或 ``None``。

        Returns:
            原样返回合法日期字符串（``None`` 表示不限）。

        Raises:
            ValueError: 日期不符合 YYYY-MM-DD 格式。
        """
        if value is None:
            return None
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"日期格式非法: {value!r}，要求 YYYY-MM-DD") from exc
        return value


class QueryOrdersParams(BaseModel):
    """``query_orders`` 的入参（同时用于生成给 LLM 的 JSON Schema）。"""

    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="客户 ID，如 'CUST-1001'。必填，不接受模糊匹配或留空查全表。",
    )
    date_range: DateRange | None = Field(
        default=None,
        description='下单日期范围 {"start": "2025-01-01", "end": "2025-03-31"}；不传则返回该客户全部订单。',
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="返回条数上限，默认 20，防止一次拉取过多行撑爆模型上下文。",
    )


class CreateTicketParams(BaseModel):
    """``create_ticket`` 的入参。"""

    title: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="工单标题，一句话概括问题，如 '订单 ORD-001 退款未到账'。",
    )
    description: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="详细描述：现象、时间、客户诉求、已尝试的操作。Agent 应把用户原话与已知订单事实一起写入。",
    )
    priority: TicketPriority = Field(
        default=TicketPriority.MEDIUM,
        description="优先级：low | medium | high | urgent。用户明确说'紧急/高优先级'时必须提升。",
    )


def parse_date_range(value: Any) -> tuple[str | None, str | None]:
    """把多种形态的日期范围统一解析为 ``(start, end)``。

    为什么要做这层宽松解析，而不是直接用 Pydantic 模型约束参数：
        MCP 协议层会用工具签名生成 JSON Schema 并在**调用前**校验。若签名写成
        ``date_range: DateRange``，模型传 ``"2025-01-01~2025-03-31"`` 或
        ``"2025-03"`` 这类字符串时会被协议层直接拒绝（实测：返回
        ``ok=True`` + ``Error executing tool query_orders: 1 validation error``），
        我们的工具根本拿不到参数、无法给出可读的修复建议，Agent 也无从重试。
        因此签名放宽为 ``str | dict | None``，在工具内部解析并返回统一信封。

    支持的形态::

        None                                    -> (None, None)
        "2025-01-01"                            -> ("2025-01-01", "2025-01-01")
        "2025-01"                               -> ("2025-01-01", "2025-01-31")
        "2025"                                  -> ("2025-01-01", "2025-12-31")
        "2025-01-01~2025-03-31" / "~" / "至" / "到" / " - "
                                                -> ("2025-01-01", "2025-03-31")
        {"start": "2025-01-01", "end": "2025-03-31"}
        {"start": ..., "end": ...}（键名也接受 from/to/begin/end_date）

    Args:
        value: 原始入参。

    Returns:
        ``(start, end)``；两者均为 ``None`` 表示不限时间。

    Raises:
        ValueError: 无法解析或日期非法（含 start 晚于 end）。
    """
    if value is None or value == "" or value == {}:
        return None, None

    if isinstance(value, dict):
        lowered = {str(k).strip().lower(): v for k, v in value.items()}
        start_raw = lowered.get("start", lowered.get("from", lowered.get("begin")))
        end_raw = lowered.get("end", lowered.get("to", lowered.get("end_date")))
        start = _normalize_bound(str(start_raw), is_start=True) if start_raw else None
        end = _normalize_bound(str(end_raw), is_start=False) if end_raw else None
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"日期范围数组必须恰好两个元素，实际 {len(value)} 个")
        start = _normalize_bound(str(value[0]), is_start=True) if value[0] else None
        end = _normalize_bound(str(value[1]), is_start=False) if value[1] else None
    else:
        text = str(value).strip()
        for sep in ("~", "～", "至", "到", " - ", " -", "- ", "..", ","):
            if sep in text and not _looks_like_iso_date(text):
                head, _, tail = text.partition(sep)
                start = _normalize_bound(head.strip(), is_start=True) if head.strip() else None
                end = _normalize_bound(tail.strip(), is_start=False) if tail.strip() else None
                break
        else:
            # 单一日期或年月：同时作为上下界
            bound = _normalize_bound(text, is_start=True)
            start, end = bound, _normalize_bound(text, is_start=False)

    if start and end and start > end:
        raise ValueError(f"日期范围非法: start({start}) 晚于 end({end})")
    return start, end


def _looks_like_iso_date(text: str) -> bool:
    """判断字符串是否形如完整 ISO 日期（避免把 ``2025-01-01`` 按 ``-`` 拆开）。

    Args:
        text: 待判断字符串。

    Returns:
        形如 ``YYYY-MM-DD`` 返回 True。
    """
    import re

    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()))


def _normalize_bound(text: str, is_start: bool) -> str:
    """把单个边界值规范化为完整 ISO 日期。

    Args:
        text: 边界值（``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD`` / ``YYYY/MM/DD``）。
        is_start: 为 True 时向上取区间首日，否则取末日。

    Returns:
        完整 ISO 日期字符串。

    Raises:
        ValueError: 无法解析或日期非法。
    """
    from calendar import monthrange

    cleaned = text.strip().replace("/", "-").replace(".", "-").replace("年", "-").replace("月", "-")
    cleaned = cleaned.replace("日", "").strip(" -")
    if not cleaned:
        raise ValueError("日期边界为空")

    parts = cleaned.split("-")
    try:
        if len(parts) == 1 and len(parts[0]) == 4:
            year = int(parts[0])
            return f"{year:04d}-01-01" if is_start else f"{year:04d}-12-31"
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            last_day = monthrange(year, month)[1]
            return f"{year:04d}-{month:02d}-01" if is_start else f"{year:04d}-{month:02d}-{last_day:02d}"
        return date.fromisoformat("-".join(parts[:3])).isoformat()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"日期无法解析: {text!r}（支持 2025 / 2025-03 / 2025-03-01 / 区间字符串）"
        ) from exc


_VALID_STATUSES: set[str] = {
    "pending",
    "paid",
    "shipped",
    "delivered",
    "refunded",
    "cancelled",
}

_PRIORITY_ALIASES: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "urgent": "urgent",
    # 中文与常见同义写法：模型经常直接输出用户原话
    "低": "low",
    "低优先级": "low",
    "普通": "medium",
    "中": "medium",
    "中优先级": "medium",
    "一般": "medium",
    "高": "high",
    "高优先级": "high",
    "紧急": "urgent",
    "特急": "urgent",
    "最高": "urgent",
    "critical": "urgent",
    "p0": "urgent",
    "p1": "high",
    "p2": "medium",
    "p3": "low",
}


def normalize_priority(value: Any) -> str:
    """把多种写法的优先级归一化为枚举值。

    宽松解析的理由与 :func:`parse_date_range` 相同：优先级若用严格枚举约束，
    模型传"高优先级""紧急"时会被协议层直接拒绝，工具无法给出可读提示。
    但**非法值仍然拒绝**，不做静默降级 —— 静默把"urgent"降成"medium"
    会让紧急工单被埋掉，这种失败必须暴露给 Agent。

    Args:
        value: 原始优先级。

    Returns:
        归一化后的优先级字符串。

    Raises:
        ValueError: 无法识别的优先级。
    """
    if value is None or str(value).strip() == "":
        return TicketPriority.MEDIUM.value
    text = str(value).strip().lower()
    if text in _PRIORITY_ALIASES:
        return _PRIORITY_ALIASES[text]
    # 去掉"优先级/工单/级别"等修饰词再试一次
    stripped = text.replace("优先级", "").replace("级别", "").replace("工单", "").strip()
    if stripped in _PRIORITY_ALIASES:
        return _PRIORITY_ALIASES[stripped]
    raise ValueError(
        f"无法识别的优先级: {value!r}（可选 low/medium/high/urgent，"
        f"也接受 低/中/高/紧急 等中文写法）"
    )


def _ok(data: Any, **extra: Any) -> dict[str, Any]:
    """构造成功信封。

    Args:
        data: 业务数据。
        **extra: 追加到信封顶层的附加字段。

    Returns:
        形如 ``{"ok": True, "data": ..., "error": None}`` 的字典。
    """
    envelope: dict[str, Any] = {"ok": True, "data": data, "error": None}
    envelope.update(extra)
    return envelope


def _fail(error: str, *, code: str, hint: str = "") -> dict[str, Any]:
    """构造失败信封（不抛异常，让 Agent 能自行决定重试或降级）。

    Args:
        error: 面向人的错误描述。
        code: 机器可判定的错误码。
        hint: 给 Agent 的修复建议。

    Returns:
        形如 ``{"ok": False, "data": None, "error": ...}`` 的字典。
    """
    message = f"[{code}] {error}" + (f" 建议: {hint}" if hint else "")
    logger.warning("tool error: %s", message)
    return {"ok": False, "data": None, "error": message, "code": code}


class BusinessDB:
    """SQLite 业务库封装（建表、演示数据、参数化查询、工单写入）。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """初始化数据库连接并确保表结构就绪。

        Args:
            db_path: 数据库路径；``None`` 时读环境变量 ``BIZ_DB_PATH``，
                再退回 ``./.data/business.db``。``":memory:"`` 表示内存库。
        """
        raw = str(db_path or os.environ.get("BIZ_DB_PATH") or DEFAULT_DB_PATH)
        self.is_memory = raw == ":memory:"
        if self.is_memory:
            self.path = raw
        else:
            path = Path(raw)
            if not path.is_absolute():
                # 相对路径按项目根解析：MCP server 由 registry 以子进程拉起，
                # cwd 未必是项目根，写死相对路径会落到不可预期的位置。
                path = Path(__file__).resolve().parent.parent.parent / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # 建表与演示数据
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """建表并在空库时写入演示数据。"""
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id     TEXT PRIMARY KEY,
                    customer_id  TEXT NOT NULL,
                    status       TEXT NOT NULL,
                    amount       REAL NOT NULL,
                    currency     TEXT NOT NULL DEFAULT 'CNY',
                    created_at   TEXT NOT NULL,
                    delivered_at TEXT,
                    items_json   TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_orders_customer
                    ON orders (customer_id, created_at);

                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id   TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority    TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'open',
                    created_at  TEXT NOT NULL,
                    session_id  TEXT
                );
                """
            )
            count = self._conn.execute("SELECT COUNT(*) AS n FROM orders").fetchone()["n"]
            if count == 0:
                self._conn.executemany(
                    "INSERT INTO orders (order_id, customer_id, status, amount, currency,"
                    " created_at, delivered_at, items_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (oid, cid, st, amt, cur, cat, dat, json.dumps(items, ensure_ascii=False))
                        for oid, cid, st, amt, cur, cat, dat, items in _DEMO_ORDERS
                    ],
                )
                logger.info("已写入 %d 条演示订单 -> %s", len(_DEMO_ORDERS), self.path)
            self._conn.commit()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query_orders(
        self,
        customer_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        top_k: int = 20,
        status: str | None = None,
        order_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按客户、状态、订单号与日期范围查询订单（全参数化）。

        Args:
            customer_id: 客户 ID；``None`` 表示不按客户过滤。
            start: 起始日期（YYYY-MM-DD，闭区间）。
            end: 结束日期（YYYY-MM-DD，闭区间）。
            top_k: 返回条数上限。
            status: 订单状态过滤；``None`` 表示不按状态过滤。
            order_ids: 订单号列表；``None`` 表示不按订单号过滤。

        Returns:
            订单字典列表，按下单时间降序。

        Raises:
            ValueError: 日期格式非法（正常路径下由工具层提前拦截）。
        """
        if start is not None:
            date.fromisoformat(start)  # 类型层校验，绝不把原始串带入 SQL
        if end is not None:
            date.fromisoformat(end)

        # 全部条件用 ? 占位符绑定，"IS NULL OR" 模式让同一段 SQL 支持任意条件组合，
        # 无需按参数拼字符串（拼字符串正是 SQL 注入的来源）。
        # 订单号是列表：用 IN (?,?,...) 并把每个值独立绑定。
        order_clause = ""
        order_params: list[Any] = []
        if order_ids:
            placeholders = ",".join("?" for _ in order_ids)
            order_clause = f" AND order_id IN ({placeholders})"
            order_params = [str(x).upper() for x in order_ids]

        sql = (
            "SELECT order_id, customer_id, status, amount, currency, created_at,"
            "       delivered_at, items_json"
            "  FROM orders"
            " WHERE (? IS NULL OR customer_id = ?)"
            "   AND (? IS NULL OR status = ?)"
            "   AND (? IS NULL OR date(created_at) >= date(?))"
            "   AND (? IS NULL OR date(created_at) <= date(?))"
            + order_clause
            + " ORDER BY date(created_at) DESC, order_id DESC"
            " LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(
                sql,
                (
                    customer_id,
                    customer_id,
                    status,
                    status,
                    start,
                    start,
                    end,
                    end,
                    *order_params,
                    int(top_k),
                ),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            try:
                items = json.loads(row["items_json"] or "[]")
            except json.JSONDecodeError:
                items = []
            results.append(
                {
                    "order_id": row["order_id"],
                    "customer_id": row["customer_id"],
                    "status": row["status"],
                    "amount": float(row["amount"]),
                    "currency": row["currency"],
                    "created_at": row["created_at"],
                    "delivered_at": row["delivered_at"],
                    "items": items,
                }
            )
        return results

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        """按订单号精确查询单条订单。

        Args:
            order_id: 订单号，如 ``ORD-001``。

        Returns:
            订单字典；不存在时 ``None``。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT customer_id, status, amount, currency, created_at, delivered_at,"
                "       items_json FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            items = json.loads(row["items_json"] or "[]")
        except json.JSONDecodeError:
            items = []
        return {
            "order_id": order_id,
            "customer_id": row["customer_id"],
            "status": row["status"],
            "amount": float(row["amount"]),
            "currency": row["currency"],
            "created_at": row["created_at"],
            "delivered_at": row["delivered_at"],
            "items": items,
        }

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def create_ticket(
        self,
        title: str,
        description: str,
        priority: str = "medium",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """创建工单并返回工单号（事务写入，失败回滚）。

        Args:
            title: 工单标题。
            description: 详细描述。
            priority: 优先级（``low`` / ``medium`` / ``high`` / ``urgent``）。
            session_id: 来源会话 ID（可选，便于回溯对话上下文）。

        Returns:
            工单记录字典。

        Raises:
            sqlite3.Error: 写入失败（事务已回滚）。
        """
        now = datetime.now().astimezone()
        day = now.strftime("%Y%m%d")
        created_at = now.isoformat(timespec="seconds")

        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM tickets WHERE ticket_id LIKE ?", (f"TKT-{day}-%",)
            )
            seq = int(cur.fetchone()["n"]) + 1
            ticket_id = f"TKT-{day}-{seq:04d}"
            try:
                self._conn.execute(
                    "INSERT INTO tickets (ticket_id, title, description, priority, status,"
                    " created_at, session_id) VALUES (?, ?, ?, ?, 'open', ?, ?)",
                    (ticket_id, title, description, priority, created_at, session_id),
                )
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise
        return {
            "ticket_id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "status": "open",
            "created_at": created_at,
            "session_id": session_id,
        }

    def list_tickets(self, limit: int = 10) -> list[dict[str, Any]]:
        """列出最近创建的工单（供测试与排障）。

        Args:
            limit: 返回条数上限。

        Returns:
            工单字典列表，按创建时间降序。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT ticket_id, title, priority, status, created_at, session_id"
                "  FROM tickets ORDER BY created_at DESC, ticket_id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass


_db_singleton: BusinessDB | None = None
_db_lock = threading.Lock()


def get_database() -> BusinessDB:
    """获取进程级数据库单例。

    Returns:
        :class:`BusinessDB` 实例。
    """
    global _db_singleton
    if _db_singleton is None:
        with _db_lock:
            if _db_singleton is None:
                _db_singleton = BusinessDB()
    return _db_singleton


# ---------------------------------------------------------------------------
# MCP 工具
# ---------------------------------------------------------------------------
def query_orders(
    customer_id: str = "",
    date_range: Any = None,
    top_k: int = 20,
    status: str = "",
    order_ids: Any = None,
) -> dict[str, Any]:
    """按客户、订单号、状态与日期范围查询订单。

    参数签名刻意放宽（``date_range`` / ``order_ids`` 均为 ``Any``）以便在工具内部
    做容错解析并返回可读错误信封，原因见 :func:`parse_date_range` 的说明。

    ``customer_id`` / ``status`` / ``order_ids`` **至少提供一个**：三者分别对应
    "查某客户的订单""查某类状态的订单""查某个具体订单"。这是评测暴露的真实需求 ——
    用户经常只说订单号（"订单 ORD-001 什么状态"），若强制要 customer_id 就得先
    反查客户，多一次往返且容易猜错。保留"至少一个"是为了禁止无条件拉全表。

    Args:
        customer_id: 客户 ID，如 ``CUST-1001``。
        date_range: 下单日期范围；支持 dict / "2025-01-01~2025-03-31" / "2025-03"
            / "2025" 等形态；``None`` 表示不限时间。
        top_k: 返回条数上限，防止一次拉取过多行撑爆模型上下文。
        status: 订单状态过滤：``pending`` / ``paid`` / ``shipped`` / ``delivered``
            / ``refunded`` / ``cancelled``；``""`` 表示不限。
        order_ids: 订单号或订单号列表，如 ``"ORD-001"`` 或 ``["ORD-001","ORD-002"]``。

    Returns:
        统一信封。成功时 ``data`` 为订单列表，并附带 ``count``、``customer_id``、
        ``status`` 与规范化后的 ``date_range``；参数非法返回 ``INVALID_ARGUMENT``；
        数据库不可用返回 ``DB_UNAVAILABLE``。
    """
    customer = (customer_id or "").strip()
    status_filter = (status or "").strip().lower()
    if status_filter in ("all", "any", "全部", "不限"):
        status_filter = ""

    # 订单号容错解析：字符串（可含逗号/顿号/空格分隔）或列表
    wanted_orders: list[str] = []
    if isinstance(order_ids, (list, tuple)):
        wanted_orders = [str(x).strip().upper() for x in order_ids if str(x).strip()]
    elif isinstance(order_ids, str) and order_ids.strip():
        import re as _re

        wanted_orders = [
            part.upper()
            for part in _re.split(r"[,，、;\s]+", order_ids.strip())
            if part.strip()
        ]

    if status_filter and status_filter not in _VALID_STATUSES:
        return _fail(
            f"订单状态非法: {status!r}",
            code="INVALID_ARGUMENT",
            hint=f"可选: {', '.join(sorted(_VALID_STATUSES))}",
        )
    if not customer and not status_filter and not wanted_orders:
        return _fail(
            "customer_id、status、order_ids 至少需要提供一个",
            code="MISSING_ARGUMENT",
            hint="给出客户 ID（CUST-xxxx）、订单状态（如 refunded）或订单号（ORD-xxx）；"
            "不允许无条件拉取全表。",
        )
    if len(customer) > 64:
        return _fail("customer_id 过长（上限 64 字符）", code="INVALID_ARGUMENT")

    try:
        limit = int(top_k)
    except (TypeError, ValueError):
        return _fail(f"top_k 必须是整数，实际 {top_k!r}", code="INVALID_ARGUMENT")
    if not 1 <= limit <= 200:
        return _fail(f"top_k 必须在 1~200 之间，实际 {limit}", code="INVALID_ARGUMENT")

    try:
        start, end = parse_date_range(date_range)
    except ValueError as exc:
        return _fail(
            str(exc),
            code="INVALID_ARGUMENT",
            hint='date_range 可写 {"start": "2025-01-01", "end": "2025-03-31"} '
            '或 "2025-01-01~2025-03-31"，日期须为 YYYY-MM-DD。',
        )

    try:
        rows = get_database().query_orders(
            customer or None,
            start=start,
            end=end,
            top_k=limit,
            status=status_filter or None,
            order_ids=wanted_orders or None,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"订单查询失败: {type(exc).__name__}: {exc}",
            code="DB_UNAVAILABLE",
            hint="请确认 BIZ_DB_PATH 指向的数据库文件可读写。",
        )

    logger.info(
        "query_orders customer=%s status=%s orders=%s range=[%s, %s] -> %d 条",
        customer or "*",
        status_filter or "*",
        wanted_orders or "*",
        start,
        end,
        len(rows),
    )
    if wanted_orders and not rows:
        return _fail(
            f"未找到订单: {', '.join(wanted_orders)}",
            code="NOT_FOUND",
            hint="请确认订单号是否正确（演示数据为 ORD-001 ~ ORD-007）。",
        )
    return _ok(
        rows,
        count=len(rows),
        customer_id=customer or None,
        status=status_filter or None,
        order_ids=wanted_orders or None,
        date_range={"start": start, "end": end},
    )

    try:
        limit = int(top_k)
    except (TypeError, ValueError):
        return _fail(f"top_k 必须是整数，实际 {top_k!r}", code="INVALID_ARGUMENT")
    if not 1 <= limit <= 200:
        return _fail(f"top_k 必须在 1~200 之间，实际 {limit}", code="INVALID_ARGUMENT")

    try:
        start, end = parse_date_range(date_range)
    except ValueError as exc:
        return _fail(
            str(exc),
            code="INVALID_ARGUMENT",
            hint='date_range 可写 {"start": "2025-01-01", "end": "2025-03-31"} '
            '或 "2025-01-01~2025-03-31"，日期须为 YYYY-MM-DD。',
        )

    try:
        rows = get_database().query_orders(customer, start=start, end=end, top_k=limit)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"订单查询失败: {type(exc).__name__}: {exc}",
            code="DB_UNAVAILABLE",
            hint="请确认 BIZ_DB_PATH 指向的数据库文件可读写。",
        )

    logger.info(
        "query_orders customer=%s range=[%s, %s] -> %d 条", customer, start, end, len(rows)
    )
    return _ok(
        rows,
        count=len(rows),
        customer_id=customer,
        date_range={"start": start, "end": end},
    )


def create_ticket(title: str, description: str, priority: Any = "medium") -> dict[str, Any]:
    """创建工单并返回 ``ticket_id``。

    Args:
        title: 工单标题（2~120 字符）。
        description: 详细描述（2~4000 字符）。
        priority: 优先级，``low`` | ``medium`` | ``high`` | ``urgent``；
            也接受"高优先级""紧急"等中文写法（见 :func:`normalize_priority`）。

    Returns:
        统一信封。成功时 ``data`` 为工单记录；参数非法返回 ``INVALID_ARGUMENT``；
        写入失败返回 ``DB_UNAVAILABLE``（事务已回滚）。
    """
    clean_title = (title or "").strip()
    clean_desc = (description or "").strip()
    if len(clean_title) < 2:
        return _fail(
            f"title 至少 2 个字符，实际 {len(clean_title)}",
            code="INVALID_ARGUMENT",
            hint="标题应含订单号或客户 ID，如 '订单 ORD-001 退款未到账'。",
        )
    if len(clean_title) > 120:
        return _fail("title 超过 120 字符", code="INVALID_ARGUMENT")
    if len(clean_desc) < 2:
        return _fail(
            "description 至少 2 个字符",
            code="INVALID_ARGUMENT",
            hint="描述应包含现象、时间、影响面与已尝试的操作。",
        )
    if len(clean_desc) > 4000:
        return _fail("description 超过 4000 字符", code="INVALID_ARGUMENT")

    try:
        normalized = normalize_priority(priority)
    except ValueError as exc:
        return _fail(str(exc), code="INVALID_ARGUMENT")

    try:
        record = get_database().create_ticket(
            title=clean_title, description=clean_desc, priority=normalized
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"工单创建失败（已回滚）: {type(exc).__name__}: {exc}",
            code="DB_UNAVAILABLE",
            hint="请确认数据库文件未被其他进程独占锁定。",
        )

    logger.info(
        "create_ticket -> %s priority=%s title=%r",
        record["ticket_id"],
        record["priority"],
        record["title"],
    )
    return _ok(record)


def build_server() -> Any:
    """创建并装配 business_db MCP server。

    Returns:
        已注册全部工具的 MCP server 实例（mcp 1.x 为 FastMCP，2.x 为 MCPServer）。
    """
    server = _ServerClass(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        **{
            _SERVER_KW_DESC: (
                "企业业务数据库服务。适用场景：订单状态/金额/时间查询、创建客服工单。"
                "查询前若缺少 customer_id 或订单号，先向用户确认，不要猜测参数。"
                "创建工单属于写操作，必须先从用户话语中抽取出标题、描述与优先级。"
            )
        },
    )

    server.add_tool(
        query_orders,
        name="query_orders",
        title="查询订单",
        description=(
            "按 customer_id、status 与可选 date_range 查询订单，返回订单号、状态、金额、"
            "下单/签收时间、商品明细。customer_id 与 status 至少提供一个："
            "问某客户的订单用 customer_id；问某类状态的订单（如'哪些订单退款了'）用 status。"
        ),
        structured_output=False,
    )
    server.add_tool(
        create_ticket,
        name="create_ticket",
        title="创建工单",
        description=(
            "创建客服/运维工单并返回 ticket_id。属于写操作，仅在用户明确要求"
            "'建单/报障/升级处理'时调用。"
        ),
        structured_output=False,
    )
    logger.info("server '%s' v%s 装配完成，已注册 2 个工具", SERVER_NAME, SERVER_VERSION)
    return server


def main() -> None:
    """以 stdio 传输启动 server。"""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
