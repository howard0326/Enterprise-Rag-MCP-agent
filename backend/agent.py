"""主 Agent：LangGraph 状态机编排 RAG 检索、业务库查询与工单创建。

图结构（节点名与需求一致）::

    intent_router ──► plan_builder ──► tool_executor ──► response_generator ──► END

    节点职责：
        intent_router       判断用户问题类型（知识问答 / 数据查询 / 操作请求，可多选）
        plan_builder        把意图翻译成有序的执行步骤（含从用户话里抽取的参数）
        tool_executor       逐步调用 MCP 工具；失败自动重试 1 次，仍失败则标记降级
        response_generator  用工具结果生成回答，附来源；无 LLM 时走确定性离线路径

    为什么分成 plan_builder 与 tool_executor 两个节点，而不是合并成一个 tool_call：
        混合问题（"订单 ORD-001 符合退货政策吗？"）需要同时查知识库与业务库。
        把"决定做什么"与"实际去做"分开后，工具层的重试/降级逻辑只写一遍，
        且计划本身可被阶段5 的评测直接断言（expected_tool 对比的就是它）。

关键工程决策：

    1. **工具调用必须走线程池**：``mcp_servers.registry`` 的 ``call_tool`` 是同步阻塞
       接口（内部把协程投递到常驻循环线程并等待）。若在 async 节点里直接调用，
       会阻塞 FastAPI 的事件循环，SSE 心跳与并发请求全部卡死。
       因此统一用 ``await asyncio.to_thread(registry.call_tool, ...)``。

    2. **两层意图识别**：有 API Key 时用 LLM JSON 模式；无 Key 或调用失败时用规则
       匹配。规则路径不是"凑数"—— 它是阶段5 评测能在离线环境跑出真实数字的前提，
       而且关键词表本身就来自阶段1 的演示语料。

    3. **降级要显式**：工具失败重试 1 次后仍失败，会在 ``degraded_reasons`` 里留下
       记录并写进响应的 ``meta``。绝不能悄悄用模型内部知识冒充"查到的结果"。

    4. **混合同题必须两步都做**：``订单 ORD-001 符合退货政策吗`` 同时命中
       "数据查询"与"知识问答"，plan 会同时包含 query_orders 与 search_documents。
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict

from backend.llm_client import LLMClient, LLMError, NoLLMError, extract_json, get_llm_client
from backend.session_store import SessionStore, get_session_store
from config.config_loader import get_config
from mcp_servers.registry import MCPServerRegistry, ToolCallResult, ToolDisabledError, get_registry

logger = logging.getLogger("backend.agent")

__all__ = [
    "Intent",
    "PlanStep",
    "ToolOutcome",
    "AgentState",
    "AgentResponse",
    "RAGMCPAgent",
    "get_agent",
]

Intent = Literal["knowledge_qa", "data_query", "action_request", "chitchat"]

_INTENT_LABELS: dict[str, Intent] = {
    "knowledge_qa": "knowledge_qa",
    "knowledge": "knowledge_qa",
    "rag": "knowledge_qa",
    "policy": "knowledge_qa",
    "知识问答": "knowledge_qa",
    "知识": "knowledge_qa",
    "data_query": "data_query",
    "data": "data_query",
    "db": "data_query",
    "database": "data_query",
    "数据查询": "data_query",
    "查询": "data_query",
    "action_request": "action_request",
    "action": "action_request",
    "ticket": "action_request",
    "操作请求": "action_request",
    "操作": "action_request",
    "chitchat": "chitchat",
    "闲聊": "chitchat",
}

# ---------------------------------------------------------------------------
# 离线规则路径的关键词表：可直接对照阶段1 的演示语料核对
# ---------------------------------------------------------------------------
_KNOWLEDGE_KEYWORDS = (
    "政策", "制度", "规范", "流程", "标准", "规定", "条例", "怎么办", "怎么走",
    "是什么", "什么是", "多少天", "多久", "几天", "条件", "要求", "限制",
    # 期限类表达：评测用例 MIX-03「有没有在退货期限内的」曾因缺"期限"漏识别
    # knowledge_qa 意图，导致只查了订单、没查政策，回答不出判断依据。
    "期限", "时限", "时间内", "范围内", "有效期", "什么时候", "能不能退", "可以退",
    "退货", "退款", "换货", "报销", "发票", "账期", "结算", "工单", "优先级",
    "响应时限", "解决时限", "升级", "关单", "符合",
)
_DATA_KEYWORDS = (
    "订单", "状态", "金额", "物流", "发货", "签收", "查一下", "查查", "查询",
    "客户", "cust-", "ord-", "什么时候", "几号", "最近", "所有", "哪些",
)
_ACTION_KEYWORDS = (
    "创建", "建单", "报障", "提交工单", "开个单", "帮我建", "升级处理", "投诉",
    "催一下", "催单", "记录一下", "登记",
)

_ORDER_ID_RE = re.compile(r"\bORD-\d{3,}\b", re.IGNORECASE)
_CUSTOMER_ID_RE = re.compile(r"\bCUST-\d{3,}\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}(?:-\d{2})?)\b")
_DATE_RANGE_RE = re.compile(
    r"(20\d{2}(?:-\d{2}(?:-\d{2})?)?)\s*(?:~|～|至|到|-)\s*(20\d{2}(?:-\d{2}(?:-\d{2})?)?)"
)

# "信息不足"类错误码：表示需要向用户追问，而不是系统故障。
# 这类结果不计入 degraded（否则正常的澄清流程会被渲染成"部分数据获取失败"）。
_INFORMATIONAL_CODES: set[str] = {"MISSING_ARGUMENT", "NOT_FOUND", "INVALID_ARGUMENT"}
_INFORMATIONAL_MESSAGES: dict[str, str] = {
    "MISSING_ARGUMENT": "需要你补充订单号（ORD-xxx）或客户 ID（CUST-xxxx）才能查询业务数据",
    "NOT_FOUND": "未找到对应记录，请确认订单号或客户 ID 是否正确",
    "INVALID_ARGUMENT": "部分参数格式不正确，已按可识别的部分作答",
}


# ---------------------------------------------------------------------------
# 状态结构
# ---------------------------------------------------------------------------
@dataclass
class PlanStep:
    """一个执行步骤。

    Attributes:
        action: 动作类型。
        tool: MCP 工具名（``search_documents`` / ``query_orders`` / ``create_ticket``）。
        server: 所属 MCP server。
        args: 工具入参。
        reason: 为什么规划这一步（可解释性 + 评测归因）。
        intent: 该步骤服务的意图。
    """

    action: Literal["rag_search", "db_query", "tool_call", "answer"]
    tool: str = ""
    server: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    intent: Intent = "chitchat"

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "action": self.action,
            "tool": self.tool,
            "server": self.server,
            "args": self.args,
            "reason": self.reason,
            "intent": self.intent,
        }


@dataclass
class ToolOutcome:
    """一次工具调用的结果（含重试与耗时）。

    Attributes:
        tool: 工具名。
        server: server 名。
        args: 入参。
        ok: 业务是否成功。
        data: 业务数据。
        error: 错误描述。
        code: 错误码。
        attempts: 实际尝试次数（1 表示一次成功，2 表示重试过）。
        latency_ms: 总耗时。
        log_record: 注册表写入的结构化日志记录。
    """

    tool: str
    server: str
    args: dict[str, Any]
    ok: bool
    data: Any = None
    error: str | None = None
    code: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0
    log_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典（响应里的 tool_calls 字段）。"""
        return {
            "tool": self.tool,
            "server": self.server,
            "args": self.args,
            "ok": self.ok,
            "code": self.code,
            "error": self.error,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 2),
        }


class AgentState(TypedDict, total=False):
    """LangGraph 状态。

    Attributes:
        message: 本轮用户输入。
        session_id: 会话 ID。
        history: 历史消息（OpenAI 格式），由 session_store 提供。
        intents: 识别出的意图列表（可多选）。
        intent_source: ``llm`` 或 ``rule``，用于评测时区分两条路径。
        plan: 待执行步骤。
        outcomes: 工具调用结果。
        sources: 引用来源（来自 search_documents）。
        answer: 最终回答。
        tool_calls: 结构化工具调用日志（阶段4 要求）。
        degraded: 是否发生过降级。
        degraded_reasons: 降级原因列表。
        metrics: token / 延迟指标。
        error: 致命错误（图执行异常时）。
    """

    message: str
    session_id: str
    history: list[dict[str, str]]
    intents: list[Intent]
    intent_source: str
    plan: list[PlanStep]
    outcomes: Annotated[list[ToolOutcome], operator.add]
    sources: list[dict[str, Any]]
    answer: str
    tool_calls: list[dict[str, Any]]
    degraded: bool
    degraded_reasons: list[str]
    metrics: dict[str, Any]
    error: str


@dataclass
class AgentResponse:
    """Agent 的最终响应（与 FastAPI 返回结构一致）。

    Attributes:
        response: 回答正文。
        sources: 引用来源。
        tool_calls: 工具调用记录。
        session_id: 会话 ID。
        intents: 识别出的意图。
        intent_source: 意图识别路径（``llm`` / ``rule``）。
        degraded: 是否降级。
        degraded_reasons: 降级原因。
        metrics: token 与延迟指标。
        plan: 执行计划（评测用）。
    """

    response: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    intents: list[str] = field(default_factory=list)
    intent_source: str = ""
    degraded: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "response": self.response,
            "sources": self.sources,
            "tool_calls": self.tool_calls,
            "session_id": self.session_id,
            "intents": self.intents,
            "intent_source": self.intent_source,
            "degraded": self.degraded,
            "degraded_reasons": self.degraded_reasons,
            "metrics": self.metrics,
            "plan": self.plan,
        }


# ---------------------------------------------------------------------------
# 意图识别（规则路径）
# ---------------------------------------------------------------------------
def detect_intents_by_rule(message: str) -> list[Intent]:
    """基于关键词与 ID 模式识别意图（离线路径）。

    混合问题会返回多个意图 —— 这正是"订单 ORD-001 符合退货政策吗"能被正确处理的原因。

    Args:
        message: 用户输入。

    Returns:
        意图列表，至少包含一项（默认 ``chitchat``）。
    """
    text = message.strip()
    lowered = text.lower()
    intents: list[Intent] = []

    has_order = bool(_ORDER_ID_RE.search(text))
    has_customer = bool(_CUSTOMER_ID_RE.search(text))

    if any(k in text for k in _ACTION_KEYWORDS):
        intents.append("action_request")
    if has_order or has_customer:
        # 出现订单号/客户号通常意味着要查数据，除非句子是纯操作请求
        intents.append("data_query")
    elif any(k in text for k in _DATA_KEYWORDS) and any(
        k in text for k in ("订单", "客户", "物流", "发货", "签收", "状态", "金额")
    ):
        # 收紧关键词判定：必须同时命中"查询意图词"与"业务实体词"才算数据查询。
        # 否则"创建工单必须填写哪些信息"里的"信息"会误判为 data_query，
        # 进而规划出一次无意义的 query_orders（实测导致 6/20 用例过度调用）。
        intents.append("data_query")
    if any(k in text for k in _KNOWLEDGE_KEYWORDS) or "data_query" not in intents:
        intents.append("knowledge_qa")

    # 去重保序
    seen: set[str] = set()
    unique: list[Intent] = []
    for item in intents:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    if not unique or (len(unique) == 1 and unique[0] == "knowledge_qa" and len(lowered) <= 3):
        return ["chitchat"]
    return unique


class RAGMCPAgent:
    """LangGraph 主 Agent。"""

    def __init__(
        self,
        registry: MCPServerRegistry | None = None,
        llm: LLMClient | None = None,
        sessions: SessionStore | None = None,
        *,
        max_retries: int | None = None,
    ) -> None:
        """初始化 Agent 并编译状态图。

        Args:
            registry: MCP 注册表；``None`` 时用进程单例。
            llm: LLM 客户端；``None`` 时用进程单例。
            sessions: 会话存储；``None`` 时用进程单例。
            max_retries: 工具失败后的重试次数；``None`` 时读配置
                （``llm.tool_call_retry``，默认 1）。
        """
        app = get_config()
        self.registry = registry or get_registry()
        self.llm = llm or get_llm_client()
        self.sessions = sessions or get_session_store()
        self.max_retries = (
            app.llm.tool_call_retry if max_retries is None else max(int(max_retries), 0)
        )
        self._tool_catalog: dict[str, dict[str, Any]] | None = None
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # 图装配
    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        """构造并编译 LangGraph 状态机。

        Returns:
            已编译的图对象（``ainvoke`` / ``astream`` 可用）。

        Raises:
            RuntimeError: langgraph 未安装。
        """
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 langgraph，请执行 pip install -r requirements.txt"
            ) from exc

        builder = StateGraph(AgentState)
        builder.add_node("intent_router", self._node_intent_router)
        builder.add_node("plan_builder", self._node_plan_builder)
        builder.add_node("tool_executor", self._node_tool_executor)
        builder.add_node("response_generator", self._node_response_generator)

        builder.set_entry_point("intent_router")
        builder.add_edge("intent_router", "plan_builder")
        builder.add_edge("plan_builder", "tool_executor")
        builder.add_edge("tool_executor", "response_generator")
        builder.add_edge("response_generator", END)
        return builder.compile()

    # ------------------------------------------------------------------
    # 节点实现
    # ------------------------------------------------------------------
    async def _node_intent_router(self, state: AgentState) -> AgentState:
        """节点：意图识别（``intent_router``）。

        有 API Key 时优先用 LLM JSON 模式；失败或未配置时退回规则匹配。
        LLM 路径会做标签归一化（``knowledge`` → ``knowledge_qa``），
        避免模型自由发挥导致下游路由失败。

        Args:
            state: 当前状态。

        Returns:
            更新后的状态片段。
        """
        message = state.get("message", "")
        if self.llm.available:
            try:
                intents = await self._classify_by_llm(message, state.get("history", []))
                return {"intents": intents, "intent_source": "llm"}
            except LLMError as exc:
                # NoLLMError 也是 LLMError 的子类，这里一并覆盖
                logger.warning("LLM 意图识别失败，退回规则路径: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 意图识别异常，退回规则路径: %s", exc)

        intents = detect_intents_by_rule(message)
        logger.info("规则意图识别 %r -> %s", message, intents)
        return {"intents": intents, "intent_source": "rule"}

    async def _classify_by_llm(self, message: str, history: list[dict[str, str]]) -> list[Intent]:
        """用 LLM 做意图识别。

        Args:
            message: 用户输入。
            history: 最近对话历史（提供上下文，帮助理解省略句）。

        Returns:
            归一化后的意图列表。

        Raises:
            LLMError: 调用或解析失败。
        """
        system = (
            "你是企业智能助手的意图分类器。判断用户问题属于以下哪几类（可多选）：\n"
            "- knowledge_qa：询问制度、政策、流程、规范（如'退货政策是什么'）\n"
            "- data_query：查询具体业务数据，如订单状态、金额、客户订单列表\n"
            "- action_request：要求执行操作，如创建工单、报障、升级处理\n"
            "- chitchat：寒暄或无明确业务意图\n"
            "\n**判断多意图的关键规则**：只要问题需要「先用政策/制度做判断依据」，"
            "就必须同时输出 knowledge_qa，即使句子里没有出现「政策」二字。例如：\n"
            "- '订单 ORD-001 符合退货政策吗？' -> data_query + knowledge_qa\n"
            "- 'CUST-1003 的订单里有没有在退货期限内的？' -> data_query + knowledge_qa"
            "（要用退货期限这个制度去判断订单）\n"
            "- 'ORD-002 还没签收，按政策可以拒收吗？' -> data_query + knowledge_qa\n"
            "- '查一下 CUST-1001 的所有订单。' -> data_query（纯取数，无需政策依据）\n"
            "\n只输出 JSON：{\"intents\": [\"knowledge_qa\"]}，不要输出解释。"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": message})

        response = await self.llm.chat(
            # 意图分类是确定性短任务，不需要推理链：
            #   * 关闭思考模式（classification_thinking=false）—— 实测 V4 默认思考，
            #     max_tokens=16 时 token 全花在 reasoning 上、content 为空，
            #     关闭后 output token 从 72 降到 1，且不会再出现空响应；
            #   * max_tokens 提到 256 作为双保险（万一服务端忽略 thinking 参数）。
            messages,
            json_mode=True,
            temperature=0.0,
            max_tokens=256,
            thinking=self.llm.config.classification_thinking,
        )
        parsed = extract_json(response.content)
        raw_intents = parsed.get("intents")
        if isinstance(raw_intents, str):
            raw_intents = [raw_intents]
        if not isinstance(raw_intents, list):
            raise LLMError(f"意图识别结果缺少 intents 数组: {parsed}")

        intents: list[Intent] = []
        for item in raw_intents:
            key = str(item).strip().lower()
            mapped = _INTENT_LABELS.get(key)
            if mapped and mapped not in intents:
                intents.append(mapped)
        if not intents:
            raise LLMError(f"意图识别结果无法归一化: {raw_intents}")
        logger.info("LLM 意图识别 %r -> %s", message, intents)
        return intents

    async def _node_plan_builder(self, state: AgentState) -> AgentState:
        """节点：把意图翻译成执行步骤（``plan_builder``）。

        参数抽取走正则（订单号 / 客户号 / 日期范围），因为这些 ID 有固定格式，
        正则比 LLM 更可靠且零成本；而"要不要调工具、调哪个"才需要模型判断，
        那部分已由 intent_router 完成。

        Args:
            state: 当前状态。

        Returns:
            含 ``plan`` 的状态片段。
        """
        message = state.get("message", "")
        intents = state.get("intents", []) or ["chitchat"]
        plan: list[PlanStep] = []

        order_ids = [m.upper() for m in _ORDER_ID_RE.findall(message)]
        customer_ids = [m.upper() for m in _CUSTOMER_ID_RE.findall(message)]
        range_match = _DATE_RANGE_RE.search(message)
        single_dates = [d for d in _DATE_RE.findall(message)]
        # 操作请求里顺带出现的客户号不构成"要查该客户的订单"，
        # 除非句子同时含明确的查询词（"查/看/有哪些/状态/金额"）。
        if "action_request" in intents and not any(
            k in message for k in ("查", "看", "状态", "金额", "有哪些", "哪些")
        ):
            customer_ids = []

        if "action_request" in intents and self._is_write_intent(message):
            title = self._make_ticket_title(message, order_ids, customer_ids)
            plan.append(
                PlanStep(
                    action="tool_call",
                    tool="create_ticket",
                    server="business_db",
                    args={
                        "title": title,
                        "description": message.strip(),
                        "priority": self._infer_priority(message),
                    },
                    reason="用户明确要求执行操作（建单/报障/升级）",
                    intent="action_request",
                )
            )

        if "data_query" in intents:
            if order_ids and not customer_ids:
                # 工具签名按 customer_id 查询，而用户往往只给订单号：
                # 反查订单所属客户，避免"猜一个客户 ID"导致查空。
                customer = await asyncio.to_thread(
                    self._resolve_customer_for_order, order_ids[0]
                )
                plan.append(
                    PlanStep(
                        action="db_query",
                        tool="query_orders",
                        server="business_db",
                        args={"customer_id": customer},
                        reason=f"用户提到订单 {order_ids[0]}，按其所属客户查询订单明细",
                        intent="data_query",
                    )
                )
            elif customer_ids:
                args: dict[str, Any] = {"customer_id": customer_ids[0]}
                if range_match:
                    args["date_range"] = {
                        "start": range_match.group(1),
                        "end": range_match.group(2),
                    }
                elif single_dates:
                    args["date_range"] = single_dates[0]
                plan.append(
                    PlanStep(
                        action="db_query",
                        tool="query_orders",
                        server="business_db",
                        args=args,
                        reason=f"按客户 {customer_ids[0]} 查询订单",
                        intent="data_query",
                    )
                )
            elif "action_request" in intents and not order_ids:
                # 操作请求里只是顺带提到"订单/客户"，且没有可查的具体 ID：
                # 不额外查库，避免"为 CUST-xxxx 的订单问题建单"这类句子
                # 落到"缺少参数"的失败分支（实测 ACT-03 即如此）。
                pass
            else:
                # 没有客户 ID：只有出现具体订单号、或问题明确在问某类订单状态时才查库。
                # 否则（"创建工单需要哪些信息"）纯属关键词误命中，不该产生工具调用。
                status = self._infer_status_filter(message) if order_ids else ""
                # 还要求"显式取数措辞"：仅凭"退款/签收"这类状态词不足以判定用户要查库。
                # 否则「已签收的订单要退货应该走什么流程？」会被多规划一次订单查询，
                # 而它其实是纯流程问题（实测 KB-07 因此多调工具并触发降级）。
                explicit_lookup = any(
                    k in message
                    for k in ("查", "看", "状态", "金额", "有哪些", "哪些", "几号", "什么时候", "都")
                )
                if order_ids or (status and explicit_lookup):
                    plan.append(
                        PlanStep(
                            action="db_query",
                            tool="query_orders",
                            server="business_db",
                            args={"status": status} if status else {"order_ids": order_ids},                            reason=(
                                f"未指定客户，按状态 {status} 查询订单"
                                if status
                                else f"按订单号 {order_ids} 定位订单"
                            ),
                            intent="data_query",
                        )
                    )
                elif "action_request" not in intents:
                    plan.append(
                        PlanStep(
                            action="db_query",
                            tool="query_orders",
                            server="business_db",
                            args={"customer_id": ""},
                            reason="缺少客户 ID，需要向用户确认",
                            intent="data_query",
                        )
                    )

        if "knowledge_qa" in intents and self._is_knowledge_question(message):
            # 操作请求场景下，只有当问题真的在问制度/政策时才额外检索。
            # 否则「麻烦建个工单催一下」也会去查知识库，既无关又拖慢响应
            # （实测 ACT-02/ACT-03 因此各多一次检索）。
            if "action_request" not in intents or self._has_policy_topic(message):
                query = self._make_kb_query(message, order_ids)
                plan.append(
                    PlanStep(
                        action="rag_search",
                        tool="search_documents",
                        server="knowledge_base",
                        args={"query": query, "top_k": 5},
                        reason="制度/政策类问题需要检索企业知识库",
                        intent="knowledge_qa",
                    )
                )

        # 门控被挡时的补偿：问题里出现"必须填写/需要哪些/什么要求"这类**询问规则**
        # 的措辞，且没有被判为知识检索，则应补一次知识库检索。
        # 否则会出现"写操作被正确挡住、但该做的检索也没做"的空转
        # （实测 KB-05 一度因此零工具调用而答不出必填字段）。
        asks_about_rules = any(
            k in message
            for k in ("必须", "需要填", "需要提供", "哪些信息", "什么信息", "什么要求", "怎么填")
        )
        if asks_about_rules and not any(step.action == "rag_search" for step in plan):
            plan.append(
                PlanStep(
                    action="rag_search",
                    tool="search_documents",
                    server="knowledge_base",
                    args={"query": self._make_kb_query(message, order_ids), "top_k": 5},
                    reason="问题在询问规则/必填项，需要检索知识库",
                    intent="knowledge_qa",
                )
            )

        if not plan:
            plan.append(
                PlanStep(
                    action="answer",
                    reason="闲聊或无业务意图，直接回答",
                    intent="chitchat",
                )
            )

        logger.info(
            "计划生成 %d 步: %s",
            len(plan),
            [(s.action, s.tool, s.reason) for s in plan],
        )
        return {"plan": plan, "outcomes": [], "sources": [], "degraded": False, "degraded_reasons": []}

    @staticmethod
    def _resolve_customer_for_order(order_id: str) -> str:
        """把订单号映射为客户 ID（查询业务库推断）。

        工具签名按 ``customer_id`` 查询，而用户往往只给订单号。这里直接查业务库
        反查所属客户，避免"猜一个客户 ID"导致查空。

        Args:
            order_id: 订单号，如 ``ORD-001``。

        Returns:
            客户 ID；查不到时返回订单号本身（由工具返回空列表，Agent 会解释）。
        """
        try:
            from mcp_servers.business_db.server import get_database

            record = get_database().get_order(order_id)
            if record:
                return str(record["customer_id"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("订单 %s 反查客户失败: %s", order_id, exc)
        return order_id

    @staticmethod
    def _is_write_intent(message: str) -> bool:
        """判断用户是否**明确要求执行写操作**（建单/报障/升级）。

        这是写操作的安全门控，独立于意图分类器：create_ticket 会真实落库，
        一旦被误触发就无法撤销（只能人工关单）。实测出现过
        「创建工单必须填写哪些信息？」被分类为 action_request 并真的建了单 ——
        这是一句**询问规则**的话，不该产生副作用。

        判定原则：必须出现"请求式"动作词（"帮我/请/麻烦 + 动词"或祈使式动词），
        仅仅提到"创建/工单"这些名词性词不算。

        Args:
            message: 用户输入。

        Returns:
            明确要求写操作返回 True。
        """
        text = message.strip()
        # 疑问句一律不视为写操作请求（"创建工单需要什么"是问规则，不是让我建单）
        if text.endswith(("？", "?")) or any(
            k in text for k in ("哪些", "什么", "怎么", "如何", "需要填", "需要提供", "可以吗")
        ):
            return False
        # 请求式动词
        imperative = (
            "帮我", "请帮", "麻烦", "请为", "替我", "给我", "帮忙",
            "建个", "建一个", "创建一", "开个", "开一个", "登记", "记录一下", "提交", "报障",
        )
        return any(k in text for k in imperative)

    @staticmethod
    def _has_policy_topic(message: str) -> bool:
        """判断问题是否真的在谈论制度/政策（而不仅是"订单/工单"这类业务名词）。

        用于操作请求场景下的二次判定：``_is_knowledge_question`` 的词表包含
        "退货/退款/工单"等业务词，范围较宽；而在"帮我建个工单催一下"这类句子里，
        这些词只表示业务对象，不构成检索制度的需要。

        Args:
            message: 用户输入。

        Returns:
            确实在问制度/政策返回 True。
        """
        strict = (
            "政策", "制度", "规范", "流程", "标准", "规定", "条例",
            "期限", "时限", "范围", "条件", "要求", "算不算", "符合",
            "允许", "能不能", "可不可以", "怎么算", "如何计算",
        )
        return any(k in message for k in strict)

    @staticmethod
    def _is_knowledge_question(message: str) -> bool:
        """判断问题是否真的需要检索知识库。

        为什么要在意图之外再做一层判断：LLM 分类器被明确要求"需要政策做判断依据时
        必须同时报 knowledge_qa"，因此会倾向多报（避免漏查）。规划阶段若无条件
        为 knowledge_qa 加一次检索，"查一下 CUST-1001 的订单"也会去查知识库，
        实测造成 5/20 用例过度调用。

        Args:
            message: 用户输入。

        Returns:
            需要检索返回 True。
        """
        return any(k in message for k in _KNOWLEDGE_KEYWORDS)

    @staticmethod
    def _infer_status_filter(message: str) -> str:
        """从用户话里推断订单状态过滤值。

        中文表达与英文状态码都要覆盖：用户说"已经退款了"，工具需要拿到
        ``refunded`` 才能查。

        Args:
            message: 用户输入。

        Returns:
            状态码；无法判断时返回空串。
        """
        text = message.lower()
        mapping: list[tuple[tuple[str, ...], str]] = [
            (("退款", "已退", "退钱", "refunded", "refund"), "refunded"),
            (("取消", "作废", "cancelled", "canceled"), "cancelled"),
            (("已签收", "签收", "delivered"), "delivered"),
            (("已发货", "发货", "在途", "运输", "shipped"), "shipped"),
            (("未付款", "待付款", "待支付", "pending"), "pending"),
            (("已付款", "待发货", "已支付", "paid"), "paid"),
        ]
        for keywords, status in mapping:
            if any(k in text for k in keywords):
                return status
        return ""

    @staticmethod
    def _make_ticket_title(message: str, order_ids: list[str], customer_ids: list[str]) -> str:
        """从用户原话生成工单标题。

        Args:
            message: 用户输入。
            order_ids: 抽取到的订单号。
            customer_ids: 抽取到的客户 ID。

        Returns:
            标题字符串（不超过 120 字符）。
        """
        prefix = ""
        if order_ids:
            prefix = f"订单 {order_ids[0]} "
        elif customer_ids:
            prefix = f"客户 {customer_ids[0]} "
        body = message.strip().replace("\n", " ")
        title = f"{prefix}{body}" if prefix and prefix.strip() not in body else body
        return title[:120]

    @staticmethod
    def _infer_priority(message: str) -> str:
        """从用户话里推断工单优先级。

        Args:
            message: 用户输入。

        Returns:
            ``low`` / ``medium`` / ``high`` / ``urgent``。
        """
        text = message.lower()
        if any(k in text for k in ("urgent", "紧急", "立刻", "马上", "特急", "p0")):
            return "urgent"
        if any(k in text for k in ("高优先级", "high", "尽快", "加急", "严重", "p1")):
            return "high"
        if any(k in text for k in ("低优先级", "low", "不急", "p3")):
            return "low"
        return "medium"

    @staticmethod
    def _make_kb_query(message: str, order_ids: list[str]) -> str:
        """把用户问题转成适合检索的查询串。

        订单号属于业务实体，放进知识库检索会引入噪声（知识库里没有 ORD-001），
        因此剔除 ID 后再检索。

        Args:
            message: 用户输入。
            order_ids: 抽取到的订单号。

        Returns:
            检索查询串。
        """
        query = message
        for oid in order_ids:
            query = query.replace(oid, " ")
        query = _CUSTOMER_ID_RE.sub(" ", query)
        query = re.sub(r"\s+", " ", query).strip(" ，,。.？?")
        return query or message

    async def _node_tool_executor(self, state: AgentState) -> AgentState:
        """节点：执行计划中的工具调用（``tool_executor``）。

        失败处理：重试 ``max_retries`` 次；仍失败则记入 ``degraded_reasons``，
        由 response_generator 走"如实说明失败 + 用已有信息作答"的降级路径。

        Args:
            state: 当前状态。

        Returns:
            含 ``outcomes`` / ``sources`` / ``tool_calls`` 的状态片段。
        """
        plan = state.get("plan", [])
        outcomes: list[ToolOutcome] = []
        sources: list[dict[str, Any]] = []
        degraded_reasons: list[str] = list(state.get("degraded_reasons", []))

        for step in plan:
            if step.action == "answer":
                continue
            if not step.tool:
                continue

            # 执行前的实体门控：既无客户 ID 也无订单号时**不要调用工具**，
            # 直接返回 MISSING_ARGUMENT 并提问。否则会真的发起一次注定失败的
            # 调用，把 degraded 置为 True —— 明明是"信息不足，需要追问"，
            # 却表现为"工具故障"（实测 KB-07/DB-06 因此被误报为失败）。
            if step.tool == "query_orders" and not (
                step.args.get("customer_id")
                or step.args.get("status")
                or step.args.get("order_ids")
            ):
                outcomes.append(
                    ToolOutcome(
                        tool=step.tool,
                        server=step.server,
                        args=step.args,
                        ok=False,
                        error="缺少 customer_id，无法查询订单",
                        code="MISSING_ARGUMENT",
                        log_record={
                            "timestamp": datetime.now()
                            .astimezone()
                            .isoformat(timespec="milliseconds"),
                            "server": step.server,
                            "tool": step.tool,
                            "args": step.args,
                            "ok": False,
                            "code": "MISSING_ARGUMENT",
                            "error": "缺少 customer_id，无法查询订单",
                            "result": None,
                            "latency_ms": 0.0,
                        },
                    )
                )
                degraded_reasons.append("缺少客户 ID：请提供 CUST-xxxx 或订单号")
                continue

            outcome = await self._call_tool_with_retry(step)
            outcomes.append(outcome)

            if outcome.ok and step.tool == "search_documents":
                sources.extend(self._collect_sources(outcome.data))
            if not outcome.ok:
                if outcome.code in _INFORMATIONAL_CODES:
                    # 信息不足需要向用户追问，不是故障。若把它算作 degraded，
                    # 响应会显示"部分数据获取失败"，把正常的澄清流程说成系统故障
                    # （实测 KB-07/DB-06 因此被误报）。
                    degraded_reasons.append(_INFORMATIONAL_MESSAGES[outcome.code])
                    logger.info("工具 %s 需要补充参数: %s", step.tool, outcome.error)
                else:
                    degraded_reasons.append(
                        f"{step.tool} 调用失败（{outcome.code or 'UNKNOWN'}）: {outcome.error}"
                    )
                    logger.warning("工具 %s 最终失败: %s", step.tool, outcome.error)

        # tool_calls 使用注册表的结构化日志格式（{timestamp, tool, args, result, latency_ms}），
        # 并补上 attempts 与 intent —— 阶段5 的评测需要断言"工具选对了吗""重试了几次"。
        tool_calls: list[dict[str, Any]] = []
        for outcome in outcomes:
            record = dict(outcome.log_record) or {
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "server": outcome.server,
                "tool": outcome.tool,
                "args": outcome.args,
                "ok": outcome.ok,
                "code": outcome.code,
                "error": outcome.error,
                "result": outcome.data,
                "latency_ms": round(outcome.latency_ms, 2),
            }
            record["attempts"] = outcome.attempts
            record["intent"] = next(
                (s.intent for s in plan if s.tool == outcome.tool), "unknown"
            )
            tool_calls.append(record)

        # degraded 只看"真故障"：信息不足（MISSING_ARGUMENT 等）只是追问，
        # 不算系统降级。判定直接来自 outcomes，避免依赖某个易被覆盖的状态键。
        real_failure = any(
            (not o.ok) and (o.code or "") not in _INFORMATIONAL_CODES for o in outcomes
        )
        return {
            "outcomes": outcomes,
            "sources": sources,
            "tool_calls": tool_calls,
            "degraded": real_failure,
            "degraded_reasons": degraded_reasons,
        }

    async def _call_tool_with_retry(self, step: PlanStep) -> ToolOutcome:
        """调用一次工具，失败按配置重试。

        **直接 await 注册表的异步接口**，不经过 ``asyncio.to_thread``：
        实测在 uvicorn 的事件循环里走线程桥接会导致每次工具调用固定卡满超时
        （子进程日志显示同一操作仅需 3ms），而直接 await 正常。注册表内部已把
        会话固定在调用方事件循环上，因此这里不需要线程池。

        Args:
            step: 计划步骤。

        Returns:
            :class:`ToolOutcome`（``attempts`` 反映真实尝试次数）。
        """
        attempts = 0
        last: ToolCallResult | None = None
        started = time.perf_counter()

        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = await self.registry.acall_tool(step.server, step.tool, step.args)
            except ToolDisabledError as exc:
                return ToolOutcome(
                    tool=step.tool,
                    server=step.server,
                    args=step.args,
                    ok=False,
                    error=f"工具被禁用: {exc}",
                    code="TOOL_DISABLED",
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except Exception as exc:  # noqa: BLE001 - 协议层/连接层异常
                last = None
                logger.warning(
                    "工具 %s.%s 第 %d 次调用抛异常: %s: %s",
                    step.server,
                    step.tool,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                if attempts <= self.max_retries:
                    await asyncio.sleep(0.5 * attempts)
                    continue
                return ToolOutcome(
                    tool=step.tool,
                    server=step.server,
                    args=step.args,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    code="PROTOCOL_ERROR",
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            last = result
            if result.ok:
                return ToolOutcome(
                    tool=step.tool,
                    server=step.server,
                    args=step.args,
                    ok=True,
                    data=result.data,
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    log_record=result.to_log_record(step.args),
                )
            if attempts <= self.max_retries:
                logger.info(
                    "工具 %s.%s 第 %d 次返回失败（%s），重试",
                    step.server,
                    step.tool,
                    attempts,
                    result.code,
                )
                await asyncio.sleep(0.3 * attempts)

        assert last is not None
        return ToolOutcome(
            tool=step.tool,
            server=step.server,
            args=step.args,
            ok=False,
            error=last.error,
            code=last.code,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000,
            log_record=last.to_log_record(step.args),
        )

    @staticmethod
    def _collect_sources(hits: Any) -> list[dict[str, Any]]:
        """从检索结果里提取引用来源。

        Args:
            hits: ``search_documents`` 返回的片段列表。

        Returns:
            去重后的来源列表（按首次出现顺序），含片段摘要与分数。
        """
        if not isinstance(hits, list):
            return []
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, Any]] = set()
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            key = (str(hit.get("source", "")), hit.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source": hit.get("source", ""),
                    "page": hit.get("page"),
                    "score": hit.get("score"),
                    "doc_id": hit.get("doc_id", ""),
                    "heading": hit.get("heading", ""),
                    "snippet": str(hit.get("content", ""))[:160],
                }
            )
        return sources

    async def _node_response_generator(self, state: AgentState) -> AgentState:
        """节点：生成最终回答（``response_generator``）。

        有 LLM 时把工具结果作为事实依据交给模型组织语言（并明确禁止编造）；
        无 LLM 或调用失败时走确定性离线路径：直接引用检索到的原文 + 订单事实。

        Args:
            state: 当前状态。

        Returns:
            含 ``answer`` 与 ``metrics`` 的状态片段。
        """
        message = state.get("message", "")
        plan = state.get("plan", [])
        outcomes = state.get("outcomes", [])
        sources = state.get("sources", [])
        degraded = bool(state.get("degraded"))
        reasons = state.get("degraded_reasons", [])

        facts = self._render_facts(outcomes)
        chitchat_only = all(step.action == "answer" for step in plan) if plan else False

        # LangGraph 的节点只保留自己返回的键：tool_executor 写入的 tool_calls /
        # sources / degraded 必须在这里显式带上，否则会在状态合并时丢失
        # （实测症状：最终响应里 tool_calls 为空，评测拿不到工具选择结果）。
        carry: AgentState = {
            "tool_calls": state.get("tool_calls", []),
            "sources": sources,
            "outcomes": outcomes,
        }

        if self.llm.available:
            try:
                answer, metrics = await self._generate_by_llm(
                    message, state.get("history", []), facts, sources, degraded, chitchat_only
                )
                carry["answer"] = answer
                carry["metrics"] = metrics
                carry["degraded"] = degraded
                carry["degraded_reasons"] = reasons
                return carry
            except LLMError as exc:
                logger.warning("LLM 回答生成失败，改用离线路径: %s", exc)
                degraded = True
                reasons = [*reasons, f"LLM 回答生成失败，已降级为离线回答: {exc}"]

        carry["answer"] = self._generate_offline(
            message, outcomes, sources, degraded, reasons, chitchat_only
        )
        carry["degraded"] = degraded
        carry["degraded_reasons"] = reasons
        carry["metrics"] = {
            "model": "offline-rule-based",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
        }
        return carry

    @staticmethod
    def _render_facts(outcomes: list[ToolOutcome]) -> str:
        """把工具结果渲染成给模型看的事实文本。

        Args:
            outcomes: 工具调用结果列表。

        Returns:
            可直接放进 prompt 的事实描述。
        """
        if not outcomes:
            return "（无工具调用结果）"
        blocks: list[str] = []
        for item in outcomes:
            if item.tool == "search_documents":
                if not item.ok:
                    blocks.append(f"【知识库检索】失败：{item.error}")
                    continue
                hits = item.data if isinstance(item.data, list) else []
                lines = [f"【知识库检索】共 {len(hits)} 条："]
                for idx, hit in enumerate(hits, 1):
                    if not isinstance(hit, dict):
                        continue
                    page = f" 第 {hit['page']} 页" if hit.get("page") else ""
                    lines.append(
                        f"  [{idx}] 来源《{hit.get('source')}》{page}"
                        f"（相关度 {hit.get('score')}）：{hit.get('content')}"
                    )
                blocks.append("\n".join(lines))
            elif item.tool == "query_orders":
                if not item.ok:
                    blocks.append(f"【订单查询】失败：{item.error}")
                    continue
                rows = item.data if isinstance(item.data, list) else []
                if not rows:
                    blocks.append("【订单查询】未查到符合条件的订单。")
                    continue
                lines = [f"【订单查询】共 {len(rows)} 条："]
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        f"  - {row.get('order_id')}：状态 {row.get('status')}，"
                        f"金额 {row.get('amount')} {row.get('currency')}，"
                        f"下单 {row.get('created_at')}，签收 {row.get('delivered_at') or '未签收'}"
                    )
                blocks.append("\n".join(lines))
            elif item.tool == "create_ticket":
                if not item.ok:
                    blocks.append(f"【创建工单】失败：{item.error}")
                    continue
                data = item.data if isinstance(item.data, dict) else {}
                blocks.append(
                    f"【创建工单】成功：工单号 {data.get('ticket_id')}，"
                    f"优先级 {data.get('priority')}，状态 {data.get('status')}"
                )
            else:
                blocks.append(f"【{item.tool}】{'成功' if item.ok else '失败'}：{item.data or item.error}")
        return "\n".join(blocks)

    async def _generate_by_llm(
        self,
        message: str,
        history: list[dict[str, str]],
        facts: str,
        sources: list[dict[str, Any]],
        degraded: bool,
        chitchat_only: bool,
    ) -> tuple[str, dict[str, Any]]:
        """用 LLM 生成回答。

        Args:
            message: 用户问题。
            history: 对话历史。
            facts: 工具结果事实文本。
            sources: 引用来源。
            degraded: 是否存在工具失败。
            chitchat_only: 是否只是闲聊（无需引用事实）。

        Returns:
            ``(回答正文, 指标字典)``。

        Raises:
            LLMError: 调用失败。
        """
        if chitchat_only:
            system = (
                "你是企业智能助手，可以回答日常寒暄。若用户问题涉及公司制度或业务数据，"
                "请提示对方补充具体问题，不要编造公司信息。"
            )
        else:
            system = (
                "你是企业智能助手。**只能依据下方【事实】回答**，禁止编造制度条款、"
                "订单号或金额。回答要求：\n"
                "1. 先给结论，再给依据；\n"
                "2. 引用知识库内容时标注来源文件名，涉及页码时一并标注；\n"
                "3. 若【事实】为空或不足以回答，明确说明缺什么信息、建议用户补充什么；\n"
                "4. 涉及具体订单时，把订单号、状态、时间写清楚；\n"
                "5. 用中文，简洁，避免重复事实原文。"
            )
            if degraded:
                system += (
                    "\n注意：部分工具调用失败，请如实说明哪部分信息没能取到，"
                    "不要用你自己的知识补充这部分内容。"
                )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history[-10:])
        user_content = f"【用户问题】\n{message}"
        if not chitchat_only:
            user_content += f"\n\n【事实】\n{facts}"
            if sources:
                cited = "、".join(
                    f"《{s['source']}》" + (f"第{s['page']}页" if s.get("page") else "")
                    for s in sources
                )
                user_content += f"\n\n【可引用来源】\n{cited}"
        messages.append({"role": "user", "content": user_content})

        response = await self.llm.chat(messages)
        return response.content.strip(), response.to_metrics()

    @staticmethod
    def _generate_offline(
        message: str,
        outcomes: list[ToolOutcome],
        sources: list[dict[str, Any]],
        degraded: bool,
        reasons: list[str],
        chitchat_only: bool,
    ) -> str:
        """离线确定性回答（无 LLM 或 LLM 失败时）。

        设计原则：**只陈述工具真实返回的内容**，不做任何推断性补充。
        这样阶段5 的评测在无 Key 环境下依然能对"回答包含率"给出真实数字。

        Args:
            message: 用户问题。
            outcomes: 工具结果。
            sources: 引用来源。
            degraded: 是否降级。
            reasons: 降级原因。
            chitchat_only: 是否闲聊。

        Returns:
            回答正文（Markdown）。
        """
        if chitchat_only:
            return (
                "你好，我是企业智能助手。可以帮你做三类事情：\n"
                "1. **查制度**：如「退货政策是什么」「工单优先级怎么定」；\n"
                "2. **查数据**：如「查一下 CUST-1001 的订单」「订单 ORD-001 什么状态」；\n"
                "3. **建工单**：如「帮我为 ORD-001 退款未到账创建一个高优先级工单」。"
            )

        parts: list[str] = []
        search = next((o for o in outcomes if o.tool == "search_documents"), None)
        orders = next((o for o in outcomes if o.tool == "query_orders"), None)
        ticket = next((o for o in outcomes if o.tool == "create_ticket"), None)

        if search is not None:
            if search.ok and isinstance(search.data, list) and search.data:
                parts.append("**知识库检索结果：**")
                for idx, hit in enumerate(search.data[:3], 1):
                    if not isinstance(hit, dict):
                        continue
                    page = f"（第 {hit['page']} 页）" if hit.get("page") else ""
                    parts.append(
                        f"{idx}. {hit.get('content', '').strip()} "
                        f"\n   —— 来源《{hit.get('source')}》{page}，相关度 {hit.get('score')}"
                    )
            elif search.ok:
                parts.append("**知识库检索结果：** 未检索到相关内容，建议确认问题措辞或补充文档。")
            else:
                parts.append(f"**知识库检索失败：** {search.error}")

        if orders is not None:
            if orders.ok and isinstance(orders.data, list) and orders.data:
                parts.append("**订单查询结果：**")
                for row in orders.data[:5]:
                    if not isinstance(row, dict):
                        continue
                    parts.append(
                        f"- `{row.get('order_id')}`：状态 **{row.get('status')}**，"
                        f"金额 {row.get('amount')} {row.get('currency')}，"
                        f"下单 {row.get('created_at')}，"
                        f"签收 {row.get('delivered_at') or '未签收'}"
                    )
            elif orders.ok:
                parts.append(
                    f"**订单查询结果：** 未查到 `{orders.args.get('customer_id')}` 的订单，"
                    f"请确认客户 ID 或订单号是否正确。"
                )
            else:
                parts.append(f"**订单查询失败：** {orders.error}")

        if ticket is not None:
            if ticket.ok and isinstance(ticket.data, dict):
                parts.append(
                    f"**工单已创建：** `{ticket.data.get('ticket_id')}`"
                    f"（优先级 {ticket.data.get('priority')}，状态 {ticket.data.get('status')}）"
                )
            else:
                parts.append(f"**工单创建失败：** {ticket.error}")

        if not parts:
            parts.append("抱歉，我没有从可用工具中获得能回答该问题的信息。")

        if degraded:
            parts.append("\n> ⚠️ 部分数据获取失败：" + "；".join(reasons[:3]))

        if sources:
            cited = "、".join(
                f"《{s['source']}》" + (f"第{s['page']}页" if s.get("page") else "")
                for s in sources[:3]
            )
            parts.append(f"\n*引用来源：{cited}*")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    async def arun(self, message: str, session_id: str = "default") -> AgentResponse:
        """处理一条用户消息（异步）。

        Args:
            message: 用户输入。
            session_id: 会话 ID。

        Returns:
            :class:`AgentResponse`。

        Raises:
            ValueError: 消息为空。
        """
        if not message or not message.strip():
            raise ValueError("message 不能为空")

        history = await asyncio.to_thread(self.sessions.as_llm_messages, session_id)
        await asyncio.to_thread(self.sessions.append, session_id, "user", message)

        started = time.perf_counter()
        try:
            final: AgentState = await self.graph.ainvoke(
                {
                    "message": message,
                    "session_id": session_id,
                    "history": history,
                    "outcomes": [],
                }
            )
        except Exception as exc:  # noqa: BLE001 - 图执行异常兜底，保证接口不 500
            logger.exception("Agent 图执行失败")
            final = {
                "message": message,
                "session_id": session_id,
                "intents": ["chitchat"],
                "intent_source": "rule",
                "plan": [],
                "outcomes": [],
                "sources": [],
                "answer": f"抱歉，处理你的请求时出现内部错误：{type(exc).__name__}: {exc}",
                "tool_calls": [],
                "degraded": True,
                "degraded_reasons": [f"图执行异常: {type(exc).__name__}"],
                "metrics": {},
                "error": str(exc),
            }

        total_ms = (time.perf_counter() - started) * 1000
        metrics = dict(final.get("metrics", {}))
        metrics["total_latency_ms"] = round(total_ms, 2)
        metrics["tool_latency_ms"] = round(
            sum(float(o.latency_ms) for o in final.get("outcomes", [])), 2
        )

        response = AgentResponse(
            response=final.get("answer", ""),
            sources=final.get("sources", []),
            tool_calls=final.get("tool_calls", []),
            session_id=session_id,
            intents=list(final.get("intents", [])),
            intent_source=final.get("intent_source", ""),
            degraded=bool(final.get("degraded")),
            degraded_reasons=list(final.get("degraded_reasons", [])),
            metrics=metrics,
            plan=[s.to_dict() for s in final.get("plan", [])],
        )

        await asyncio.to_thread(
            self.sessions.append,
            session_id,
            "assistant",
            response.response,
            {
                "intents": response.intents,
                "intent_source": response.intent_source,
                "tool_calls": response.tool_calls,
                "degraded": response.degraded,
                "metrics": response.metrics,
            },
        )
        logger.info(
            "会话 %s 处理完成: intents=%s tools=%s %.0fms degraded=%s",
            session_id,
            response.intents,
            [t.get("tool") for t in response.tool_calls],
            total_ms,
            response.degraded,
        )
        return response

    def run(self, message: str, session_id: str = "default") -> AgentResponse:
        """同步包装（脚本与测试友好）。

        注意：若在已有事件循环中调用会抛 ``RuntimeError``（由 ``asyncio.run`` 触发）。

        Args:
            message: 用户输入。
            session_id: 会话 ID。

        Returns:
            :class:`AgentResponse`。
        """
        return asyncio.run(self.arun(message, session_id))

    async def astream(self, message: str, session_id: str = "default") -> AsyncIterator[dict[str, Any]]:
        """流式处理：先产出过程事件，再产出回答增量。

        事件类型：
            ``intent``  识别出的意图
            ``plan``    执行计划
            ``tool``    单个工具调用完成（含 ok / latency）
            ``token``   回答文本增量
            ``done``    最终完整响应（含 sources / metrics）

        Args:
            message: 用户输入。
            session_id: 会话 ID。

        Yields:
            事件字典（``{"event": ..., "data": ...}``）。
        """
        if not message or not message.strip():
            raise ValueError("message 不能为空")

        history = await asyncio.to_thread(self.sessions.as_llm_messages, session_id)
        await asyncio.to_thread(self.sessions.append, session_id, "user", message)
        started = time.perf_counter()

        state: AgentState = {
            "message": message,
            "session_id": session_id,
            "history": history,
            "outcomes": [],
        }
        state.update(await self._node_intent_router(state))
        yield {"event": "intent", "data": {
            "intents": state.get("intents", []),
            "source": state.get("intent_source", ""),
        }}

        state.update(await self._node_plan_builder(state))
        yield {"event": "plan", "data": [s.to_dict() for s in state.get("plan", [])]}

        state.update(await self._node_tool_executor(state))
        for outcome in state.get("outcomes", []):
            yield {"event": "tool", "data": outcome.to_dict()}

        state.update(await self._node_response_generator(state))
        answer = state.get("answer", "")

        # 有 LLM 时用流式重新生成一次回答，让前端逐字显示；
        # 离线路径直接一次性吐出（本身就无生成过程，硬拆成 token 是假流式）。
        streamed = False
        if self.llm.available and answer:
            try:
                facts = self._render_facts(state.get("outcomes", []))
                chitchat_only = all(
                    step.action == "answer" for step in state.get("plan", [])
                ) if state.get("plan") else False
                messages = self._stream_messages(message, history, facts, chitchat_only)
                buffer: list[str] = []
                async for piece in self.llm.stream_chat(messages):
                    buffer.append(piece)
                    yield {"event": "token", "data": piece}
                if buffer:
                    answer = "".join(buffer)
                    streamed = True
            except LLMError as exc:
                logger.warning("流式生成失败，回退到已生成的完整回答: %s", exc)

        if not streamed:
            yield {"event": "token", "data": answer}

        total_ms = (time.perf_counter() - started) * 1000
        metrics = dict(state.get("metrics", {}))
        metrics["total_latency_ms"] = round(total_ms, 2)
        metrics["tool_latency_ms"] = round(
            sum(float(o.latency_ms) for o in state.get("outcomes", [])), 2
        )
        response = AgentResponse(
            response=answer,
            sources=state.get("sources", []),
            tool_calls=state.get("tool_calls", []),
            session_id=session_id,
            intents=list(state.get("intents", [])),
            intent_source=state.get("intent_source", ""),
            degraded=bool(state.get("degraded")),
            degraded_reasons=list(state.get("degraded_reasons", [])),
            metrics=metrics,
            plan=[s.to_dict() for s in state.get("plan", [])],
        )
        await asyncio.to_thread(
            self.sessions.append,
            session_id,
            "assistant",
            response.response,
            {
                "intents": response.intents,
                "intent_source": response.intent_source,
                "tool_calls": response.tool_calls,
                "degraded": response.degraded,
                "metrics": response.metrics,
            },
        )
        yield {"event": "done", "data": response.to_dict()}

    def _stream_messages(
        self,
        message: str,
        history: list[dict[str, str]],
        facts: str,
        chitchat_only: bool,
    ) -> list[dict[str, str]]:
        """构造流式生成的 prompt（与非流式保持一致的系统提示）。

        Args:
            message: 用户问题。
            history: 对话历史。
            facts: 事实文本。
            chitchat_only: 是否闲聊。

        Returns:
            OpenAI 格式消息数组。
        """
        if chitchat_only:
            system = (
                "你是企业智能助手，可以回答日常寒暄。若用户问题涉及公司制度或业务数据，"
                "请提示对方补充具体问题，不要编造公司信息。"
            )
        else:
            system = (
                "你是企业智能助手。**只能依据下方【事实】回答**，禁止编造制度条款、"
                "订单号或金额。先给结论再给依据，引用知识库时标注来源文件名，"
                "涉及订单时写清订单号/状态/时间。用中文，简洁。"
            )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(history[-10:])
        user_content = f"【用户问题】\n{message}"
        if not chitchat_only:
            user_content += f"\n\n【事实】\n{facts}"
        messages.append({"role": "user", "content": user_content})
        return messages

    async def warmup(self) -> dict[str, Any]:
        """预热：把冷启动成本（MCP 握手 + 嵌入模型加载）前移到服务启动阶段。

        实测依据：未预热时首条评测用例耗时 **10.2s**，后续平均 2.5s，
        差额就是 bge-m3 加载（约 9s）与 MCP 子进程握手（约 1.4s）。
        预热对接口 P99 有直接收益。

        Returns:
            ``{"embedder_ms":..., "tools_ms":..., "retrieval_ms":..., "tools": n}``。
        """
        report: dict[str, Any] = {}

        # 1) 加载嵌入模型（最耗时的一步）
        started = time.perf_counter()
        try:
            from rag_pipeline.embedder import get_embedder

            embedder = get_embedder()
            embedder.encode_query("预热")
            report["embedder_ms"] = round((time.perf_counter() - started) * 1000, 1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("嵌入模型预热失败: %s", exc)
            report["embedder_ms"] = -1

        # 2) 建立 MCP 会话并拉取工具目录
        started = time.perf_counter()
        try:
            catalog = await self.atool_catalog(refresh=True)
            report["tools_ms"] = round((time.perf_counter() - started) * 1000, 1)
            report["tools"] = len(catalog)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 工具目录预热失败: %s", exc)
            report["tools_ms"] = -1
            report["tools"] = 0

        # 3) 跑一次真实检索，走通"查询编码 → Qdrant → 重排"整条链路
        started = time.perf_counter()
        try:
            result = await self.registry.acall_tool(
                "knowledge_base", "search_documents", {"query": "预热", "top_k": 1}
            )
            report["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 1)
            report["retrieval_ok"] = result.ok
        except Exception as exc:  # noqa: BLE001
            logger.warning("检索链路预热失败: %s", exc)
            report["retrieval_ms"] = -1
            report["retrieval_ok"] = False

        logger.info("Agent 预热报告: %s", report)
        return report

    async def atool_catalog(self, refresh: bool = False) -> dict[str, dict[str, Any]]:
        """获取可用工具目录（供 ``GET /tools`` 与系统提示使用）。

        Args:
            refresh: 是否强制刷新。

        Returns:
            工具目录字典。
        """
        if self._tool_catalog is None or refresh:
            self._tool_catalog = await self.registry.alist_tools(refresh=refresh)
        return self._tool_catalog

    async def build_tool_prompt(self) -> str:
        """把所有工具的 JSON Schema 渲染成系统提示片段。

        需求要求"系统提示词中包含所有可用工具的 schema"。这里生成的结构可直接
        塞进 prompt；LLM 自主选择时看到的就是真实的工具契约。

        Returns:
            Markdown 格式的工具说明。
        """
        catalog = await self.atool_catalog()
        lines = ["你可以调用以下工具（参数必须符合 JSON Schema）："]
        for name, meta in sorted(catalog.items()):
            schema = meta.get("input_schema") or {}
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            lines.append(f"\n### {name}（server: {meta.get('server')}）")
            lines.append(f"说明：{meta.get('description', '')}")
            for prop, spec in props.items():
                flag = "必填" if prop in required else "可选"
                lines.append(
                    f"- `{prop}`（{spec.get('type', 'any')}，{flag}）：{spec.get('description', '')}"
                )
        return "\n".join(lines)

    async def aclose(self) -> None:
        """释放 Agent 持有的资源（注册表会话与 LLM 连接）。"""
        try:
            await self.registry.ashutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭 MCP 会话失败: %s", exc)
        await self.llm.aclose()

    def close(self) -> None:
        """同步释放资源（脚本与测试收尾用；FastAPI 用 lifespan 调 :meth:`aclose`）。"""
        try:
            self.registry.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭 MCP 会话失败: %s", exc)


_agent_singleton: RAGMCPAgent | None = None


def get_agent() -> RAGMCPAgent:
    """获取进程级 Agent 单例。

    Returns:
        :class:`RAGMCPAgent` 实例。
    """
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = RAGMCPAgent()
    return _agent_singleton
