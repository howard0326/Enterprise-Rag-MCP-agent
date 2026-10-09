"""FastAPI 服务：RAG + MCP 智能助手的 HTTP 接口（含 SSE 流式输出）。

接口一览::

    POST /chat        {"message": "...", "session_id": "..."}  → 非流式完整响应
    POST /chat/stream {"message": "...", "session_id": "..."}  → SSE 事件流
    GET  /health      服务与依赖健康检查
    GET  /tools       MCP 工具目录（含 JSON Schema）
    GET  /sessions    最近活跃会话
    DELETE /sessions/{session_id}  清空指定会话历史

SSE 事件序列（``POST /chat/stream``）::

    event: intent   data: {"intents": [...], "source": "llm|rule"}
    event: plan     data: [{"action": "rag_search", "tool": "search_documents", ...}]
    event: tool     data: {"tool": "search_documents", "ok": true, "latency_ms": 168.2, ...}
    event: token    data: "回答文本增量"
    event: done     data: {"response": "...", "sources": [...], "tool_calls": [...], "metrics": {...}}
    event: error    data: {"message": "..."}   （仅在致命错误时出现）

关键工程点：
    1. **工具调用不能阻塞事件循环**：MCP 注册表是同步阻塞接口，Agent 内部已用
       ``asyncio.to_thread`` 包裹（见 backend/agent.py）。SSE 心跳因此不会被卡住。
    2. **lifespan 统一收尾**：Agent 持有 MCP 子进程与 LLM 长连接，必须在进程退出时
       关闭，否则会残留 python.exe 子进程（阶段3 实测过）。
    3. **session_id 可选**：不传时自动生成 UUID，响应里回传，便于前端保持上下文。
    4. **SSE 采用 POST**：请求体带 message/session_id，GET 只能靠 query 传参，
       长文本会被 URL 长度限制；因此用 POST + ``text/event-stream`` 响应。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import RAGMCPAgent, get_agent
from backend.session_store import get_session_store
from config.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("backend.api")

__all__ = ["app", "ChatRequest", "ChatResponse"]


class ChatRequest(BaseModel):
    """``POST /chat`` 请求体。"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户消息",
        examples=["订单 ORD-001 符合退货政策吗？"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="会话 ID；不传则自动生成并随响应返回",
    )


class ChatResponse(BaseModel):
    """``POST /chat`` 响应体。"""

    response: str = Field(..., description="回答正文（Markdown）")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="引用来源（文件、页码、相关度）"
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="工具调用记录（含耗时与重试次数）"
    )
    session_id: str = Field(..., description="会话 ID")
    intents: list[str] = Field(default_factory=list, description="识别出的意图")
    intent_source: str = Field(default="", description="意图识别路径：llm / rule")
    degraded: bool = Field(default=False, description="是否发生降级（工具或 LLM 失败）")
    degraded_reasons: list[str] = Field(default_factory=list, description="降级原因")
    metrics: dict[str, Any] = Field(default_factory=dict, description="token 与延迟指标")
    plan: list[dict[str, Any]] = Field(default_factory=list, description="执行计划")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时预热 Agent，退出时关闭 MCP 子进程与 LLM 连接。

    为什么必须预热（实测数据）：
        首个请求要付两次冷启动成本 —— MCP 子进程握手（约 1.4s）+ bge-m3 加载
        （约 9s），实测第一条评测用例耗时 **10.2s**，而后续请求平均只要 2.5s。
        若把这份成本留在首个用户请求上，接口的 P99 会被一个与业务无关的
        加载过程支配。因此在 lifespan 里主动触发一次检索与一次工具调用目录拉取，
        把成本前移到启动阶段。

    Args:
        application: FastAPI 实例。

    Yields:
        无（控制权交还框架）。
    """
    config = get_config()
    agent = get_agent()
    logger.info(
        "服务启动: %s v%s, MCP servers=%s, LLM=%s(%s)",
        config.project.name,
        config.project.version,
        agent.registry.enabled_server_names(),
        config.llm.model,
        "已配置 Key" if agent.llm.available else "未配置 Key（走离线降级路径）",
    )
    if not agent.llm.available:
        logger.warning(
            "未配置 DEEPSEEK_API_KEY：意图识别与回答生成将走规则式离线路径，"
            "响应里的 intent_source 会是 rule、metrics.model 会是 offline-rule-based。"
        )

    # ---- 预热：把冷启动成本移到启动阶段 ----
    warmup_started = time.perf_counter()
    try:
        await agent.warmup()
        logger.info("预热完成，耗时 %.1fs", time.perf_counter() - warmup_started)
    except Exception as exc:  # noqa: BLE001 - 预热失败不应阻止服务启动
        logger.warning("预热失败（服务仍可启动，但首个请求会变慢）: %s", exc)

    try:
        yield
    finally:
        logger.info("服务关闭，正在释放 MCP 会话与 LLM 连接…")
        try:
            await agent.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("释放资源失败: %s", exc)


app = FastAPI(
    title="企业级 RAG + MCP 智能助手",
    version="0.1.0",
    description="LangGraph 编排 + 混合检索知识库 + 业务数据库工具调用",
    lifespan=lifespan,
)

_cors = get_config().api.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 演示控制台：挂静态目录，与 API 同源（避免用 file:// 打开时的跨域限制）。
# 访问 http://127.0.0.1:8000/demo
# 为什么需要它：Qdrant 的 Windows 单文件版**不带 Web UI 静态资源**，
# http://127.0.0.1:6333/dashboard 会返回 404（Docker 镜像里才有 dashboard），
# 所以这里提供一个等价的展示页，顺便把"看向量库"和"对话"合到一屏。
_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_web_dir), html=True), name="demo")


def _agent() -> RAGMCPAgent:
    """获取 Agent 单例（供路由使用，便于测试时替换）。"""
    return get_agent()


def _sse(event: str, data: Any) -> str:
    """构造一条 SSE 消息。

    Args:
        event: 事件名。
        data: 事件负载（自动 JSON 序列化；字符串原样作为文本）。

    Returns:
        SSE 格式字符串（以双换行结尾）。
    """
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    # SSE 规范：data 字段内不能出现裸换行，需按行加 data: 前缀
    lines = payload.split("\n")
    body = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{body}\n\n"


@app.post("/chat", response_model=ChatResponse, summary="对话（非流式）")
async def chat(request: ChatRequest) -> ChatResponse:
    """处理一条用户消息并返回完整响应。

    Args:
        request: 请求体。

    Returns:
        :class:`ChatResponse`。

    Raises:
        HTTPException: 400 消息为空；500 处理异常。
    """
    session_id = request.session_id or f"sess-{uuid.uuid4().hex[:12]}"
    try:
        result = await _agent().arun(request.message, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("处理 /chat 失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return ChatResponse(**result.to_dict())


@app.post("/chat/stream", summary="对话（SSE 流式）")
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """以 SSE 流式返回处理过程与回答。

    事件序列见模块 docstring。客户端可用 ``curl -N`` 或浏览器 ``EventSource``
    （注意 EventSource 不支持 POST，前端需用 fetch + ReadableStream 解析）。

    Args:
        request: 请求体。
        http_request: 原始请求（用于检测客户端断开）。

    Returns:
        ``text/event-stream`` 响应。

    Raises:
        HTTPException: 400 消息为空。
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    session_id = request.session_id or f"sess-{uuid.uuid4().hex[:12]}"

    async def generator() -> AsyncIterator[str]:
        """产出 SSE 事件。"""
        try:
            async for event in _agent().astream(request.message, session_id=session_id):
                if await http_request.is_disconnected():
                    logger.info("客户端断开，停止推送 session=%s", session_id)
                    return
                yield _sse(str(event.get("event", "message")), event.get("data"))
        except asyncio.CancelledError:  # pragma: no cover - 客户端断开
            logger.info("SSE 连接被取消 session=%s", session_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE 流处理失败")
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 避免反向代理缓冲导致"流式"变批量
            "X-Session-Id": session_id,
        },
    )


@app.get("/health", summary="健康检查")
async def health() -> dict[str, Any]:
    """检查服务、MCP server 与依赖状态。

    Returns:
        ``{"status": "ok|degraded", "servers": [...], "llm": {...}, "sessions": n}``。
    """
    agent = _agent()
    try:
        servers = await agent.registry.ahealth_check()
    except Exception as exc:  # noqa: BLE001
        servers = [{"server": "registry", "ok": False, "error": f"{type(exc).__name__}: {exc}"}]
    all_ok = all(bool(item.get("ok")) for item in servers) if servers else False
    sessions = await asyncio.to_thread(agent.sessions.list_sessions, 50)
    return {
        "status": "ok" if all_ok else "degraded",
        "servers": servers,
        "llm": {
            "model": agent.llm.config.model,
            "configured": agent.llm.available,
            "intent_source": "llm" if agent.llm.available else "rule",
        },
        "sessions": len(sessions),
    }


@app.get("/tools", summary="MCP 工具目录")
async def tools(refresh: bool = Query(default=False, description="是否强制刷新")) -> dict[str, Any]:
    """返回所有已启用工具及其 JSON Schema。

    Args:
        refresh: 是否绕过缓存重新拉取。

    Returns:
        ``{"count": n, "tools": {...}}``。
    """
    catalog = await _agent().atool_catalog(refresh=refresh)
    return {"count": len(catalog), "tools": catalog}


@app.get("/sessions", summary="最近会话")
async def sessions(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    """列出最近活跃的会话。

    Args:
        limit: 返回条数上限。

    Returns:
        ``{"count": n, "sessions": [...]}``。
    """
    items = await asyncio.to_thread(get_session_store().list_sessions, limit)
    return {"count": len(items), "sessions": items}


@app.get("/sessions/{session_id}", summary="会话历史")
async def session_history(
    session_id: str, turns: int = Query(default=10, ge=1, le=50)
) -> dict[str, Any]:
    """读取指定会话的历史轮次。

    Args:
        session_id: 会话 ID。
        turns: 读取最近 N 轮。

    Returns:
        ``{"session_id":..., "turns": [...]}``。
    """
    store = get_session_store()
    items = await asyncio.to_thread(store.turns, session_id)
    return {
        "session_id": session_id,
        "turns": [t.to_dict() for t in items[-turns:]],
    }


@app.delete("/sessions/{session_id}", summary="清空会话")
async def clear_session(session_id: str) -> dict[str, Any]:
    """清空指定会话的历史消息。

    Args:
        session_id: 会话 ID。

    Returns:
        ``{"session_id":..., "deleted": n}``。
    """
    deleted = await asyncio.to_thread(get_session_store().clear, session_id)
    return {"session_id": session_id, "deleted": deleted}


def main() -> None:
    """以 uvicorn 启动服务（``python -m backend.main``）。"""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "backend.main:app",
        host=config.api.host,
        port=config.api.port,
        log_level=config.observability.level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
