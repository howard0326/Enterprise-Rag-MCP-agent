"""结构化日志器：把每次请求的模型、token、工具调用与延迟写成 JSONL。

设计要点：
    1. **一行一条 JSON（JSONL）**：便于 ``pandas.read_json(lines=True)`` 直接分析，
       也便于追加写入（不需要读改写整个文件）。
    2. **写入线程安全**：FastAPI 会并发处理请求，用锁串行化追加写，避免行交错。
    3. **默认脱敏**：``log_prompts=False``（配置默认）时只记录消息长度与字符数，
       不落盘原始 prompt —— 企业场景下 prompt 里可能含客户 PII。
    4. **与阶段3 的工具日志互补**：
       ``logs/tool_calls.jsonl`` 是工具粒度（每次 MCP 调用一条），
       ``logs/requests.jsonl`` 是请求粒度（一次 /chat 一条，聚合 token 与总延迟）。
       两者用 ``request_id`` 关联，阶段5 的评测就是靠这个把"回答对不对"与
       "哪次工具调用慢/失败"对上。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.config_loader import PROJECT_ROOT, get_config

__all__ = [
    "StructuredLogger",
    "RequestLogRecord",
    "get_structured_logger",
]

logger = logging.getLogger("backend.logging")


@dataclass
class RequestLogRecord:
    """一次请求的结构化记录。

    Attributes:
        request_id: 请求 ID（与工具日志关联）。
        session_id: 会话 ID。
        message: 用户消息（``log_prompts=False`` 时仅记录长度）。
        message_chars: 用户消息字符数。
        intent_source: 意图识别路径（``llm`` / ``rule``）。
        intents: 识别出的意图。
        tools: 调用的工具名列表。
        tool_calls: 工具调用明细（含耗时与重试次数）。
        model: 实际使用的模型。
        prompt_tokens: 输入 token 数。
        completion_tokens: 输出 token 数。
        llm_latency_ms: LLM 调用耗时。
        tool_latency_ms: 工具调用总耗时。
        total_latency_ms: 端到端耗时。
        degraded: 是否降级。
        degraded_reasons: 降级原因。
        source_count: 引用来源数量。
        answer_chars: 回答字符数。
        error: 致命错误信息。
        timestamp: 记录时间。
    """

    request_id: str
    session_id: str
    message_chars: int
    intent_source: str = ""
    intents: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    degraded: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    source_count: int = 0
    answer_chars: int = 0
    error: str | None = None
    message: str | None = None
    timestamp: str = ""

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 总数。"""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典（工具明细做裁剪，避免日志过大）。"""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "message": self.message,
            "message_chars": self.message_chars,
            "intent_source": self.intent_source,
            "intents": self.intents,
            "tools": self.tools,
            "tool_calls": [
                {
                    "tool": t.get("tool"),
                    "server": t.get("server"),
                    "ok": t.get("ok"),
                    "code": t.get("code"),
                    "attempts": t.get("attempts"),
                    "latency_ms": t.get("latency_ms"),
                    "args": t.get("args"),
                }
                for t in self.tool_calls
            ],
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_latency_ms": round(self.llm_latency_ms, 2),
            "tool_latency_ms": round(self.tool_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "degraded": self.degraded,
            "degraded_reasons": self.degraded_reasons[:3],
            "source_count": self.source_count,
            "answer_chars": self.answer_chars,
            "error": self.error,
        }


class StructuredLogger:
    """JSONL 结构化日志写入器。"""

    def __init__(self, log_file: str | Path | None = None) -> None:
        """初始化写入器。

        Args:
            log_file: 日志路径；``None`` 时读配置
                （``observability.request_log_file``）。
        """
        app = get_config()
        raw = log_file or app.observability.request_log_file
        self.path: Path | None = None
        if raw:
            path = Path(raw)
            self.path = path if path.is_absolute() else PROJECT_ROOT / path
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_prompts = bool(app.observability.log_prompts)
        self._lock = threading.Lock()

    def new_request_id(self) -> str:
        """生成请求 ID。

        Returns:
            形如 ``req-1a2b3c4d5e6f`` 的短 ID。
        """
        return f"req-{uuid.uuid4().hex[:12]}"

    def build_record(
        self,
        *,
        request_id: str,
        session_id: str,
        message: str,
        response: Any = None,
        error: str | None = None,
        started_at: float | None = None,
    ) -> RequestLogRecord:
        """从 Agent 响应构造日志记录。

        Args:
            request_id: 请求 ID。
            session_id: 会话 ID。
            message: 用户消息。
            response: ``AgentResponse`` 或 ``None``（出错时）。
            error: 错误信息。
            started_at: ``time.perf_counter()`` 起点；``None`` 时不计算总耗时。

        Returns:
            :class:`RequestLogRecord`。
        """
        metrics: dict[str, Any] = {}
        if response is not None:
            metrics = dict(getattr(response, "metrics", {}) or {})

        tool_calls = list(getattr(response, "tool_calls", []) or []) if response else []
        total_ms = float(metrics.get("total_latency_ms", 0.0) or 0.0)
        if started_at is not None:
            total_ms = max(total_ms, (time.perf_counter() - started_at) * 1000)

        return RequestLogRecord(
            request_id=request_id,
            session_id=session_id,
            message=message if self.log_prompts else None,
            message_chars=len(message or ""),
            intent_source=str(getattr(response, "intent_source", "") or ""),
            intents=list(getattr(response, "intents", []) or []),
            tools=[str(t.get("tool", "")) for t in tool_calls if isinstance(t, dict)],
            tool_calls=[t for t in tool_calls if isinstance(t, dict)],
            model=str(metrics.get("model", "") or ""),
            prompt_tokens=int(metrics.get("prompt_tokens", 0) or 0),
            completion_tokens=int(metrics.get("completion_tokens", 0) or 0),
            llm_latency_ms=float(metrics.get("latency_ms", 0.0) or 0.0),
            tool_latency_ms=float(metrics.get("tool_latency_ms", 0.0) or 0.0),
            total_latency_ms=total_ms,
            degraded=bool(getattr(response, "degraded", False)),
            degraded_reasons=list(getattr(response, "degraded_reasons", []) or []),
            source_count=len(getattr(response, "sources", []) or []),
            answer_chars=len(str(getattr(response, "response", "") or "")),
            error=error,
            timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        )

    def write(self, record: RequestLogRecord) -> None:
        """追加写入一条记录（线程安全，失败只告警不抛）。

        Args:
            record: 日志记录。
        """
        if self.path is None:
            return
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            logger.warning("结构化日志写入失败: %s", exc)

    def log_response(
        self,
        *,
        session_id: str,
        message: str,
        response: Any = None,
        error: str | None = None,
        started_at: float | None = None,
        request_id: str | None = None,
    ) -> RequestLogRecord:
        """构造并写入一条请求记录。

        Args:
            session_id: 会话 ID。
            message: 用户消息。
            response: Agent 响应。
            error: 错误信息。
            started_at: 计时起点。
            request_id: 请求 ID；``None`` 时自动生成。

        Returns:
            写入的记录。
        """
        record = self.build_record(
            request_id=request_id or self.new_request_id(),
            session_id=session_id,
            message=message,
            response=response,
            error=error,
            started_at=started_at,
        )
        self.write(record)
        return record


_logger_singleton: StructuredLogger | None = None
_singleton_lock = threading.Lock()


def get_structured_logger() -> StructuredLogger:
    """获取进程级结构化日志器单例。

    Returns:
        :class:`StructuredLogger` 实例。
    """
    global _logger_singleton
    if _logger_singleton is None:
        with _singleton_lock:
            if _logger_singleton is None:
                _logger_singleton = StructuredLogger()
    return _logger_singleton
