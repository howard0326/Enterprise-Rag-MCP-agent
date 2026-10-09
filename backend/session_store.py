"""会话存储：SQLite 持久化对话历史，每个 session 保留最近 N 轮。

设计要点：
    1. **"轮"的定义**：一轮 = 一条 user 消息 + 其对应的 assistant 回复。
       ``max_turns=10`` 表示保留最近 20 条消息（10 轮）。裁剪按"整轮"进行，
       保证喂给模型的上下文永远从 user 消息开始 —— 否则历史会以 assistant
       消息开头，模型容易误判对话结构。
    2. **写放大控制**：裁剪在写入时执行（DELETE 旧行），而不是读取时过滤，
       避免表无限增长；同时保证同一 session 的历史查询是 O(保留量)。
    3. **线程安全**：``check_same_thread=False`` + 线程锁。FastAPI 会把同步
       存储调用丢到线程池执行，必须自己保证串行。
    4. **与阶段3 的 business_db 分离**：会话库（``.data/sessions.db``）与业务库
       （``.data/business.db``）是两份独立文件，避免评测/测试清空对话时误删业务数据。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.config_loader import PROJECT_ROOT, SessionConfig, get_config

__all__ = [
    "SessionStoreError",
    "SessionMessage",
    "Turn",
    "SessionStore",
    "get_session_store",
]


class SessionStoreError(RuntimeError):
    """会话存储失败（路径不可写、SQL 错误等）。"""


@dataclass
class SessionMessage:
    """单条消息。

    Attributes:
        role: ``user`` / ``assistant`` / ``system``。
        content: 消息正文。
        created_at: 写入时间（ISO 8601）。
        meta: 附加信息（意图、工具调用、来源等），序列化为 JSON 存储。
    """

    role: str
    content: str
    created_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "meta": self.meta,
        }


@dataclass
class Turn:
    """一轮对话（user + assistant）。"""

    user: SessionMessage
    assistant: SessionMessage | None = None

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "user": self.user.to_dict(),
            "assistant": self.assistant.to_dict() if self.assistant else None,
        }


class SessionStore:
    """SQLite 会话存储。"""

    def __init__(self, config: SessionConfig | None = None) -> None:
        """初始化存储并建表。

        Args:
            config: 会话配置；``None`` 时读 ``config/settings.yaml``。
        """
        self.config = config or get_config().api.session
        path = Path(self.config.db_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.max_turns = max(1, int(self.config.max_turns))
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        except sqlite3.Error as exc:
            raise SessionStoreError(f"会话库初始化失败: {path} -> {exc}") from exc

    def _init_schema(self) -> None:
        """建表与索引。"""
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    meta_json  TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages (session_id, id);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> SessionMessage:
        """追加一条消息并按需裁剪历史。

        Args:
            session_id: 会话 ID。
            role: 角色（``user`` / ``assistant`` / ``system``）。
            content: 消息正文。
            meta: 附加信息（意图、工具调用、来源等）。

        Returns:
            写入的消息对象。

        Raises:
            SessionStoreError: 写入失败。
        """
        if not session_id:
            raise SessionStoreError("session_id 不能为空")
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO messages (session_id, role, content, meta_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, meta_json, created_at),
                )
                self._conn.commit()
                self._trim_locked(session_id)
            except sqlite3.Error as exc:
                self._conn.rollback()
                raise SessionStoreError(f"消息写入失败: {exc}") from exc
        return SessionMessage(role=role, content=content, created_at=created_at, meta=meta or {})

    def _trim_locked(self, session_id: str) -> None:
        """裁剪历史，只保留最近 ``max_turns`` 轮（调用方需已持锁）。

        按"整轮"裁剪：找出第 max_turns 轮之前的 user 消息 id，删除更早的行。
        若历史以 assistant 开头（异常情况），也一并清掉，保证上下文结构正确。

        Args:
            session_id: 会话 ID。
        """
        keep = self.max_turns * 2  # 每轮 2 条消息
        rows = self._conn.execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, keep),
        ).fetchall()
        if not rows:
            return
        cutoff = min(int(r["id"]) for r in rows)
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id < ?", (session_id, cutoff)
        )
        # 保证历史以 user 消息开头：起始若是 assistant，则删除该条
        first = self._conn.execute(
            "SELECT id, role FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        if first is not None and first["role"] != "user":
            self._conn.execute("DELETE FROM messages WHERE id = ?", (int(first["id"]),))
        self._conn.commit()

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def history(self, session_id: str, limit_turns: int | None = None) -> list[SessionMessage]:
        """读取会话历史（时间正序）。

        Args:
            session_id: 会话 ID。
            limit_turns: 只取最近 N 轮；``None`` 时取配置中的上限。

        Returns:
            :class:`SessionMessage` 列表。
        """
        turns = limit_turns or self.max_turns
        limit = max(1, turns) * 2
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta_json, created_at FROM ("
                "  SELECT id, role, content, meta_json, created_at FROM messages"
                "   WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (session_id, limit),
            ).fetchall()

        messages: list[SessionMessage] = []
        for row in rows:
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            messages.append(
                SessionMessage(
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    meta=meta if isinstance(meta, dict) else {},
                )
            )
        return messages

    def as_llm_messages(self, session_id: str) -> list[dict[str, str]]:
        """把历史转成 OpenAI 兼容的消息数组（只含 role/content）。

        Args:
            session_id: 会话 ID。

        Returns:
            ``[{"role": "user", "content": "..."}, ...]``。
        """
        return [
            {"role": m.role, "content": m.content}
            for m in self.history(session_id)
            if m.role in ("user", "assistant")
        ]

    def turns(self, session_id: str) -> list[Turn]:
        """把历史配对成轮次（便于评测与前端渲染）。

        Args:
            session_id: 会话 ID。

        Returns:
            :class:`Turn` 列表；落单的 user 消息 assistant 为 ``None``。
        """
        result: list[Turn] = []
        pending: SessionMessage | None = None
        for msg in self.history(session_id):
            if msg.role == "user":
                if pending is not None:
                    result.append(Turn(user=pending))
                pending = msg
            elif msg.role == "assistant" and pending is not None:
                result.append(Turn(user=pending, assistant=msg))
                pending = None
        if pending is not None:
            result.append(Turn(user=pending))
        return result

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """列出最近活跃的会话。

        Args:
            limit: 返回条数上限。

        Returns:
            ``[{"session_id":..., "messages":..., "updated_at":...}]``。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, COUNT(*) AS n, MAX(created_at) AS updated_at"
                "  FROM messages GROUP BY session_id ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "messages": int(row["n"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def clear(self, session_id: str | None = None) -> int:
        """清空会话历史。

        Args:
            session_id: 指定会话；``None`` 时清空全部。

        Returns:
            删除的消息条数。
        """
        with self._lock:
            if session_id is None:
                cur = self._conn.execute("DELETE FROM messages")
            else:
                cur = self._conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (session_id,)
                )
            self._conn.commit()
            return int(cur.rowcount or 0)

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass


_store_singleton: SessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """获取进程级会话存储单例。

    Returns:
        :class:`SessionStore` 实例。
    """
    global _store_singleton
    if _store_singleton is None:
        with _store_lock:
            if _store_singleton is None:
                _store_singleton = SessionStore()
    return _store_singleton
