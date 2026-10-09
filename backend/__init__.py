"""后端服务包：LangGraph Agent 编排 + FastAPI/SSE 接口 + 会话与可观测性。

阶段4 交付文件：
    agent.py                LangGraph 状态机（intent_router → rag_search / db_query /
                            tool_call → response_generator）、工具重试与降级
    main.py                 POST /chat（SSE 流式）、GET /health、GET /tools
    session_store.py        SQLite 会话存储，每 session 保留最近 10 轮

阶段5 交付文件：
    structured_logger.py    结构化 JSONL 日志（model / prompt_tokens /
                            completion_tokens / tool_calls / latency）

阶段1 仅占用包位置与导入路径。
"""

from __future__ import annotations

__all__: list[str] = []
