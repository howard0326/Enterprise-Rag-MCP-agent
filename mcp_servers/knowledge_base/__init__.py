"""knowledge_base MCP server 包。

启动方式（stdio 传输）::

    python -m mcp_servers.knowledge_base.server

阶段3：接入 rag_pipeline.retriever，实现真实检索与元数据查询。

注意：这里刻意**不在顶层导入** server 模块的符号。否则
``python -m mcp_servers.knowledge_base.server`` 会先通过本包导入 server，
runpy 再执行一次同名模块，抛出
``found in sys.modules after import of package`` 警告并可能产生两个
独立的 server 实例。改用 PEP 562 的模块级 ``__getattr__`` 惰性导出。
"""

from __future__ import annotations

from typing import Any

__all__ = ["SERVER_NAME", "SERVER_VERSION", "build_server"]


def __getattr__(name: str) -> Any:
    """惰性导出 server 模块中的公开符号。

    Args:
        name: 被访问的属性名。

    Returns:
        server 模块中对应对象的引用。

    Raises:
        AttributeError: 访问了未导出的属性名。
    """
    if name in __all__:
        from mcp_servers.knowledge_base import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
