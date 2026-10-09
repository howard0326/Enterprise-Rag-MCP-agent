"""business_db MCP server 包。

启动方式（stdio 传输）::

    python -m mcp_servers.business_db.server

阶段3：接入 SQLite（自动建表 + demo 数据），实现参数化查询与工单写入。

注意：与 knowledge_base 同理，这里使用 PEP 562 惰性导出，避免
``python -m mcp_servers.business_db.server`` 触发 runpy 重复导入警告。
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
        from mcp_servers.business_db import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
