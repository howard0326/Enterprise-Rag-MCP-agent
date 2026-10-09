"""配置包。

注意：这里刻意**不在顶层导入** config_loader 的符号。
原因：``python -m config.config_loader`` 会先导入本包再执行子模块，
若顶层已 re-export config_loader，runpy 会抛出
``found in sys.modules after import of package 'config'`` 警告。
因此改用 PEP 562 的模块级 ``__getattr__`` 做惰性导出。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅供类型检查器与 IDE 补全使用，运行时不触发导入
    from config.config_loader import AppConfig, ConfigError

__all__ = [
    "AppConfig",
    "ConfigError",
    "get_config",
    "get_mcp_server",
    "get_mcp_servers",
    "load_settings",
    "reload_config",
    "resolve_path",
]


def __getattr__(name: str) -> Any:
    """惰性导出 config_loader 中的公开符号。

    Args:
        name: 被访问的属性名。

    Returns:
        config_loader 中对应对象的引用。

    Raises:
        AttributeError: 访问了未导出的属性名。
    """
    if name in __all__:
        from config import config_loader

        return getattr(config_loader, name)
    raise AttributeError(f"module 'config' has no attribute {name!r}")


def __dir__() -> list[str]:
    """返回包内可访问的符号列表（供 dir() 与 IDE 使用）。"""
    return sorted(__all__)
