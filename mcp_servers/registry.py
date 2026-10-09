"""MCP server 注册表：加载 settings.yaml、管理 stdio 会话池、统一调用接口。

对外接口：
    list_tools()                              聚合所有已启用工具（含 JSON Schema）
    call_tool(server_name, tool_name, args)   定位 server → 校验开关 → 调用 → 打日志
    health_check()                            逐个 server 握手并报告耗时
    shutdown()                                关闭全部会话

两个关键设计决策（有明确理由，不是随手写的）：

    1. **会话池而不是"每次调用起进程"**：
       MCP 的 stdio 传输下，``initialize`` 握手与子进程启动都有固定成本
       （本机实测：拉起 Python 子进程 + 导入依赖约 1~3 秒）。若每次工具调用都
       重新起进程，阶段4 Agent 的多步调用会慢到不可用。因此按 server 名维护
       常驻会话，首次调用时懒加载，进程退出时统一关闭。

    2. **工具开关在注册表层强制**：
       ``settings.yaml`` 里 ``mcp.servers[].enabled`` 与 ``tools[].enabled`` 两级开关
       都在这里生效。禁用不是"从目录里隐藏"那么简单 —— 直接调用被禁用的工具
       同样会被拒绝，否则 Agent 绕过目录仍能触发（阶段4 的降级测试会用到）。

并发与事件循环约束（本模块最容易踩的坑，两次实测踩过）：
    1. 会话内部的 anyio 任务组**绑定在创建它的事件循环上** —— 跨循环关闭会抛
       ``Attempted to exit a cancel scope that isn't the current task's current cancel scope``
       并挂死。因此本模块维护一个常驻事件循环线程（``_LoopRunner``）。
    2. 更严格的是：进入与退出 ``stdio_client`` / ``ClientSession`` 的
       **必须是同一个 task**，仅同一个循环还不够，否则抛
       ``Attempted to exit cancel scope in a different task than it was entered in``。
       因此每个 server 由一个**专职常驻任务**持有会话上下文，所有调用通过队列
       投递给该任务顺序执行（`_session_worker`）。

    对外接口因此分成两层：
        * ``run(coro)``：同步上下文阻塞等待循环线程结果；在已有事件循环中调用会
          抛 ``RuntimeError``（提示改用 await acall_tool 或 asyncio.to_thread）。
        * ``acall_tool`` 等异步接口：可在 FastAPI 中直接 await。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine

from config.config_loader import MCPServerConfig, PROJECT_ROOT, get_config, resolve_path

__all__ = [
    "MCPRegistryError",
    "ServerNotRegisteredError",
    "ToolDisabledError",
    "ToolNotAvailableError",
    "ToolCallResult",
    "MCPServerRegistry",
    "get_registry",
    "reset_registry",
]

logger = logging.getLogger("mcp.registry")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class MCPRegistryError(RuntimeError):
    """注册表级错误基类。"""


class ServerNotRegisteredError(MCPRegistryError):
    """请求的 server 未在 settings.yaml 中注册。"""


class ToolDisabledError(MCPRegistryError):
    """工具或所属 server 已被配置禁用。"""


class ToolNotAvailableError(MCPRegistryError):
    """server 已注册但未暴露该工具。"""


@dataclass
class ToolCallResult:
    """一次工具调用的结果。

    Attributes:
        server: server 名。
        tool: 工具名。
        ok: 业务是否成功（由工具返回信封的 ``ok`` 字段决定）。
        data: 业务数据（``ok=True`` 时）。
        error: 错误描述（``ok=False`` 时）。
        code: 错误码（``ok=False`` 时）。
        raw: 工具返回的原始信封。
        latency_ms: 端到端耗时（毫秒），含协议往返。
        timestamp: 调用时间（ISO 8601）。
    """

    server: str
    tool: str
    ok: bool
    data: Any = None
    error: str | None = None
    code: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: str = ""

    def to_log_record(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """构造结构化日志记录（阶段4/5 直接消费该结构）。

        Args:
            args: 本次调用的入参。

        Returns:
            ``{timestamp, server, tool, args, result, latency_ms, ok, code}``。
        """
        return {
            "timestamp": self.timestamp,
            "server": self.server,
            "tool": self.tool,
            "args": args or {},
            "ok": self.ok,
            "code": self.code,
            "error": self.error,
            "result": self.data
            if isinstance(self.data, (dict, list))
            else (str(self.data)[:500] if self.data is not None else None),
            "latency_ms": round(self.latency_ms, 2),
        }


class _LoopRunner:
    """常驻事件循环线程：保证所有 MCP 会话都在同一个循环内创建与关闭。"""

    def __init__(self, heartbeat_seconds: float = 0.0) -> None:
        """启动后台事件循环线程。

        Args:
            heartbeat_seconds: 大于 0 时启动心跳任务，按该间隔检测事件循环是否
                被长时间占用并打印告警。用于诊断"工具调用固定耗时等于超时阈值"
                这类问题（那是循环线程被阻塞的典型特征）。默认关闭。
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-registry-loop", daemon=True
        )
        self._thread.start()
        if heartbeat_seconds > 0:
            asyncio.run_coroutine_threadsafe(self._heartbeat(heartbeat_seconds), self._loop)
        logger.debug("MCP 注册表事件循环线程已启动")

    async def _heartbeat(self, interval: float) -> None:
        """周期性检测事件循环是否被阻塞。

        Args:
            interval: 检测间隔（秒）。
        """
        loop = asyncio.get_running_loop()
        last = loop.time()
        while True:
            try:
                await asyncio.sleep(interval)
                now = loop.time()
                drift = now - last - interval
                if drift > interval:
                    logger.warning(
                        "[heartbeat] 事件循环线程被阻塞约 %.1fs（有长任务占住了线程）", drift
                    )
                last = now
            except asyncio.CancelledError:  # pragma: no cover
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("[heartbeat] 异常: %s", exc)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """把协程投递到循环线程并返回 future。

        Args:
            coro: 待执行的协程。

        Returns:
            ``concurrent.futures.Future``。
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        """停止事件循环线程（幂等）。"""
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()


class MCPServerRegistry:
    """MCP server 注册表与会话池。"""

    def __init__(self, log_file: str | Path | None = None) -> None:
        """初始化注册表。

        Args:
            log_file: 结构化日志路径；``None`` 时读配置
                （``observability.structured_log_file``）。传空字符串可禁用落盘。
        """
        app = get_config()
        self.app_config = app
        self.servers: dict[str, MCPServerConfig] = {
            s.name: s for s in app.mcp.servers if s.enabled
        }
        self.disabled_servers: dict[str, MCPServerConfig] = {
            s.name: s for s in app.mcp.servers if not s.enabled
        }
        self._log_path: Path | None = (
            resolve_path(log_file or app.observability.structured_log_file)
            if (log_file or app.observability.structured_log_file)
            else None
        )
        self._sessions: dict[str, Any] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._worker_ready: dict[str, asyncio.Event] = {}
        self._worker_errors: dict[str, Exception] = {}
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._tool_catalog: dict[str, dict[str, Any]] | None = None
        self._runner = _LoopRunner()

    def _assert_owner_loop(self) -> asyncio.AbstractEventLoop:
        """确认（或首次绑定）会话所属的事件循环。

        为什么必须绑定：MCP 会话的 anyio 任务组与它所在的循环/任务强绑定，
        跨循环使用会挂死。因此**第一个发起异步调用的循环**成为属主，
        后续都必须在同一循环里调用。

        实测背景：FastAPI 里若经 ``asyncio.to_thread`` → 同步 ``call_tool``
        触发跨线程桥接，工具调用会固定卡满超时（子进程日志显示同一操作仅需 3ms）；
        改为直接 await ``acall_tool`` 后恢复正常。

        Returns:
            当前运行中的事件循环。

        Raises:
            MCPRegistryError: 在非属主循环中调用。
        """
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            logger.info("MCP 注册表会话绑定到事件循环 %s", id(loop))
        elif self._owner_loop is not loop:
            raise MCPRegistryError(
                "MCPServerRegistry 的会话绑定在另一个事件循环上，不能跨循环使用。"
                "请在同一事件循环内 await acall_tool(...)；同步上下文请用 call_tool(...)，"
                "它会通过独立的循环线程桥接。"
            )
        return loop

    # ------------------------------------------------------------------
    # 目录
    # ------------------------------------------------------------------
    def enabled_server_names(self) -> list[str]:
        """返回已启用的 server 名列表。"""
        return list(self.servers.keys())

    def is_tool_enabled(self, server_name: str, tool_name: str) -> bool:
        """判断某工具是否被配置启用（server 级与 tool 级双重判定）。

        Args:
            server_name: server 名。
            tool_name: 工具名。

        Returns:
            启用返回 True。
        """
        server = self.servers.get(server_name)
        if server is None:
            return False
        tool = server.get_tool(tool_name)
        return bool(tool and tool.enabled)

    async def alist_tools(self, refresh: bool = False) -> dict[str, dict[str, Any]]:
        """聚合所有已启用工具，返回扁平化目录。

        Args:
            refresh: 为 True 时忽略缓存，重新握手拉取。

        Returns:
            ``{tool_name: {"server":..., "description":..., "input_schema":..., "title":...}}``。
            同名工具以先注册的 server 为准（配置已保证 server 内不重名）。
        """
        if self._tool_catalog is not None and not refresh:
            return self._tool_catalog

        self._assert_owner_loop()
        catalog: dict[str, dict[str, Any]] = {}
        for name, server in self.servers.items():
            try:
                listed = await self._call_in_session(name, lambda s: s.list_tools())
            except Exception as exc:  # noqa: BLE001
                logger.error("server '%s' 工具目录拉取失败: %s: %s", name, type(exc).__name__, exc)
                continue

            for tool in listed.tools:
                schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", {})
                if not self.is_tool_enabled(name, tool.name):
                    logger.info("工具 '%s.%s' 已按配置禁用，跳过", name, tool.name)
                    continue
                if tool.name in catalog:
                    logger.warning(
                        "工具名冲突: '%s' 同时存在于 %s 与 %s，保留前者",
                        tool.name,
                        catalog[tool.name]["server"],
                        name,
                    )
                    continue
                catalog[tool.name] = {
                    "server": name,
                    "title": getattr(tool, "title", "") or "",
                    "description": getattr(tool, "description", "") or "",
                    "input_schema": schema,
                }

        self._tool_catalog = catalog
        return catalog

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def _server_params(self, name: str) -> Any:
        """把配置转换为 StdioServerParameters。

        Args:
            name: server 名。

        Returns:
            ``StdioServerParameters`` 实例。

        Raises:
            ServerNotRegisteredError: 未注册或已禁用。
        """
        from mcp import StdioServerParameters

        server = self.servers.get(name)
        if server is None:
            if name in self.disabled_servers:
                raise ToolDisabledError(f"server '{name}' 已在 settings.yaml 中禁用")
            raise ServerNotRegisteredError(
                f"server '{name}' 未注册；已启用: {sorted(self.servers)}"
            )

        command = server.command
        # 配置里写 "python" 时，改用当前解释器：venv 环境下裸 python 可能
        # 指向系统解释器，导致子进程缺少本项目依赖。
        if command in ("python", "python3"):
            command = sys.executable

        env = dict(server.env)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUNBUFFERED", "1")

        cwd = server.cwd or "."
        cwd_path = Path(cwd)
        if not cwd_path.is_absolute():
            cwd_path = PROJECT_ROOT / cwd_path

        return StdioServerParameters(
            command=command,
            args=list(server.args),
            cwd=str(cwd_path),
            env=env,
        )

    async def _session_worker(self, name: str) -> None:
        """专职任务：持有某个 server 的会话上下文并顺序处理调用请求。

        为什么必须这样写（实测结论，见 .tmp/probe_shutdown.py 的最小复现）：
            * 进入与退出 ``stdio_client`` / ``ClientSession`` 必须在**同一个 task**；
            * 更关键的是**不能用 ``AsyncExitStack`` 手动组合**：实测
              ``stack.aclose()`` 会让 anyio 任务组在非属主 task 中退出，抛
              ``RuntimeError: Attempted to exit a cancel scope that isn't the
              current tasks's current cancel scope``（即使调用方就是同一个 worker 任务）。
              只有原生嵌套 ``async with`` 才能保证退出发生在进入的同一语句块内。

        因此这里用两层 ``async with`` 固定住生命周期，内部循环处理队列请求。

        Args:
            name: server 名。
        """
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        ready = self._worker_ready[name]
        try:
            params = self._server_params(name)
        except MCPRegistryError as exc:
            self._worker_errors[name] = exc
            ready.set()
            return

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._sessions[name] = session
                    logger.info("MCP server '%s' 会话已建立（command=%s）", name, params.command)
                    ready.set()

                    while True:
                        item = await self._queues[name].get()
                        if item is None:  # 关闭哨兵
                            break
                        coro_factory, reply = item
                        try:
                            reply.set_result(await coro_factory(session))
                        except Exception as exc:  # noqa: BLE001
                            reply.set_exception(exc)
                    logger.info("MCP server '%s' 收到关闭哨兵，准备释放会话", name)
        except Exception as exc:  # noqa: BLE001
            logger.error("server '%s' 会话异常: %s: %s", name, type(exc).__name__, exc)
            self._worker_errors[name] = MCPRegistryError(
                f"连接 MCP server '{name}' 失败: {type(exc).__name__}: {exc}"
            )
            ready.set()
        finally:
            self._sessions.pop(name, None)
            logger.info("MCP server '%s' 会话已关闭", name)

    async def _get_session(self, name: str) -> Any:
        """获取（或懒加载）指定 server 的常驻会话。

        Args:
            name: server 名。

        Returns:
            已 ``initialize`` 的 ``ClientSession``。

        Raises:
            ToolDisabledError: server 被禁用。
            ServerNotRegisteredError: server 未注册。
            MCPRegistryError: 握手失败或会话已崩溃。
        """
        if name in self._sessions:
            return self._sessions[name]

        self._assert_owner_loop()
        if name in self._tasks:
            # 已有 worker：等待其就绪或报错
            await self._worker_ready[name].wait()
            if name in self._worker_errors:
                raise self._worker_errors[name]
            if name in self._sessions:
                return self._sessions[name]
            raise MCPRegistryError(f"server '{name}' 会话不可用")

        self._server_params(name)  # 提前抛出未注册/已禁用
        self._worker_ready[name] = asyncio.Event()
        self._queues[name] = asyncio.Queue()
        self._tasks[name] = asyncio.create_task(self._session_worker(name))
        await self._worker_ready[name].wait()
        if name in self._worker_errors:
            raise self._worker_errors[name]
        session = self._sessions.get(name)
        if session is None:
            raise MCPRegistryError(f"server '{name}' 会话建立后不可用")
        return session

    async def _call_in_session(self, name: str, factory: Any) -> Any:
        """把一次会话操作投递给该 server 的专职任务执行。

        Args:
            name: server 名。
            factory: 接收 session 返回协程的可调用对象。

        Returns:
            操作结果。

        Raises:
            MCPRegistryError: 会话不可用或 worker 已退出。
        """
        await self._get_session(name)
        reply: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queues[name].put((factory, reply))
        return await reply

    async def ashutdown(self) -> None:
        """关闭全部会话与子进程（幂等）。

        注意：不主动 ``task.cancel()``。会话上下文（anyio 任务组）必须由持有它的
        那个 task 自行退出，外部 cancel 会触发
        ``Attempted to exit a cancel scope that isn't the current task's current cancel scope``。
        因此这里只投递关闭哨兵并等待；超时也只告警，把任务留给循环收尾。
        """
        for name, queue in list(self._queues.items()):
            task = self._tasks.get(name)
            if task is None or task.done():
                continue
            try:
                await queue.put(None)
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
                logger.debug("server '%s' 会话已正常关闭", name)
            except asyncio.TimeoutError:
                logger.warning(
                    "server '%s' 会话在 10 秒内未关闭，交由事件循环收尾", name
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("关闭 server '%s' 会话时出错: %s", name, exc)
        self._sessions.clear()
        self._tasks.clear()
        self._queues.clear()
        self._worker_errors.clear()
        self._worker_ready.clear()
        self._tool_catalog = None

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    async def _reset_session(self, name: str) -> None:
        """强制丢弃某个 server 的会话与子进程，下次调用会重建。

        用于"会话卡死"的自愈：实测出现过工具调用恰好耗时等于超时阈值
        （60s / 120s，且对端日志显示同一操作仅需 3ms）的情况，说明 channel
        已不可用而不是对端慢。此时重建会话是唯一可靠的恢复手段。

        Args:
            name: server 名。
        """
        task = self._tasks.get(name)
        queue = self._queues.get(name)
        if queue is not None and task is not None and not task.done():
            try:
                await queue.put(None)
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("重建 server '%s' 会话时关闭旧会话失败: %s", name, exc)
        self._sessions.pop(name, None)
        self._tasks.pop(name, None)
        self._queues.pop(name, None)
        self._worker_ready.pop(name, None)
        self._worker_errors.pop(name, None)
        logger.info("server '%s' 会话已重置", name)

    async def acall_tool(
        self, server_name: str, tool_name: str, args: dict[str, Any] | None = None
    ) -> ToolCallResult:
        """异步调用工具（同一事件循环内复用常驻会话）。

        通道级失败（超时/协议错误）时会**重建会话并重试一次**：这类失败说明
        channel 已不可用，重试同一个会话没有意义，但换新会话往往立即成功。

        Args:
            server_name: server 名。
            tool_name: 工具名。
            args: 工具入参。

        Returns:
            :class:`ToolCallResult`。

        Raises:
            ServerNotRegisteredError: server 未注册。
            ToolDisabledError: server 或工具被禁用。
            ToolNotAvailableError: 工具不存在。
        """
        if server_name in self.disabled_servers:
            raise ToolDisabledError(f"server '{server_name}' 已在 settings.yaml 中禁用")
        if server_name not in self.servers:
            raise ServerNotRegisteredError(
                f"server '{server_name}' 未注册；已启用: {sorted(self.servers)}"
            )
        if not self.is_tool_enabled(server_name, tool_name):
            tool_cfg = self.servers[server_name].get_tool(tool_name)
            if tool_cfg is None:
                listed = self.servers[server_name].enabled_tool_names()
                raise ToolNotAvailableError(
                    f"server '{server_name}' 未暴露工具 '{tool_name}'；可用: {listed}"
                )
            raise ToolDisabledError(
                f"工具 '{server_name}.{tool_name}' 已在 settings.yaml 中禁用"
            )

        await self._get_session(server_name)
        payload = dict(args or {})

        total_started = time.perf_counter()
        transcript: list[dict[str, Any]] = []
        result: ToolCallResult | None = None
        max_transport_tries = 2

        for attempt in range(1, max_transport_tries + 1):
            started = time.perf_counter()
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            try:
                raw_result = await self._call_in_session(
                    server_name, lambda s: s.call_tool(tool_name, payload)
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = (time.perf_counter() - started) * 1000
                logger.warning(
                    "server '%s' 调用 %s 第 %d 次失败: %s: %s（%.0fms）",
                    server_name,
                    tool_name,
                    attempt,
                    type(exc).__name__,
                    exc,
                    elapsed,
                )
                transcript.append(
                    {"try": attempt, "transport_error": f"{type(exc).__name__}: {exc}"}
                )
                if attempt < max_transport_tries:
                    await self._reset_session(server_name)
                    continue
                result = ToolCallResult(
                    server=server_name,
                    tool=tool_name,
                    ok=False,
                    error=f"协议层调用失败（已重建会话重试）: {type(exc).__name__}: {exc}",
                    code="PROTOCOL_ERROR",
                    latency_ms=(time.perf_counter() - total_started) * 1000,
                    timestamp=timestamp,
                    raw={"transport_attempts": transcript},
                )
                break

            result = self._parse_result(
                server_name, tool_name, raw_result, started, timestamp, total_started
            )
            if result.ok or result.code not in (None, "PROTOCOL_ERROR"):
                break
            if attempt < max_transport_tries:
                logger.info(
                    "server '%s' 调用 %s 返回 %s，重建会话后重试",
                    server_name,
                    tool_name,
                    result.code,
                )
                transcript.append({"try": attempt, "tool_error": result.error})
                await self._reset_session(server_name)
                continue
            break

        assert result is not None
        result.raw = {**result.raw, "transport_attempts": transcript}
        self._write_log(result, payload)
        return result

    def _parse_result(
        self,
        server_name: str,
        tool_name: str,
        raw_result: Any,
        started: float,
        timestamp: str,
        total_started: float | None = None,
    ) -> ToolCallResult:
        """把 MCP 返回内容解析为统一信封。

        Args:
            server_name: server 名。
            tool_name: 工具名。
            raw_result: ``CallToolResult``。
            started: 本次尝试的起始时间戳（``time.perf_counter``）。
            timestamp: ISO 8601 调用时间。
            total_started: 整个调用（含重试）的起始时间戳；``None`` 时用 ``started``。

        Returns:
            :class:`ToolCallResult`。
        """
        elapsed = (time.perf_counter() - (total_started or started)) * 1000
        text = ""
        for block in getattr(raw_result, "content", []) or []:
            text += getattr(block, "text", "") or ""

        envelope: dict[str, Any] = {}
        if text.strip():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    envelope = parsed
                else:
                    envelope = {"ok": True, "data": parsed, "error": None}
            except json.JSONDecodeError:
                envelope = {"ok": True, "data": text, "error": None}

        # mcp 1.x 为 isError，2.x 改为 is_error
        is_error = getattr(raw_result, "is_error", None)
        if is_error is None:
            is_error = getattr(raw_result, "isError", False)
        ok = bool(envelope.get("ok", not is_error))
        return ToolCallResult(
            server=server_name,
            tool=tool_name,
            ok=ok,
            data=envelope.get("data"),
            error=envelope.get("error"),
            code=envelope.get("code"),
            raw=envelope,
            latency_ms=elapsed,
            timestamp=timestamp,
        )

    def _write_log(self, result: ToolCallResult, args: dict[str, Any]) -> None:
        """把调用结果追加写入结构化日志（JSONL）。

        Args:
            result: 调用结果。
            args: 调用入参。
        """
        record = result.to_log_record(args)
        if result.ok:
            logger.info(
                "call %s.%s -> ok（%.0fms）", result.server, result.tool, result.latency_ms
            )
        else:
            logger.warning(
                "call %s.%s -> fail code=%s（%.0fms）: %s",
                result.server,
                result.tool,
                result.code,
                result.latency_ms,
                result.error,
            )
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("结构化日志写入失败: %s", exc)

    # ------------------------------------------------------------------
    # 同步接口（脚本/测试友好）
    # ------------------------------------------------------------------
    async def ahealth_check(self) -> list[dict[str, Any]]:
        """逐个 server 握手并报告状态。

        Returns:
            ``[{"server":..., "ok":..., "tools":[...], "latency_ms":..., "error":...}]``。
        """
        report: list[dict[str, Any]] = []
        for name in self.servers:
            started = time.perf_counter()
            entry: dict[str, Any] = {"server": name, "ok": False, "tools": [], "error": None}
            try:
                listed = await self._call_in_session(name, lambda s: s.list_tools())
                entry["tools"] = [t.name for t in listed.tools]
                entry["ok"] = True
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            report.append(entry)
        return report

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """在常驻循环线程中执行协程并阻塞等待结果。

        Args:
            coro: 待执行的协程对象。

        Returns:
            协程返回值。

        Raises:
            RuntimeError: 在已有事件循环的线程中调用（如 FastAPI 路由内直接调用
                同步接口）。此时请改用 ``await registry.acall_tool(...)``，
                或用 ``await asyncio.to_thread(registry.call_tool, ...)``。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # 无运行中的循环，可安全阻塞等待
        else:
            raise RuntimeError(
                "MCPServerRegistry.run() 不能在已有事件循环中调用（会阻塞该循环）。"
                "请改用 await registry.acall_tool(...)，"
                "或用 await asyncio.to_thread(registry.call_tool, ...)。"
            )
        # 加超时兜底：MCP 协议层若因对端异常而永不返回，不能把调用方一起挂死
        timeout = float(self.app_config.mcp.call_timeout) + 30.0
        try:
            return self._runner.submit(coro).result(timeout=timeout)
        except TimeoutError as exc:
            raise MCPRegistryError(
                f"MCP 调用超时（{timeout:.0f}s）：对端 server 未在预期时间内返回。"
                f"请检查 server 子进程是否卡住或崩溃。"
            ) from exc

    def list_tools(self, refresh: bool = False) -> dict[str, dict[str, Any]]:
        """同步版 :meth:`alist_tools`（复用常驻会话）。

        Args:
            refresh: 是否强制刷新。

        Returns:
            工具目录字典。
        """
        return self.run(self.alist_tools(refresh=refresh))

    def call_tool(
        self, server_name: str, tool_name: str, args: dict[str, Any] | None = None
    ) -> ToolCallResult:
        """同步版 :meth:`acall_tool`（复用常驻会话，不关闭会话）。

        Args:
            server_name: server 名。
            tool_name: 工具名。
            args: 工具入参。

        Returns:
            :class:`ToolCallResult`。
        """
        return self.run(self.acall_tool(server_name, tool_name, args))

    def health_check(self) -> list[dict[str, Any]]:
        """同步版 :meth:`ahealth_check`。

        Returns:
            各 server 状态列表。
        """
        return self.run(self.ahealth_check())

    def shutdown(self) -> None:
        """同步版 :meth:`ashutdown`：关闭全部会话后停止循环线程。"""
        try:
            self.run(self.ashutdown())
        except RuntimeError as exc:
            logger.warning("shutdown 时事件循环异常: %s", exc)
        self._runner.stop()


_registry_singleton: MCPServerRegistry | None = None


def get_registry() -> MCPServerRegistry:
    """获取进程级注册表单例。

    Returns:
        :class:`MCPServerRegistry` 实例。
    """
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = MCPServerRegistry()
    return _registry_singleton


def reset_registry() -> None:
    """销毁单例（测试用；会关闭既有会话）。"""
    global _registry_singleton
    if _registry_singleton is not None:
        _registry_singleton.shutdown()
        _registry_singleton = None
