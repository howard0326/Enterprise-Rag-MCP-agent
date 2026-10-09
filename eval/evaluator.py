"""评测执行器：对 test_cases.json 逐条调用 Agent，统计指标并生成 Markdown 报告。

四项指标（与需求一致）：
    intent_accuracy           意图准确率 —— Agent 识别出的意图是否覆盖期望意图（子集匹配）
    tool_selection_accuracy   工具选择准确率 —— 实际调用的工具是否覆盖期望工具（子集匹配）
    answer_contains_rate      回答包含率 —— 回答中是否出现期望关键内容（任一命中即通过）
    e2e_latency               端到端延迟 —— 均值 / P50 / P95 / 最大

为什么用"子集匹配"而不是精确相等：
    混合问题（"订单 ORD-001 符合退货政策吗"）天然需要多个意图与多个工具。
    若用精确相等，"多识别一个相关意图"会被判错，掩盖真实能力。
    子集匹配衡量的是"该做的有没有做"，同时报告"多做"的比例（``extra_intents``），
    让过度调用（例如对纯知识问题也去查库）暴露出来而不是被隐藏。

失败分类（用于失败案例分析）：
    intent_miss     意图漏识别
    extra_tool      调用了非期望工具（过度调用）
    tool_miss       缺少期望工具
    content_miss    回答未包含期望内容
    tool_error      工具调用失败（含重试后仍失败）
    latency_high    单条耗时超过阈值

用法::

    .venv\\Scripts\\python.exe eval\\evaluator.py                    # 全量 20 条
    .venv\\Scripts\\python.exe eval\\evaluator.py --limit 5           # 只跑前 5 条
    .venv\\Scripts\\python.exe eval\\evaluator.py --category mixed    # 只跑某一类
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _name in ("httpx", "httpcore", "qdrant_client", "mcp"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from backend.agent import RAGMCPAgent  # noqa: E402
from backend.structured_logger import get_structured_logger  # noqa: E402
from config.config_loader import get_config  # noqa: E402

__all__ = ["EvalCase", "CaseResult", "EvalReport", "Evaluator", "main"]

LATENCY_HIGH_MS = 15000.0

# 与 backend.agent._INFORMATIONAL_CODES 保持一致：这些错误码表示"信息不足，需要追问"，
# 不是系统故障。评测器若把它们算作 tool_error，报告就会出现"全通过但仍有工具失败"
# 的自相矛盾（实测踩过）。
_INFORMATIONAL_CODES = {"MISSING_ARGUMENT", "NOT_FOUND", "INVALID_ARGUMENT"}


@dataclass
class EvalCase:
    """单条测试用例。"""

    id: str
    category: str
    input: str
    expected_intent: list[str] = field(default_factory=list)
    expected_tool: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CaseResult:
    """单条用例的执行与判定结果。"""

    case: EvalCase
    actual_intents: list[str] = field(default_factory=list)
    actual_tools: list[str] = field(default_factory=list)
    intent_source: str = ""
    intent_ok: bool = False
    tool_ok: bool = False
    contains_ok: bool = False
    missing_intents: list[str] = field(default_factory=list)
    extra_intents: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    extra_tools: list[str] = field(default_factory=list)
    hit_contains: list[str] = field(default_factory=list)
    missed_contains: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tool_error: bool = False
    degraded: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    answer: str = ""
    failure_types: list[str] = field(default_factory=list)
    exception: str = ""

    @property
    def passed(self) -> bool:
        """三条正确性指标是否全部通过。"""
        return self.intent_ok and self.tool_ok and self.contains_ok

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "id": self.case.id,
            "category": self.case.category,
            "input": self.case.input,
            "expected_intent": self.case.expected_intent,
            "actual_intents": self.actual_intents,
            "intent_ok": self.intent_ok,
            "intent_source": self.intent_source,
            "expected_tool": self.case.expected_tool,
            "actual_tools": self.actual_tools,
            "tool_ok": self.tool_ok,
            "expected_contains": self.case.expected_contains,
            "hit_contains": self.hit_contains,
            "missed_contains": self.missed_contains,
            "contains_ok": self.contains_ok,
            "latency_ms": round(self.latency_ms, 1),
            "degraded": self.degraded,
            "tool_error": self.tool_error,
            "failure_types": self.failure_types,
            "answer_chars": len(self.answer),
            "exception": self.exception,
        }


@dataclass
class EvalReport:
    """整份评测报告。"""

    started_at: str = ""
    finished_at: str = ""
    total: int = 0
    results: list[CaseResult] = field(default_factory=list)
    llm_configured: bool = False
    model: str = ""

    def metric(self, name: str) -> float:
        """计算某项指标（0~1）。

        Args:
            name: ``intent`` / ``tool`` / ``contains`` / ``passed``。

        Returns:
            通过率；无样本时返回 0。
        """
        if not self.results:
            return 0.0
        key = {
            "intent": "intent_ok",
            "tool": "tool_ok",
            "contains": "contains_ok",
            "passed": "passed",
        }[name]
        return sum(1 for r in self.results if getattr(r, key)) / len(self.results)

    def latencies(self) -> list[float]:
        """返回所有用例的端到端延迟。"""
        return [r.latency_ms for r in self.results if r.latency_ms > 0]

    def latency_stats(self) -> dict[str, float]:
        """延迟统计（均值 / P50 / P95 / 最大 / 最小）。

        Returns:
            统计字典；无样本时全部为 0。
        """
        values = self.latencies()
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": ordered[p95_index],
            "max": max(values),
            "min": min(values),
        }

    def by_category(self) -> dict[str, dict[str, Any]]:
        """按分类汇总指标。

        Returns:
            ``{category: {"count":..., "intent":..., "tool":..., "contains":..., "latency":...}}``。
        """
        groups: dict[str, list[CaseResult]] = {}
        for result in self.results:
            groups.setdefault(result.case.category, []).append(result)
        summary: dict[str, dict[str, Any]] = {}
        for category, items in sorted(groups.items()):
            n = len(items)
            summary[category] = {
                "count": n,
                "intent": sum(1 for r in items if r.intent_ok) / n,
                "tool": sum(1 for r in items if r.tool_ok) / n,
                "contains": sum(1 for r in items if r.contains_ok) / n,
                "passed": sum(1 for r in items if r.passed) / n,
                "latency_mean": statistics.fmean([r.latency_ms for r in items]),
            }
        return summary

    def failure_histogram(self) -> dict[str, int]:
        """统计失败类型分布。"""
        histogram: dict[str, int] = {}
        for result in self.results:
            for kind in result.failure_types:
                histogram[kind] = histogram.get(kind, 0) + 1
        return dict(sorted(histogram.items(), key=lambda kv: -kv[1]))


class Evaluator:
    """评测执行器。"""

    def __init__(self, cases: list[EvalCase], agent: RAGMCPAgent | None = None) -> None:
        """初始化。

        Args:
            cases: 测试用例列表。
            agent: 复用已有 Agent；``None`` 时新建。
        """
        self.cases = cases
        self.agent = agent or RAGMCPAgent()
        self.logger = get_structured_logger()

    @staticmethod
    def load_cases(path: str | Path | None = None) -> list[EvalCase]:
        """从 JSON 加载测试用例。

        Args:
            path: 用例文件路径；``None`` 时读配置。

        Returns:
            :class:`EvalCase` 列表。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: JSON 结构非法。
        """
        target = Path(path) if path else ROOT / get_config().eval.test_cases_file.lstrip("./")
        if not target.is_file():
            raise FileNotFoundError(f"测试用例文件不存在: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"测试用例文件结构非法（缺少 cases 数组）: {target}")
        cases: list[EvalCase] = []
        for index, item in enumerate(raw_cases, 1):
            if not isinstance(item, dict) or "input" not in item:
                raise ValueError(f"第 {index} 条用例缺少 input 字段")
            cases.append(
                EvalCase(
                    id=str(item.get("id") or f"CASE-{index:02d}"),
                    category=str(item.get("category") or "uncategorized"),
                    input=str(item["input"]),
                    expected_intent=[str(x) for x in item.get("expected_intent", [])],
                    expected_tool=[str(x) for x in item.get("expected_tool", [])],
                    expected_contains=[str(x) for x in item.get("expected_contains", [])],
                    notes=str(item.get("notes", "")),
                )
            )
        return cases

    async def run_case(self, case: EvalCase, index: int, total: int) -> CaseResult:
        """执行单条用例。

        Args:
            case: 用例。
            index: 序号（从 1 开始，用于日志）。
            total: 总数。

        Returns:
            :class:`CaseResult`。
        """
        result = CaseResult(case=case)
        session_id = f"eval-{case.id.lower()}"
        started = time.perf_counter()
        request_id = self.logger.new_request_id()
        print(f"  [{index:2d}/{total}] {case.id:8s} {case.input[:34]:36s}", end="", flush=True)

        try:
            response = await self.agent.arun(case.input, session_id=session_id)
            elapsed = (time.perf_counter() - started) * 1000
            result.latency_ms = float(response.metrics.get("total_latency_ms", elapsed) or elapsed)
            result.actual_intents = [str(x) for x in response.intents]
            result.intent_source = response.intent_source
            result.actual_tools = [
                str(t.get("tool", "")) for t in response.tool_calls if isinstance(t, dict)
            ]
            result.tool_error = any(
                (not t.get("ok", False)) and str(t.get("code") or "") not in _INFORMATIONAL_CODES
                for t in response.tool_calls
                if isinstance(t, dict)
            )
            result.degraded = response.degraded
            result.degraded_reasons = list(response.degraded_reasons)
            result.answer = response.response
            self.logger.log_response(
                session_id=session_id,
                message=case.input,
                response=response,
                started_at=started,
                request_id=request_id,
            )
        except Exception as exc:  # noqa: BLE001
            result.latency_ms = (time.perf_counter() - started) * 1000
            result.exception = f"{type(exc).__name__}: {exc}"
            self.logger.log_response(
                session_id=session_id,
                message=case.input,
                error=result.exception,
                started_at=started,
                request_id=request_id,
            )

        # ---- 判定 ----
        result.missing_intents = [i for i in case.expected_intent if i not in result.actual_intents]
        result.extra_intents = [i for i in result.actual_intents if i not in case.expected_intent]
        result.intent_ok = not result.missing_intents and not result.exception

        result.missing_tools = [t for t in case.expected_tool if t not in result.actual_tools]
        result.extra_tools = [t for t in result.actual_tools if t not in case.expected_tool]
        result.tool_ok = not result.missing_tools and not result.exception

        result.hit_contains = [k for k in case.expected_contains if k in result.answer]
        result.missed_contains = [k for k in case.expected_contains if k not in result.answer]
        result.contains_ok = bool(result.hit_contains) and not result.exception

        # ---- 失败分类 ----
        if result.exception:
            result.failure_types.append("exception")
        if result.missing_intents:
            result.failure_types.append("intent_miss")
        if result.missing_tools:
            result.failure_types.append("tool_miss")
        if result.extra_tools:
            result.failure_types.append("extra_tool")
        if not result.contains_ok:
            result.failure_types.append("content_miss")
        if result.tool_error:
            result.failure_types.append("tool_error")
        if result.latency_ms > LATENCY_HIGH_MS:
            result.failure_types.append("latency_high")

        mark = "PASS" if result.passed else "FAIL"
        print(f" {mark} {result.latency_ms / 1000:.1f}s tools={result.actual_tools}")
        return result

    async def run(self) -> EvalReport:
        """执行全部用例。

        Returns:
            :class:`EvalReport`。
        """
        report = EvalReport(
            started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            total=len(self.cases),
            llm_configured=self.agent.llm.available,
            model=self.agent.llm.config.model if self.agent.llm.available else "offline-rule-based",
        )
        print(
            f"[eval] 共 {len(self.cases)} 条用例；LLM="
            f"{'已配置（意图识别走模型）' if report.llm_configured else '未配置（走规则路径）'}"
        )
        # 先预热：否则第一条用例要独自承担 MCP 握手 + bge-m3 加载的冷启动成本
        # （实测首条 10.2s vs 后续均值 2.5s），会把首条延迟误记成模型能力问题。
        started = time.perf_counter()
        try:
            warm = await self.agent.warmup()
            print(f"[eval] 预热完成 {time.perf_counter() - started:.1f}s: {warm}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] 预热失败（首条会变慢）: {exc}\n")
        for index, case in enumerate(self.cases, 1):
            report.results.append(await self.run_case(case, index, len(self.cases)))
        report.finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return report


def _percent(value: float) -> str:
    """格式化为百分比。"""
    return f"{value * 100:.1f}%"


def render_report(report: EvalReport) -> str:
    """把评测报告渲染成 Markdown。

    Args:
        report: 评测报告。

    Returns:
        Markdown 文本。
    """
    lat = report.latency_stats()
    histogram = report.failure_histogram()
    lines: list[str] = [
        "# RAG + MCP 智能助手评测报告",
        "",
        f"- 执行时间：{report.started_at} → {report.finished_at}",
        f"- 用例总数：**{report.total}**",
        f"- 意图识别路径：**{'LLM（' + report.model + '）' if report.llm_configured else '规则匹配（未配置 API Key）'}**",
        "",
        "## 1. 总体指标",
        "",
        "| 指标 | 数值 | 通过的用例数 |",
        "| --- | --- | --- |",
        f"| 意图准确率 | **{_percent(report.metric('intent'))}** | "
        f"{sum(1 for r in report.results if r.intent_ok)}/{report.total} |",
        f"| 工具选择准确率 | **{_percent(report.metric('tool'))}** | "
        f"{sum(1 for r in report.results if r.tool_ok)}/{report.total} |",
        f"| 回答包含率 | **{_percent(report.metric('contains'))}** | "
        f"{sum(1 for r in report.results if r.contains_ok)}/{report.total} |",
        f"| 三项全通过 | **{_percent(report.metric('passed'))}** | "
        f"{sum(1 for r in report.results if r.passed)}/{report.total} |",
        f"| 端到端延迟（均值） | **{lat['mean'] / 1000:.2f}s** | "
        f"P50 {lat['p50'] / 1000:.2f}s / P95 {lat['p95'] / 1000:.2f}s / "
        f"max {lat['max'] / 1000:.2f}s |",
        "",
        "## 2. 分类指标",
        "",
        "| 分类 | 用例数 | 意图准确率 | 工具选择准确率 | 回答包含率 | 全通过 | 平均延迟 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for category, stat in report.by_category().items():
        lines.append(
            f"| {category} | {stat['count']} | {_percent(stat['intent'])} | "
            f"{_percent(stat['tool'])} | {_percent(stat['contains'])} | "
            f"{_percent(stat['passed'])} | {stat['latency_mean'] / 1000:.2f}s |"
        )

    lines += ["", "## 3. 失败类型分布", ""]
    if histogram:
        lines += ["| 失败类型 | 次数 | 说明 |", "| --- | --- | --- |"]
        meaning = {
            "intent_miss": "意图漏识别",
            "tool_miss": "缺少期望的工具调用",
            "extra_tool": "调用了非期望工具（过度调用）",
            "content_miss": "回答未包含期望的关键内容",
            "tool_error": "工具调用失败",
            "latency_high": f"单条耗时超过 {LATENCY_HIGH_MS / 1000:.0f}s",
            "exception": "执行抛异常",
        }
        for kind, count in histogram.items():
            lines.append(f"| {kind} | {count} | {meaning.get(kind, '')} |")
    else:
        lines.append("无失败用例。")

    lines += ["", "## 4. 逐条结果", "",
              "| ID | 分类 | 输入 | 期望意图 | 实际意图 | 期望工具 | 实际工具 | 回答命中 | 延迟 | 判定 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for result in report.results:
        hit = "/".join(result.hit_contains) if result.hit_contains else "—"
        lines.append(
            f"| {result.case.id} | {result.case.category} | {result.case.input[:26]} | "
            f"{','.join(result.case.expected_intent)} | {','.join(result.actual_intents)} | "
            f"{','.join(result.case.expected_tool) or '—'} | {','.join(result.actual_tools) or '—'} | "
            f"{hit} | {result.latency_ms / 1000:.1f}s | {'✅' if result.passed else '❌'} |"
        )

    failures = [r for r in report.results if not r.passed]
    lines += ["", "## 5. 失败案例分析", ""]
    if not failures:
        lines.append("全部用例通过，无失败案例。")
    for result in failures:
        lines += [
            f"### {result.case.id} — {result.case.input}",
            "",
            f"- **失败类型**：{', '.join(result.failure_types) or '未分类'}",
            f"- **出题意图**：{result.case.notes}",
            f"- **期望**：意图 {result.case.expected_intent} / 工具 {result.case.expected_tool} / "
            f"包含 {result.case.expected_contains}",
            f"- **实际**：意图 {result.actual_intents}（来源 {result.intent_source}）/ "
            f"工具 {result.actual_tools} / 命中 {result.hit_contains or '无'}",
        ]
        if result.missing_intents or result.missing_tools or result.missed_contains:
            lines.append(
                f"- **差距**：漏意图 {result.missing_intents or '无'}；"
                f"漏工具 {result.missing_tools or '无'}；"
                f"漏内容 {result.missed_contains or '无'}"
            )
        if result.extra_tools:
            lines.append(f"- **过度调用**：{result.extra_tools}")
        if result.tool_error or result.degraded:
            lines.append(f"- **工具/降级**：{result.degraded_reasons[:2] or '工具调用失败'}")
        if result.exception:
            lines.append(f"- **异常**：`{result.exception}`")
        lines += ["", "**回答摘录**：", "", "```", result.answer[:600].strip() or "(空)", "```", ""]

    lines += [
        "",
        "## 6. 结论与改进方向",
        "",
        f"- 意图准确率 {_percent(report.metric('intent'))}、工具选择准确率 "
        f"{_percent(report.metric('tool'))}、回答包含率 {_percent(report.metric('contains'))}。",
    ]
    if histogram.get("content_miss"):
        lines.append(
            f"- `content_miss` 出现 {histogram['content_miss']} 次：说明检索召回的片段"
            "不足以覆盖期望数值，优先检查分块粒度与 Top-K。"
        )
    if histogram.get("extra_tool"):
        lines.append(
            f"- `extra_tool` 出现 {histogram['extra_tool']} 次：存在过度调用，"
            "会增加延迟；可在规划阶段加「是否需要」的二次确认。"
        )
    if histogram.get("intent_miss"):
        lines.append(
            f"- `intent_miss` 出现 {histogram['intent_miss']} 次：意图识别漏项，"
            "检查规则关键词表或 LLM 分类 prompt。"
        )
    if lat["p95"] > 10000:
        lines.append(
            f"- P95 延迟 {lat['p95'] / 1000:.1f}s 偏高：主要来自冷启动（MCP 握手 + "
            "bge-m3 加载）。生产环境应在服务启动时预热，而非首个请求时加载。"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码：0 表示所有用例三项指标全通过，1 存在失败。
    """
    parser = argparse.ArgumentParser(description="RAG + MCP 智能助手评测")
    parser.add_argument("--cases", default=None, help="测试用例 JSON 路径")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--category", default=None, help="只跑指定分类")
    parser.add_argument("--output", default=None, help="报告输出路径（默认 eval/eval_report.md）")
    parser.add_argument("--json-output", default=None, help="逐条结果 JSON 输出路径")
    args = parser.parse_args(argv)

    cases = Evaluator.load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print("[eval] 没有匹配的用例", file=sys.stderr)
        return 2

    evaluator = Evaluator(cases)
    report = asyncio.run(evaluator.run())

    markdown = render_report(report)
    out_path = Path(args.output) if args.output else ROOT / "eval" / "eval_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    json_path = Path(args.json_output) if args.json_output else ROOT / "eval" / "eval_results.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": {
                    "started_at": report.started_at,
                    "finished_at": report.finished_at,
                    "total": report.total,
                    "llm_configured": report.llm_configured,
                    "model": report.model,
                    "intent_accuracy": report.metric("intent"),
                    "tool_selection_accuracy": report.metric("tool"),
                    "answer_contains_rate": report.metric("contains"),
                    "passed_rate": report.metric("passed"),
                    "latency": report.latency_stats(),
                    "failure_histogram": report.failure_histogram(),
                },
                "results": [r.to_dict() for r in report.results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lat = report.latency_stats()
    print("\n" + "=" * 66)
    print(f"  意图准确率    : {_percent(report.metric('intent'))}")
    print(f"  工具选择准确率: {_percent(report.metric('tool'))}")
    print(f"  回答包含率    : {_percent(report.metric('contains'))}")
    print(f"  三项全通过    : {_percent(report.metric('passed'))}")
    print(f"  端到端延迟    : 均值 {lat['mean'] / 1000:.2f}s / P95 {lat['p95'] / 1000:.2f}s")
    print(f"  报告          : {out_path}")
    print(f"  逐条结果      : {json_path}")
    print("=" * 66)
    return 0 if report.metric("passed") == 1.0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # 规避 MCP 子进程清理在解释器退出阶段的长时间阻塞
    os._exit(code)
