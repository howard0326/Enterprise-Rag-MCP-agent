"""评测包：20 条测试用例、指标计算与 Markdown 报告生成。

阶段5 交付文件：
    test_cases.json    20 条用例，覆盖纯知识问答 / 纯数据查询 / 混合问题 / 操作请求
    evaluator.py       逐条调用 Agent，统计意图准确率、工具选择准确率、
                       回答包含率、端到端延迟，输出 eval_report.md

阶段1 仅占用包位置与导入路径。
"""

from __future__ import annotations

__all__: list[str] = []
