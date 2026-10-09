"""RAG 管道包：文档解析 → 分块 → 向量/关键词索引 → 混合检索。

阶段2 交付文件：
    document_parser.py    PDF / Markdown / 纯文本解析 + 512/64 分块 + 元数据抽取
    indexer.py            Qdrant collection 建库、batch=100 批量写入、索引统计
    retriever.py          向量 + 关键词混合检索、重排序（向量相似度 > 0.75 优先）
    ingestion_cli.py      增量索引命令行入口（基于文件 hash 跳过已索引文档）

阶段1 仅占用包位置与导入路径。
"""

from __future__ import annotations

__all__: list[str] = []
