"""增量摄取 CLI：把目录下的文档解析、分块、索引进 Qdrant。

用法::

    python -m rag_pipeline.ingestion_cli --path ./docs --collection enterprise_docs
    python -m rag_pipeline.ingestion_cli --path ./docs --recreate        # 清空重建
    python -m rag_pipeline.ingestion_cli --path ./docs --dry-run         # 只解析不入库
    python -m rag_pipeline.ingestion_cli --stats                         # 只看集合现状

增量索引的两道判重（互为补充，缺一不可）：

    1. **本地状态文件**（``.state/ingestion_state.json``）：记录
       ``doc_id -> {file_hash, mtime, size, source, chunk_count, indexed_at}``。
       同一文件内容未变则直接跳过，省掉解析与嵌入的开销。
    2. **Qdrant 元数据集合**：状态文件可能被删除或换机器，此时靠集合里已有的
       ``file_hash`` 判重。两道判重都通过才真正写入，保证
       "重复执行同一命令不会产生重复片段"。

退出码：0 成功（允许存在被跳过的文档），1 存在失败文档，2 参数或环境错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.config_loader import PROJECT_ROOT, IngestionConfig, get_config, resolve_path
from rag_pipeline.document_parser import DocumentParser, ParsedDocument
from rag_pipeline.indexer import IndexStats, QdrantIndexer
from rag_pipeline.retriever import HybridRetriever

__all__ = ["IngestionState", "IngestionRunner", "main"]


@dataclass
class StateRecord:
    """单个文档的索引状态。

    Attributes:
        file_hash: 文件内容哈希（判重依据）。
        source: 相对路径。
        chunk_count: 片段数。
        size: 文件字节数。
        mtime: 文件修改时间（Unix 时间戳）。
        indexed_at: 入库时间（ISO 8601）。
    """

    file_hash: str
    source: str
    chunk_count: int
    size: int
    mtime: float
    indexed_at: str

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化字典。"""
        return {
            "file_hash": self.file_hash,
            "source": self.source,
            "chunk_count": self.chunk_count,
            "size": self.size,
            "mtime": self.mtime,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateRecord:
        """从字典恢复状态记录。

        Args:
            data: 状态字典（缺失字段用默认值兜底）。

        Returns:
            :class:`StateRecord` 实例。
        """
        return cls(
            file_hash=str(data.get("file_hash", "")),
            source=str(data.get("source", "")),
            chunk_count=int(data.get("chunk_count", 0)),
            size=int(data.get("size", 0)),
            mtime=float(data.get("mtime", 0.0)),
            indexed_at=str(data.get("indexed_at", "")),
        )


class IngestionState:
    """本地摄取状态（JSON 文件）。"""

    def __init__(self, path: str | Path) -> None:
        """初始化状态存储。

        Args:
            path: 状态文件路径（相对路径按项目根解析）。
        """
        self.path = resolve_path(path)
        self.records: dict[str, StateRecord] = {}
        self._load()

    def _load(self) -> None:
        """从磁盘加载状态；文件不存在或损坏时以空状态启动。"""
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            documents = raw.get("documents", {}) if isinstance(raw, dict) else {}
            for doc_id, item in documents.items():
                if isinstance(item, dict):
                    self.records[str(doc_id)] = StateRecord.from_dict(item)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[warn] 状态文件读取失败，将以空状态继续（不会丢数据，但首次会重新判定）："
                f"{self.path} -> {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            self.records = {}

    def save(self) -> None:
        """把状态写回磁盘（自动创建父目录）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "documents": {doc_id: rec.to_dict() for doc_id, rec in sorted(self.records.items())},
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def is_unchanged(self, doc_id: str, file_hash: str) -> bool:
        """判断文档是否与上次索引一致。

        Args:
            doc_id: 文档 ID。
            file_hash: 当前文件哈希。

        Returns:
            一致返回 True（调用方应跳过该文件）。
        """
        record = self.records.get(doc_id)
        return bool(record and record.file_hash == file_hash)

    def update(self, doc_id: str, source: str, file_hash: str, stat: Any, chunk_count: int) -> None:
        """更新单个文档的状态。

        Args:
            doc_id: 文档 ID。
            source: 相对路径。
            file_hash: 文件哈希。
            stat: ``os.stat_result``。
            chunk_count: 片段数。
        """
        self.records[doc_id] = StateRecord(
            file_hash=file_hash,
            source=source,
            chunk_count=chunk_count,
            size=int(getattr(stat, "st_size", 0)),
            mtime=float(getattr(stat, "st_mtime", 0.0)),
            indexed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )


@dataclass
class RunReport:
    """整次摄取的结果报告。"""

    stats: IndexStats = field(default_factory=IndexStats)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> str:
        """生成人类可读报告。

        Returns:
            多行文本报告。
        """
        lines = [
            "",
            "=" * 68,
            "摄取报告",
            "=" * 68,
            f"  新增/更新文档 : {self.stats.documents}",
            f"  跳过(未变化)  : {self.stats.skipped}",
            f"  失败          : {self.stats.failed}",
            f"  写入片段      : {self.stats.chunks}",
            f"  写入批次      : {self.stats.batches}",
            f"  总耗时        : {self.stats.elapsed_sec:.2f}s "
            f"({self.stats.chunks_per_sec:.1f} 片段/s)",
        ]
        if self.skipped_files:
            lines.append("  跳过的文件    :")
            lines.extend(f"    - {name}" for name in self.skipped_files)
        if self.failed_files:
            lines.append("  失败的文件    :")
            lines.extend(f"    - {name}: {err}" for name, err in self.failed_files)
        lines.append("=" * 68)
        return "\n".join(lines)


class IngestionRunner:
    """摄取流程编排。"""

    def __init__(
        self,
        collection: str | None = None,
        ingestion_config: IngestionConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        """初始化摄取器。

        Args:
            collection: 覆盖配置中的集合名。
            ingestion_config: 摄取配置。
            dry_run: 为 True 时只解析不入库（用于验证解析质量）。
        """
        app = get_config()
        self.ingestion = ingestion_config or app.ingestion
        self.dry_run = dry_run

        vector_config = app.vector_store.model_copy(
            update={"collection": collection} if collection else {}
        )
        self.indexer = QdrantIndexer(vector_config=vector_config, ingestion_config=self.ingestion)
        self.parser = DocumentParser(self.ingestion)
        self.state = IngestionState(self.ingestion.state_file)

    def run(self, directory: str | Path, recursive: bool = True, recreate: bool = False) -> RunReport:
        """执行摄取。

        Args:
            directory: 待摄取目录。
            recursive: 是否递归子目录。
            recreate: 是否先重建集合（清空既有数据与状态）。

        Returns:
            :class:`RunReport`。

        Raises:
            FileNotFoundError: 目录不存在。
            IndexerError: Qdrant 连接失败。
        """
        report = RunReport()
        started = time.perf_counter()
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"目录不存在: {root}")

        if recreate and not self.dry_run:
            print(f"[info] 重建集合: {self.indexer.collection} / {self.indexer.meta_collection}")
            self.indexer.ensure_collections(recreate=True)
            self.state.records = {}
        elif not self.dry_run:
            self.indexer.ensure_collections()

        # 第二道判重：Qdrant 侧已有的 hash（状态文件丢失时兜底）
        remote_hashes: dict[str, str] = {}
        if not self.dry_run:
            try:
                remote_hashes = self.indexer.indexed_doc_hashes()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 读取远端索引状态失败，仅依赖本地状态: {exc}", file=sys.stderr)

        print(f"[info] 扫描目录: {root.resolve()}")
        for file_path, parsed in self.parser.parse_directory(root, recursive=recursive):
            rel = str(file_path)
            if isinstance(parsed, Exception):
                report.stats.failed += 1
                report.failed_files.append((rel, f"{type(parsed).__name__}: {parsed}"))
                print(f"  [FAIL] {rel} -> {type(parsed).__name__}: {parsed}", file=sys.stderr)
                continue

            assert isinstance(parsed, ParsedDocument)
            doc_id = parsed.doc_id
            file_hash = parsed.metadata.file_hash

            unchanged_local = self.state.is_unchanged(doc_id, file_hash)
            unchanged_remote = remote_hashes.get(doc_id) == file_hash
            if unchanged_local or unchanged_remote:
                report.stats.skipped += 1
                report.skipped_files.append(f"{parsed.metadata.source} (hash={file_hash[:12]})")
                print(
                    f"  [SKIP] {parsed.metadata.source} "
                    f"({'本地状态一致' if unchanged_local else '远端索引一致'})"
                )
                continue

            if self.dry_run:
                report.stats.documents += 1
                report.stats.chunks += len(parsed.chunks)
                print(
                    f"  [DRY ] {parsed.metadata.source} -> {len(parsed.chunks)} 片段, "
                    f"{parsed.metadata.page_count} 页, 类型 {parsed.metadata.content_type}"
                )
                continue

            try:
                stats = self.indexer.index_document(parsed)
            except Exception as exc:  # noqa: BLE001
                report.stats.failed += 1
                report.failed_files.append((rel, f"{type(exc).__name__}: {exc}"))
                print(f"  [FAIL] {rel} -> {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            report.stats.merge(stats)
            self.state.update(
                doc_id, parsed.metadata.source, file_hash, file_path.stat(), len(parsed.chunks)
            )
            print(
                f"  [OK  ] {parsed.metadata.source} -> {stats.chunks} 片段 "
                f"({stats.elapsed_sec:.1f}s)"
            )

        report.stats.elapsed_sec = time.perf_counter() - started
        if not self.dry_run:
            self.state.save()
        return report


def _print_stats(runner: IngestionRunner) -> None:
    """打印集合现状。

    Args:
        runner: 已构造的摄取器。
    """
    retriever = HybridRetriever(indexer=runner.indexer)
    stats = retriever.stats()
    print("集合现状：")
    for key in ("collection", "points", "meta_collection", "documents"):
        print(f"  {key:16s}: {stats.get(key)}")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。

    Args:
        argv: 命令行参数（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码：0 成功，1 有失败文件，2 参数/环境错误。
    """
    app = get_config()
    parser = argparse.ArgumentParser(
        description="企业级 RAG 文档摄取（增量索引，基于文件 hash 判重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path", default=app.ingestion.docs_dir, help="文档目录")
    parser.add_argument(
        "--collection", default=app.vector_store.collection, help="Qdrant 集合名"
    )
    parser.add_argument("--recreate", action="store_true", help="重建集合（清空既有数据）")
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument("--dry-run", action="store_true", help="只解析不入库")
    parser.add_argument("--stats", action="store_true", help="只打印集合现状后退出")
    args = parser.parse_args(argv)

    try:
        runner = IngestionRunner(collection=args.collection, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 初始化失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.stats:
            _print_stats(runner)
            return 0

        report = runner.run(
            args.path, recursive=not args.no_recursive, recreate=args.recreate
        )
        print(report.render())
        if not args.dry_run:
            _print_stats(runner)
        return 1 if report.stats.failed else 0
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            f"[error] 摄取失败: {type(exc).__name__}: {exc}\n"
            f"提示：确认 Qdrant 已启动（scripts\\start_qdrant.ps1），"
            f"且 embedding.dimension 与集合维度一致。",
            file=sys.stderr,
        )
        return 2
    finally:
        runner.indexer.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
