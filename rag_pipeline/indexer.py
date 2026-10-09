"""索引器：把解析后的 chunk 写入 Qdrant，并记录索引统计。

集合设计（两个集合，职责分离）：

    enterprise_docs          片段级：稠密向量(bge-m3) + 稀疏向量(BM25 词频项)
    enterprise_docs_meta     文档级：标题/来源/页数/创建时间等，供阶段3 元数据工具直接读取

为什么元数据要单独一个集合：
    如果把文档级字段冗余在每个 chunk 的 payload 里，"查文档元数据"就得做一次
    全集合扫描 + 去重；单独集合可以用一次 ``retrieve`` 按 doc_id 精确命中。
    代价是写入时需要维护两处，故本模块把两者放在同一事务性流程里。

稀疏向量为什么只存词频项：
    集合启用 ``SparseVectorParams(modifier=Modifier.IDF)`` 后，Qdrant 会在查询时
    自动按"词在该集合中的稀有度"加权。客户端因此不需要回传 IDF 表，
    入库与查询两侧复用同一套 BM25 编码函数即可，打分口径天然一致。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from config.config_loader import IngestionConfig, VectorStoreConfig, get_config
from rag_pipeline.document_parser import DocumentChunk, DocumentMetadata, ParsedDocument
from rag_pipeline.embedder import Embedder, SparseEncoder, get_embedder, get_sparse_encoder

__all__ = [
    "IndexerError",
    "CollectionNotReadyError",
    "IndexStats",
    "QdrantIndexer",
    "META_COLLECTION_SUFFIX",
]

META_COLLECTION_SUFFIX = "_meta"
# 元数据集合的向量维度取 1：该集合只按 ID 精确检索，不参与语义检索
_META_VECTOR_SIZE = 1


class IndexerError(RuntimeError):
    """索引过程失败（连接、维度、写入等）。"""


class CollectionNotReadyError(IndexerError):
    """目标集合不存在（通常是尚未执行入库）。"""


@dataclass
class IndexStats:
    """单批/单次索引的统计信息。

    Attributes:
        documents: 处理文档数。
        chunks: 写入片段数。
        skipped: 因增量判重跳过的文档数。
        failed: 失败文档数。
        elapsed_sec: 耗时（秒）。
        batches: 写入批次数。
        errors: 失败详情 ``[(来源, 错误信息)]``。
    """

    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_sec: float = 0.0
    batches: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def merge(self, other: IndexStats) -> None:
        """合并另一份统计。

        Args:
            other: 待合并的统计对象。
        """
        self.documents += other.documents
        self.chunks += other.chunks
        self.skipped += other.skipped
        self.failed += other.failed
        self.batches += other.batches
        self.errors.extend(other.errors)

    @property
    def chunks_per_sec(self) -> float:
        """写入吞吐（片段/秒）；耗时为 0 时返回 0。"""
        return self.chunks / self.elapsed_sec if self.elapsed_sec > 0 else 0.0

    def summary(self) -> str:
        """生成人类可读的一行摘要。

        Returns:
            形如 ``文档 3（跳过 0/失败 0），片段 42，批次 1，耗时 12.3s（3.4 片段/s）``。
        """
        return (
            f"文档 {self.documents}（跳过 {self.skipped}/失败 {self.failed}），"
            f"片段 {self.chunks}，批次 {self.batches}，"
            f"耗时 {self.elapsed_sec:.1f}s（{self.chunks_per_sec:.1f} 片段/s）"
        )


def _point_id(key: str) -> str:
    """把业务键稳定映射为 Qdrant 可接受的 UUID 点 ID。

    Args:
        key: 业务键，如 ``chunk_id`` 或 ``doc_id``。

    Returns:
        UUID5 字符串（同一 key 永远得到同一个点 ID，重复写入即覆盖）。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"enterprise-rag-mcp::{key}"))


class QdrantIndexer:
    """Qdrant 索引器。"""

    def __init__(
        self,
        vector_config: VectorStoreConfig | None = None,
        ingestion_config: IngestionConfig | None = None,
        embedder: Embedder | None = None,
        sparse: SparseEncoder | None = None,
    ) -> None:
        """初始化索引器（不建立连接，连接在首次使用时创建）。

        Args:
            vector_config: 向量库配置。
            ingestion_config: 摄取配置（批大小等）。
            embedder: 稠密嵌入器；``None`` 时使用进程单例。
            sparse: 稀疏编码器；``None`` 时使用进程单例。
        """
        app = get_config()
        self.config = vector_config or app.vector_store
        self.ingestion = ingestion_config or app.ingestion
        self.embedder = embedder or get_embedder()
        self.sparse = sparse or get_sparse_encoder()
        self._client: Any | None = None
        self._sparse_ready: set[str] = set()

    # ------------------------------------------------------------------
    # 连接与集合管理
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        """懒加载 Qdrant 客户端。

        Returns:
            ``QdrantClient`` 实例。

        Raises:
            IndexerError: 连接失败（服务未启动或地址错误）。
        """
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover
                raise IndexerError("未安装 qdrant-client，请执行 pip install -r requirements.txt") from exc

            try:
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    api_key=self.config.api_key or None,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc,
                )
                # 主动探活：QdrantClient 构造本身不会连接，失败要在这里暴露
                self._client.get_collections()
            except Exception as exc:  # noqa: BLE001
                raise IndexerError(
                    f"Qdrant 连接失败: {self.config.url}（{type(exc).__name__}: {exc}）。"
                    f"请确认服务已启动：powershell -ExecutionPolicy Bypass "
                    f"-File scripts\\start_qdrant.ps1"
                ) from exc
        return self._client

    @property
    def collection(self) -> str:
        """片段集合名。"""
        return self.config.collection

    @property
    def meta_collection(self) -> str:
        """文档元数据集合名。"""
        return self.config.collection + META_COLLECTION_SUFFIX

    def collection_exists(self, name: str | None = None) -> bool:
        """判断集合是否存在。

        Args:
            name: 集合名；``None`` 时用片段集合。

        Returns:
            存在返回 True。
        """
        target = name or self.collection
        try:
            existing = {c.name for c in self.client.get_collections().collections}
        except Exception as exc:  # noqa: BLE001
            raise IndexerError(f"读取集合列表失败: {exc}") from exc
        return target in existing

    def ensure_collections(self, recreate: bool = False) -> None:
        """确保两个集合存在，并按配置建立 payload 索引。

        Args:
            recreate: 为 True 时先删除再重建（清空数据，谨慎使用）。

        Raises:
            IndexerError: 建集合或建索引失败。
        """
        from qdrant_client import models as qm

        dim = self.embedder.dimension

        if recreate:
            for name in (self.collection, self.meta_collection):
                if self.collection_exists(name):
                    self.client.delete_collection(name)

        if not self.collection_exists(self.collection):
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        "dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        # IDF 修饰符：让 Qdrant 按词稀有度加权，客户端只提供词频项
                        "sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)
                    },
                    hnsw_config=qm.HnswConfigDiff(
                        m=self.config.hnsw.m, ef_construct=self.config.hnsw.ef_construct
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                raise IndexerError(
                    f"创建集合失败: {self.collection}（维度 {dim}）-> {type(exc).__name__}: {exc}"
                ) from exc

        if not self.collection_exists(self.meta_collection):
            try:
                self.client.create_collection(
                    collection_name=self.meta_collection,
                    vectors_config=qm.VectorParams(
                        size=_META_VECTOR_SIZE, distance=qm.Distance.COSINE
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                raise IndexerError(f"创建元数据集合失败: {self.meta_collection} -> {exc}") from exc

        for field_name in self.config.payload_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=qm.PayloadSchemaType.KEYWORD
                    if field_name != "page"
                    else qm.PayloadSchemaType.INTEGER,
                )
            except Exception:  # noqa: BLE001 - 索引已存在时 Qdrant 会报错，属幂等场景
                pass

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def index_document(self, document: ParsedDocument) -> IndexStats:
        """索引单个文档（含元数据与全部片段）。

        Args:
            document: 解析结果。

        Returns:
            :class:`IndexStats` 统计。

        Raises:
            IndexerError: 集合不可用、嵌入失败或写入失败。
        """
        from qdrant_client import models as qm

        started = time.perf_counter()
        stats = IndexStats()
        self.ensure_collections()

        chunks = document.chunks
        if not chunks:
            stats.skipped += 1
            stats.elapsed_sec = time.perf_counter() - started
            return stats

        # 重新索引同一文档时先清掉旧片段，避免内容变更后残留孤儿片段
        self.delete_document(document.doc_id)

        # 编码用 embedding_text()（含来源文件名与标题上下文），
        # 存储与展示仍用 content：让模型看到结构，让用户看到干净正文。
        embed_texts = [c.embedding_text() for c in chunks]
        try:
            dense_vectors = self.embedder.encode_documents(embed_texts)
        except Exception as exc:  # noqa: BLE001
            raise IndexerError(
                f"文档嵌入失败: {document.metadata.source} -> {type(exc).__name__}: {exc}"
            ) from exc

        # 用真实文档长度更新平均长度，提升 BM25 长度归一化的准确性
        token_lens = [len(self._sparse_token_count(t)) for t in embed_texts]
        self.sparse.update_corpus_stats(token_lens)

        batch_size = self.ingestion.index_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points: list[Any] = []
            for offset, chunk in enumerate(batch):
                dense = dense_vectors[start + offset]
                # 稀疏向量同样基于 embedding_text，保证与稠密分支的输入一致
                sparse_vec = self.sparse.encode(embed_texts[start + offset])
                points.append(
                    qm.PointStruct(
                        id=_point_id(chunk.chunk_id),
                        vector={
                            "dense": dense,
                            "sparse": qm.SparseVector(
                                indices=sparse_vec.indices, values=sparse_vec.values
                            ),
                        },
                        payload=chunk.to_payload(),
                    )
                )
            try:
                self.client.upsert(collection_name=self.collection, points=points, wait=True)
            except Exception as exc:  # noqa: BLE001
                raise IndexerError(
                    f"写入 Qdrant 失败（批次 {start // batch_size + 1}，"
                    f"{len(points)} 个点）-> {type(exc).__name__}: {exc}"
                ) from exc
            stats.batches += 1

        # 写入文档级元数据
        try:
            self.client.upsert(
                collection_name=self.meta_collection,
                points=[
                    qm.PointStruct(
                        id=_point_id(f"meta::{document.doc_id}"),
                        vector=[0.0] * _META_VECTOR_SIZE,
                        payload=document.metadata.to_payload(),
                    )
                ],
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise IndexerError(f"写入文档元数据失败: {document.metadata.source} -> {exc}") from exc

        stats.documents += 1
        stats.chunks += len(chunks)
        stats.elapsed_sec = time.perf_counter() - started
        return stats

    @staticmethod
    def _sparse_token_count(text: str) -> list[str]:
        """对文本分词（用于 BM25 长度归一化的平均长度统计）。

        Args:
            text: 待分词文本（此处传入的是 ``embedding_text()``，
                与稀疏编码的输入保持严格一致）。

        Returns:
            token 列表。
        """
        from rag_pipeline.embedder import tokenize

        return tokenize(text)

    def delete_document(self, doc_id: str) -> None:
        """删除某文档的全部片段与元数据（幂等）。

        Args:
            doc_id: 文档 ID。
        """
        from qdrant_client import models as qm

        if self.collection_exists(self.collection):
            self.client.delete(
                collection_name=self.collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                    )
                ),
                wait=True,
            )
        if self.collection_exists(self.meta_collection):
            try:
                self.client.delete(
                    collection_name=self.meta_collection,
                    points_selector=[_point_id(f"meta::{doc_id}")],
                    wait=True,
                )
            except Exception:  # noqa: BLE001 - 元数据不存在时忽略
                pass

    # ------------------------------------------------------------------
    # 查询辅助
    # ------------------------------------------------------------------
    def indexed_doc_hashes(self) -> dict[str, str]:
        """读取已索引文档的 ``doc_id -> file_hash`` 映射（供增量索引判重）。

        Returns:
            映射字典；元数据集合不存在时返回空字典。
        """
        if not self.collection_exists(self.meta_collection):
            return {}
        mapping: dict[str, str] = {}
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.meta_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                doc_id = str(payload.get("doc_id", ""))
                file_hash = str(payload.get("file_hash", ""))
                if doc_id:
                    mapping[doc_id] = file_hash
            if offset is None:
                break
        return mapping

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """按 doc_id 读取文档元数据。

        Args:
            doc_id: 文档 ID。

        Returns:
            元数据字典；不存在时返回 ``None``。

        Raises:
            CollectionNotReadyError: 元数据集合不存在（尚未执行过入库）。
        """
        if not self.collection_exists(self.meta_collection):
            raise CollectionNotReadyError(
                f"元数据集合不存在: {self.meta_collection}。请先执行入库："
                f"python -m rag_pipeline.ingestion_cli --path ./docs"
            )
        records = self.client.retrieve(
            collection_name=self.meta_collection,
            ids=[_point_id(f"meta::{doc_id}")],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return dict(records[0].payload or {})

    def collection_stats(self) -> dict[str, Any]:
        """统计集合规模。

        Returns:
            ``{"collection":..., "points":..., "documents":..., "vectors":...}``；
            集合不存在时各计数为 0。

        Raises:
            IndexerError: 读取失败。
        """
        info: dict[str, Any] = {
            "collection": self.collection,
            "meta_collection": self.meta_collection,
            "points": 0,
            "documents": 0,
            "indexed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if self.collection_exists(self.collection):
            info["points"] = self.client.count(self.collection, exact=True).count
        if self.collection_exists(self.meta_collection):
            info["documents"] = self.client.count(self.meta_collection, exact=True).count
        return info

    def close(self) -> None:
        """关闭客户端连接（幂等）。"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


def build_indexer(recreate: bool = False) -> QdrantIndexer:
    """便捷函数：构造索引器并确保集合就绪。

    Args:
        recreate: 是否重建集合（会清空已有数据）。

    Returns:
        就绪的 :class:`QdrantIndexer`。
    """
    indexer = QdrantIndexer()
    indexer.ensure_collections(recreate=recreate)
    return indexer
