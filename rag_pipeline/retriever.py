"""混合检索器：稠密向量 + 稀疏 BM25（服务端 IDF），RRF 融合后重排序。

检索流程（一次 Qdrant 请求完成召回，重排放在本地以便可解释、可消融）：

    1. **召回**：``query_points`` 带两个 prefetch 分支
         - dense  : bge-m3 查询向量，取 ``prefetch_limit`` 条
         - sparse : BM25 词频项，Qdrant 用 IDF 修饰符自动加权
       然后用 ``Fusion.RRF`` 做倒数排名融合（Reciprocal Rank Fusion）。
       RRF 的价值：稠密分（余弦，0~1）与稀疏分（BM25，无上界）量纲不同，
       直接加权需要调参；RRF 只看排名，天然免疫量纲问题。

    2. **重排序**（本模块核心，规则明确且可评测）：
         a. 高置信直通：``vector_score >= 0.75`` 的片段进入第一梯队，内部按向量分降序。
            —— 这是需求"优先返回向量相似度 > 0.75 的片段"的落地方式。
         b. 其余片段按 ``score = 0.6 * 归一化RRF + 0.25 * 向量分 + 0.15 * 关键词分`` 排序。
         c. 绝对下限：``vector_score`` 低于 ``min_vector_score`` 的片段即使关键词命中
            也直接丢弃 —— 否则"退货政策"这类高频词会让大量无关片段靠关键词混进上下文。
         d. 精确短语命中：查询串完整出现在片段中时加 ``phrase_bonus``。

    3. **分数可归因**：返回结果同时携带 ``vector_score`` / ``keyword_score`` /
       ``rrf_score``，阶段5 才能做"向量 vs 关键词各贡献多少"的消融分析。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from config.config_loader import VectorStoreConfig, get_config
from rag_pipeline.embedder import Embedder, SparseEncoder, get_embedder, get_sparse_encoder
from rag_pipeline.indexer import QdrantIndexer

__all__ = [
    "RetrieverError",
    "SearchFilters",
    "SearchResult",
    "SearchDiagnostics",
    "HybridRetriever",
]


class RetrieverError(RuntimeError):
    """检索过程失败（集合缺失、连接失败、查询编码失败等）。"""


@dataclass
class SearchFilters:
    """检索过滤条件（对应 Qdrant payload 索引字段）。

    Attributes:
        source: 限定来源文件（精确匹配）。
        doc_ids: 限定文档 ID 列表。
        page: 限定 PDF 页码。
    """

    source: str | None = None
    doc_ids: list[str] = field(default_factory=list)
    page: int | None = None

    def to_qdrant_filter(self) -> Any | None:
        """构造 Qdrant Filter 对象。

        Returns:
            ``Filter`` 实例；无任何条件时返回 ``None``。
        """
        from qdrant_client import models as qm

        must: list[Any] = []
        if self.source:
            must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=self.source)))
        if self.doc_ids:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(self.doc_ids))))
        if self.page is not None:
            must.append(qm.FieldCondition(key="page", match=qm.MatchValue(value=int(self.page))))
        return qm.Filter(must=must) if must else None


@dataclass
class SearchResult:
    """单条检索结果（字段与阶段1 ``SearchHit`` 严格对齐）。

    Attributes:
        content: 片段正文。
        source: 来源文件相对路径。
        score: 融合后的最终得分（0~1，越大越相关）。
        page: PDF 页码；非分页文档为 ``None``。
        doc_id: 所属文档 ID。
        chunk_id: 片段 ID。
        vector_score: 向量相似度原始分（可归因）。
        keyword_score: 关键词（稀疏）得分，已归一化（可归因）。
        rrf_score: RRF 融合原始分（可归因）。
        heading: 片段所属标题。
        created_at: 文档创建时间。
        tier: ``high``（向量高置信直通）或 ``fused``（融合排序）。
    """

    content: str
    source: str
    score: float
    page: int | None
    doc_id: str
    chunk_id: str = ""
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    heading: str = ""
    created_at: str = ""
    tier: str = "fused"

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化字典（MCP 工具直接返回该结构）。"""
        return {
            "content": self.content,
            "source": self.source,
            "score": round(self.score, 6),
            "page": self.page,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "vector_score": round(self.vector_score, 6),
            "keyword_score": round(self.keyword_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "heading": self.heading,
            "created_at": self.created_at,
            "tier": self.tier,
        }


@dataclass
class SearchDiagnostics:
    """检索过程诊断信息（供评测与排障）。"""

    elapsed_ms: float = 0.0
    candidates: int = 0
    kept: int = 0
    filtered_low_vector: int = 0
    high_confidence_hits: int = 0
    collection: str = ""
    mode: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化字典。"""
        return {
            "elapsed_ms": round(self.elapsed_ms, 2),
            "candidates": self.candidates,
            "kept": self.kept,
            "filtered_low_vector": self.filtered_low_vector,
            "high_confidence_hits": self.high_confidence_hits,
            "collection": self.collection,
            "mode": self.mode,
        }


class HybridRetriever:
    """向量 + 关键词混合检索器。"""

    def __init__(
        self,
        indexer: QdrantIndexer | None = None,
        vector_config: VectorStoreConfig | None = None,
        embedder: Embedder | None = None,
        sparse: SparseEncoder | None = None,
        *,
        high_confidence: float = 0.75,
        min_vector_score: float = 0.40,
        phrase_bonus: float = 0.05,
        prefetch_limit: int = 40,
        weight_rrf: float = 0.40,
        weight_vector: float = 0.55,
        weight_keyword: float = 0.05,
    ) -> None:
        """初始化检索器。

        Args:
            indexer: 复用已有索引器（共享 Qdrant 连接）；``None`` 时新建。
            vector_config: 向量库配置。
            embedder: 稠密嵌入器。
            sparse: 稀疏编码器。
            high_confidence: 向量相似度"高置信"阈值，达到即进第一梯队。
            min_vector_score: 向量分绝对下限，低于该值的片段直接丢弃。
            phrase_bonus: 查询短语精确命中时对最终分的加分。
            prefetch_limit: 每个召回分支的候选数量。
            weight_rrf / weight_vector / weight_keyword: 融合分权重，
                三者之和必须为 1。默认值 (0.40, 0.55, 0.05) 来自
                ``.tmp/tune_rerank.py`` 在 8 条查询上的网格搜索结果：
                R@3 从 (0.60, 0.25, 0.15) 的 0.750/MRR 0.604 提升到
                0.750/MRR 0.667（MRR 提升代表正确片段排得更靠前）。

        Raises:
            ValueError: 三个权重之和不为 1。
        """
        total = weight_rrf + weight_vector + weight_keyword
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"融合权重之和必须为 1，当前 {weight_rrf}+{weight_vector}+"
                f"{weight_keyword}={total}"
            )
        app = get_config()
        self.config = vector_config or app.vector_store
        self.indexer = indexer or QdrantIndexer(vector_config=self.config)
        self.embedder = embedder or get_embedder()
        self.sparse = sparse or get_sparse_encoder()
        self.high_confidence = high_confidence
        self.min_vector_score = min_vector_score
        self.phrase_bonus = phrase_bonus
        self.prefetch_limit = prefetch_limit
        self.weight_rrf = weight_rrf
        self.weight_vector = weight_vector
        self.weight_keyword = weight_keyword

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """执行检索并返回重排后的结果。

        Args:
            query: 查询语句。
            top_k: 返回条数上限。
            filters: 过滤条件。
            mode: ``hybrid``（向量+关键词，默认）/ ``vector`` / ``keyword``。
                后两者用于阶段5 的消融评测，定位"混合检索到底贡献了多少"。

        Returns:
            :class:`SearchResult` 列表，按 ``score`` 降序；无命中时返回空列表。

        Raises:
            RetrieverError: 集合不存在、连接失败或查询编码失败。
        """
        results, _ = self.search_with_diagnostics(query, top_k, filters, mode)
        return results

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        mode: str = "hybrid",
    ) -> tuple[list[SearchResult], SearchDiagnostics]:
        """执行检索并返回结果与诊断信息。

        Args:
            query: 查询语句。
            top_k: 返回条数上限。
            filters: 过滤条件。
            mode: 检索模式，见 :meth:`search`。

        Returns:
            ``(结果列表, 诊断信息)``。

        Raises:
            RetrieverError: 集合不存在、连接失败或查询编码失败。
        """
        if not query or not query.strip():
            raise RetrieverError("查询语句不能为空")
        if top_k < 1:
            raise RetrieverError(f"top_k 必须 >= 1，实际为 {top_k}")
        if mode not in ("hybrid", "vector", "keyword"):
            raise RetrieverError(f"未知检索模式: {mode}（可选 hybrid/vector/keyword）")

        started = time.perf_counter()
        collection = self.config.collection
        if not self.indexer.collection_exists(collection):
            raise RetrieverError(
                f"集合不存在: {collection}。请先入库："
                f"python -m rag_pipeline.ingestion_cli --path ./docs"
            )

        qdrant_filter = filters.to_qdrant_filter() if filters else None
        try:
            dense_points, sparse_points = self._query_qdrant(query, mode, qdrant_filter)
        except RetrieverError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(
                f"Qdrant 查询失败: {type(exc).__name__}: {exc}"
            ) from exc

        diagnostics = SearchDiagnostics(
            candidates=len(dense_points) + len(sparse_points), collection=collection, mode=mode
        )
        rescored = self._rerank(query, dense_points, sparse_points, diagnostics)
        final = rescored[:top_k]
        diagnostics.kept = len(final)
        diagnostics.elapsed_ms = (time.perf_counter() - started) * 1000
        return final, diagnostics

    def _query_qdrant(
        self, query: str, mode: str, qdrant_filter: Any | None
    ) -> tuple[list[Any], list[Any]]:
        """向 Qdrant 发起召回，返回稠密与稀疏两个结果集。

        为什么不用服务端 ``Fusion.RRF``：
            服务端融合后 Qdrant 只回传融合分，**原始的稠密余弦分会被丢弃**；
            而"向量相似度 > 0.75 优先"与"低向量分剔除"两条规则都依赖原始分。
            因此召回仍分两个分支执行（各自保留原始分），融合在本地完成 ——
            代价是多一次往返，换来完整的分数归因能力（阶段5 消融评测需要）。

        Args:
            query: 查询语句。
            mode: 检索模式。
            qdrant_filter: Qdrant Filter 对象或 ``None``。

        Returns:
            ``(dense_points, sparse_points)``，未启用的分支为空列表。

        Raises:
            RetrieverError: 查询向量编码失败或关键词查询无法分词。
        """
        from qdrant_client import models as qm

        dense_points: list[Any] = []
        sparse_points: list[Any] = []

        if mode in ("hybrid", "vector"):
            try:
                dense_query = self.embedder.encode_query(query)
            except Exception as exc:  # noqa: BLE001
                raise RetrieverError(
                    f"查询向量编码失败: {type(exc).__name__}: {exc}"
                ) from exc
            dense_points = self.indexer.client.query_points(
                collection_name=self.config.collection,
                query=dense_query,
                using="dense",
                limit=self.prefetch_limit,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points

        if mode in ("hybrid", "keyword"):
            sv = self.sparse.encode(query)
            if len(sv) == 0:
                if mode == "keyword":
                    raise RetrieverError(f"查询无法分词，无法执行关键词检索: {query!r}")
            else:
                sparse_points = self.indexer.client.query_points(
                    collection_name=self.config.collection,
                    query=qm.SparseVector(indices=sv.indices, values=sv.values),
                    using="sparse",
                    limit=self.prefetch_limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                ).points

        return dense_points, sparse_points

    def _rerank(
        self,
        query: str,
        dense_points: Sequence[Any],
        sparse_points: Sequence[Any],
        diagnostics: SearchDiagnostics,
    ) -> list[SearchResult]:
        """对两个召回分支做 RRZ 融合与重排序。

        融合公式（本地计算，保证原始分可归因）::

            rrf(rank)  = 1 / (rrf_k + rank)                      # 倒数排名融合
            norm_rrf   = rrf / 该查询内的最大 rrf                 # 归一化到 0~1
            score      = 0.6 * norm_rrf + 0.25 * vector_score + 0.15 * keyword_norm

        重排规则：
            a. ``vector_score >= high_confidence`` 的片段进第一梯队（按向量分降序），
               这是需求"优先返回向量相似度 > 0.75 的片段"的落地；
            b. 其余按融合分降序；
            c. ``vector_score < min_vector_score`` 的片段直接丢弃 —— 只有关键词命中
               但语义无关的片段不得进入上下文；
            d. 查询短语完整出现在片段中时加 ``phrase_bonus``。

        Args:
            query: 查询语句。
            dense_points: 稠密召回结果（有序，含原始余弦分）。
            sparse_points: 稀疏召回结果（有序，含 BM25 分）。
            diagnostics: 诊断对象（就地更新过滤计数）。

        Returns:
            重排后的结果列表。
        """
        rrf_k = 60.0
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        rrf_acc: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        def _key(point: Any) -> str:
            payload = dict(getattr(point, "payload", None) or {})
            return str(payload.get("chunk_id") or getattr(point, "id", ""))

        for rank, point in enumerate(dense_points):
            key = _key(point)
            payloads[key] = dict(getattr(point, "payload", None) or {})
            dense_scores[key] = float(getattr(point, "score", 0.0) or 0.0)
            rrf_acc[key] = rrf_acc.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, point in enumerate(sparse_points):
            key = _key(point)
            payloads.setdefault(key, dict(getattr(point, "payload", None) or {}))
            sparse_scores[key] = float(getattr(point, "score", 0.0) or 0.0)
            rrf_acc[key] = rrf_acc.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        if not rrf_acc:
            return []

        max_rrf = max(rrf_acc.values())
        max_sparse = max(sparse_scores.values()) if sparse_scores else 0.0
        query_norm = query.strip()

        results: list[SearchResult] = []
        for key, rrf in rrf_acc.items():
            payload = payloads.get(key, {})
            vector_score = dense_scores.get(key, 0.0)
            raw_sparse = sparse_scores.get(key, 0.0)
            keyword_norm = (raw_sparse / max_sparse) if max_sparse > 0 else 0.0
            content = str(payload.get("content", ""))

            # (c) 绝对下限：关键词命中但语义无关的片段直接丢弃
            if vector_score < self.min_vector_score:
                diagnostics.filtered_low_vector += 1
                continue

            norm_rrf = rrf / max_rrf if max_rrf > 0 else 0.0
            fused = (
                self.weight_rrf * norm_rrf
                + self.weight_vector * vector_score
                + self.weight_keyword * keyword_norm
            )
            if query_norm and query_norm in content:
                fused += self.phrase_bonus

            tier = "high" if vector_score >= self.high_confidence else "fused"
            if tier == "high":
                diagnostics.high_confidence_hits += 1

            results.append(
                SearchResult(
                    content=content,
                    source=str(payload.get("source", "")),
                    score=min(fused, 1.0),
                    page=payload.get("page"),
                    doc_id=str(payload.get("doc_id", "")),
                    chunk_id=str(payload.get("chunk_id", "")),
                    vector_score=vector_score,
                    keyword_score=keyword_norm,
                    rrf_score=rrf,
                    heading=str(payload.get("heading", "")),
                    created_at=str(payload.get("created_at", "")),
                    tier=tier,
                )
            )

        # (a) 高置信直通第一梯队，其内按向量分降序；(b) 其余按融合分降序
        high = sorted(
            (r for r in results if r.tier == "high"), key=lambda r: (-r.vector_score, -r.score)
        )
        fused = sorted((r for r in results if r.tier != "high"), key=lambda r: -r.score)
        return high + fused

    @staticmethod
    def _extract_scores(point: Any) -> tuple[float, float]:
        """已废弃：原实现会在分量缺失时用融合分冒充向量分，导致错误的高置信判定。

        保留此方法仅为兼容外部可能的调用，新代码请使用 :meth:`_rerank` 中的
        分数归因逻辑（召回分两个分支执行，原始分不会丢失）。

        Args:
            point: Qdrant ``ScoredPoint``。

        Returns:
            ``(融合分, 0.0)``，不携带任何语义。
        """
        return float(getattr(point, "score", 0.0) or 0.0), 0.0

    # ------------------------------------------------------------------
    # 便捷查询
    # ------------------------------------------------------------------
    def search_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """MCP 工具友好的检索入口（返回字典列表）。

        Args:
            query: 查询语句。
            top_k: 返回条数。

        Returns:
            :class:`SearchResult` 转换后的字典列表。
        """
        return [r.to_dict() for r in self.search(query, top_k=top_k)]

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """转发文档元数据查询（供阶段3 元数据工具使用）。

        Args:
            doc_id: 文档 ID。

        Returns:
            元数据字典；不存在时 ``None``。
        """
        return self.indexer.get_document_metadata(doc_id)

    def stats(self) -> dict[str, Any]:
        """转发集合统计。"""
        return self.indexer.collection_stats()
