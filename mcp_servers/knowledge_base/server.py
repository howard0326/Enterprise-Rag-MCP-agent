"""knowledge_base MCP server —— 企业知识库检索工具服务。

暴露的工具（阶段3 已接入阶段2 的真实检索实现）：
    - search_documents(query, top_k)     混合检索（向量 + BM25，本地 RRF 融合）
    - get_document_metadata(doc_id)      按文档 ID 返回元数据

工程约束（与 business_db 一致，理由见该文件）：
    1. **异常收敛在工具内部**：mcp 2.x 会把未捕获异常转成 ``UnexpectedToolError``
       抛给客户端并带崩其 task group（阶段1 实测），因此必须返回
       ``{"ok": false, "error": ..., "code": ...}`` 信封。
    2. **stdio 下 stdout 只放 JSON-RPC 报文**，日志走 stderr。
    3. **检索器进程级单例**：bge-m3 加载约 6~30 秒（实测），每次调用重新加载
       会让工具延迟不可接受；懒加载 + 单例复用。

阶段2 检索器返回的字段（可直接用于溯源）：
    content / source / score / page / doc_id / chunk_id /
    vector_score / keyword_score / rrf_score / heading / tier

手动自测::

    python -m mcp_servers.knowledge_base.server
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# MCP SDK 版本兼容层（同 business_db/server.py）
#   mcp 1.x : FastMCP      mcp 2.x : mcp.server.mcpserver.MCPServer
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 取决于运行环境安装的 SDK 版本
    from mcp.server.mcpserver import MCPServer as _ServerClass  # type: ignore[attr-defined]

    _SERVER_KW_DESC = "description"
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as _ServerClass  # type: ignore[no-redef]

    _SERVER_KW_DESC = "instructions"

SERVER_NAME = "knowledge_base"
SERVER_VERSION = "0.1.0"

__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "GetDocumentMetadataParams",
    "SearchDocumentsParams",
    "ToolBackendError",
    "build_server",
    "get_document_metadata",
    "get_retriever",
    "search_documents",
    "main",
]

logger = logging.getLogger(f"mcp.{SERVER_NAME}")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class ToolBackendError(RuntimeError):
    """检索后端不可用（集合缺失、Qdrant 连接失败、模型加载失败等）。"""


class SearchDocumentsParams(BaseModel):
    """``search_documents`` 的入参。

    这些 Field 的 description 就是给 LLM 看的 JSON Schema 描述，
    是模型选择工具、填写参数时的唯一依据，必须写"什么时候用""怎么填"。
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="用户问题的检索语句，保留业务术语原文（如'退货政策''账期'），不要改写为关键词堆砌。",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="返回的文档片段数量上限，默认 5。问题越宽泛可适当调大（8~10）。",
    )


class GetDocumentMetadataParams(BaseModel):
    """``get_document_metadata`` 的入参。"""

    doc_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="文档 ID（search_documents 返回结果里的 doc_id 字段，即入库时计算的内容哈希前 16 位）。",
    )


def _ok(data: Any, **extra: Any) -> dict[str, Any]:
    """构造成功信封。

    Args:
        data: 业务数据。
        **extra: 追加到信封顶层的附加字段。

    Returns:
        形如 ``{"ok": True, "data": ..., "error": None}`` 的字典。
    """
    envelope: dict[str, Any] = {"ok": True, "data": data, "error": None}
    envelope.update(extra)
    return envelope


def _fail(error: str, *, code: str, hint: str = "") -> dict[str, Any]:
    """构造失败信封（不抛异常，让 Agent 能自行决定重试或降级）。

    Args:
        error: 面向人的错误描述。
        code: 机器可判定的错误码。
        hint: 给 Agent 的修复建议。

    Returns:
        形如 ``{"ok": False, "data": None, "error": ...}`` 的字典。
    """
    message = f"[{code}] {error}" + (f" 建议: {hint}" if hint else "")
    logger.warning("tool error: %s", message)
    return {"ok": False, "data": None, "error": message, "code": code}


_retriever: Any | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> Any:
    """获取进程级混合检索器单例（懒加载）。

    Returns:
        :class:`rag_pipeline.retriever.HybridRetriever` 实例。

    Raises:
        ToolBackendError: 检索器构造失败（依赖缺失、配置非法等）。
    """
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                try:
                    from rag_pipeline.retriever import HybridRetriever

                    _retriever = HybridRetriever()
                    logger.info("检索器已初始化（集合=%s）", _retriever.config.collection)
                except Exception as exc:  # noqa: BLE001
                    raise ToolBackendError(
                        f"检索器初始化失败: {type(exc).__name__}: {exc}"
                    ) from exc
    return _retriever


def search_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """在企业知识库中检索与 ``query`` 相关的文档片段。

    内部流程：稠密向量（bge-m3）+ 稀疏 BM25 双路召回 → 本地 RRF 融合 →
    高置信直通（向量分 ≥ 0.75）→ 低向量分剔除 → 返回 Top-K。

    Args:
        query: 用户问题的检索语句，建议保留业务术语原文。
        top_k: 返回片段数量上限，取值 1~50，默认 5。

    Returns:
        统一信封。成功时 ``data`` 为片段列表（按相关性降序），每项含
        ``content`` / ``source`` / ``score`` / ``page`` / ``doc_id`` 及分数归因字段；
        另附 ``hit_count``、``collection`` 与 ``elapsed_ms``。
        参数非法返回 ``INVALID_ARGUMENT``；知识库不可用返回 ``BACKEND_UNAVAILABLE``。
    """
    try:
        params = SearchDocumentsParams(query=query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"检索参数非法: {exc}",
            code="INVALID_ARGUMENT",
            hint="query 不能为空；top_k 取值范围 1~50。",
        )

    try:
        retriever = get_retriever()
        results, diagnostics = retriever.search_with_diagnostics(
            params.query, top_k=params.top_k
        )
    except ToolBackendError as exc:
        return _fail(
            str(exc),
            code="BACKEND_UNAVAILABLE",
            hint="请确认 Qdrant 已启动（scripts\\start_qdrant.ps1）且已完成入库。",
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"检索失败: {type(exc).__name__}: {exc}",
            code="BACKEND_UNAVAILABLE",
            hint="请确认 Qdrant 已启动且集合 enterprise_docs 已建立。",
        )

    hits = [r.to_dict() for r in results]
    logger.info(
        "search_documents query=%r top_k=%d -> %d 条（候选 %d，低向量剔除 %d，耗时 %.0fms）",
        params.query,
        params.top_k,
        len(hits),
        diagnostics.candidates,
        diagnostics.filtered_low_vector,
        diagnostics.elapsed_ms,
    )
    return _ok(
        hits,
        hit_count=len(hits),
        collection=diagnostics.collection,
        elapsed_ms=round(diagnostics.elapsed_ms, 1),
        mode=diagnostics.mode,
    )


def get_document_metadata(doc_id: str) -> dict[str, Any]:
    """按 ``doc_id`` 查询文档元数据。

    数据来源是入库时写入的独立元数据集合（``enterprise_docs_meta``），
    因此是精确命中，不需要遍历片段。

    Args:
        doc_id: 文档 ID（``search_documents`` 结果中的 ``doc_id``）。

    Returns:
        统一信封。成功时 ``data`` 为元数据字典（标题、来源、类型、页数、片段数、
        创建时间、标签等）；文档不存在返回 ``NOT_FOUND``；知识库不可用返回
        ``BACKEND_UNAVAILABLE``。
    """
    try:
        params = GetDocumentMetadataParams(doc_id=doc_id)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"参数非法: {exc}", code="INVALID_ARGUMENT", hint="doc_id 不能为空。")

    try:
        retriever = get_retriever()
        metadata = retriever.get_document_metadata(params.doc_id)
    except ToolBackendError as exc:
        return _fail(
            str(exc),
            code="BACKEND_UNAVAILABLE",
            hint="请确认 Qdrant 已启动（scripts\\start_qdrant.ps1）且已完成入库。",
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"元数据查询失败: {type(exc).__name__}: {exc}",
            code="BACKEND_UNAVAILABLE",
            hint="请确认元数据集合 enterprise_docs_meta 已建立。",
        )

    if metadata is None:
        logger.info("get_document_metadata doc_id=%r -> NOT_FOUND", params.doc_id)
        return _fail(
            f"文档不存在: doc_id={params.doc_id}",
            code="NOT_FOUND",
            hint="请先用 search_documents 检索，使用返回结果中的 doc_id 再查询。",
        )

    logger.info("get_document_metadata doc_id=%r -> %s", params.doc_id, metadata.get("title"))
    return _ok(metadata)


def build_server() -> Any:
    """创建并装配 knowledge_base MCP server。

    Returns:
        已注册全部工具的 MCP server 实例（mcp 1.x 为 FastMCP，2.x 为 MCPServer）。
    """
    server = _ServerClass(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        **{
            _SERVER_KW_DESC: (
                "企业知识库检索服务。适用场景：制度/政策/流程/SOP/产品手册类问题。"
                "先用 search_documents 拿到片段，需要文档级信息（页数、来源、创建时间）"
                "时再用 get_document_metadata。"
            )
        },
    )

    server.add_tool(
        search_documents,
        name="search_documents",
        title="企业知识库混合检索",
        description=(
            "在企业知识库中检索与 query 相关的文档片段，返回 content/source/score/page。"
            "适合回答'退货政策是什么''报销流程怎么走'等制度类问题。"
            "回答时请引用 source 与 page 以便溯源。"
        ),
        structured_output=False,
    )
    server.add_tool(
        get_document_metadata,
        name="get_document_metadata",
        title="查询文档元数据",
        description=(
            "按 doc_id 返回文档元数据：标题、来源、类型、页数、片段数、创建时间等。"
            "当需要引用出处或判断文档新旧程度时调用。"
        ),
        structured_output=False,
    )
    logger.info("server '%s' v%s 装配完成，已注册 2 个工具", SERVER_NAME, SERVER_VERSION)
    return server


def main() -> None:
    """以 stdio 传输启动 server。"""
    build_server().run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
