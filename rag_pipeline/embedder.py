"""嵌入与稀疏编码层：稠密向量（可插拔 provider）+ 服务端 IDF 稀疏 BM25。

设计要点（均有实测依据）：
    1. **可插拔稠密嵌入**：``local`` 走 sentence-transformers 本地推理；
       ``openai_compatible`` 走任意兼容 ``/v1/embeddings`` 的服务。DeepSeek 官方
       目前未开放 embedding 接口，因此默认走本地 bge-m3；将来切换只改配置。
    2. **版本兼容**：sentence-transformers 5.x 起把
       ``get_sentence_embedding_dimension`` 改名为 ``get_embedding_dimension``，
       这里做兼容取值，避免 FutureWarning 变成硬失败。
    3. **稀疏向量只算词频项**：BM25 的 IDF 由 Qdrant 服务端的
       ``SparseVectorParams(modifier=Modifier.IDF)`` 自动施加，
       因此客户端只提供 ``tf * (k1+1) / (tf + k1*(1-b+b*len/avg_len))``。
       这是关键取舍：把 IDF 交给向量库，服务端无需回传语料统计；
       且同一套编码函数在入库与检索两侧复用，保证打分口径一致。
    4. **中英文分词**：中文用字符 bigram（无需词典，工程上最稳），
       英文/数字用小写词。带 offset 的 token 便于将来做高亮。
    5. **预置 HF 环境变量**：huggingface.co 在部分网络不可达，需走镜像；
       ``HF_HUB_DISABLE_SYMLINKS`` 规避受限环境下符号链接创建失败。
       均在导入时就地设置（仅当用户未显式指定时）。

环境变量（可在 .env 中配置）：
    HF_ENDPOINT              默认 https://hf-mirror.com
    HF_HOME                  默认 <项目根>/.cache/huggingface
    HF_HUB_DISABLE_SYMLINKS  默认 1
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from config.config_loader import PROJECT_ROOT, EmbeddingConfig, get_config

# ---------------------------------------------------------------------------
# 在导入 sentence_transformers 之前设置缓存与镜像
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
# 强制禁用符号链接（不用 setdefault）：受限环境下 file_download 的
# "能否建 symlink" 探测会在 tmp 目录 touch 时抛 PermissionError，
# 直接让 huggingface_hub 走真实文件复制，绕过探测。
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

__all__ = [
    "EmbeddingError",
    "Embedder",
    "SparseEncoder",
    "SparseVectorData",
    "get_embedder",
    "get_sparse_encoder",
    "tokenize",
]

# BM25 超参：k1 控制词频饱和，b 控制长度归一化强度
BM25_K1 = 1.5
BM25_B = 0.75
# 稀疏词表哈希空间：2^24，冲突概率对本项目规模可忽略
SPARSE_VOCAB_SIZE = 1 << 24

_DEFAULT_AVG_LEN = 200.0

# 中日韩统一表意文字 + 常用中文标点：按 bigram 切分
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


class EmbeddingError(RuntimeError):
    """嵌入模型加载或编码失败。"""


class SparseVectorData:
    """稀疏向量容器：与 qdrant_client.models.SparseVector 字段一一对应。

    Attributes:
        indices: 词表下标（有序，与 values 等长）。
        values: 对应的权重。
    """

    __slots__ = ("indices", "values")

    def __init__(self, indices: Sequence[int], values: Sequence[float]) -> None:
        self.indices = list(indices)
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return f"SparseVectorData(nnz={len(self.indices)})"

    def as_dict(self) -> dict[str, list[Any]]:
        """转为可序列化字典（便于调试与日志）。"""
        return {"indices": self.indices, "values": self.values}


def tokenize(text: str) -> list[str]:
    """对中英混排文本分词：中文取字符 bigram，英文数字取小写词。

    Args:
        text: 原始文本。

    Returns:
        token 列表。中文按相邻字符两两成词（单字文本退化为单 token），
        英文按 ``[a-z0-9]`` 词边界切分。例如
        ``"退货 policy 2025"`` → ``['退货', '货 ', ' policy', 'policy', '2025']``
        的近似形式（标点与空格会被过滤）。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []

    # 英文/数字词
    tokens.extend(_LATIN_RE.findall(lowered))

    # 中文 bigram：只保留汉字序列，标点/空格作为分隔
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return tokens


def _hash_token(token: str) -> int:
    """把 token 稳定映射到稀疏词表下标。

    Args:
        token: 分词结果中的单个 token。

    Returns:
        ``[0, SPARSE_VOCAB_SIZE)`` 范围内的整数下标。
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=6).digest()
    return int.from_bytes(digest, "big") % SPARSE_VOCAB_SIZE


class SparseEncoder:
    """BM25 词频项编码器（IDF 由 Qdrant 服务端施加）。

    该类刻意不做任何语料统计：同一实例既用于入库编码，也用于查询编码，
    因此两侧打分口径天然一致。
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        """初始化编码器。

        Args:
            k1: BM25 词频饱和参数。
            b: BM25 文档长度归一化参数。
        """
        self.k1 = k1
        self.b = b
        self._avg_len = _DEFAULT_AVG_LEN
        self._avg_len_lock = threading.Lock()

    def update_corpus_stats(self, lengths: Iterable[int]) -> None:
        """用已编码文档的长度更新平均长度（影响长度归一化）。

        Args:
            lengths: 各文档的 token 数序列。
        """
        values = [n for n in lengths if n > 0]
        if not values:
            return
        with self._avg_len_lock:
            self._avg_len = sum(values) / len(values)

    @property
    def avg_len(self) -> float:
        """当前使用的平均文档长度（token 数）。"""
        return self._avg_len

    def encode(self, text: str) -> SparseVectorData:
        """把单条文本编码为稀疏向量。

        Args:
            text: 待编码文本。

        Returns:
            :class:`SparseVectorData`；无有效 token 时返回空向量。
        """
        tokens = tokenize(text)
        if not tokens:
            return SparseVectorData([], [])

        term_freq = Counter(tokens)
        doc_len = len(tokens)
        norm = self.k1 * (1.0 - self.b + self.b * doc_len / self._avg_len)

        indices: list[int] = []
        values: list[float] = []
        for token, freq in term_freq.items():
            weight = freq * (self.k1 + 1.0) / (freq + norm)
            indices.append(_hash_token(token))
            values.append(round(weight, 6))

        # 合并哈希冲突（罕见，但必须处理，否则 Qdrant 会拒绝重复下标）
        merged: dict[int, float] = {}
        for idx, val in zip(indices, values):
            merged[idx] = merged.get(idx, 0.0) + val
        return SparseVectorData(sorted(merged), [merged[i] for i in sorted(merged)])


class Embedder:
    """稠密嵌入器：统一 ``local`` 与 ``openai_compatible`` 两种 provider。

    实例内部对模型做懒加载与缓存，构造本身不做任何重活。
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """初始化嵌入器。

        Args:
            config: 嵌入配置；为 ``None`` 时读取 ``config/settings.yaml``。
        """
        self.config = config or get_config().embedding
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._actual_dim: int | None = None

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _resolve_local_model_path(self) -> str:
        """准备本地模型目录，返回可直接加载的路径或 HF 仓库 ID。

        默认走"非镜像模式"：先用 ``snapshot_download`` + ``local_dir`` 把模型落成
        **真实文件目录**，再让 SentenceTransformer 从该目录加载。
        这样彻底绕开缓存镜像（blob + symlink）机制 —— 受限环境下
        huggingface_hub 的"能否创建符号链接"探测会在 tmp 目录 touch 时抛
        PermissionError，而该探测发生在 hf_hub_download 内部，无法通过参数关闭。

        Returns:
            本地目录绝对路径（已存在或下载成功），或原模型名（下载失败时回退，
            让 SentenceTransformer 自行处理，以便给出更贴近根因的报错）。
        """
        model = self.config.model
        # 已是本地路径则直接使用
        if Path(model).exists():
            return model
        if "/" not in model:  # 非 HF 仓库 ID，交给下游
            return model

        local_dir = PROJECT_ROOT / ".cache" / "models" / model.replace("/", "--")
        marker = local_dir / "config.json"
        if marker.is_file():
            return str(local_dir)

        try:
            from huggingface_hub import snapshot_download

            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model,
                local_dir=str(local_dir),
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.bin",
                    "*.safetensors",
                    "*.onnx",
                ],
            )
            if marker.is_file():
                return str(local_dir)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] 模型 {model} 预下载到本地目录失败（将回退为直接加载）："
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return model

    def _load_local_model(self) -> Any:
        """懒加载本地 sentence-transformers 模型。

        Returns:
            已加载的 ``SentenceTransformer`` 实例（同一实例被复用）。

        Raises:
            EmbeddingError: 模型下载/加载失败。
        """
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "未安装 sentence-transformers。请执行 pip install -r requirements.txt，"
                    "或把 embedding.provider 改为 openai_compatible 走远程接口。"
                ) from exc

            cache_dir = Path(self.config.cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = PROJECT_ROOT / cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_ref = self._resolve_local_model_path()
            try:
                self._model = SentenceTransformer(
                    model_ref,
                    device=self.config.device,
                    cache_folder=str(cache_dir),
                )
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"加载嵌入模型失败: model={self.config.model}, device={self.config.device}. "
                    f"若为网络问题，请设置 HF_ENDPOINT=https://hf-mirror.com；"
                    f"原始错误: {type(exc).__name__}: {exc}"
                ) from exc

            self._actual_dim = self._detect_dimension(self._model)
            self._configure_inference()
            return self._model

    def _configure_inference(self) -> None:
        """收敛推理期开销。

        实测：单条编码 0.27s，但每次查询端到端却要 ~4.2s，说明开销不在矩阵计算。
        这里做三件事：
            1. 固定 torch 线程数：默认线程数在每次调用时可能触发线程池重建，
               Windows 上这一项开销可观且抖动大；
            2. 把模型 ``max_seq_length`` 收敛到配置的 ``max_length``：
               bge-m3 默认上限很高，但我们的 chunk 只有 512 字符，
               多余的位置编码容量会按最长输入分配显存/内存；
            3. 关闭 tokenizer 的并行告警噪声。
        """
        try:
            import torch

            if self.config.device == "cpu":
                # 用小线程数换取稳定的低延迟：16 条短文本用不满多核，
                # 反而会因线程争用变慢。
                torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
        except Exception:  # noqa: BLE001 - 没有 torch 时（远程 provider）静默跳过
            pass

        try:
            max_len = int(self.config.max_length)
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = max_len
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _detect_dimension(model: Any) -> int | None:
        """读取模型输出维度，兼容 ST 5.x/6.x 的方法改名。

        Args:
            model: SentenceTransformer 实例。

        Returns:
            维度整数；两个方法都不存在时返回 ``None``。
        """
        for method in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            fn = getattr(model, method, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:  # noqa: BLE001
                    continue
        return None

    @property
    def dimension(self) -> int:
        """实际使用的向量维度（本地模型以实测为准，远程以配置为准）。"""
        if self.config.provider == "local":
            self._load_local_model()
            return self._actual_dim or self.config.dimension
        return self.config.dimension

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode_documents(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """编码文档片段。

        Args:
            texts: 文档文本列表。
            batch_size: 批大小；``None`` 时取配置值。

        Returns:
            与 ``texts`` 等长的向量列表，每个向量为 ``list[float]``。

        Raises:
            EmbeddingError: 编码失败或维度与配置不符。
        """
        return self._encode(texts, batch_size, kind="document")

    def encode_query(self, text: str) -> list[float]:
        """编码单条查询。

        Args:
            text: 查询文本。

        Returns:
            查询向量。

        Raises:
            EmbeddingError: 编码失败。
        """
        vectors = self._encode([text], None, kind="query")
        if not vectors:
            raise EmbeddingError("查询编码返回空结果")
        return vectors[0]

    def _encode(
        self, texts: Sequence[str], batch_size: int | None, kind: str
    ) -> list[list[float]]:
        """编码实现：分派到本地或远程 provider，并做维度自检。

        Args:
            texts: 文本列表。
            batch_size: 批大小。
            kind: ``document`` 或 ``query``（当前两者编码方式一致，
                bge-m3 无需指令前缀；保留参数以便将来接入需要前缀的模型）。

        Returns:
            向量列表。

        Raises:
            EmbeddingError: provider 未知、编码失败或维度不一致。
        """
        if not texts:
            return []
        size = batch_size or self.config.batch_size

        if self.config.provider == "local":
            model = self._load_local_model()
            try:
                raw = self._encode_local(model, list(texts), size)
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    f"本地嵌入编码失败（{len(texts)} 条文本）: {type(exc).__name__}: {exc}"
                ) from exc
            vectors = [[float(x) for x in row] for row in raw]
        elif self.config.provider == "openai_compatible":
            vectors = self._encode_remote(texts)
        else:  # pragma: no cover - Pydantic 已限定枚举
            raise EmbeddingError(f"未知的 embedding.provider: {self.config.provider}")

        got = len(vectors[0]) if vectors else 0
        if got != self.config.dimension:
            raise EmbeddingError(
                f"嵌入维度不一致：模型实际输出 {got} 维，但配置 embedding.dimension="
                f"{self.config.dimension}。Qdrant 集合维度必须与之一致，"
                f"请修正 config/settings.yaml。"
            )
        if self.config.normalize:
            # 显式 L2 归一化，而不是依赖模型库的 normalize_embeddings 参数：
            # 检索侧用余弦相似度，向量必须是单位长度，这一步不能靠库的默认行为。
            vectors = [self._l2_normalize(v) for v in vectors]
        return vectors

    @staticmethod
    def _l2_normalize(vector: Sequence[float]) -> list[float]:
        """L2 归一化单个向量。

        Args:
            vector: 原始向量。

        Returns:
            单位长度向量；零向量原样返回（避免除零）。
        """
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        if norm == 0.0:
            return [float(x) for x in vector]
        return [float(x) / norm for x in vector]

    @staticmethod
    def _encode_local(model: Any, texts: list[str], batch_size: int) -> Any:
        """本地编码：包一层 ``torch.inference_mode()`` 去掉自动求导开销。

        Args:
            model: SentenceTransformer 实例。
            texts: 文本列表。
            batch_size: 批大小。

        Returns:
            形如 ``(n, dim)`` 的二维数组。

        Raises:
            Exception: 透传底层编码异常（由调用方包装为 EmbeddingError）。
        """
        try:
            import torch

            with torch.inference_mode():
                return model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        except ImportError:
            return model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

    def _encode_remote(self, texts: Sequence[str]) -> list[list[float]]:
        """通过 OpenAI 兼容的 ``/v1/embeddings`` 接口编码。

        Args:
            texts: 文本列表。

        Returns:
            向量列表。

        Raises:
            EmbeddingError: 缺少 api_key、网络失败或响应格式异常。
        """
        import httpx

        if not self.config.api_key:
            raise EmbeddingError(
                "embedding.provider=openai_compatible 但未配置 api_key，"
                "请在 .env 中设置对应密钥。"
            )
        url = self.config.base_url.rstrip("/") + "/embeddings"
        try:
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.config.model, "input": list(texts)},
                timeout=60.0,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"远程嵌入接口调用失败: {url} -> {type(exc).__name__}: {exc}"
            ) from exc

        try:
            data = sorted(payload["data"], key=lambda item: item.get("index", 0))
            return [[float(x) for x in item["embedding"]] for item in data]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"远程嵌入接口响应格式异常: {str(payload)[:200]}") from exc

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------
    @staticmethod
    def l2_norm(vector: Sequence[float]) -> float:
        """计算向量 L2 范数（用于自检归一化是否生效）。

        Args:
            vector: 向量。

        Returns:
            L2 范数。
        """
        return math.sqrt(sum(float(x) * float(x) for x in vector))


_embedder_singleton: Embedder | None = None
_sparse_singleton: SparseEncoder | None = None


def get_embedder() -> Embedder:
    """获取进程级嵌入器单例。

    Returns:
        :class:`Embedder` 实例（避免重复加载 2 GB 级模型）。
    """
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = Embedder()
    return _embedder_singleton


def get_sparse_encoder() -> SparseEncoder:
    """获取进程级稀疏编码器单例。

    Returns:
        :class:`SparseEncoder` 实例。
    """
    global _sparse_singleton
    if _sparse_singleton is None:
        _sparse_singleton = SparseEncoder()
    return _sparse_singleton
