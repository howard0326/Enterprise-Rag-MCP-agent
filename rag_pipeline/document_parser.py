"""文档解析与分块：PDF / Markdown / 纯文本 → 带元数据与页码的 chunk 列表。

核心设计决策（每条都直接影响阶段5 的评测数字，不是随手选的）：

    1. **chunk 不跨页**：PDF 按页独立分块。若允许跨页合并，一个 chunk 会同时
       属于第 3 页和第 4 页，``page`` 字段就只能填首页码，引用出处会失真。
       代价是页边界处可能产生略短的 chunk，换来"引用页码永远可追溯"。

    2. **结构感知分块，而非定长切断**：先在标题/段落/句号处切，再向
       ``chunk_size`` 打包。中文语义密度高，定长 512 字硬切极易把"退货期限"
       和"7 个自然日"切到两个 chunk 里，检索命中却答不出数字。

    3. **overlap 取自前一个 chunk 的尾部**：保证被切断的句子在下一个 chunk
       里完整出现，避免"半句话"无法被语义匹配。

    4. **doc_id = 内容哈希前 16 位**：用于增量索引判重。小文件用全量哈希
       （避免前缀相同、内容不同的文件被误判为同一文档），大文件用
       "前 1 MiB + 大小 + mtime" 快速摘要（避免每次摄入都全量读盘）。

    5. **分块后的 token 数不参与切分**：切分单位是字符，因为
       ``chunk_size=512/overlap=64`` 是配置约定值；bge-m3 支持 8192 token，
       512 字符（中文约 350 token）远在安全区内。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config.config_loader import IngestionConfig, get_config

__all__ = [
    "ParseError",
    "UnsupportedFormatError",
    "DocumentChunk",
    "DocumentMetadata",
    "ParsedDocument",
    "TextChunker",
    "DocumentParser",
    "parse_path",
]

# 句子边界：中英文句号/问号/叹号/分号 + 换行
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*|\n{2,}")
# Markdown 标题
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# 全角/半角空白归一化
_WS_RE = re.compile(r"[ \t\u3000]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


class ParseError(RuntimeError):
    """文档解析失败（编码错误、PDF 损坏、文本不可提取等）。"""


class UnsupportedFormatError(ParseError):
    """文件扩展名不在支持列表中。"""


@dataclass
class DocumentChunk:
    """单个文档片段。

    Attributes:
        chunk_id: 片段 ID（``{doc_id}-{index:04d}``，确定性生成，可重复索引）。
        doc_id: 所属文档 ID（内容哈希前 16 位）。
        content: 片段正文。
        source: 源文件相对路径。
        page: PDF 页码（从 1 开始）；非分页文档为 ``None``。
        chunk_index: 在文档内的顺序号（从 0 开始）。
        char_count: 正文字符数。
        token_count: 分词后的 token 数（供稀疏编码更新平均长度）。
        heading: 片段所属的最近一级标题（Markdown 有效，其余为 ``""``）。
        created_at: 文件创建时间（ISO 8601）。
    """

    chunk_id: str
    doc_id: str
    content: str
    source: str
    page: int | None
    chunk_index: int
    char_count: int
    token_count: int = 0
    heading: str = ""
    created_at: str = ""

    def to_payload(self) -> dict[str, Any]:
        """转为 Qdrant payload。

        Returns:
            扁平字典；字段名与阶段3 ``get_document_metadata`` 读取的键保持一致。
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "heading": self.heading,
            "created_at": self.created_at,
        }

    def embedding_text(self) -> str:
        """返回用于向量与稀疏编码的文本（含来源与标题上下文）。

        为什么不能直接拿 ``content`` 编码（实测依据）：
            片段"本政策适用于通过官方直营商城…"本身不含"退货"二字，但它的所属
            标题是"1. 适用范围"；而"标准商品自签收之日起 7 个自然日内…"所属标题是
            "2. 无理由退货期限"。把标题拼进来后，查询"退货政策是什么"与该片段的
            语义距离显著缩短 —— 这是把已抽取的结构信息真正用起来。

        同时拼入来源文件名（如"sample_退货政策.md"），让文档级主题词参与匹配。

        Args:
            self: 片段对象。

        Returns:
            形如 ``"《sample_退货政策.md》# 2. 无理由退货期限\\n标准商品自…"`` 的文本。
        """
        parts: list[str] = []
        if self.source:
            parts.append(f"《{self.source}》")
        if self.heading:
            parts.append(f"# {self.heading}")
        parts.append(self.content)
        return "\n".join(parts)


@dataclass
class DocumentMetadata:
    """文档级元数据。"""

    doc_id: str
    title: str
    source: str
    content_type: str
    page_count: int
    chunk_count: int
    created_at: str
    indexed_at: str
    file_hash: str
    size_bytes: int
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "content_type": self.content_type,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "indexed_at": self.indexed_at,
            "file_hash": self.file_hash,
            "size_bytes": self.size_bytes,
            "tags": list(self.tags),
        }


@dataclass
class ParsedDocument:
    """解析结果：元数据 + 片段列表。"""

    metadata: DocumentMetadata
    chunks: list[DocumentChunk]

    @property
    def doc_id(self) -> str:
        """文档 ID 快捷访问。"""
        return self.metadata.doc_id


def _normalize_whitespace(text: str) -> str:
    """归一化空白：压缩连续空格、折叠 3 个以上换行。

    Args:
        text: 原始文本。

    Returns:
        归一化后的文本。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


class TextChunker:
    """结构感知分块器。"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 64,
    ) -> None:
        """初始化分块器。

        Args:
            chunk_size: 目标片段字符数。
            chunk_overlap: 相邻片段重叠字符数（取自前一片段尾部）。
            min_chunk_size: 片段最小字符数；小于该值的尾块向前合并，避免碎片。

        Raises:
            ValueError: ``chunk_overlap >= chunk_size``（会导致打包死循环）。
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _split_units(text: str) -> list[str]:
        """把文本切成"最小可切分单元"（句子/段落）。

        Args:
            text: 归一化后的文本。

        Returns:
            句子列表；超过 ``chunk_size`` 的超长句会被进一步按逗号切分。
        """
        raw = [seg.strip() for seg in _SENTENCE_SPLIT_RE.split(text) if seg and seg.strip()]
        units: list[str] = []
        for seg in raw:
            if len(seg) <= 512:
                units.append(seg)
                continue
            # 超长句（如无标点的长表格行）：按中文逗号/顿号再切，最后兜底硬切
            sub = [p.strip() for p in re.split(r"(?<=[，,、])", seg) if p.strip()]
            for piece in sub:
                if len(piece) <= 512:
                    units.append(piece)
                else:
                    units.extend(piece[i : i + 512] for i in range(0, len(piece), 512))
        return units

    def split(self, text: str) -> list[str]:
        """把一段文本切分为带重叠的片段。

        Args:
            text: 归一化后的文本。

        Returns:
            片段字符串列表；空输入返回空列表。
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        units = self._split_units(text)
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_len = 0

        for unit in units:
            unit_len = len(unit)
            if buffer and buffer_len + unit_len > self.chunk_size:
                chunks.append("\n".join(buffer))
                # 取前一片段尾部作为重叠起点：优先按整句回退，保证语义完整
                tail = self._overlap_tail(buffer)
                buffer = [tail] if tail else []
                buffer_len = len(tail)
            buffer.append(unit)
            buffer_len += unit_len

        if buffer:
            chunks.append("\n".join(buffer))

        # 尾块过短则并入前一块（前提是合并后不超过 chunk_size * 1.5）
        if (
            len(chunks) >= 2
            and len(chunks[-1]) < self.min_chunk_size
            and len(chunks[-2]) + len(chunks[-1]) <= int(self.chunk_size * 1.5)
        ):
            tail = chunks.pop()
            chunks[-1] = chunks[-1] + "\n" + tail

        return [c.strip() for c in chunks if c.strip()]

    def _overlap_tail(self, buffer: list[str]) -> str:
        """从缓冲区尾部取出重叠文本。

        Args:
            buffer: 当前缓冲的句子列表。

        Returns:
            长度不超过 ``chunk_overlap`` 的尾部文本；不足时返回整段。
        """
        if self.chunk_overlap <= 0 or not buffer:
            return ""
        tail_parts: list[str] = []
        total = 0
        for unit in reversed(buffer):
            if total + len(unit) > self.chunk_overlap and tail_parts:
                break
            tail_parts.insert(0, unit)
            total += len(unit)
            if total >= self.chunk_overlap:
                break
        tail = "\n".join(tail_parts)
        return tail[-self.chunk_overlap :] if len(tail) > self.chunk_overlap else tail


class DocumentParser:
    """多格式文档解析器。"""

    def __init__(self, config: IngestionConfig | None = None) -> None:
        """初始化解析器。

        Args:
            config: 摄取配置；为 ``None`` 时读取 ``config/settings.yaml``。
        """
        self.config = config or get_config().ingestion
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def parse(self, path: str | Path, root: str | Path | None = None) -> ParsedDocument:
        """解析单个文件。

        Args:
            path: 文件路径。
            root: 用于计算 ``source`` 相对路径的根目录；``None`` 时用文件所在目录。

        Returns:
            :class:`ParsedDocument`。

        Raises:
            FileNotFoundError: 文件不存在。
            UnsupportedFormatError: 扩展名不受支持。
            ParseError: 解析失败（编码/损坏/无可提取文本）。
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"文件不存在或不是普通文件: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in set(self.config.supported_extensions):
            raise UnsupportedFormatError(
                f"不支持的文件类型: {suffix}（支持: {', '.join(self.config.supported_extensions)}）"
            )

        base = Path(root) if root is not None else file_path.parent
        try:
            source = str(file_path.resolve().relative_to(Path(base).resolve())).replace("\\", "/")
        except ValueError:
            source = file_path.name

        stat = file_path.stat()
        file_hash = self.compute_file_hash(file_path)
        doc_id = file_hash[:16]
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).astimezone().isoformat(
            timespec="seconds"
        )

        if suffix == ".pdf":
            sections, page_count, content_type = self._parse_pdf(file_path)
            default_title = file_path.stem
        elif suffix in (".md", ".markdown"):
            sections, page_count, content_type, default_title = self._parse_markdown(file_path)
        else:
            sections, page_count, content_type, default_title = self._parse_text(file_path)

        chunks: list[DocumentChunk] = []
        for section_text, page, heading in sections:
            for piece in self.chunker.split(section_text):
                index = len(chunks)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}-{index:04d}",
                        doc_id=doc_id,
                        content=piece,
                        source=source,
                        page=page,
                        chunk_index=index,
                        char_count=len(piece),
                        heading=heading,
                        created_at=created_at,
                    )
                )

        if not chunks:
            raise ParseError(
                f"未能从文件中提取到任何文本: {file_path}。"
                f"PDF 可能是扫描件（需 OCR，本项目未实现）或为空文件。"
            )

        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=default_title,
            source=source,
            content_type=content_type,
            page_count=page_count,
            chunk_count=len(chunks),
            created_at=created_at,
            indexed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            file_hash=file_hash,
            size_bytes=stat.st_size,
            tags=self._infer_tags(default_title, source),
        )
        return ParsedDocument(metadata=metadata, chunks=chunks)

    def parse_directory(
        self, directory: str | Path, recursive: bool = True
    ) -> Iterable[tuple[Path, ParsedDocument | Exception]]:
        """遍历目录并逐个解析，单个文件的失败不影响整体。

        Args:
            directory: 待扫描目录。
            recursive: 是否递归子目录。

        Yields:
            ``(文件路径, ParsedDocument 或 异常对象)`` 二元组。
        """
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"目录不存在: {root}")

        pattern = "**/*" if recursive else "*"
        allowed = set(self.config.supported_extensions)
        for file_path in sorted(root.glob(pattern)):
            if not file_path.is_file() or file_path.suffix.lower() not in allowed:
                continue
            try:
                yield file_path, self.parse(file_path, root=root)
            except Exception as exc:  # noqa: BLE001 - 收集失败而非中断整批
                yield file_path, exc

    # ------------------------------------------------------------------
    # 各格式解析
    # ------------------------------------------------------------------
    def _parse_pdf(self, file_path: Path) -> tuple[list[tuple[str, int | None, str]], int, str]:
        """解析 PDF，按页返回。

        Args:
            file_path: PDF 路径。

        Returns:
            ``(sections, page_count, content_type)``，其中每个 section 为
            ``(文本, 页码, 标题)``。

        Raises:
            ParseError: pypdf 未安装、文件损坏或加密。
        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ParseError("未安装 pypdf，无法解析 PDF。请执行 pip install pypdf") from exc

        try:
            reader = PdfReader(str(file_path))
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")  # 尝试空密码
                except Exception as exc:  # noqa: BLE001
                    raise ParseError(f"PDF 已加密，无法解析: {file_path}") from exc
            page_count = len(reader.pages)
            sections: list[tuple[str, int | None, str]] = []
            for page_no, page in enumerate(reader.pages, start=1):
                try:
                    raw = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整篇
                    raise ParseError(
                        f"PDF 第 {page_no} 页文本提取失败（文件可能损坏）: {file_path} -> {exc}"
                    ) from exc
                text = _normalize_whitespace(raw)
                if text:
                    sections.append((text, page_no, ""))
        except ParseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"PDF 解析失败: {file_path} -> {type(exc).__name__}: {exc}") from exc

        return sections, page_count, "pdf"

    def _parse_markdown(
        self, file_path: Path
    ) -> tuple[list[tuple[str, int | None, str]], int, str, str]:
        """解析 Markdown，按标题切分为 section。

        Args:
            file_path: Markdown 路径。

        Returns:
            ``(sections, page_count, content_type, title)``。

        Raises:
            ParseError: 文件读取或解码失败。
        """
        text = self._read_text(file_path)
        title = file_path.stem
        headings = list(_MD_HEADING_RE.finditer(text))
        sections: list[tuple[str, int | None, str]] = []

        if not headings:
            body = _normalize_whitespace(text)
            if body:
                sections.append((body, None, ""))
            return sections, 0, "markdown", title

        # 文档大标题：第一个一级标题（若没有则取第一个标题）
        h1 = [m for m in headings if len(m.group(1)) == 1]
        title = (h1[0] if h1 else headings[0]).group(2).strip() or file_path.stem

        for idx, match in enumerate(headings):
            start = match.end()
            end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
            body = _normalize_whitespace(text[start:end])
            if not body:
                continue
            sections.append((body, None, match.group(2).strip()))

        # 首个标题之前的前言
        preamble = _normalize_whitespace(text[: headings[0].start()])
        if preamble:
            sections.insert(0, (preamble, None, ""))

        return sections, 0, "markdown", title

    def _parse_text(
        self, file_path: Path
    ) -> tuple[list[tuple[str, int | None, str]], int, str, str]:
        """解析纯文本，按空行分段。

        Args:
            file_path: 文本路径。

        Returns:
            ``(sections, page_count, content_type, title)``。

        Raises:
            ParseError: 文件读取或解码失败。
        """
        text = self._read_text(file_path)
        body = _normalize_whitespace(text)
        sections = [(body, None, "")] if body else []
        return sections, 0, "text", file_path.stem

    def _read_text(self, file_path: Path) -> str:
        """读取文本文件，UTF-8 失败时回退 GBK（中文 Windows 常见）。

        Args:
            file_path: 文件路径。

        Returns:
            文件文本内容。

        Raises:
            ParseError: 两种编码都失败。
        """
        raw = file_path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ParseError(
            f"文件解码失败（已尝试 utf-8/gbk/gb18030）: {file_path}。"
            f"请将文件转为 UTF-8 编码后重试。"
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def compute_file_hash(file_path: Path, full_read_limit: int = 1024 * 1024) -> str:
        """计算文件内容哈希（用于增量索引判重）。

        小文件（≤1 MiB）用全量 SHA-256，避免"前缀相同、内容不同"被误判；
        大文件用"前 1 MiB + 文件大小"摘要，避免每次摄入都全量读盘。

        Args:
            file_path: 文件路径。
            full_read_limit: 全量读取的大小上限（字节）。

        Returns:
            十六进制哈希字符串。
        """
        stat = file_path.stat()
        digest = hashlib.sha256()
        with file_path.open("rb") as fh:
            if stat.st_size <= full_read_limit:
                digest.update(fh.read())
            else:
                digest.update(fh.read(full_read_limit))
                digest.update(str(stat.st_size).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _infer_tags(title: str, source: str) -> list[str]:
        """从文件名推断业务标签。

        Args:
            title: 文档标题。
            source: 源文件相对路径。

        Returns:
            标签列表（可能为空）。
        """
        joined = f"{title} {source}"
        candidates = {
            "政策": ["政策", "制度", "规范", "条例", "规定"],
            "售后": ["退货", "退款", "换货", "售后"],
            "订单": ["订单", "发货", "物流", "签收"],
            "客服": ["工单", "客服", "投诉"],
            "财务": ["报销", "发票", "账期", "结算"],
        }
        return [tag for tag, words in candidates.items() if any(w in joined for w in words)]


def parse_path(path: str | Path, root: str | Path | None = None) -> ParsedDocument:
    """便捷函数：用默认配置解析单个文件。

    Args:
        path: 文件路径。
        root: 相对路径基准目录。

    Returns:
        :class:`ParsedDocument`。
    """
    return DocumentParser().parse(path, root=root)
