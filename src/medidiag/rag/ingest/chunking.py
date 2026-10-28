"""分块与稳定 ID（M6）：递归切分（recursive）与父子分块（parent_child）。

两种策略共用约定：

- 输入是解析后的统一文档模型（Document → Section → Element），不跨章节切分。
- ``chunk_id`` 格式为 ``{doc_id}:{章节序号}:{策略}:{序号}``，章节序号为文档内
  从 0 起的全局下标（含空章节，保证跨文档一致）；策略短码：``rc`` = 递归切分，
  ``pc`` = 父子分块的子块（child），``pp`` = 父子分块的父块（parent）。
  序号在每个 ``(doc_id, 章节序号, 策略)`` 组内从 0 起按生成顺序编号，
  因此同样输入得到同样的 chunk_id 与输出顺序（幂等）。
- ``chunk_hash`` 为文本的 sha256；``source_id`` 用 ``doc_id``，``source`` 用
  文档标题（沿用 v3「来源显示名」的约定），字段集合与 ``KnowledgeChunk`` 一致；
  新增字段（parent_id、section_path、strategy、chunk_hash、role、source_type、
  published_at、period、license、content_kind 等）放进 ``metadata``。
- 引用校验以子块为准：``chunks.jsonl`` 只含可检索块（递归切分块与父子的子块），
  父块单独写入 ``parents.jsonl``（``metadata.role = "parent"``），不进检索索引，
  只作为命中子块后交给模型的上下文。
- 表格：递归切分与父子分块都把表格行（序列化文本「列名：值；…」）单独成块，
  不与正文混切；父子分块另把整表摘要作为父块。取舍：表格行是独立的结构化事实，
  与上下文正文主题可能不同，混切会稀释检索信号，且引用校验对比的就是序列化文本。
- 列表元素按条目切分（WHO 重要事实多为列表），条目不跨元素合并；
  父子分块把整个列表元素作为父块。
- 跳过 M5 的剔除内容：文档级去重剔除的文档不出现在输入里（调用方按
  ``kept_doc_ids`` 过滤）；段落级跨来源重复中被判剔除的一方由
  ``dropped_paragraphs``（来自 dedup.DedupReport 或 ingest 报告）跳过。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medidiag.rag.ingest.document import Document, Element, serialize_table_row
from medidiag.schemas import KnowledgeChunk

# ---- 参数（验收条款；改参数会改变全部 chunk_id） ----
RECURSIVE_MAX_CHARS = 512
RECURSIVE_OVERLAP_RATIO = 0.15  # 重叠约 15%
RECURSIVE_OVERLAP_CHARS = int(RECURSIVE_MAX_CHARS * RECURSIVE_OVERLAP_RATIO)  # 76
CHILD_MAX_CHARS = 110
# 短碎片合并时允许的子块上限放宽值（2026-10-28 数据质量修复：不足 15 字的
# 相邻短句合并，合并后仍不超过 110 字优先；仅合并需要时可到 130 字并说明）
CHILD_MERGE_MAX_CHARS = 130
PARENT_MAX_CHARS = 600
# 低于该长度的子块视为短碎片，需与相邻句合并（仍遵守上限）；纯日期/纯控件文字
# 直接丢弃；合并不了的孤立短碎片仍保留（一句完整但短的正文不丢弃，如「疟疾会致命。」）
CHILD_MIN_CHARS = 15

_STRATEGY_CODES = {"recursive": "rc", "child": "pc", "parent": "pp"}

# 句子边界：句号类（。！？；）、逗号类（，、：）；括号包裹的标点不误切
_SENTENCE_END_RE = re.compile(r"[。！？；][”’」』）)]*")
_COMMA_RE = re.compile(r"[，、：][”’」』）)]*")

# 页面控件残留文字（清洗阶段按来源裁剪，这里兜底，避免进入分块）：
# 含 WHO 页首杂质（跳转链接、「实况报道」栏目标签、页面日期行、罗马数字/纯数字页码）。
_CONTROL_LINE_PATTERNS = (
    r"^Skip to main content$", r"^阅读时间$", r"^打印$", r"^分享$", r"^分享本页",
    r"^世卫组织\s*/\s*媒体中心", r"^媒体中心\s*/", r"^实况报道$",
    r"^\d{4}年\d{1,2}月\d{1,2}日$", r"^\d{4}年\d{1,2}月$",
    r"^[ivxlcdm]{1,8}$", r"^\d{1,4}$",
)
_CONTROL_LINE_RE = re.compile("|".join(f"(?:{p})" for p in _CONTROL_LINE_PATTERNS), re.I)


def _squash(text: str) -> str:
    """去掉全部空白，用于与去重清单、v3 摘录的逐字比较。"""
    return re.sub(r"\s+", "", text)


def _split_sentences(text: str) -> list[str]:
    """按句号类标点切句，标点保留在句尾；无标点的文本整体为一句。"""
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return [p for p in (p.strip() for p in parts) if p]


def _split_commas(text: str) -> list[str]:
    """按逗号类标点切分（递归切分的次级边界），标点保留在段尾。"""
    parts: list[str] = []
    start = 0
    for match in _COMMA_RE.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return [p for p in parts if p]


def _pack_units(units: list[str], max_chars: int) -> list[str]:
    """把有序单元按上限装进块：优先整单元装入，单单元超限则按字符硬切。

    硬切起点按单元内逗号类边界对齐（取不超过起点的最近逗号），减少断词。
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for unit in units:
        unit_len = len(unit)
        if buf_len + unit_len <= max_chars:
            buf.append(unit)
            buf_len += unit_len
            continue
        if buf:
            chunks.append("".join(buf))
            buf, buf_len = [], 0
        if unit_len <= max_chars:
            buf, buf_len = [unit], unit_len
            continue
        # 单单元超限：硬切
        start = 0
        while start < unit_len:
            end = min(start + max_chars, unit_len)
            chunks.append(unit[start:end])
            start = end
    if buf:
        chunks.append("".join(buf))
    return chunks


def _recursive_split(text: str, max_chars: int = RECURSIVE_MAX_CHARS) -> list[str]:
    """递归切分：按段落 → 句号类 → 逗号类的优先级，目标不超过 max_chars。

    输入在这里是单个元素或拼接后的缓冲文本；段落优先级的递归在
    ``_recursive_section_chunks`` 的打包里体现（先整段装入），本函数处理
    段落内部的句子 → 逗号两级。
    """
    if len(text) <= max_chars:
        return [text]
    sentences = _split_sentences(text)
    if len(sentences) > 1:
        # 句子级：整句装入，单句超限再往下递归
        out: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for sentence in sentences:
            if len(sentence) > max_chars:
                if buf:
                    out.append("".join(buf))
                    buf, buf_len = [], 0
                out.extend(_recursive_split_by_comma(sentence, max_chars))
            elif buf_len + len(sentence) <= max_chars:
                buf.append(sentence)
                buf_len += len(sentence)
            else:
                out.append("".join(buf))
                buf, buf_len = [sentence], len(sentence)
        if buf:
            out.append("".join(buf))
        return out
    return _recursive_split_by_comma(text, max_chars)


def _recursive_split_by_comma(text: str, max_chars: int) -> list[str]:
    """逗号类层级：整段装入，单段超限按字符硬切。"""
    if len(text) <= max_chars:
        return [text]
    segments = _split_commas(text)
    if len(segments) <= 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for segment in segments:
        if len(segment) > max_chars:
            if buf:
                out.append("".join(buf))
                buf, buf_len = [], 0
            out.extend(segment[i:i + max_chars] for i in range(0, len(segment), max_chars))
        elif buf_len + len(segment) <= max_chars:
            buf.append(segment)
            buf_len += len(segment)
        else:
            out.append("".join(buf))
            buf, buf_len = [segment], len(segment)
    if buf:
        out.append("".join(buf))
    return out


def _add_overlap(chunks: list[str], overlap_chars: int = RECURSIVE_OVERLAP_CHARS) -> list[str]:
    """给相邻块加字符级重叠：第 i 块前缀 = 第 i-1 块末尾的 overlap_chars 个字符。

    字符级重叠对中文稳妥（不引入跨块断句的歧义），重叠不计入长度上限；
    首块不加前缀。块数 ≤1 时原样返回。
    """
    if len(chunks) <= 1 or overlap_chars <= 0:
        return chunks
    out = [chunks[0]]
    # chunks[1:] 比 chunks 短一个，逐对（前块, 后块）给后块加前缀，不能用 strict
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        prefix = prev[-overlap_chars:]
        out.append(prefix + cur)
    return out


# ---- 章节内的内容单元 ----


@dataclass(frozen=True)
class _Unit:
    """章节内按出现顺序的内容单元：正文段落、列表元素、表格行、表格本身。"""

    kind: str  # paragraph / list / table_row / table
    text: str
    element: Element
    row_index: int | None = None  # 表格的第几行（仅 table_row）


def _section_units(elements: list[Element]) -> list[_Unit]:
    units: list[_Unit] = []
    for element in elements:
        if element.kind == "paragraph":
            text = element.text.strip()
            if text and not _CONTROL_LINE_RE.match(text):
                units.append(_Unit("paragraph", text, element))
        elif element.kind == "list":
            items = [item.strip() for item in element.items
                     if item.strip() and not _CONTROL_LINE_RE.match(item.strip())]
            if items:
                units.append(_Unit("list", "", element))
        elif element.kind == "table":
            if element.rows:
                for idx, row in enumerate(element.rows):
                    text = serialize_table_row(element.header, row)
                    if text:
                        units.append(_Unit("table_row", text, element, idx))
            elif element.text.strip():  # 整表摘要块（XLSX/CSV 无逐行 rows）
                units.append(_Unit("table", element.text.strip(), element))
    return units


def _unit_texts(unit: _Unit) -> list[str]:
    """单元参与正文打包的文本：列表按条目拆开，其余为单文本。"""
    if unit.kind == "list":
        return [item.strip() for item in unit.element.items if item.strip()]
    return [unit.text]


# ---- 分块结果 ----


@dataclass
class ChunkOutput:
    """一种策略的分块产物：检索块（children）与上下文块（parents）。"""

    children: list[KnowledgeChunk] = field(default_factory=list)
    parents: list[KnowledgeChunk] = field(default_factory=list)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_metadata(document: Document, section_path: list[str], strategy: str,
                   role: str) -> dict[str, Any]:
    meta = document.metadata
    title = meta.title if meta.title and meta.title != "None" else document.doc_id
    return {
        "doc_id": document.doc_id,
        "title": title,
        "source_url": meta.source_url,
        "section_path": section_path,
        "strategy": strategy,
        "role": role,
        "source_type": meta.source_type,
        "published_at": meta.published_at,
        "accessed_at": meta.accessed_at,
        "period": meta.period,
        "license": meta.license,
        "language": meta.language,
        "topic": meta.topic,
        "content_hash": meta.content_hash,
    }


def _make_chunk(document: Document, section_index: int, code: str, seq: int,
                text: str, section_path: list[str], strategy: str, role: str,
                extra_metadata: dict[str, Any] | None = None) -> KnowledgeChunk:
    metadata = _base_metadata(document, section_path, strategy, role)
    metadata["chunk_hash"] = _hash(text)
    if extra_metadata:
        metadata.update(extra_metadata)
    return KnowledgeChunk(
        chunk_id=f"{document.doc_id}:{section_index}:{code}:{seq}",
        source=metadata["title"],
        source_id=document.doc_id,
        text=text,
        evidence_level=document.metadata.evidence_level or "level_5_other",
        metadata=metadata,
    )


def chunk_document(document: Document, strategy: str,
                   dropped_paragraphs: dict[str, list[str]] | None = None) -> ChunkOutput:
    """对单篇文档分块。dropped_paragraphs 为 M5 段落级去重的剔除清单。"""
    _handled_tables.clear()
    dropped = {_squash(t) for t in (dropped_paragraphs or {}).get(document.doc_id, [])}
    paths = document.iter_section_paths()
    if strategy == "recursive":
        return _chunk_recursive(document, paths, dropped)
    if strategy == "parent_child":
        return _chunk_parent_child(document, paths, dropped)
    raise ValueError(f"未知分块策略：{strategy}")


def _chunk_recursive(document: Document,
                     paths: list[tuple[Any, list[str]]],
                     dropped: set[str]) -> ChunkOutput:
    """递归切分基线：512 字、重叠约 15%，不跨章节；表格行单独成块。"""
    out = ChunkOutput()
    for section_index, (section, path) in enumerate(paths):
        seq = 0

        def emit(text: str, content_kind: str,
                 _si: int = section_index, _path: list[str] = path) -> None:
            nonlocal seq
            out.children.append(_make_chunk(
                document, _si, "rc", seq, text, _path,
                strategy="recursive", role="child",
                extra_metadata={"content_kind": content_kind}))
            seq += 1

        buffer: list[str] = []  # 跨元素正文缓冲（不超过上限时合并相邻小元素）
        buffer_len = 0

        def flush() -> None:
            nonlocal buffer, buffer_len
            if not buffer:
                return
            pieces = _add_overlap(_recursive_split("".join(buffer)))
            for piece in pieces:
                emit(piece, "text")
            buffer, buffer_len = [], 0

        for unit in _section_units(section.elements):
            if unit.kind == "table_row":
                flush()
                # 表格行单独成块、不与正文混切；个别 PDF 把整段正文误抽成单行表格，
                # 超长行按同一递归规则切分（仍是独立块，只是内容被拆开）
                for piece in (_recursive_split(unit.text)
                              if len(unit.text) > RECURSIVE_MAX_CHARS else [unit.text]):
                    emit(piece, "table_row")
                continue
            if unit.kind == "table":
                flush()
                for piece in (_recursive_split(unit.text)
                              if len(unit.text) > RECURSIVE_MAX_CHARS else [unit.text]):
                    emit(piece, "table_summary")
                continue
            for text in _unit_texts(unit):
                if unit.kind == "paragraph" and _squash(text) in dropped:
                    continue
                if len(text) > RECURSIVE_MAX_CHARS:
                    # 超长按句号 → 逗号递归切；不与缓冲合并（保持句子边界优先），块间仍加重叠
                    flush()
                    for piece in _add_overlap(_recursive_split(text)):
                        emit(piece, "text")
                elif buffer_len + len(text) > RECURSIVE_MAX_CHARS:
                    flush()
                    buffer, buffer_len = [text], len(text)
                else:
                    buffer.append(text)
                    buffer_len += len(text)
        flush()
    return out


def _group_child_sentences(sentences: list[str], max_chars: int = CHILD_MAX_CHARS) -> list[str]:
    """把句子聚成子块：每块 1–2 句，不超过 max_chars。

    单句超限时按逗号类边界硬切（取不超过上限的最近逗号，否则字符切）。
    不足 ``CHILD_MIN_CHARS`` 的短碎片与相邻句合并（合并上限放宽到
    ``CHILD_MERGE_MAX_CHARS``，只用于合并、不用于硬切）；纯日期/纯控件文字
    直接丢弃（不进检索）。丢弃与合并不改变句子顺序，保证 chunk_id 幂等。
    """
    children: list[str] = []
    i = 0
    while i < len(sentences):
        first = sentences[i]
        if len(first) > max_chars:
            children.extend(_hard_split_sentence(first, max_chars))
            i += 1
            continue
        if (i + 1 < len(sentences)
                and len(first) + len(sentences[i + 1]) <= max_chars):
            children.append(first + sentences[i + 1])
            i += 2
        else:
            children.append(first)
            i += 1
    # 不足 15 字的短碎片与相邻子块合并（合并上限放宽到 130 字，只用于合并）；
    # 纯日期/纯控件文字丢弃；合并不了的孤立短碎片保留（短句正文不静默丢弃）
    children = [c for c in _merge_short_children(children)
                if not _CONTROL_LINE_RE.match(c.strip())]
    return children


def _merge_short_children(children: list[str]) -> list[str]:
    """把不足 ``CHILD_MIN_CHARS`` 的子块并入前一个子块；合并不了的孤立短碎片保留。

    只在左邻本身是短/中等块时合并：合并后不超过 ``CHILD_MERGE_MAX_CHARS``
    且左邻短于 ``CHILD_MAX_CHARS``（硬切好的超限单句尾巴不并回，避免把
    刚按 110 字上限切开的单句重新并成超限块）。合并不了的孤立短碎片保留。
    """
    out: list[str] = []
    for child in children:
        if (len(child) < CHILD_MIN_CHARS and out
                and len(out[-1]) < CHILD_MAX_CHARS
                and len(out[-1]) + len(child) <= CHILD_MERGE_MAX_CHARS):
            out[-1] = out[-1] + child
            continue
        out.append(child)
    return out


def _pack_by_limit(items: list[str], max_chars: int) -> list[list[str]]:
    """把有序条目按上限分组成多组，整组拼接不超过 max_chars（单条目超限也自成一组）。"""
    groups: list[list[str]] = []
    buf: list[str] = []
    buf_len = 0
    for item in items:
        if buf and buf_len + len(item) > max_chars:
            groups.append(buf)
            buf, buf_len = [], 0
        buf.append(item)
        buf_len += len(item)
    if buf:
        groups.append(buf)
    return groups


def _hard_split_sentence(sentence: str, max_chars: int) -> list[str]:
    """超限单句：优先在逗号类边界断开，断不开则字符硬切。"""
    out: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind(p) for p in ("，", "、", "：", ",", "；", ";"))
        if cut < max_chars // 3:  # 太靠前的逗号会造成大量碎块，宁可硬切
            cut = max_chars - 1
        out.append(rest[:cut + 1])
        rest = rest[cut + 1:]
    if rest:
        out.append(rest)
    return out


def _chunk_parent_child(document: Document,
                        paths: list[tuple[Any, list[str]]],
                        dropped: set[str]) -> ChunkOutput:
    """父子分块：子块 1–2 句（≤110 字）进索引；父块（≤600 字）只作上下文。

    父块为同一章节内的段落（整段一个父块）；超过 600 字的段落按句切成
    多个父块，子块归属于覆盖它的那个父块。列表元素整体作为一个父块
    （条目是并列事实，合并后也不超过父块上限的常见长度）；列表条目本身
    作为句子参与子块切分。表格行固定作为子块，整表摘要作为父块。
    """
    out = ChunkOutput()
    for section_index, (section, path) in enumerate(paths):
        child_seq = 0
        parent_seq = 0

        def emit_parent(text: str, content_kind: str,
                        _si: int = section_index, _path: list[str] = path) -> str:
            nonlocal parent_seq
            chunk = _make_chunk(
                document, _si, "pp", parent_seq, text, _path,
                strategy="parent_child", role="parent",
                extra_metadata={"content_kind": content_kind})
            parent_seq += 1
            out.parents.append(chunk)
            return chunk.chunk_id

        def emit_child(text: str, parent_id: str, content_kind: str,
                       _si: int = section_index, _path: list[str] = path) -> None:
            nonlocal child_seq
            out.children.append(_make_chunk(
                document, _si, "pc", child_seq, text, _path,
                strategy="parent_child", role="child",
                extra_metadata={"parent_id": parent_id, "content_kind": content_kind}))
            child_seq += 1

        for unit in _section_units(section.elements):
            if unit.kind == "table_row":
                # 同一表格的多个行单元只在首行处理一次：一个摘要父块 + 每行一个子块
                if id(unit.element) in _handled_tables:
                    continue
                _handled_tables.add(id(unit.element))
                parent_id = emit_parent(_table_summary_text(unit.element), "table_summary")
                for row in unit.element.rows:
                    text = serialize_table_row(unit.element.header, row)
                    if text:
                        emit_child(text, parent_id, "table_row")
                continue
            if unit.kind == "table":
                parent_id = emit_parent(unit.text, "table_summary")
                # 摘要本身也作为子块可检索（整表概览类问题）
                emit_child(unit.text, parent_id, "table_summary")
                continue
            if unit.kind == "list":
                items = _unit_texts(unit)
                # 父块尽量含整个列表；超过 600 字时按条目切成多个父块，
                # 子块归属于覆盖它的那个父块（参考文献类长列表常见）
                for group in _pack_by_limit(items, PARENT_MAX_CHARS):
                    parent_id = emit_parent("".join(group), "list")
                    for child in _group_child_sentences(group):
                        emit_child(child, parent_id, "list")
                continue
            # 正文段落：段落级去重剔除的跳过
            if _squash(unit.text) in dropped:
                continue
            _emit_paragraph(unit.text, emit_parent, emit_child)
    return out


def _table_summary_text(element: Element) -> str:
    """HTML/PDF 内嵌表格的整表摘要文本（父块）：表头与行列规模。"""
    header = "、".join(h for h in element.header if h)
    summary = f"表格共 {len(element.rows)} 行"
    if header:
        summary += f"，列：{header}"
    if element.text and element.metadata.get("content_kind") == "table_summary":
        return element.text  # XLSX/CSV 的摘要文本已有更完整描述
    return summary


# 父子分块里同一表格的多个行单元只在首行处理一次，用元素对象 id 标记；
# 按 corpus 分块时清空（单进程内对象 id 唯一，跨调用不复用）
_handled_tables: set[int] = set()


def _emit_paragraph(text: str, emit_parent: Any, emit_child: Any) -> None:
    """一个正文段落：父块为段落本身（≤600 字），超长段落按句切成多个父块。"""
    sentences = _split_sentences(text)
    if len(text) <= PARENT_MAX_CHARS:
        parent_id = emit_parent(text, "text")
        for child in _group_child_sentences(sentences):
            emit_child(child, parent_id, "text")
        return
    # 超长段落：按句装进 ≤600 字的多个父块，子块归属覆盖它的父块；
    # 单句本身超过 600 字时按逗号/字符硬切到 ≤600 的父块
    parent_groups: list[list[str]] = []
    buf: list[str] = []
    buf_len = 0
    for sentence in sentences:
        if len(sentence) > PARENT_MAX_CHARS:
            if buf:
                parent_groups.append(buf)
                buf, buf_len = [], 0
            for piece in _hard_split_sentence(sentence, PARENT_MAX_CHARS):
                parent_groups.append([piece])
            continue
        if buf and buf_len + len(sentence) > PARENT_MAX_CHARS:
            parent_groups.append(buf)
            buf, buf_len = [], 0
        buf.append(sentence)
        buf_len += len(sentence)
    if buf:
        parent_groups.append(buf)
    for group in parent_groups:
        parent_id = emit_parent("".join(group), "text")
        for child in _group_child_sentences(group):
            emit_child(child, parent_id, "text")


def chunk_corpus(documents: list[Document], strategy: str,
                 dropped_paragraphs: dict[str, list[str]] | None = None) -> ChunkOutput:
    """对整批文档分块；文档顺序即输出顺序，保证幂等。"""
    _handled_tables.clear()
    out = ChunkOutput()
    for document in documents:
        result = chunk_document(document, strategy, dropped_paragraphs)
        out.children.extend(result.children)
        out.parents.extend(result.parents)
    return out


# ---- 编排：读解析产物 → 分块 → 写 chunks/<strategy>/ ----


@dataclass
class ChunkReport:
    """分块运行报告。"""

    strategy: str
    document_count: int = 0
    child_count: int = 0
    parent_count: int = 0
    child_chars_mean: float = 0.0
    parent_chars_mean: float = 0.0
    table_row_chunks: int = 0
    table_summary_parents: int = 0
    dropped_paragraph_count: int = 0


def load_parsed_documents(parsed_path: Path) -> list[Document]:
    """读取 M5 的 documents.jsonl。"""
    from medidiag.rag.ingest.document import Document as _Doc

    documents: list[_Doc] = []
    with parsed_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                documents.append(_Doc.from_json(line))
    return documents


def run_chunking(parsed_path: Path, strategy: str, out_dir: Path) -> ChunkReport:
    """执行分块并写产物：``chunks.jsonl``（检索块）与 ``parents.jsonl``（父子策略）。

    段落级去重清单通过在解析文档上重跑 ``dedup.dedup_paragraphs`` 重建
    （确定性算法），跳过被判剔除的一方。
    """
    import statistics

    from medidiag.rag.ingest import dedup
    from medidiag.schemas import chunk_to_dict, write_jsonl

    documents = load_parsed_documents(parsed_path)
    dropped = dedup.dedup_paragraphs(documents).dropped_paragraphs
    output = chunk_corpus(documents, strategy, dropped)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl((chunk_to_dict(c) for c in output.children), out_dir / "chunks.jsonl")
    if output.parents:
        write_jsonl((chunk_to_dict(c) for c in output.parents), out_dir / "parents.jsonl")

    child_lens = [len(c.text) for c in output.children]
    parent_lens = [len(c.text) for c in output.parents]
    return ChunkReport(
        strategy=strategy,
        document_count=len(documents),
        child_count=len(output.children),
        parent_count=len(output.parents),
        child_chars_mean=round(statistics.mean(child_lens), 1) if child_lens else 0.0,
        parent_chars_mean=round(statistics.mean(parent_lens), 1) if parent_lens else 0.0,
        table_row_chunks=sum(1 for c in output.children
                             if c.metadata.get("content_kind") == "table_row"),
        table_summary_parents=sum(1 for c in output.parents
                                  if c.metadata.get("content_kind") == "table_summary"),
        dropped_paragraph_count=sum(len(v) for v in dropped.values()),
    )
