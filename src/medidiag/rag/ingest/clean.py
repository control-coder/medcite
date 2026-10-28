"""清洗：统一全半角、空白与换行，去参考文献编号、残留链接文字、控件文字。

清洗规则按 source_type 配置（``CLEAN_RULES``），不硬编码在通用函数中；
通用函数只做全半角、空白与换行归一。表格行的序列化文本不做去参考文献处理，
避免误删表格里的正常数值。

2026-10-28 数据质量修复新增：
- ``\\r\\n``/``\\r`` 统一为 ``\\n``；
- 参考文献类章节（参考文献/参考资料/References/外部链接/延伸阅读/注释等）按来源整节丢弃；
- WHO 出版物 PDF 中无标题的参考文献列表（连续编号 + 年份/URL 的行）按规则丢弃；
- 页码类孤立行（罗马数字、纯数字、「第 X 页」）删除。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medidiag.rag.ingest.document import Document, Element


# 全角 → 半角仅限：数字（U+FF10–FF19）、大写字母（FF21–FF3A）、小写字母（FF41–FF5A）、
# 全角空格（U+3000）。中文全角标点（，：；？！（）【】、。％等）保持全角不变，
# 保证展示给用户的证据原文与来源一致（可溯源、逐字原文）。
def _full_alnum_space_to_half(text: str) -> str:
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if o == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


_WS_RUN = re.compile(r"[ \t　\xa0]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """通用归一：全角字母/数字/空格转半角、统一空白与换行；中文全角标点保持原样。

    - 不做整体 NFKC（它会把中文全角标点转成 ASCII 半角，导致证据原文与来源不一致）；
    - ``\\r\\n``/``\\r`` 统一为 ``\\n``（WHO 快照与个别政府页含 ``\\r``，否则
      引用校验与逐字比较会把换行差异当成内容差异）。
    所有来源都应用。
    """
    text = _full_alnum_space_to_half(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class CleanRules:
    """按来源配置的清洗规则。所有正则为可选，按出现顺序应用。"""

    # 参考文献编号：(1) （1） [1] 上标式数字
    strip_reference_markers: bool = True
    # 残留链接文字：「相关链接」「更多信息见」「点击这里」等
    link_residue_patterns: tuple[str, ...] = ()
    # 页面控件文字：「打印」「分享」「收藏」「字号：大中小」等整行
    control_line_patterns: tuple[str, ...] = ()
    # 行内噪声正则 → 替换为空串
    inline_noise_patterns: tuple[str, ...] = ()
    # 页码类孤立行：罗马数字（vi、xi）、纯数字、「第 X 页」，PDF 抽取残留的页码碎片
    drop_page_number_lines: bool = False
    # 参考文献类章节标题：命中即整节丢弃（参考文献/参考资料/外部链接/延伸阅读/注释等）
    drop_section_titles: tuple[str, ...] = ()
    # 无标题参考文献列表：连续 ≥ 该值的「编号 + 年份/URL」行按参考文献丢弃
    drop_untitled_reference_runs: int = 0


_REFERENCE_MARKER = re.compile(r"\s*[(（\[]\d{1,3}[)）\]](?=\s*[,，.。；;]|$)")
_SUPERSCRIPT_REF = re.compile(r"(?<=[一-鿿])\s*\[\d{1,2}(-\d{1,2})?\]")

# 页码类孤立行：罗马数字（WHO 出版物前言页码 vi、xi 等）、纯数字、「第 X 页」
_PAGE_NUMBER_LINE = re.compile(
    r"^(?:[ivxlcdm]{1,8}|\d{1,4}|第\s*\d{1,4}\s*页(?:[-—–]?\s*共\s*\d{1,4}\s*页)?)$", re.I)

# 参考文献类章节标题（按来源启用；标题命中即整节丢弃）
_REFERENCE_SECTION_TITLES = (
    "参考文献", "参考资料", "参考文献列表", "外部链接", "延伸阅读", "注释",
    "References", "Bibliography", "External links", "Further reading", "Notes",
)
_REFERENCE_SECTION_RE = re.compile(
    r"^(?:" + "|".join(re.escape(t) for t in _REFERENCE_SECTION_TITLES) + r")$", re.I)

# 无标题参考文献行：「1.」「(1)」编号开头，且含年份/URL/英文书目特征的书目条目。
# WHO 出版物与事实清单的参考文献列表常无「参考文献」标题，按连续行识别
# （连续 2 行以上才丢弃；正文里偶尔出现的单个编号引用不触发）。
# 条目按词边界截断判断（PDF 把长条目断成多行时，只有首行带编号）。
_REF_LINE_START = re.compile(r"^(?:\d{1,3}[.、．]|[(（]\d{1,3}[)）])\s*\S")
_REF_LINE_HINT = re.compile(
    r"(?:\d{4}\s*年|https?://|www\.|ISBN\s*\d|doi\s*[:.]|et\s+al\.|"
    r"Lancet|BMJ|Geneva|Washington|MMWR|PLoS|Cochrane)", re.I)
# 参考文献条目前 12 个词内是否出现书目特征（用于截断条目行）
_REF_HEAD_WORDS = 12


# WHO 页首杂质：跳转链接、「实况报道」栏目标签、页面日期行（YYYY年M月D日 整行）。
# 日期行已由解析阶段提取为 metadata.published_at，清洗时按控件文字从正文删除。
_WHO_CONTROL_LINES = (
    r"^打印本页?$", r"^分享(本页|到)?$", r"^收藏本页$", r"^字号[:：].*$",
    r"^阅读时间.*$", r"^返回顶部$", r"^Skip to main content$",
    r"^实况报道$", r"^\d{4}年\d{1,2}月\d{1,2}日$",
)

_WHO_RULES = CleanRules(
    link_residue_patterns=(
        r"相关链接[:：]?.*$", r"更多信息[见请].*$", r"点击[这此]处.*$",
    ),
    control_line_patterns=_WHO_CONTROL_LINES,
    # WHO 事实清单的「参考文献/参考资料」节是书目列表，不进检索；
    # 正文段落不受影响（v3 摘录所在的正文保留，见 examples/public_health_v4/README.md）
    drop_section_titles=_REFERENCE_SECTION_TITLES,
    # 事实清单常把参考文献列表塞在正文末尾章节（无标题、英文书目），按连续行丢弃
    drop_untitled_reference_runs=2,
)

_WHO_PUBLICATION_RULES = CleanRules(
    link_residue_patterns=(
        r"相关链接[:：]?.*$", r"更多信息[见请].*$", r"点击[这此]处.*$",
    ),
    control_line_patterns=_WHO_CONTROL_LINES,
    # 出版物前言页码（vi、xi 等罗马数字孤立行）
    drop_page_number_lines=True,
    drop_section_titles=_REFERENCE_SECTION_TITLES,
    # 无标题参考文献列表：连续 3 行以上「编号 + 年份/URL」的段落
    drop_untitled_reference_runs=3,
)

_WIKIPEDIA_RULES = CleanRules(
    link_residue_patterns=(
        r"^跳到导航.*$", r"^跳到搜索.*$", r"\[编辑\]",
    ),
    control_line_patterns=(
        r"^\[编辑\]$", r"^编辑$", r"^本条目.*$", r"^此条目.*$",
    ),
    inline_noise_patterns=(
        r"\[\d{1,3}\]",  # 维基行内参考文献编号
    ),
    # 维基的参考文献/外部链接/延伸阅读/注释节是书目与站外链接，不进检索
    drop_section_titles=_REFERENCE_SECTION_TITLES,
)

_NHC_RULES = CleanRules(
    control_line_patterns=(
        r"^打印.*$", r"^分享到.*$", r"^字号[:：].*$", r"^来源[:：].*$",
        r"^发布时间[:：].*$", r"^分享本页.*$",
    ),
    drop_section_titles=_REFERENCE_SECTION_TITLES,
)

_DEFAULT_RULES = CleanRules()

CLEAN_RULES: dict[str, CleanRules] = {
    "who_factsheet": _WHO_RULES,
    "who_publication": _WHO_PUBLICATION_RULES,
    "nhc_document": _NHC_RULES,
    "epidemic_report": _NHC_RULES,
    "wikipedia": _WIKIPEDIA_RULES,
}


def _is_reference_line(text: str) -> bool:
    """该行是否像无标题参考文献条目（编号开头且前几个词内含书目特征）。

    书目特征只在条目前 ``_REF_HEAD_WORDS`` 个词内判定：PDF 把长条目断行后
    只有首行带编号，截断条目按前几个词的人名-期刊模式识别；正文里偶尔
    出现的单个编号引用仍不触发（需连续 ≥ min_run）。
    """
    stripped = text.strip()
    if not _REF_LINE_START.match(stripped):
        return False
    head = " ".join(stripped.split()[:_REF_HEAD_WORDS])
    return bool(_REF_LINE_HINT.search(head))


def _drop_reference_runs_texts(texts: list[str], min_run: int) -> list[str]:
    """从有序文本中丢弃连续 ≥ min_run 的参考文献行；不足 min_run 的保留。"""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) < min_run:
            out.extend(run)
        run.clear()

    for text in texts:
        if _is_reference_line(text):
            run.append(text)
        else:
            flush()
            out.append(text)
    flush()
    return out


def _drop_untitled_reference_runs(elements: list[Element], min_run: int) -> list[Element]:
    """丢弃连续 ≥ min_run 的无标题参考文献行（段落元素，列表元素按条目处理）。

    连续段中间夹杂的非匹配元素重置计数；整个连续段长度达到阈值时整段丢弃。
    列表元素逐条目判断（参考文献常抽成列表）；表格不处理（避免误删数据行）。
    """
    if min_run <= 0:
        return elements
    out: list[Element] = []
    run: list[Element] = []

    def flush() -> None:
        if len(run) < min_run:
            out.extend(run)
        run.clear()

    for element in elements:
        if element.kind == "table":
            flush()
            out.append(element)
        elif element.kind == "list":
            items = _drop_reference_runs_texts(element.items, min_run)
            if not items:
                continue
            element.items = items
            element.text = "\n".join(items)
            flush()
            out.append(element)
        elif _is_reference_line(element.text):
            run.append(element)
        else:
            flush()
            out.append(element)
    flush()
    return out


def _clean_line(line: str, rules: CleanRules) -> str:
    """清洗单行文本；返回空串表示整行丢弃。"""
    text = normalize_text(line)
    if not text:
        return ""
    for pattern in rules.control_line_patterns:
        if re.match(pattern, text):
            return ""
    if rules.drop_page_number_lines and _PAGE_NUMBER_LINE.match(text):
        return ""
    for pattern in rules.link_residue_patterns:
        text = re.sub(pattern, "", text).strip()
    for pattern in rules.inline_noise_patterns:
        text = re.sub(pattern, "", text).strip()
    if rules.strip_reference_markers:
        text = _REFERENCE_MARKER.sub("", text)
        text = _SUPERSCRIPT_REF.sub("", text)
    return text.strip()


def _clean_element(element: Element, rules: CleanRules) -> Element | None:
    """清洗单个元素；整元素被清空时返回 None。表格元素不清洗内容。"""
    if element.kind == "table":
        return element
    if element.kind == "list":
        items = [t for t in (_clean_line(i, rules) for i in element.items) if t]
        if not items:
            return None
        element.items = items
        element.text = "\n".join(items)
        return element
    # 段落逐行清洗：多行段落（如含 \\r 的页首杂质段）删除其中的控件行，
    # 只保留正文行；全部是控件行时整段丢弃
    lines = [t for t in (_clean_line(ln, rules) for ln in element.text.split("\n")) if t]
    if not lines:
        return None
    element.text = "\n".join(lines)
    return element


def clean_document(document: Document, source_type: str | None = None) -> Document:
    """按 source_type 清洗整个文档；未知来源用默认规则。"""
    stype = source_type or document.metadata.source_type
    rules = CLEAN_RULES.get(stype, _DEFAULT_RULES)
    drop_titles = frozenset(t.lower() for t in rules.drop_section_titles)
    kept_sections = []
    for section in document.sections:
        # 参考文献类章节整节丢弃（按来源配置；正文章节不受影响）
        if drop_titles and section.title.strip().lower() in drop_titles:
            continue
        cleaned: list[Element] = []
        for element in section.elements:
            result = _clean_element(element, rules)
            if result is not None:
                cleaned.append(result)
        if rules.drop_untitled_reference_runs:
            cleaned = _drop_untitled_reference_runs(cleaned, rules.drop_untitled_reference_runs)
        section.elements = cleaned
        kept_sections.append(section)
    # 清洗后标题也过一遍控件文字（不常见，但政府站有「打印」误入标题的情况）
    document.sections = [s for s in kept_sections if s.title or s.elements]
    return document
