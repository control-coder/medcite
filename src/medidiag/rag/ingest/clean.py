"""清洗：统一全半角与空白，去参考文献编号、残留链接文字、控件文字。

清洗规则按 source_type 配置（``CLEAN_RULES``），不硬编码在通用函数中；
通用函数只做全半角与空白归一。表格行的序列化文本不做去参考文献处理，
避免误删表格里的正常数值。
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
    """通用归一：全角字母/数字/空格转半角、统一空白；中文全角标点保持原样。

    不做整体 NFKC（它会把中文全角标点转成 ASCII 半角，导致证据原文与来源不一致）。
    所有来源都应用。
    """
    text = _full_alnum_space_to_half(text)
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


_REFERENCE_MARKER = re.compile(r"\s*[(（\[]\d{1,3}[)）\]](?=\s*[,，.。；;]|$)")
_SUPERSCRIPT_REF = re.compile(r"(?<=[\u4e00-\u9fff])\s*\[\d{1,2}(-\d{1,2})?\]")

_WHO_RULES = CleanRules(
    link_residue_patterns=(
        r"相关链接[:：]?.*$", r"更多信息[见请].*$", r"点击[这此]处.*$",
    ),
    control_line_patterns=(
        r"^打印本页?$", r"^分享(本页|到)?$", r"^收藏本页$", r"^字号[:：].*$",
        r"^阅读时间.*$", r"^返回顶部$",
    ),
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
)

_NHC_RULES = CleanRules(
    control_line_patterns=(
        r"^打印.*$", r"^分享到.*$", r"^字号[:：].*$", r"^来源[:：].*$",
        r"^发布时间[:：].*$", r"^分享本页.*$",
    ),
)

_DEFAULT_RULES = CleanRules()

CLEAN_RULES: dict[str, CleanRules] = {
    "who_factsheet": _WHO_RULES,
    "who_publication": _WHO_RULES,
    "nhc_document": _NHC_RULES,
    "epidemic_report": _NHC_RULES,
    "wikipedia": _WIKIPEDIA_RULES,
}


def _clean_line(line: str, rules: CleanRules) -> str:
    """清洗单行文本；返回空串表示整行丢弃。"""
    text = normalize_text(line)
    if not text:
        return ""
    for pattern in rules.control_line_patterns:
        if re.match(pattern, text):
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
    text = _clean_line(element.text, rules)
    if not text:
        return None
    element.text = text
    return element


def clean_document(document: Document, source_type: str | None = None) -> Document:
    """按 source_type 清洗整个文档；未知来源用默认规则。"""
    stype = source_type or document.metadata.source_type
    rules = CLEAN_RULES.get(stype, _DEFAULT_RULES)
    for section in document.sections:
        cleaned: list[Element] = []
        for element in section.elements:
            result = _clean_element(element, rules)
            if result is not None:
                cleaned.append(result)
        section.elements = cleaned
    # 清洗后标题也过一遍控件文字（不常见，但政府站有「打印」误入标题的情况）
    document.sections = [s for s in document.sections if s.title or s.elements]
    return document
