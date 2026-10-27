"""HTML 解析器：把网页正文解析为统一文档模型。

迁移并改进 scripts/public_corpus.py 与 scripts/extend_corpus.py 的自研抽取逻辑：
- 去掉导航、页脚、脚本、分享控件等噪声（跳过规则按来源配置）；
- 按 h1–h4 建章节层级；
- ``<table>`` 转为表格元素；
- ``<ul>/<ol>`` 转为列表元素。

维基百科（MediaWiki API 返回的正文 HTML）走同一个解析器，只是换一套
跳过规则（编辑链接、参考文献、导航框、信息框等），见 ``SOURCE_RULES``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from medidiag.rag.ingest.document import (
    Document,
    DocumentMetadata,
    Element,
    Section,
    serialize_table_row,
)

# 任何来源都跳过的标签
_BASE_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button", "iframe"}
)

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}

# 出现这些标签时隐式闭合未闭合的 <p>（HTML5 规范：p 不能包含块级元素）
_P_CLOSING_TAGS = frozenset(
    {"p", "div", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
     "section", "article", "blockquote", "pre", "hr", "form"}
)


@dataclass(frozen=True)
class HtmlRules:
    """按来源配置的解析规则。"""

    skip_classes: frozenset[str] = frozenset()  # class 含任一值即跳过整块
    skip_ids: frozenset[str] = frozenset()  # id 含任一值即跳过整块
    skip_placeholder_labels: frozenset[str] = frozenset()  # data-placeholder-label 匹配即跳过
    drop_line_patterns: tuple[str, ...] = ()  # 抽取后行文本命中即丢弃（控件文字等）
    content_ids: frozenset[str] = frozenset()  # 非空时只取这些 id 的容器作为正文
    content_classes: frozenset[str] = frozenset()  # 非空时只取含这些 class 的容器


# 世界卫生组织实况报道：正文在 article 里，toolbox/面包屑/语言选择器是噪声
_WHO_RULES = HtmlRules(
    skip_classes=frozenset(
        {
            "social-share", "addthis", "language-selector", "breadcrumb",
            "readTime", "cookie", "related-items", "sf-comments", "tags", "page-tools",
        }
    ),
    skip_ids=frozenset({"readTime", "language-selector-item-details"}),
    skip_placeholder_labels=frozenset({"Toolbox", "Images"}),
    drop_line_patterns=(
        r"^阅读时间$", r"^打印$", r"^分享$", r"^分享本页", r"^Skip to main content",
        r"^世卫组织\s*/\s*媒体中心", r"^媒体中心\s*/",
    ),
)

# 维基百科（MediaWiki action=parse 的正文 HTML）
_WIKIPEDIA_RULES = HtmlRules(
    skip_classes=frozenset(
        {
            "mw-editsection", "reference", "navbox", "infobox", "metadata", "ambox",
            "hatnote", "mw-references-wrap", "refbegin", "noprint", "mw-jump-link",
            "sister-box", "sidebar", "vertical-navbox", "thumb", "gallery",
        }
    ),
    skip_ids=frozenset({"toc", "References", "参考文献", "External_links"}),
    drop_line_patterns=(
        r"^\[编辑\]$", r"^编辑$", r"^\[\d+\]$", r"^跳到导航$", r"^跳到搜索$",
    ),
)

# 疫情月报（中国疾控中心）：正文在 <div class="content" id="articleCon">，
# 页面其余部分是主导航、相关新闻与页脚，直接限定正文容器
_EPIDEMIC_RULES = HtmlRules(
    content_ids=frozenset({"articleCon"}),
    content_classes=frozenset(),
    drop_line_patterns=(
        r"^字体[:：].*$", r"^时间[:：].*$", r"^文件附件[:：]?$", r"^相关新闻$",
        r"^分享.*$", r"^打印.*$",
    ),
)

# 国家文件（中国政府网/卫健委）：正文在 <div class="pages_content" id="UCAP-CONTENT">
_NHC_RULES_HTML = HtmlRules(
    content_ids=frozenset({"UCAP-CONTENT"}),
    content_classes=frozenset({"pages_content"}),
    drop_line_patterns=(
        r"^字号[:：].*$", r"^打印$", r"^收藏$", r"^分享.*$", r"^扫一扫.*$",
        r"^发布时间[:：].*$", r"^来源[:：].*$",
    ),
)

_DEFAULT_RULES = HtmlRules()

# 按 source_type 选择规则；未知来源用默认规则
SOURCE_RULES: dict[str, HtmlRules] = {
    "who_factsheet": _WHO_RULES,
    "who_publication": _WHO_RULES,
    "nhc_document": _NHC_RULES_HTML,
    "epidemic_report": _EPIDEMIC_RULES,
    "wikipedia": _WIKIPEDIA_RULES,
}


@dataclass
class _Node:
    """HTML 节点的轻量表示，只保留提取正文所需的结构。

    children 按文档顺序混合存放子元素与文本片段（``_Text``），保证遍历
    顺序与原文一致。
    """

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[_Node | _Text] = field(default_factory=list)
    skipped: bool = False

    def append_text(self, data: str) -> None:
        if self.children and isinstance(self.children[-1], _Text):
            self.children[-1].data += data
        else:
            self.children.append(_Text(data))

    @property
    def text_parts(self) -> list[str]:
        """该节点直属的文本片段（不含子元素的文本）。"""
        return [c.data for c in self.children if isinstance(c, _Text)]

    @property
    def element_children(self) -> list[_Node]:
        return [c for c in self.children if isinstance(c, _Node)]


@dataclass
class _Text:
    """节点直属的一段文本，保持在 children 序列中的位置。"""

    data: str


class _TreeBuilder(HTMLParser):
    """把 HTML 解析成 _Node 树，同时按规则标记跳过的子树。"""

    def __init__(self, rules: HtmlRules) -> None:
        super().__init__(convert_charrefs=True)
        self.rules = rules
        self.root = _Node("__root__")
        self._stack: list[_Node] = [self.root]
        self._skip_depth = 0

    def _should_skip(self, tag: str, attrs: dict[str, str]) -> bool:
        if tag in _BASE_SKIP_TAGS:
            return True
        classes = set(attrs.get("class", "").split())
        node_id = attrs.get("id", "")
        if classes & self.rules.skip_classes:
            return True
        if attrs.get("data-placeholder-label", "") in self.rules.skip_placeholder_labels:
            return True
        # 较长的 skip 词允许子串匹配（如 addthis_inline_share_toolbox）；
        # 短词只做完整匹配，避免 "share" 误伤 "shared" 之类的 class
        if any(
            len(skip) >= 5 and any(skip in cls for cls in classes)
            for skip in self.rules.skip_classes
        ):
            return True
        return any(skip and skip in node_id for skip in self.rules.skip_ids)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        if self._skip_depth:
            self._skip_depth += 1
            return
        if self._should_skip(tag, attr_map):
            self._skip_depth = 1
            return
        # HTML 容错：<p> 不能包含块级元素；遇到时按规范隐式闭合未闭合的 p
        if tag in _P_CLOSING_TAGS and self._stack[-1].tag == "p":
            self._stack.pop()
        node = _Node(tag=tag, attrs=attr_map)
        self._stack[-1].children.append(node)
        # void 元素不进栈
        if tag not in {"br", "hr", "img", "meta", "link", "input", "col", "wbr", "source"}:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            return
        attr_map = {k: (v or "") for k, v in attrs}
        if self._should_skip(tag, attr_map):
            return
        node = _Node(tag=tag, attrs=attr_map)
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return
        # 允许标签不闭合（HTML 容错）：找到最近匹配的就弹栈
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._stack[-1].append_text(data)


_WS = re.compile(r"[ \t　\xa0]+")


def _collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _node_text(node: _Node) -> str:
    """递归取节点全部文本（用于标题、段落、单元格），保持文档顺序。"""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, _Text):
            parts.append(child.data)
        else:
            parts.append(_node_text(child))
    return _collapse_ws("".join(parts))


def _extract_table(node: _Node) -> Element:
    """把 <table> 节点转成表格元素。

    表头识别：thead/th 优先；都没有时把第一个非空行当表头（政府统计表
    与维基常见模式），行序列化时以它为列名。
    """
    header: list[str] = []
    rows: list[list[str]] = []
    for child in node.element_children:
        if child.tag == "thead":
            for tr in child.element_children:
                if tr.tag == "tr":
                    header = [_node_text(c) for c in tr.element_children if c.tag in {"th", "td"}]
        elif child.tag in {"tbody", "tfoot"} or child.tag == "tr":
            tr_list = ([child] if child.tag == "tr"
                       else [c for c in child.element_children if c.tag == "tr"])
            for tr in tr_list:
                cells = [_node_text(c) for c in tr.element_children if c.tag in {"th", "td"}]
                if not cells or not any(cells):
                    continue
                if not header and not rows and any(c.tag == "th" for c in tr.element_children):
                    header = cells
                else:
                    rows.append(cells)
    # thead/th 缺失：第一行作表头（至少两列且首行无长文本时才判定，防误伤布局表）
    if not header and rows and len(rows[0]) >= 2 and all(len(c) <= 30 for c in rows[0]):
        header, rows = rows[0], rows[1:]
    text = "\n".join(serialize_table_row(header, row) for row in rows)
    return Element(kind="table", text=text, header=header, rows=rows)


def _extract_list(node: _Node, ordered: bool) -> Element:
    """把 <ul>/<ol> 节点转成列表元素，每项一条文本。

    嵌套列表（WHO 常用 ``<ul><li>分组：<ul class="disc">…``）内层项
    与外层项拉平收集；外层 ``<li>`` 的文本只取直属文本，避免与内层重复。
    """
    items: list[str] = []

    def li_own_text(li: _Node) -> str:
        """<li> 自身的文本：递归取全部文本，但排除嵌套列表的内容。"""
        parts: list[str] = []
        for child in li.children:
            if isinstance(child, _Text):
                parts.append(child.data)
            elif child.tag not in {"ul", "ol"}:
                parts.append(_node_text(child))
        return _collapse_ws("".join(parts))

    def collect(list_node: _Node) -> None:
        for child in list_node.element_children:
            if child.tag == "li":
                own = li_own_text(child)
                if own:
                    items.append(own)
                # li 里的嵌套列表递归收集
                for grandchild in child.element_children:
                    if grandchild.tag in {"ul", "ol"}:
                        collect(grandchild)
            elif child.tag in {"ul", "ol"}:
                collect(child)

    collect(node)
    return Element(kind="list", text="\n".join(items), items=items,
                   metadata={"ordered": ordered})


def _list_contains_titles(list_node: _Node) -> bool:
    """列表内部是否嵌着小节标题（``<p><strong>标题</strong></p>``）。

    WHO 部分新页面把整个正文包在一个 ``<ul>`` 里，小节标题嵌在 ``<li>``
    内部；这种「假列表」不能收为列表元素，要按容器递归展开。
    """
    for li in list_node.element_children:
        if li.tag != "li":
            continue
        for child in li.element_children:
            if child.tag == "p" and _strong_only_title(child):
                return True
    return False


def _strong_only_title(p_node: _Node) -> str:
    """识别 ``<p><strong>标题</strong></p>`` 模式；不是则返回空串。

    要求：段落只含 strong/b 元素（WHO 常写成 ``<strong>标题</strong><strong></strong>``，
    允许空 strong 并存）、无其他正文文本、合并文本 2–20 字、不带句末标点。
    WHO 部分新页面用它代替 h2 作小节标题。
    """
    element_children = p_node.element_children
    if not element_children or any(c.tag not in {"strong", "b"} for c in element_children):
        return ""
    if "".join(p_node.text_parts).strip():
        return ""
    title = _collapse_ws("".join(_node_text(c) for c in element_children))
    if not 2 <= len(title) <= 20:
        return ""
    if re.search(r"[。；，：、…]", title):
        return ""
    return title


def _flush_paragraph(buffer: list[str], sections: list[Section]) -> None:
    text = _collapse_ws("".join(buffer))
    buffer.clear()
    if text:
        sections[-1].elements.append(Element(kind="paragraph", text=text))


def _walk(node: _Node, sections: list[Section], buffer: list[str]) -> None:
    """深度优先遍历节点树，产出 Section 与 Element。buffer 累积行内文本成段落。

    children 混合存放元素与文本片段（``_Text``），按文档顺序遍历；
    块级标签先把累积的行内文本冲刷成段落，再处理自身。
    """
    for child in node.children:
        if isinstance(child, _Text):
            buffer.append(child.data)
            continue
        tag = child.tag
        if tag in _HEADING_TAGS:
            _flush_paragraph(buffer, sections)
            level = _HEADING_TAGS[tag]
            title = _node_text(child)
            sections.append(Section(title=title, level=level))
        elif tag == "p":
            _flush_paragraph(buffer, sections)
            # WHO 部分新页面小节标题退化为 <p><strong>标题</strong></p>（无 h2）：
            # 段落只含一个 strong 且文本短、无句末标点时按 2 级标题处理
            strong_title = _strong_only_title(child)
            if strong_title:
                sections.append(Section(title=strong_title, level=2))
                continue
            text = _node_text(child)
            if text:
                sections[-1].elements.append(Element(kind="paragraph", text=text))
        elif tag in {"ul", "ol"}:
            _flush_paragraph(buffer, sections)
            if _list_contains_titles(child):
                # 假列表：内部嵌着小节标题，按容器递归展开（li 里再遇标题、段落、子列表）
                _walk(child, sections, buffer)
                _flush_paragraph(buffer, sections)
                continue
            element = _extract_list(child, ordered=(tag == "ol"))
            if element.items:
                sections[-1].elements.append(element)
        elif tag == "table":
            _flush_paragraph(buffer, sections)
            element = _extract_table(child)
            if element.rows or element.header:
                sections[-1].elements.append(element)
        elif tag in {"div", "section", "article", "main", "blockquote", "hr", "li"}:
            _flush_paragraph(buffer, sections)
            _walk(child, sections, buffer)
            _flush_paragraph(buffer, sections)
        else:
            # 行内元素（a、em、strong、span 等）：递归，其文本经 _Text 进缓冲区
            _walk(child, sections, buffer)


def _drop_noise_lines(sections: list[Section], rules: HtmlRules) -> None:
    """按来源规则丢弃控件文字等噪声行。"""
    if not rules.drop_line_patterns:
        return
    patterns = [re.compile(p) for p in rules.drop_line_patterns]

    def keep(text: str) -> bool:
        return not any(p.search(text) for p in patterns)

    for section in sections:
        kept: list[Element] = []
        for element in section.elements:
            if element.kind in {"paragraph", "heading"} and not keep(element.text):
                continue
            if element.kind == "list":
                element.items = [i for i in element.items if keep(i)]
                element.text = "\n".join(element.items)
                if not element.items:
                    continue
            kept.append(element)
        section.elements = kept


def _find_content_roots(root: _Node, rules: HtmlRules) -> list[_Node]:
    """按 content_ids/content_classes 找正文容器；找不到时回退为整棵树。

    id 匹配优先：页面常同时有正文版与移动/打印版两份内容（class 相同、
    只有正文版有目标 id），有 id 命中时不再用 class 匹配，避免重复抽取。
    """
    if not rules.content_ids and not rules.content_classes:
        return [root]
    by_id: list[_Node] = []
    by_class: list[_Node] = []

    def visit(node: _Node) -> None:
        node_id = node.attrs.get("id", "")
        classes = set(node.attrs.get("class", "").split())
        if node_id and node_id in rules.content_ids:
            by_id.append(node)
            return  # 容器内部不再找嵌套容器
        if classes & rules.content_classes:
            by_class.append(node)
            return
        for child in node.element_children:
            visit(child)

    visit(root)
    matches = by_id or by_class
    return matches or [root]


def parse_html(
    html: str,
    doc_id: str,
    metadata: DocumentMetadata | None = None,
    source_type: str = "",
) -> Document:
    """把 HTML 正文解析为 Document。

    ``source_type`` 决定噪声规则；未知来源用默认规则（只去脚本导航等通用噪声）。
    """
    rules = SOURCE_RULES.get(source_type, _DEFAULT_RULES)
    builder = _TreeBuilder(rules)
    builder.feed(html)

    sections: list[Section] = [Section(title="", level=1)]  # 兜底章节，收标题前的内容
    buffer: list[str] = []
    for content_root in _find_content_roots(builder.root, rules):
        _walk(content_root, sections, buffer)
    _flush_paragraph(buffer, sections)
    _drop_noise_lines(sections, rules)

    # 丢掉没有任何内容的兜底空章节
    sections = [s for s in sections if s.title or s.elements]

    meta = metadata or DocumentMetadata()
    if source_type and not meta.source_type:
        meta.source_type = source_type
    # h1 即文档标题时回填 metadata.title
    if not meta.title:
        for section in sections:
            if section.level == 1 and section.title:
                meta.title = section.title
                break
    return Document(doc_id=doc_id, metadata=meta, sections=sections)


def parse_mediawiki_json(payload: dict[str, Any], doc_id: str,
                         metadata: DocumentMetadata | None = None) -> Document:
    """解析 MediaWiki API action=parse 的响应：取出正文 HTML 后走 HTML 解析器。"""
    parse = payload.get("parse", {})
    text_field = parse.get("text", "")
    # text 可能是 {"*": "..."}（旧格式）或直接是字符串
    if isinstance(text_field, dict):
        html = str(text_field.get("*", ""))
    else:
        html = str(text_field)
    meta = metadata or DocumentMetadata()
    if not meta.title:
        meta.title = str(parse.get("title", ""))
    return parse_html(html, doc_id, meta, source_type="wikipedia")
