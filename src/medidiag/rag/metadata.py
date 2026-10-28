"""元数据过滤（M10）：主题词表、章节类型映射、过滤值校验与放宽重查。

块的 ``topic`` 与 ``section_type`` 在运行时从 ``title`` 与 ``section_path`` 派生，只保存在
``MetadataIndex`` 里，不改写语料 jsonl，也不改块的 metadata（不影响向量缓存键与语料哈希）。
过滤值必须属于已知枚举，否则丢弃并记录；检索器只接收 ``allowed_ids``，不关心元数据。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SECTION_TYPES = ("definition", "symptoms", "transmission", "prevention", "treatment", "epidemiology", "key_facts")
SOURCE_TYPES = ("who_factsheet", "who_publication", "nhc_document", "epidemic_report", "wikipedia")
FILTER_KEYS = ("topic", "section_type", "source_type", "period")
# 放宽顺序：先丢最细的条件（章节类型），最后丢统计周期
RELAX_ORDER = ("section_type", "topic", "source_type", "period")
LATEST = "latest"

# 章节标题关键词，按顺序匹配，先中者胜出
_SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("transmission", re.compile(r"传播|感染途径|传染途径")),
    ("symptoms", re.compile(r"症状|临床表现|体征")),
    ("prevention", re.compile(r"预防|疫苗")),
    ("treatment", re.compile(r"治疗|诊疗|处理")),
    ("epidemiology", re.compile(r"流行|发病|负担|统计|概况|问题的范围")),
    ("key_facts", re.compile(r"要点|重要事实")),
    ("definition", re.compile(r"概述|简介|是什么|^定义$|^.{0,6}的定义$")),
)

# 手写别名：任一别名（归一化后）与过滤值互相包含时，改用右侧关键词匹配文档标题
_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("流感", "流行性感冒", "季节性流感"): ("流感", "流行性感冒"),
    ("禽流感", "人感染禽流感"): ("禽流感",),
    ("新冠", "新型冠状病毒", "新型冠状病毒感染", "2019冠状病毒病", "covid"): ("冠状病毒", "covid"),
    ("猴痘", "mpox"): ("猴痘", "mpox"),
    ("蛇咬伤", "蛇咬"): ("蛇咬",),
    ("抗生素耐药", "抗菌素耐药", "耐药"): ("耐药",),
    ("高血压",): ("高血压", "高血壓"),
    ("艾滋病", "艾滋"): ("艾滋",),
    ("肺结核", "结核病", "结核"): ("结核", "結核"),
    ("乙肝", "乙型肝炎"): ("乙型肝炎",),
    ("肥胖", "超重"): ("肥胖",),
    ("中风", "卒中"): ("中风", "卒中"),
    ("心梗", "心肌梗死", "心脏病发作"): ("心脏病发作", "心肌梗死"),
}

_VERSION_BRACKET = re.compile(r"[（(][^）)]*(?:版|试行|附件|含)[^）)]*[）)]")
_BRACKET = re.compile(r"[（(]([^）)]*)[）)]")
_NOISE = re.compile(r"[\s《》“”\"'·、，,：:\u200e\u200f\-–—/]+")


def _norm(text: str) -> str:
    return _NOISE.sub("", text).lower()


def title_names(title: str) -> set[str]:
    """文档标题的主题名集合：去版本号括号，括号外与括号内各算一个名字，再归一化。"""
    title = _VERSION_BRACKET.sub("", title)
    names = {_BRACKET.sub("", title)} | set(_BRACKET.findall(title))
    return {name for name in (_norm(n) for n in names) if len(name) >= 2}


def section_type_of(section_path: Sequence[str] | str | None) -> str | None:
    """章节路径从最深一级往上找第一个命中规则的标题；无匹配返回 None。"""
    if not section_path:
        return None
    parts = [section_path] if isinstance(section_path, str) else list(section_path)
    for heading in reversed(parts):
        heading = str(heading).strip()
        for section_type, pattern in _SECTION_RULES:
            if heading and pattern.search(heading):
                return section_type
    return None


@dataclass
class FilterTrace:
    """一次过滤检索的记录：请求的条件、被丢弃的条件、实际生效的条件与放宽轨迹。"""

    requested: dict[str, Any]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    applied: dict[str, str] = field(default_factory=dict)
    relaxed: list[str] = field(default_factory=list)  # 按顺序被放宽掉的条件名
    attempts: list[dict[str, Any]] = field(default_factory=list)  # 每次尝试的条件与结果数
    allowed_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"requested": self.requested, "dropped": self.dropped, "applied": self.applied,
                "relaxed": self.relaxed, "attempts": self.attempts, "allowed_count": self.allowed_count}


class MetadataIndex:
    """按块派生 topic / section_type，并把校验后的过滤条件解析成允许的 chunk_id 集合。"""

    def __init__(self, chunks: Iterable[Any]) -> None:
        self._rows: list[tuple[str, str, str, str | None, str | None]] = []  # chunk_id, doc_id, source_type, period, section_type
        self._doc_names: dict[str, set[str]] = {}
        periods: set[str] = set()
        for chunk in chunks:
            meta = chunk.metadata
            doc_id = str(meta.get("doc_id") or chunk.source_id)
            period = meta.get("period") or None
            if period:
                periods.add(str(period))
            self._doc_names.setdefault(doc_id, title_names(str(meta.get("title") or chunk.source)))
            self._rows.append((chunk.chunk_id, doc_id, str(meta.get("source_type") or ""),
                               str(period) if period else None, section_type_of(meta.get("section_path"))))
        self.periods = tuple(sorted(periods))
        self.latest_period = self.periods[-1] if self.periods else None

    def resolve_topic(self, value: str) -> set[str]:
        """把主题词解析为文档 id 集合；空集合表示不在已知主题里，调用方应丢弃该过滤值。"""
        needle = _norm(value)
        if len(needle) < 2:
            return set()
        keywords = {needle}
        for aliases, targets in _ALIASES.items():
            if any(a in needle for a in map(_norm, aliases)):
                keywords.update(_norm(t) for t in targets)
        return {doc_id for doc_id, names in self._doc_names.items()
                if any(k in name or (len(name) >= 3 and name in k) for k in keywords for name in names)}

    def validate(self, raw: Mapping[str, Any] | None) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """校验过滤条件：未知键、非字符串值、不在枚举内的值一律丢弃并记录原因；latest 解析为语料中最大的 period。"""
        valid: dict[str, str] = {}
        dropped: list[dict[str, Any]] = []
        for key, value in (raw or {}).items():
            reason: str | None = None
            if key not in FILTER_KEYS:
                reason = "unknown_key"
            elif not isinstance(value, str) or not value.strip():
                reason = "invalid_value"
            else:
                value = value.strip()
                if key == "section_type" and value not in SECTION_TYPES:
                    reason = "not_in_enum"
                elif key == "source_type" and value not in SOURCE_TYPES:
                    reason = "not_in_enum"
                elif key == "period":
                    if value == LATEST:
                        if self.latest_period is None:
                            reason = "no_period_in_corpus"
                        else:
                            value = self.latest_period
                    elif value not in self.periods:
                        reason = "not_in_corpus"
                elif key == "topic" and not self.resolve_topic(value):
                    reason = "unknown_topic"
            if reason is None:
                valid[key] = str(value)
            else:
                dropped.append({"key": key, "value": value if isinstance(value, str) else repr(value)[:40],
                                "reason": reason})
        return valid, dropped

    def allowed_ids(self, filters: Mapping[str, str]) -> set[str]:
        """满足全部（已校验的）过滤条件的 chunk_id 集合；没有条件时为全部块。"""
        topic_docs = self.resolve_topic(filters["topic"]) if "topic" in filters else None
        out: set[str] = set()
        for chunk_id, doc_id, source_type, period, section_type in self._rows:
            if topic_docs is not None and doc_id not in topic_docs:
                continue
            if "section_type" in filters and section_type != filters["section_type"]:
                continue
            if "source_type" in filters and source_type != filters["source_type"]:
                continue
            if "period" in filters and period != filters["period"]:
                continue
            out.add(chunk_id)
        return out


def search_with_relaxation(
    search: Callable[[set[str] | None], Sequence[Any]],
    index: MetadataIndex,
    raw_filters: Mapping[str, Any] | None,
    min_results: int,
) -> tuple[Sequence[Any], FilterTrace]:
    """带过滤检索；结果少于 ``min_results`` 时按 RELAX_ORDER 逐个放宽条件重查，直到够数或没有条件。

    ``search(allowed_ids)`` 在 ``allowed_ids`` 内检索（None 表示不过滤）。放宽过程记在返回的 FilterTrace 里；
    最后一次尝试的结果原样返回（即使仍不足 ``min_results``）。
    """
    valid, dropped = index.validate(raw_filters)
    trace = FilterTrace(requested=dict(raw_filters or {}), dropped=dropped)
    current = dict(valid)
    while True:
        allowed = index.allowed_ids(current) if current else None
        results = search(allowed)
        trace.attempts.append({"filters": dict(current), "allowed": None if allowed is None else len(allowed),
                               "results": len(results)})
        if len(results) >= min_results or not current:
            trace.applied = dict(current)
            trace.allowed_count = None if allowed is None else len(allowed)
            return results, trace
        drop = next(key for key in RELAX_ORDER if key in current)
        del current[drop]
        trace.relaxed.append(drop)
