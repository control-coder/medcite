"""专科路由器：确定性规则打分，不依赖 LLM。

打分公式（2026-07-27 修正，DD-023）:
    单专科总分 = 3.0*关键词分 + 2.0*归一化术语分 + 2.0*证据加权分

三个分项都按 `min(命中数 / MATCH_SATURATION_COUNT, 1.0)` 归一，分母与词表长度无关。
原第四个分项 `plan_hint` 已删除：它的唯一取值来源是 `route()` 的入参，而所有调用点
都传空串，该分项恒为 0。

Top2 筛选阈值（数值**未改动**）:
    MIN_PRIMARY_SCORE = 2.0
    MIN_SECONDARY_SCORE = 1.2
    SCORE_GAP = 3.0

原规则 4（`置信度 < LOW_CONFIDENCE → 兜底`）已于 2026-07-28 删除，其语义是颠倒的
（DD-025）：`confidence = top2.total / top1.total` 越小表示 top1 越占优，也就是路由
**越明确**，而该规则恰恰在这时兜底。`LOW_CONFIDENCE` 常量随之删除——它没有别的
消费者，保留一个不再生效的阈值属于 DD-019 明确反对的做法。比值本身仍作为歧义度
诊断量保留在 `RoutingResult.confidence` 与路由诊断输出中，只是不再当门禁。

全局默认执行策略:
    if 第一名分数 >= 2.0 且 第二名分数 >= 1.2 且 分差 < 3.0: 动态Top2
    elif 第一名分数 >= 2.0: Top1 + evidence_skeptic
    else: 兜底组合 general_internal + evidence_skeptic
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from medidiag.agents.specialty_data import (
    FALLBACK_PAIR,
    MATCH_SATURATION_COUNT,
    ROUTING_WEIGHTS,
    SPECIALTIES,
    SPECIALTY_KEYWORDS,
    THRESHOLDS,
)
from medidiag.rag.normalizer import NormalizedQuery, TerminologyNormalizer
from medidiag.schemas import KnowledgeChunk

_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _pattern(term: str) -> re.Pattern[str]:
    """词边界正则的进程内缓存（每个样本都会重建一次 router）。"""
    cached = _PATTERN_CACHE.get(term)
    if cached is None:
        cached = re.compile(r"\b" + re.escape(term) + r"\b")
        _PATTERN_CACHE[term] = cached
    return cached


def _contains_word(text: str, term: str) -> bool:
    """按词边界在已小写的 ``text`` 中查找已小写的 ``term``。"""
    return _pattern(term).search(text) is not None


@dataclass
class SpecialtyScore:
    """单个专科的分项得分。"""

    specialty: str
    keyword_score: float = 0.0
    normalized_term_score: float = 0.0
    evidence_score: float = 0.0
    total: float = 0.0

    def compute_total(self) -> None:
        """计算加权总分。"""
        w = ROUTING_WEIGHTS
        self.total = (
            w["keyword"] * self.keyword_score
            + w["normalized_term"] * self.normalized_term_score
            + w["evidence"] * self.evidence_score
        )

    def to_dict(self) -> dict[str, Any]:
        """转为字典（用于日志/trace）。

        2026-07-27 起不再输出 `plan_hint_score`（分项已删除，DD-023）。历史
        `reports/routing_diagnostics_*_{before,after}.json` 里仍带该键。
        """
        return {
            "specialty": self.specialty,
            "keyword_score": round(self.keyword_score, 4),
            "normalized_term_score": round(self.normalized_term_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "total": round(self.total, 4),
        }


@dataclass
class RoutingResult:
    """路由结果。"""

    specialty_pair: tuple[str, str]
    """选中的双专科。"""

    scores: dict[str, SpecialtyScore] = field(default_factory=dict)
    """所有专科的分项得分。"""

    confidence: float = 0.0
    """歧义度比值 = top2.total / top1.total。

    **该值不参与路由决策**（原规则 4 已删除，DD-025）。名字沿用 `confidence` 是为了
    不破坏 `eval/runner.py` 的 `routing_confidence` 字段与历史产物的键名，但它的方向
    与"置信度"直觉相反：值越大表示前两名越接近，也就是越难判断。
    """

    is_fallback: bool = False
    """是否使用了兜底组合。"""

    reason: str = ""
    """选择理由。"""

    @property
    def specialty_names(self) -> list[str]:
        """双专科名称列表。"""
        return list(self.specialty_pair)

    def to_dict(self) -> dict[str, Any]:
        """转为字典（用于日志/trace）。"""
        return {
            "specialty_pair": list(self.specialty_pair),
            "confidence": round(self.confidence, 4),
            "is_fallback": self.is_fallback,
            "reason": self.reason,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
        }


class SpecialistRouter:
    """专科路由器。

    确定性规则打分，不依赖 LLM，输出可复现、可消融。

    用法:
        router = SpecialistRouter(normalizer=normalizer)
        result = router.route(query, evidence_chunks, plan_hint)
        # result.specialty_pair = ("cardiology", "respiratory")
    """

    def __init__(
        self,
        normalizer: TerminologyNormalizer | None = None,
        evidence_level_scores: dict[str, float] | None = None,
    ) -> None:
        self.normalizer = normalizer
        # Evaluation callers pass the locked YAML mapping. The neutral default
        # keeps this runtime component usable without importing eval config.
        self.evidence_level_scores = evidence_level_scores or {}

    def route(
        self,
        normalized_query: str | NormalizedQuery,
        evidence_chunks: list[KnowledgeChunk] | list[dict[str, Any]],
    ) -> RoutingResult:
        """路由到双专科。

        Args:
            normalized_query: 归一化后的查询（字符串或 NormalizedQuery）。
            evidence_chunks: 检索证据 chunks。

        Returns:
            RoutingResult，包含双专科、分项得分、置信度、理由。
        """
        # 1. 计算所有专科得分
        scores = self._compute_all_scores(normalized_query, evidence_chunks)

        # 2. 按总分降序排序
        sorted_scores = sorted(
            scores.values(), key=lambda s: s.total, reverse=True
        )

        top1 = sorted_scores[0] if sorted_scores else None
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else None

        th = THRESHOLDS

        # 3. 歧义度比值 = top2.total / top1.total。只作为诊断量记录，不参与决策
        # （原规则 4 已删除，DD-025）。
        if top1 and top2 and top1.total > 0:
            confidence = top2.total / top1.total
        else:
            confidence = 0.0

        # 4. 路由决策（全局默认执行策略）
        # 规则1: 最高分 < MIN_PRIMARY_SCORE → 兜底
        if not top1 or top1.total < th["MIN_PRIMARY_SCORE"]:
            return RoutingResult(
                specialty_pair=FALLBACK_PAIR,
                scores={s.specialty: s for s in sorted_scores[:5]},
                confidence=confidence,
                is_fallback=True,
                reason=(
                    f"最高分 {top1.total if top1 else 0:.2f} < "
                    f"{th['MIN_PRIMARY_SCORE']}, 兜底组合"
                ),
            )

        # 规则2: 第二名不足 或 分差过大 → Top1 + evidence_skeptic
        if (
            not top2
            or top2.total < th["MIN_SECONDARY_SCORE"]
            or (top1.total - top2.total) >= th["SCORE_GAP"]
        ):
            pair = (top1.specialty, "evidence_skeptic")
            # 如果 top1 已经是 evidence_skeptic，用 general_internal 替代
            if top1.specialty == "evidence_skeptic":
                pair = (top1.specialty, "general_internal")
            return RoutingResult(
                specialty_pair=pair,
                scores={s.specialty: s for s in sorted_scores[:5]},
                confidence=confidence,
                is_fallback=False,
                reason=(
                    f"Top1={top1.specialty}({top1.total:.2f}), "
                    f"第二名不足或分差>= {th['SCORE_GAP']}, "
                    f"Top1+evidence_skeptic"
                ),
            )

        # 规则3: 动态Top2（均达标）
        return RoutingResult(
            specialty_pair=(top1.specialty, top2.specialty),
            scores={s.specialty: s for s in sorted_scores[:5]},
            confidence=confidence,
            is_fallback=False,
            reason=(
                f"动态Top2: {top1.specialty}({top1.total:.2f}) + "
                f"{top2.specialty}({top2.total:.2f})"
            ),
        )

    def _compute_all_scores(
        self,
        normalized_query: str | NormalizedQuery,
        evidence_chunks: list[Any],
    ) -> dict[str, SpecialtyScore]:
        """计算所有专科的得分。"""
        # 归一化查询
        if isinstance(normalized_query, str):
            if self.normalizer:
                nq = self.normalizer.normalize(normalized_query)
            else:
                nq = NormalizedQuery(
                    original=normalized_query, normalized=normalized_query
                )
        else:
            nq = normalized_query

        # 关键词分同时看归一化前后的文本：归一化器会把 `ECG` 改写成
        # `electrocardiogram`、`dyspnea` 改写成 `shortness of breath`，只打分
        # 改写后的文本会丢掉被改写掉的那些词条的命中（详见 `_compute_keyword_score`）。
        query_text = nq.normalized.lower()
        original_text = nq.original.lower()

        # 证据文本
        evidence_texts: list[str] = []
        for chunk in evidence_chunks:
            if isinstance(chunk, dict):
                evidence_texts.append(chunk.get("text", ""))
            else:
                evidence_texts.append(chunk.text)

        scores: dict[str, SpecialtyScore] = {}
        for specialty in SPECIALTIES:
            score = SpecialtyScore(specialty=specialty)

            score.keyword_score = self._compute_keyword_score(
                specialty, query_text, original_text
            )
            score.normalized_term_score = self._compute_term_score(
                specialty, nq
            )
            score.evidence_score = self._compute_evidence_score(
                specialty, evidence_texts, evidence_chunks
            )

            score.compute_total()
            scores[specialty] = score

        return scores

    def _compute_keyword_score(
        self, specialty: str, query_text: str, original_text: str = ""
    ) -> float:
        """关键词分：命中词条数按 ``MATCH_SATURATION_COUNT`` 饱和归一至 0~1。

        同时在归一化后文本与原文中查找，取并集去重计数。只看归一化后的文本会
        丢命中：归一化器把 `ECG`/`EKG` 改写成 `electrocardiogram`、`dyspnea`
        改写成 `shortness of breath`，被改写掉的原词条在改写结果里不再存在。
        并集去重保证同一词条不会因为两处都在而被数两次，也保证「同义词组里命中
        几个变体」只算一次命中（`electrocardiogram` 与 `ECG` 都在心内科表内，
        并集会把它们算成 2 次，这一点见下方 `_dedupe_by_preferred`）。
        """
        keywords = SPECIALTY_KEYWORDS.get(specialty, [])
        if not keywords:
            return 0.0
        hits: set[str] = set()
        for kw in keywords:
            lowered = kw.lower()
            if _contains_word(query_text, lowered) or (
                original_text and _contains_word(original_text, lowered)
            ):
                hits.add(lowered)
        matches = len(self._dedupe_by_preferred(hits))
        return min(matches / MATCH_SATURATION_COUNT, 1.0)

    def _dedupe_by_preferred(self, hits: set[str]) -> set[str]:
        """把命中的词条折叠到归一化首选形式，同义词组只计一次。

        `ECG`、`EKG`、`electrocardiogram` 三条都在心内科表内且互为同义词；同时
        对原文与归一化文本计数时它们会同时命中，把一个临床事实数成 3 次命中。
        没有 normalizer 时退化为原样返回。
        """
        if not self.normalizer:
            return hits
        return {self.normalizer.preferred_form(hit).lower() for hit in hits}

    def _compute_term_score(
        self, specialty: str, nq: NormalizedQuery
    ) -> float:
        """归一化术语分：命中的首选术语落在该专科词表内的**数量**，饱和归一。

        2026-07-27 修正（DD-023）：原实现除以 `len(SPECIALTY_KEYWORDS[specialty])`，
        于是给词表加词会**降低**一个已经命中的专科的分数。现在与关键词分共用
        `MATCH_SATURATION_COUNT`，分母与词表长度无关。
        """
        keywords = SPECIALTY_KEYWORDS.get(specialty, [])
        if not keywords or not nq.matched_terms:
            return 0.0
        matched_preferred = {m.preferred.lower() for m in nq.matched_terms}
        kw_lower = {kw.lower() for kw in keywords}
        overlap = len(matched_preferred & kw_lower)
        return min(overlap / MATCH_SATURATION_COUNT, 1.0)

    def _compute_evidence_score(
        self,
        specialty: str,
        evidence_texts: list[str],
        evidence_chunks: list[Any],
    ) -> float:
        """证据加权分：Top 检索证据的关键词命中数 × 证据等级权重，饱和归一 0~1。

        2026-07-27 修正（DD-023）：每条 chunk 的命中数原先除以
        `len(SPECIALTY_KEYWORDS[specialty])`，与 `_compute_term_score` 同病。现在
        每条 chunk 用 `min(命中数 / MATCH_SATURATION_COUNT, 1.0)` 得到 0~1 的
        chunk 级分，乘以证据等级权重后按 chunk 数取平均，因此分母只与 chunk 数
        有关，与词表长度无关。
        """
        keywords = SPECIALTY_KEYWORDS.get(specialty, [])
        if not keywords or not evidence_texts:
            return 0.0

        top_k = min(5, len(evidence_texts))
        total_score = 0.0

        for i in range(top_k):
            text = evidence_texts[i].lower()
            hits: set[str] = set()
            for kw in keywords:
                lowered = kw.lower()
                if _contains_word(text, lowered):
                    hits.add(lowered)
            matches = len(self._dedupe_by_preferred(hits))

            # 证据等级分数
            if i < len(evidence_chunks):
                chunk = evidence_chunks[i]
                if isinstance(chunk, dict):
                    level = chunk.get(
                        "evidence_level", "level_5_other"
                    )
                else:
                    level = chunk.evidence_level
                ev_score = self.evidence_level_scores.get(level, 1.0)
            else:
                ev_score = 0.3

            total_score += min(matches / MATCH_SATURATION_COUNT, 1.0) * ev_score

        return min(total_score / max(top_k, 1), 1.0)
