"""专科路由器：确定性规则打分，不依赖 LLM。

打分公式:
    单专科总分 = 3.0*关键词分 + 2.0*归一化术语分 + 2.0*证据加权分 + 1.0*诊断规划提示分

Top2 筛选阈值:
    MIN_PRIMARY_SCORE = 2.0
    MIN_SECONDARY_SCORE = 1.2
    SCORE_GAP = 3.0
    LOW_CONFIDENCE = 0.45

全局默认执行策略:
    if 置信度 >= 0.45 且 第二名分数 >= 1.2: 动态Top2
    elif 第一名分数 >= 2.0: Top1 + evidence_skeptic
    else: 兜底组合 general_internal + evidence_skeptic
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medidiag.rag.normalizer import NormalizedQuery, TerminologyNormalizer
from medidiag.rag.retrieval import EVIDENCE_LEVEL_SCORES
from medidiag.schemas import KnowledgeChunk

from medidiag.agents.specialty_data import (
    FALLBACK_PAIR,
    ROUTING_WEIGHTS,
    SPECIALTIES,
    SPECIALTY_KEYWORDS,
    SYSTEM_MATCH_KEYWORDS,
    THRESHOLDS,
)


@dataclass
class SpecialtyScore:
    """单个专科的分项得分。"""

    specialty: str
    keyword_score: float = 0.0
    normalized_term_score: float = 0.0
    evidence_score: float = 0.0
    plan_hint_score: float = 0.0
    total: float = 0.0

    def compute_total(self) -> None:
        """计算加权总分。"""
        w = ROUTING_WEIGHTS
        self.total = (
            w["keyword"] * self.keyword_score
            + w["normalized_term"] * self.normalized_term_score
            + w["evidence"] * self.evidence_score
            + w["plan_hint"] * self.plan_hint_score
        )

    def to_dict(self) -> dict:
        """转为字典（用于日志/trace）。"""
        return {
            "specialty": self.specialty,
            "keyword_score": round(self.keyword_score, 4),
            "normalized_term_score": round(self.normalized_term_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "plan_hint_score": round(self.plan_hint_score, 4),
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
    """路由置信度 = top2.total / top1.total。"""

    is_fallback: bool = False
    """是否使用了兜底组合。"""

    reason: str = ""
    """选择理由。"""

    @property
    def specialty_names(self) -> list[str]:
        """双专科名称列表。"""
        return list(self.specialty_pair)

    def to_dict(self) -> dict:
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

    def __init__(self, normalizer: TerminologyNormalizer | None = None) -> None:
        self.normalizer = normalizer

    def route(
        self,
        normalized_query: str | NormalizedQuery,
        evidence_chunks: list[KnowledgeChunk] | list[dict],
        plan_hint: str = "",
    ) -> RoutingResult:
        """路由到双专科。

        Args:
            normalized_query: 归一化后的查询（字符串或 NormalizedQuery）。
            evidence_chunks: 检索证据 chunks。
            plan_hint: 前置诊断规划提示（系统匹配信息）。

        Returns:
            RoutingResult，包含双专科、分项得分、置信度、理由。
        """
        # 1. 计算所有专科得分
        scores = self._compute_all_scores(
            normalized_query, evidence_chunks, plan_hint
        )

        # 2. 按总分降序排序
        sorted_scores = sorted(
            scores.values(), key=lambda s: s.total, reverse=True
        )

        top1 = sorted_scores[0] if sorted_scores else None
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else None

        th = THRESHOLDS

        # 3. 计算置信度 = top2.total / top1.total
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

        # 规则4: 置信度 < LOW_CONFIDENCE → 兜底
        if confidence < th["LOW_CONFIDENCE"]:
            return RoutingResult(
                specialty_pair=FALLBACK_PAIR,
                scores={s.specialty: s for s in sorted_scores[:5]},
                confidence=confidence,
                is_fallback=True,
                reason=(
                    f"置信度 {confidence:.2f} < "
                    f"{th['LOW_CONFIDENCE']}, 兜底组合"
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
        evidence_chunks: list,
        plan_hint: str,
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

        query_text = nq.normalized.lower()

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
                specialty, query_text
            )
            score.normalized_term_score = self._compute_term_score(
                specialty, nq
            )
            score.evidence_score = self._compute_evidence_score(
                specialty, evidence_texts, evidence_chunks
            )
            score.plan_hint_score = self._compute_plan_hint_score(
                specialty, plan_hint
            )

            score.compute_total()
            scores[specialty] = score

        return scores

    def _compute_keyword_score(self, specialty: str, query_text: str) -> float:
        """关键词分：匹配数量，上限4条归一至 0~1。"""
        keywords = SPECIALTY_KEYWORDS.get(specialty, [])
        if not keywords:
            return 0.0
        matches = 0
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, query_text):
                matches += 1
        return min(matches / 4.0, 1.0)

    def _compute_term_score(
        self, specialty: str, nq: NormalizedQuery
    ) -> float:
        """归一化术语分：该专科关键词在归一化术语中的匹配占比。"""
        keywords = SPECIALTY_KEYWORDS.get(specialty, [])
        if not keywords or not nq.matched_terms:
            return 0.0
        matched_preferred = {m.preferred.lower() for m in nq.matched_terms}
        kw_lower = {kw.lower() for kw in keywords}
        overlap = len(matched_preferred & kw_lower)
        return min(overlap / max(len(keywords), 1), 1.0)

    def _compute_evidence_score(
        self,
        specialty: str,
        evidence_texts: list[str],
        evidence_chunks: list,
    ) -> float:
        """证据加权分：Top检索证据文本关键词匹配量 × 证据得分，归一 0~1。"""
        keywords = SPECIALTY_KEYWORDS.get(specialty, [])
        if not keywords or not evidence_texts:
            return 0.0

        top_k = min(5, len(evidence_texts))
        total_score = 0.0

        for i in range(top_k):
            text = evidence_texts[i].lower()
            matches = 0
            for kw in keywords:
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, text):
                    matches += 1

            # 证据等级分数
            if i < len(evidence_chunks):
                chunk = evidence_chunks[i]
                if isinstance(chunk, dict):
                    level = chunk.get(
                        "evidence_level", "level_5_other"
                    )
                else:
                    level = chunk.evidence_level
                ev_score = EVIDENCE_LEVEL_SCORES.get(level, 0.3)
            else:
                ev_score = 0.3

            total_score += (
                matches / max(len(keywords), 1)
            ) * ev_score

        return min(total_score / max(top_k, 1), 1.0)

    def _compute_plan_hint_score(
        self, specialty: str, plan_hint: str
    ) -> float:
        """诊断规划提示分：疑似系统匹配则 1，否则 0。"""
        if not plan_hint:
            return 0.0
        system_keywords = SYSTEM_MATCH_KEYWORDS.get(specialty, [])
        for sk in system_keywords:
            if sk.lower() in plan_hint.lower():
                return 1.0
        return 0.0
