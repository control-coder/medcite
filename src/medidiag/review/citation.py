"""引用校验模块：判定 claim 与 evidence 的支撑关系。

使用 NLI 模型（microsoft/deberta-v3-base-mnli）判定 SUPPORTED/PARTIAL/UNSUPPORTED。
NLI 模型不可用时降级为规则判定（关键词重叠）。
LLM judge 只作为辅助解释，不作为唯一真值。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from medidiag.schemas import KnowledgeChunk


class CitationVerdict(str, Enum):
    """引用校验判定结果。"""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class CitationResult:
    """单个 claim 的校验结果。"""

    claim_text: str
    evidence_chunk_id: str
    verdict: CitationVerdict
    confidence: float = 0.0
    method: str = "rule"  # "nli" | "rule"
    detail: str = ""


class CitationVerifier:
    """引用校验器。

    优先使用 NLI 模型判定，不可用时降级为规则判定。
    规则判定基于关键词重叠率，不是真值，标注 method="rule"。

    PLAN.md 要求:
        - SUPPORTED/PARTIAL/UNSUPPORTED 不得只靠 LLM judge
        - 必须使用固定 NLI / cross-encoder / judge 模型
        - 抽样人工复核 + Cohen's Kappa 一致性统计
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base-mnli",
        use_nli: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_nli = use_nli
        self._nli_pipeline = None
        self._nli_available: bool | None = None

    def _load_nli_model(self) -> bool:
        """尝试加载 NLI 模型，返回是否成功。"""
        if not self.use_nli:
            self._nli_available = False
            return False
        if self._nli_available is not None:
            return self._nli_available
        try:
            from transformers import pipeline

            self._nli_pipeline = pipeline(
                "text-classification", model=self.model_name
            )
            self._nli_available = True
            return True
        except Exception:
            self._nli_available = False
            return False

    def verify(
        self,
        claim_text: str,
        evidence_text: str,
        evidence_chunk_id: str = "",
    ) -> CitationResult:
        """校验单个 claim 与 evidence 的支撑关系。"""
        # 优先用 NLI 模型
        if self._nli_available is None:
            self._load_nli_model()

        if self._nli_available and self._nli_pipeline:
            return self._verify_nli(
                claim_text, evidence_text, evidence_chunk_id
            )
        else:
            return self._verify_rule(
                claim_text, evidence_text, evidence_chunk_id
            )

    def verify_batch(
        self,
        claims: list[dict],
        evidence_chunks: list[KnowledgeChunk] | list[dict],
    ) -> list[CitationResult]:
        """批量校验 claims。

        Args:
            claims: [{"text": ..., "citation_chunk_ids": [...]}]
            evidence_chunks: 知识库 chunks
        """
        # 构建 chunk_id -> text 映射
        chunk_map: dict[str, str] = {}
        for chunk in evidence_chunks:
            if isinstance(chunk, dict):
                chunk_map[chunk.get("chunk_id", "")] = chunk.get("text", "")
            else:
                chunk_map[chunk.chunk_id] = chunk.text

        results: list[CitationResult] = []
        for claim in claims:
            text = claim.get("text", "")
            citation_ids = claim.get("citation_chunk_ids", [])
            if not citation_ids:
                results.append(
                    CitationResult(
                        claim_text=text,
                        evidence_chunk_id="",
                        verdict=CitationVerdict.UNSUPPORTED,
                        method="rule",
                        detail="no citation",
                    )
                )
                continue

            # 对每个 citation 分别校验，取最佳结果
            best_verdict = CitationVerdict.UNSUPPORTED
            best_confidence = 0.0
            best_chunk_id = citation_ids[0]
            best_detail = ""

            for cid in citation_ids:
                ev_text = chunk_map.get(cid, "")
                if not ev_text:
                    continue
                result = self.verify(text, ev_text, cid)
                if result.verdict == CitationVerdict.SUPPORTED:
                    best_verdict = CitationVerdict.SUPPORTED
                    best_confidence = result.confidence
                    best_chunk_id = cid
                    best_detail = result.detail
                    break
                elif result.verdict == CitationVerdict.PARTIAL:
                    if best_verdict == CitationVerdict.UNSUPPORTED:
                        best_verdict = CitationVerdict.PARTIAL
                        best_confidence = result.confidence
                        best_chunk_id = cid
                        best_detail = result.detail

            results.append(
                CitationResult(
                    claim_text=text,
                    evidence_chunk_id=best_chunk_id,
                    verdict=best_verdict,
                    confidence=best_confidence,
                    method="nli" if self._nli_available else "rule",
                    detail=best_detail,
                )
            )

        return results

    def _verify_nli(
        self,
        claim: str,
        evidence: str,
        chunk_id: str,
    ) -> CitationResult:
        """NLI 模型判定。"""
        try:
            result = self._nli_pipeline(
                f"{evidence} [SEP] {claim}"
            )
            label = result[0]["label"].upper()
            score = result[0]["score"]

            if "ENTAIL" in label:
                verdict = CitationVerdict.SUPPORTED
            elif "NEUTRAL" in label:
                verdict = CitationVerdict.PARTIAL
            else:
                verdict = CitationVerdict.UNSUPPORTED

            return CitationResult(
                claim_text=claim,
                evidence_chunk_id=chunk_id,
                verdict=verdict,
                confidence=score,
                method="nli",
                detail=f"label={label}, score={score:.4f}",
            )
        except Exception as e:
            return self._verify_rule(claim, evidence, chunk_id)

    def _verify_rule(
        self,
        claim: str,
        evidence: str,
        chunk_id: str,
    ) -> CitationResult:
        """规则判定：关键词重叠率（降级方案）。

        不是真值，标注 method="rule"。
        """
        # 提取关键词（去掉停用词）
        stop_words = {
            "the", "and", "for", "with", "has", "have", "was", "were",
            "are", "not", "but", "from", "this", "that", "patient",
            "shows", "showed", "presented", "been", "were", "will",
            "would", "could", "should", "may", "might", "can",
        }
        claim_words = set(
            w.lower()
            for w in re.findall(r"\b\w{3,}\b", claim)
        ) - stop_words
        evidence_words = set(
            w.lower()
            for w in re.findall(r"\b\w{3,}\b", evidence)
        ) - stop_words

        if not claim_words:
            return CitationResult(
                claim_text=claim,
                evidence_chunk_id=chunk_id,
                verdict=CitationVerdict.UNSUPPORTED,
                method="rule",
                detail="no keywords in claim",
            )

        overlap = len(claim_words & evidence_words)
        ratio = overlap / len(claim_words)

        if ratio >= 0.6:
            verdict = CitationVerdict.SUPPORTED
        elif ratio >= 0.3:
            verdict = CitationVerdict.PARTIAL
        else:
            verdict = CitationVerdict.UNSUPPORTED

        return CitationResult(
            claim_text=claim,
            evidence_chunk_id=chunk_id,
            verdict=verdict,
            confidence=ratio,
            method="rule",
            detail=f"overlap_ratio={ratio:.4f} ({overlap}/{len(claim_words)})",
        )
