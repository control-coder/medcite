"""Claim-citation verification with explicit formal/development behavior."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

from medidiag.schemas import KnowledgeChunk


class CitationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class JudgeInitializationError(RuntimeError):
    """Raised when the configured formal judge cannot be loaded."""


class JudgeInferenceError(RuntimeError):
    """Raised when formal judge inference fails."""


@dataclass
class CitationResult:
    """Auditable verdict for one claim-citation pair (or an uncited claim)."""

    claim_id: str
    claim_text: str
    evidence_chunk_id: str
    verdict: CitationVerdict
    confidence: float = 0.0
    method: str = "rule_fallback"
    model_name: str = ""
    model_revision: str = ""
    detail: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


class CitationVerifier:
    """Verify citations using a fixed NLI judge or an explicit dev fallback.

    ``method=nli`` is fail-closed: loading or inference errors raise and abort
    the run. ``method=rule_fallback`` never claims to be NLI and is allowed
    only by the evaluation configuration's development mode.
    """

    def __init__(
        self,
        model_name: str,
        model_revision: str,
        method: str = "nli",
    ) -> None:
        if method not in {"nli", "rule_fallback"}:
            raise ValueError("method must be 'nli' or 'rule_fallback'")
        self.model_name = model_name
        self.model_revision = model_revision
        self.method = method
        self._nli_pipeline = None

    def initialize(self) -> None:
        """Eagerly load the formal judge so a run fails before producing output."""
        if self.method == "rule_fallback" or self._nli_pipeline is not None:
            return
        try:
            from transformers import pipeline

            self._nli_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                revision=self.model_revision,
            )
        except Exception as exc:
            raise JudgeInitializationError(
                f"failed to load judge {self.model_name}@{self.model_revision}: {exc}"
            ) from exc

    def verify(
        self,
        claim_text: str,
        evidence_text: str,
        evidence_chunk_id: str = "",
        claim_id: str = "",
    ) -> CitationResult:
        if self.method == "nli":
            self.initialize()
            return self._verify_nli(claim_id, claim_text, evidence_text, evidence_chunk_id)
        return self._verify_rule(claim_id, claim_text, evidence_text, evidence_chunk_id)

    def verify_batch(
        self,
        claims: list[dict],
        evidence_chunks: list[KnowledgeChunk] | list[dict],
    ) -> list[CitationResult]:
        """Return one result per emitted claim-citation pair.

        An uncited claim produces one ``UNSUPPORTED`` record with an empty
        ``evidence_chunk_id`` so claim-level metrics retain the claim.
        """
        chunk_map: dict[str, str] = {}
        for chunk in evidence_chunks:
            if isinstance(chunk, dict):
                chunk_map[str(chunk.get("chunk_id", ""))] = str(chunk.get("text", ""))
            else:
                chunk_map[chunk.chunk_id] = chunk.text

        results: list[CitationResult] = []
        for index, claim in enumerate(claims):
            claim_id = str(claim.get("claim_id") or f"claim_{index:04d}")
            text = str(claim.get("text", ""))
            citation_ids = [str(value) for value in claim.get("citation_chunk_ids", [])]
            if not citation_ids:
                results.append(
                    CitationResult(
                        claim_id=claim_id,
                        claim_text=text,
                        evidence_chunk_id="",
                        verdict=CitationVerdict.UNSUPPORTED,
                        method=self.method,
                        model_name=self.model_name if self.method == "nli" else "",
                        model_revision=self.model_revision if self.method == "nli" else "",
                        detail="no citation",
                    )
                )
                continue

            for chunk_id in citation_ids:
                evidence_text = chunk_map.get(chunk_id)
                if not evidence_text:
                    results.append(
                        CitationResult(
                            claim_id=claim_id,
                            claim_text=text,
                            evidence_chunk_id=chunk_id,
                            verdict=CitationVerdict.UNSUPPORTED,
                            method=self.method,
                            model_name=self.model_name if self.method == "nli" else "",
                            model_revision=self.model_revision if self.method == "nli" else "",
                            detail="citation chunk not found",
                            error="CITATION_CHUNK_NOT_FOUND",
                        )
                    )
                    continue
                results.append(self.verify(text, evidence_text, chunk_id, claim_id))
        return results

    def _verify_nli(
        self, claim_id: str, claim: str, evidence: str, chunk_id: str
    ) -> CitationResult:
        try:
            output = self._nli_pipeline({"text": evidence, "text_pair": claim})
            result = output[0] if isinstance(output, list) else output
            label = str(result["label"]).upper()
            score = float(result["score"])
        except Exception as exc:
            raise JudgeInferenceError(
                f"judge inference failed for claim={claim_id}, chunk={chunk_id}: {exc}"
            ) from exc

        if "ENTAIL" in label:
            verdict = CitationVerdict.SUPPORTED
        elif "NEUTRAL" in label:
            verdict = CitationVerdict.PARTIAL
        else:
            verdict = CitationVerdict.UNSUPPORTED
        return CitationResult(
            claim_id=claim_id,
            claim_text=claim,
            evidence_chunk_id=chunk_id,
            verdict=verdict,
            confidence=score,
            method="nli",
            model_name=self.model_name,
            model_revision=self.model_revision,
            detail=f"label={label}, score={score:.4f}",
        )

    def _verify_rule(
        self, claim_id: str, claim: str, evidence: str, chunk_id: str
    ) -> CitationResult:
        stop_words = {
            "the", "and", "for", "with", "has", "have", "was", "were",
            "are", "not", "but", "from", "this", "that", "patient",
            "shows", "showed", "presented", "been", "will", "would",
            "could", "should", "may", "might", "can",
        }
        claim_words = {
            word.lower() for word in re.findall(r"\b\w{3,}\b", claim)
        } - stop_words
        evidence_words = {
            word.lower() for word in re.findall(r"\b\w{3,}\b", evidence)
        } - stop_words
        if not claim_words:
            return CitationResult(
                claim_id=claim_id,
                claim_text=claim,
                evidence_chunk_id=chunk_id,
                verdict=CitationVerdict.UNSUPPORTED,
                method="rule_fallback",
                detail="no keywords in claim",
            )

        overlap = len(claim_words & evidence_words)
        ratio = overlap / len(claim_words)
        verdict = (
            CitationVerdict.SUPPORTED
            if ratio >= 0.6
            else CitationVerdict.PARTIAL
            if ratio >= 0.3
            else CitationVerdict.UNSUPPORTED
        )
        return CitationResult(
            claim_id=claim_id,
            claim_text=claim,
            evidence_chunk_id=chunk_id,
            verdict=verdict,
            confidence=ratio,
            method="rule_fallback",
            detail=f"overlap_ratio={ratio:.4f} ({overlap}/{len(claim_words)})",
        )
