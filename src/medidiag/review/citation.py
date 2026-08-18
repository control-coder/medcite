"""Claim-citation verification with explicit formal/development behavior."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from medidiag.acceleration import resolve_torch_device
from medidiag.schemas import KnowledgeChunk


class CitationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class JudgeInitializationError(RuntimeError):
    """Raised when the configured formal judge cannot be loaded."""


class JudgeInferenceError(RuntimeError):
    """Raised when formal judge inference fails."""


class JudgeInputLanguageError(RuntimeError):
    """固定语言 NLI judge 收到不兼容的输入语言时抛出。"""


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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


NLI_MAX_LENGTH = 512
# 汉字命中表示输入包含中文；在英文 NLI judge 下这会造成中英混用 pair。
_HAN_CHARACTER_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# 固定 NLI judge 必须暴露的三分类语义。只接受可识别为这三类的标签集，
# 拒绝 transformers 在 config 缺少显式标签时使用的 LABEL_0/1/2 默认值。
NLI_CANONICAL_LABELS = ("entailment", "neutral", "contradiction")

_NLI_VERDICT_BY_LABEL = {
    "entailment": CitationVerdict.SUPPORTED,
    "neutral": CitationVerdict.PARTIAL,
    "contradiction": CitationVerdict.UNSUPPORTED,
}


def canonical_nli_label(label: str) -> str | None:
    """把 provider 标签归一为三分类之一，无法识别时返回 None。

    只做前缀识别（``ENTAILMENT`` / ``entail`` / ``CONTRADICTION`` 等大小写与
    连字符变体），不做包含匹配：``LABEL_0``、``not_entailment`` 这类标签必须
    识别失败，否则会把未知语义静默折叠成 ``UNSUPPORTED``。
    """
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    for prefix, canonical in (
        ("entail", "entailment"),
        ("neutral", "neutral"),
        ("contradict", "contradiction"),
    ):
        if normalized.startswith(prefix):
            return canonical
    return None


def validate_nli_label_set(id2label: object, judge_ref: str) -> dict[int, str]:
    """校验 judge 暴露的 ``id2label`` 是完整的三分类 NLI 标签集。

    Raises:
        JudgeInitializationError: 标签缺失、数量不为 3 或无法识别为
            entailment / neutral / contradiction。
    """
    if not isinstance(id2label, dict) or not id2label:
        raise JudgeInitializationError(
            f"judge {judge_ref} does not expose an id2label mapping; "
            "a fixed NLI judge must declare its label set"
        )
    canonical = {}
    for key, raw_label in id2label.items():
        label = canonical_nli_label(raw_label)
        if label is None:
            raise JudgeInitializationError(
                f"judge {judge_ref} exposes unrecognised NLI label {raw_label!r}; "
                f"expected labels resolvable to {NLI_CANONICAL_LABELS}"
            )
        canonical[key] = label
    if set(canonical.values()) != set(NLI_CANONICAL_LABELS):
        raise JudgeInitializationError(
            f"judge {judge_ref} exposes label set {sorted(set(canonical.values()))}; "
            f"expected exactly {sorted(NLI_CANONICAL_LABELS)}"
        )
    return canonical


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
        device: str = "auto",
        batch_size: int = 32,
        input_language: str = "en",
    ) -> None:
        if method not in {"nli", "rule_fallback"}:
            raise ValueError("method must be 'nli' or 'rule_fallback'")
        if input_language != "en":
            raise ValueError("input_language 目前只支持 'en'")
        self.model_name = model_name
        self.model_revision = model_revision
        self.method = method
        self.input_language = input_language
        self.requested_device = device
        self.actual_device = resolve_torch_device(device)
        self.batch_size = max(1, int(batch_size))
        self._nli_pipeline: Any = None

    def initialize(self) -> None:
        """Eagerly load the formal judge so a run fails before producing output."""
        if self.method == "rule_fallback" or self._nli_pipeline is not None:
            return
        try:
            loaded = self._load_pipeline()
        except Exception as exc:
            raise JudgeInitializationError(
                f"failed to load judge {self.judge_ref}: {exc}"
            ) from exc
        # 标签集校验必须在任何 claim 被判定之前完成：若模型只暴露
        # LABEL_0/1/2，子串映射会把全部 claim 静默判为 UNSUPPORTED，
        # 产出一个看似合法的 Citation Precision。
        validate_nli_label_set(self._pipeline_id2label(loaded), self.judge_ref)
        self._nli_pipeline = loaded

    @property
    def judge_ref(self) -> str:
        return f"{self.model_name}@{self.model_revision}"

    def _load_pipeline(self) -> object:
        """Load the fixed judge. Overridden in tests to avoid a model download."""
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model=self.model_name,
            revision=self.model_revision,
            device=0 if self.actual_device == "cuda" else -1,
        )

    @staticmethod
    def _pipeline_id2label(loaded: object) -> object:
        config = getattr(getattr(loaded, "model", None), "config", None)
        return getattr(config, "id2label", None)

    def _validate_nli_input_language(
        self,
        claim_text: str,
        evidence_text: str,
        claim_id: str,
        evidence_chunk_id: str,
    ) -> None:
        """在执行 NLI 前拒绝与固定英文 judge 不兼容的文本 pair。"""
        fields_with_han = [
            field_name
            for field_name, value in (
                ("claim_text", claim_text),
                ("evidence_text", evidence_text),
            )
            if _HAN_CHARACTER_PATTERN.search(value)
        ]
        if fields_with_han:
            raise JudgeInputLanguageError(
                "NLI_JUDGE_LANGUAGE_MISMATCH: "
                f"judge_input_language={self.input_language}; "
                f"fields_with_han={','.join(fields_with_han)}; "
                f"claim_id={claim_id}; evidence_chunk_id={evidence_chunk_id}. "
                "英文 NLI judge 不接受包含中文或中英混用的 claim/evidence pair"
            )

    def verify(
        self,
        claim_text: str,
        evidence_text: str,
        evidence_chunk_id: str = "",
        claim_id: str = "",
    ) -> CitationResult:
        if self.method == "nli":
            self._validate_nli_input_language(
                claim_text, evidence_text, claim_id, evidence_chunk_id
            )
            self.initialize()
            return self._verify_nli(claim_id, claim_text, evidence_text, evidence_chunk_id)
        return self._verify_rule(claim_id, claim_text, evidence_text, evidence_chunk_id)

    def verify_batch(
        self,
        claims: list[dict[str, Any]],
        evidence_chunks: list[KnowledgeChunk] | list[dict[str, Any]],
    ) -> list[CitationResult]:
        """按原始 claim-citation 顺序返回结果；NLI 路径使用真正批推理。"""
        chunk_map: dict[str, str] = {}
        for chunk in evidence_chunks:
            if isinstance(chunk, dict):
                chunk_map[str(chunk.get("chunk_id", ""))] = str(chunk.get("text", ""))
            else:
                chunk_map[chunk.chunk_id] = chunk.text

        ordered: list[CitationResult | tuple[str, str, str, str]] = []
        pending_inputs: list[dict[str, str]] = []
        pending_meta: list[tuple[str, str, str]] = []
        for index, claim in enumerate(claims):
            claim_id = str(claim.get("claim_id") or f"claim_{index:04d}")
            text = str(claim.get("text", ""))
            citation_ids = [str(value) for value in claim.get("citation_chunk_ids", [])]
            if not citation_ids:
                ordered.append(
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
                    ordered.append(
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

                if self.method == "nli":
                    self._validate_nli_input_language(
                        text, evidence_text, claim_id, chunk_id
                    )
                    marker = (claim_id, text, chunk_id, evidence_text)
                    ordered.append(marker)
                    pending_meta.append((claim_id, text, chunk_id))
                    pending_inputs.append({"text": evidence_text, "text_pair": text})
                else:
                    ordered.append(self._verify_rule(claim_id, text, evidence_text, chunk_id))

        if self.method != "nli" or not pending_inputs:
            return [item for item in ordered if isinstance(item, CitationResult)]

        self.initialize()
        try:
            raw_outputs = self._nli_pipeline(
                pending_inputs,
                truncation=True,
                max_length=NLI_MAX_LENGTH,
                batch_size=self.batch_size,
            )
            if not isinstance(raw_outputs, list) or len(raw_outputs) != len(pending_meta):
                raise ValueError(
                    f"judge returned {len(raw_outputs) if isinstance(raw_outputs, list) else 'non-list'} "
                    f"outputs for {len(pending_meta)} inputs"
                )
        except Exception as exc:
            raise JudgeInferenceError(f"judge batch inference failed: {exc}") from exc

        batch_results = iter(
            self._citation_result_from_nli_output(meta, output)
            for meta, output in zip(pending_meta, raw_outputs, strict=True)
        )
        final: list[CitationResult] = []
        for item in ordered:
            final.append(item if isinstance(item, CitationResult) else next(batch_results))
        return final

    def _citation_result_from_nli_output(
        self,
        meta: tuple[str, str, str],
        output: object,
    ) -> CitationResult:
        claim_id, claim, chunk_id = meta
        result = output[0] if isinstance(output, list) else output
        if not isinstance(result, dict):
            raise JudgeInferenceError(
                f"judge output is not a mapping for claim={claim_id}, chunk={chunk_id}"
            )
        label = str(result.get("label", ""))
        score = float(result.get("score", 0.0))
        canonical = canonical_nli_label(label)
        if canonical is None:
            # fail-closed：未知标签不得折叠为 UNSUPPORTED，否则判定结果不可解释。
            raise JudgeInferenceError(
                f"judge {self.judge_ref} returned unrecognised label {label!r} "
                f"for claim={claim_id}, chunk={chunk_id}"
            )
        verdict = _NLI_VERDICT_BY_LABEL[canonical]
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

    def _verify_nli(
        self, claim_id: str, claim: str, evidence: str, chunk_id: str
    ) -> CitationResult:
        try:
            output = self._nli_pipeline(
                {"text": evidence, "text_pair": claim},
                truncation=True,
                max_length=NLI_MAX_LENGTH,
            )
            return self._citation_result_from_nli_output(
                (claim_id, claim, chunk_id), output
            )
        except JudgeInferenceError:
            raise
        except Exception as exc:
            raise JudgeInferenceError(
                f"judge inference failed for claim={claim_id}, chunk={chunk_id}: {exc}"
            ) from exc

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
