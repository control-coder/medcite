"""阶段 5 审核+合规测试。

覆盖:
1. CitationVerifier（规则判定 SUPPORTED/PARTIAL/UNSUPPORTED + 批量校验）
2. ClinicalLogicReviewer（完整性检查 + 弃权 + 升级阈值）
3. ComplianceGuard（绝对化拦截 + 免责声明 + 超范围拒答）
"""

from __future__ import annotations

import pytest

from medidiag.agents.base import AgentOutput, Claim, DiagnosisItem
from medidiag.compliance.guard import ComplianceGuard
from medidiag.review.citation import (
    CitationResult,
    CitationVerdict,
    CitationVerifier,
    JudgeInferenceError,
)
from medidiag.review.logic import ClinicalLogicReviewer

# ===== CitationVerifier 测试 =====


class TestCitationVerifier:
    @pytest.fixture
    def verifier(self) -> CitationVerifier:
        return CitationVerifier(
            model_name="test-judge",
            model_revision="test-revision",
            method="rule_fallback",
        )

    def test_rule_supported(self, verifier: CitationVerifier) -> None:
        """高重叠 → SUPPORTED。"""
        result = verifier.verify(
            "chest pain and myocardial infarction",
            "The patient has chest pain and myocardial infarction requiring treatment.",
        )
        assert result.verdict == CitationVerdict.SUPPORTED
        assert result.method == "rule_fallback"

    def test_rule_unsupported(self, verifier: CitationVerifier) -> None:
        """低重叠 → UNSUPPORTED。"""
        result = verifier.verify(
            "headache and dizziness",
            "The patient has abdominal pain and nausea with vomiting.",
        )
        assert result.verdict == CitationVerdict.UNSUPPORTED

    def test_rule_partial(self, verifier: CitationVerifier) -> None:
        """中等重叠 → PARTIAL。"""
        result = verifier.verify(
            "chest pain fever",
            "The patient presented with chest pain and cough.",
        )
        # "chest pain" 匹配，"fever" 不匹配 → 部分匹配
        assert result.verdict in [
            CitationVerdict.PARTIAL,
            CitationVerdict.SUPPORTED,
        ]

    def test_no_citation_unsupported(self, verifier: CitationVerifier) -> None:
        """无引用 claim → UNSUPPORTED。"""
        results = verifier.verify_batch(
            [{"text": "claim1", "citation_chunk_ids": []}], []
        )
        assert results[0].verdict == CitationVerdict.UNSUPPORTED
        assert "no citation" in results[0].detail

    def test_batch_verify(self, verifier: CitationVerifier) -> None:
        """批量校验。"""
        claims = [
            {"text": "chest pain cardiac", "citation_chunk_ids": ["c1"]},
            {"text": "headache neurological", "citation_chunk_ids": ["c2"]},
        ]
        chunks = [
            {"chunk_id": "c1", "text": "patient has chest pain and cardiac issues"},
            {"chunk_id": "c2", "text": "patient has headache and neurological symptoms"},
        ]
        results = verifier.verify_batch(claims, chunks)
        assert len(results) == 2
        assert results[0].verdict == CitationVerdict.SUPPORTED
        assert results[1].verdict == CitationVerdict.SUPPORTED

    def test_result_to_dict(self, verifier: CitationVerifier) -> None:
        """结果可序列化。"""
        result = verifier.verify("test claim", "test evidence")
        assert isinstance(result, CitationResult)
        assert isinstance(result.verdict, CitationVerdict)
        assert result.to_dict()["verdict"] in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}

    def test_batch_preserves_each_claim_citation_pair(
        self, verifier: CitationVerifier
    ) -> None:
        claims = [{
            "claim_id": "claim-1",
            "text": "cardiac chest pain",
            "citation_chunk_ids": ["c1", "c2"],
        }]
        chunks = [
            {"chunk_id": "c1", "text": "cardiac chest pain"},
            {"chunk_id": "c2", "text": "unrelated abdominal finding"},
        ]
        results = verifier.verify_batch(claims, chunks)
        assert len(results) == 2
        assert {result.evidence_chunk_id for result in results} == {"c1", "c2"}
        assert {result.claim_id for result in results} == {"claim-1"}

    def test_nli_inference_failure_does_not_fallback(self) -> None:
        verifier = CitationVerifier(
            model_name="test-judge",
            model_revision="immutable-revision",
            method="nli",
        )
        verifier._nli_pipeline = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(JudgeInferenceError, match="judge inference failed"):
            verifier.verify("claim", "evidence", "c1", "claim-1")


# ===== ClinicalLogicReviewer 测试 =====


class TestClinicalLogicReviewer:
    @pytest.fixture
    def reviewer(self) -> ClinicalLogicReviewer:
        return ClinicalLogicReviewer()

    def _make_output(self, **kwargs) -> AgentOutput:
        defaults = dict(
            specialty="cardiology",
            differential_diagnosis=[
                DiagnosisItem(diagnosis="MI", probability=0.9)
            ],
            claims=[
                Claim(text="claim1", citation_chunk_ids=["c1"], confidence=0.8)
            ],
            risk_flags=["risk1"],
            recommended_tests=["ECG"],
            uncertainty="some uncertainty",
        )
        defaults.update(kwargs)
        return AgentOutput(**defaults)

    def test_complete_output_approved(self, reviewer: ClinicalLogicReviewer) -> None:
        """完整输出 → APPROVED。"""
        result = reviewer.check(self._make_output())
        assert result.verdict == "APPROVED"
        assert result.is_approved

    def test_missing_risk_flags(self, reviewer: ClinicalLogicReviewer) -> None:
        """缺失风险提示 → REVISION_REQUIRED。"""
        result = reviewer.check(self._make_output(risk_flags=[]))
        assert result.verdict == "REVISION_REQUIRED"
        assert any("risk" in i for i in result.issues)

    def test_missing_uncertainty(self, reviewer: ClinicalLogicReviewer) -> None:
        """缺失不确定性 → REVISION_REQUIRED。"""
        result = reviewer.check(self._make_output(uncertainty=""))
        assert result.verdict == "REVISION_REQUIRED"

    def test_missing_recommended_tests(self, reviewer: ClinicalLogicReviewer) -> None:
        """缺失建议检查 → REVISION_REQUIRED。"""
        result = reviewer.check(self._make_output(recommended_tests=[]))
        assert result.verdict == "REVISION_REQUIRED"

    def test_abstain(self, reviewer: ClinicalLogicReviewer) -> None:
        """弃权 → REVISION_REQUIRED。"""
        result = reviewer.check(
            self._make_output(abstain=True, abstain_reason="no evidence")
        )
        assert result.verdict == "REVISION_REQUIRED"
        assert "abstained" in result.issues[0]

    def test_claims_without_citation(self, reviewer: ClinicalLogicReviewer) -> None:
        """claim 无引用 → REVISION_REQUIRED。"""
        result = reviewer.check(
            self._make_output(claims=[Claim(text="claim1", citation_chunk_ids=[])])
        )
        assert result.verdict == "REVISION_REQUIRED"
        assert any("citation" in i for i in result.issues)

    def test_no_diagnosis(self, reviewer: ClinicalLogicReviewer) -> None:
        """无诊断 → REVISION_REQUIRED。"""
        result = reviewer.check(self._make_output(differential_diagnosis=[]))
        assert result.verdict == "REVISION_REQUIRED"

    def test_should_escalate(self, reviewer: ClinicalLogicReviewer) -> None:
        """连续失败 3 次 → 升级。"""
        assert reviewer.should_escalate(2) is False
        assert reviewer.should_escalate(3) is True
        assert reviewer.should_escalate(5) is True

    def test_result_to_dict(self, reviewer: ClinicalLogicReviewer) -> None:
        """结果可序列化。"""
        result = reviewer.check(self._make_output())
        d = result.to_dict()
        assert "verdict" in d
        assert "issues" in d


# ===== ComplianceGuard 测试 =====


class TestComplianceGuard:
    @pytest.fixture
    def guard(self) -> ComplianceGuard:
        return ComplianceGuard()

    def test_absolute_term_blocked(self, guard: ComplianceGuard) -> None:
        """绝对化措辞拦截。"""
        result = guard.check("确诊为心肌梗死，保证治愈")
        assert result.blocked is True
        assert any("absolute_term" in r for r in result.block_reasons)
        assert "***" in result.sanitized_text

    def test_disclaimer_added(self, guard: ComplianceGuard) -> None:
        """强制免责声明。"""
        result = guard.check("patient has fever")
        assert result.disclaimer_added is True
        assert "仅供学习" in result.sanitized_text

    def test_disclaimer_not_duplicated(self, guard: ComplianceGuard) -> None:
        """已有免责声明不重复添加。"""
        text = "patient has fever\n仅供学习和工程演示，不构成医疗建议。"
        result = guard.check(text)
        assert result.disclaimer_added is False

    def test_out_of_scope(self, guard: ComplianceGuard) -> None:
        """超范围问题。"""
        result = guard.check("请推荐股票投资策略")
        assert result.out_of_scope is True
        assert result.blocked is True

    def test_normal_text_not_blocked(self, guard: ComplianceGuard) -> None:
        """正常文本不拦截。"""
        result = guard.check("patient has chest pain and fever")
        assert result.blocked is False
        assert result.disclaimer_added is True

    def test_reject_template(self, guard: ComplianceGuard) -> None:
        """拒答模板。"""
        template = guard.get_out_of_scope_reject_template()
        assert "超出" in template or "模拟范围" in template
        assert "仅供学习" in template

    def test_check_output_dict(self, guard: ComplianceGuard) -> None:
        """检查 Agent 输出字典。"""
        output_dict = {
            "differential_diagnosis": [{"diagnosis": "确诊心肌梗死"}],
            "claims": [{"text": "保证治愈"}],
            "risk_flags": [],
            "recommended_tests": ["ECG"],
            "uncertainty": "some",
        }
        result = guard.check_output(output_dict)
        assert result.blocked is True
        assert any("absolute_term" in r for r in result.block_reasons)

    def test_result_to_dict(self, guard: ComplianceGuard) -> None:
        """结果可序列化。"""
        result = guard.check("test")
        d = result.to_dict()
        assert "blocked" in d
        assert "disclaimer_added" in d


class _RecordingNliPipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return [
            {"label": "ENTAILMENT", "score": 0.91},
            {"label": "NEUTRAL", "score": 0.72},
        ]


def test_nli_verify_batch_uses_one_batched_call_and_preserves_order() -> None:
    """NLI 必须批量推理，并保留缺失引用与有效 pair 的原始顺序。"""
    verifier = CitationVerifier(
        model_name="test-judge",
        model_revision="immutable-revision",
        method="nli",
        device="cpu",
        batch_size=8,
    )
    fake = _RecordingNliPipeline()
    verifier._nli_pipeline = fake
    claims = [
        {"claim_id": "c-1", "text": "claim one", "citation_chunk_ids": ["e1"]},
        {"claim_id": "c-2", "text": "missing", "citation_chunk_ids": ["missing"]},
        {"claim_id": "c-3", "text": "claim three", "citation_chunk_ids": ["e2"]},
    ]
    chunks = [
        {"chunk_id": "e1", "text": "evidence " * 700},
        {"chunk_id": "e2", "text": "other evidence"},
    ]

    results = verifier.verify_batch(claims, chunks)

    assert [item.claim_id for item in results] == ["c-1", "c-2", "c-3"]
    assert [item.verdict for item in results] == [
        CitationVerdict.SUPPORTED,
        CitationVerdict.UNSUPPORTED,
        CitationVerdict.PARTIAL,
    ]
    assert len(fake.calls) == 1
    inputs, kwargs = fake.calls[0]
    assert len(inputs) == 2
    assert kwargs == {"truncation": True, "max_length": 512, "batch_size": 8}
