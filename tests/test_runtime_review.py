"""P5 运行时 citation、合规与 AssistantReport 收口测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select

from medidiag.compliance.status import ComplianceStatus
from medidiag.db.models import CaseReport, Citation
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.review.citation import CitationVerifier
from medidiag.review.runtime import RuntimeMedicalReview
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


class _FixedNliPipeline:
    """不下载模型的固定 NLI 测试 double。"""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.calls: list[Any] = []
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
            )
        )

    def __call__(self, inputs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((inputs, kwargs))
        size = len(inputs) if isinstance(inputs, list) else 1
        labels = self.labels[:size]
        if len(labels) != size:
            raise AssertionError("测试标签数量不足")
        return [
            {"label": label, "score": 0.95 - index * 0.05}
            for index, label in enumerate(labels)
        ]


def _runtime(*labels: str) -> RuntimeMedicalReview:
    verifier = CitationVerifier(
        model_name="fixed-nli-test",
        model_revision="immutable-test-revision",
        method="nli",
        device="cpu",
        input_language="en",
    )
    verifier._nli_pipeline = _FixedNliPipeline(list(labels))
    return RuntimeMedicalReview(verifier)


def _retrieval() -> dict[str, Any]:
    return {
        "query": "deidentified simulated chest pain question",
        "evidence_bundle_id": "bundle-p5",
        "chunks": [
            {
                "chunk_id": "e1",
                "source": "public_source",
                "source_id": "source-1",
                "chunk_hash": "hash-e1",
                "evidence_level": "level_2_review",
                "text": "Chest pain can require clinical assessment and evidence review.",
            },
            {
                "chunk_id": "e2",
                "source": "public_source",
                "source_id": "source-2",
                "chunk_hash": "hash-e2",
                "evidence_level": "level_3_observational",
                "text": "A headache can have multiple causes and needs contextual evaluation.",
            },
            {
                "chunk_id": "e3",
                "source": "public_source",
                "source_id": "source-3",
                "chunk_hash": "hash-e3",
                "evidence_level": "level_5_other",
                "text": "This evidence is unrelated to the asserted treatment conclusion.",
            },
        ],
    }


def _generation(texts: list[str] | None = None) -> dict[str, Any]:
    values = texts or [
        "Chest pain can require assessment by a qualified clinician.",
        "Headache findings require cautious contextual interpretation.",
        "This treatment conclusion is not established by the cited evidence.",
    ]
    claims = [
        {
            "claim_id": f"claim-{index}",
            "text": text,
            "citation_chunk_ids": [f"e{index}"],
            "confidence": 0.8,
            "specialty": "cardiology",
            "agent_run_id": "agent-1",
        }
        for index, text in enumerate(values, start=1)
    ]
    return {
        "claims": claims,
        "claims_before_arbitration": claims,
        "evidence_hash": "evidence-hash-p5",
        "risk_flags": ["qualified clinician review required"],
        "recommended_tests": ["clinical assessment"],
        "uncertainty": "The available evidence does not establish a diagnosis.",
    }


def _arbitration(generation: dict[str, Any]) -> dict[str, Any]:
    claims = generation["claims"]
    return {
        "selected_claim_ids": [item["claim_id"] for item in claims],
        "claims_after_arbitration": claims,
    }


def test_runtime_review_rejects_rule_fallback_config() -> None:
    with pytest.raises(MediDiagError, match="CITATION_REVIEW_INVALID"):
        RuntimeMedicalReview.from_config(
            {
                "judge": {
                    "model": "fixed-nli",
                    "revision": "immutable",
                    "method": "rule_fallback",
                    "input_language": "en",
                }
            }
        )


def test_canonical_claim_rejects_chinese_before_nli() -> None:
    runtime = _runtime("ENTAILMENT")
    generation = _generation(["该 claim 不得进入英文 NLI judge。"])
    with pytest.raises(MediDiagError, match="CLAIM_LANGUAGE_INVALID"):
        runtime.review(generation, _arbitration(generation), _retrieval())


def test_fixed_nli_keeps_pair_verdicts_and_filters_non_supported_claims() -> None:
    runtime = _runtime("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
    generation = _generation()
    review = runtime.review(generation, _arbitration(generation), _retrieval())

    assert review["verdict"] == "APPROVED"
    assert review["compliance_status"] == ComplianceStatus.PASSED_FIXED_NLI.value
    assert [item["verdict"] for item in review["citation_verdicts"]] == [
        "SUPPORTED",
        "PARTIAL",
        "UNSUPPORTED",
    ]
    assert review["approved_claim_ids"] == ["claim-1"]
    assert review["filtered_claim_ids"] == ["claim-2", "claim-3"]
    assert review["judge"] == {
        "method": "nli",
        "model": "fixed-nli-test",
        "revision": "immutable-test-revision",
        "input_language": "en",
        "device": "cpu",
    }

    report = runtime.report("case-p5", generation, review)
    assert report["schema_version"] == "assistant-report-v1"
    assert [item["claim_id"] for item in report["claims"]] == ["claim-1"]
    assert report["claims"][0]["canonical_text_en"] == generation["claims"][0]["text"]
    assert report["claims"][0]["citation_verdict"] == "SUPPORTED"
    assert "claim-2" not in str(report)
    assert "not established" not in str(report)
    assert report["provenance"]["claim_text_policy"] == (
        "exact_canonical_english_no_translation"
    )


def test_no_supported_claim_requires_revision_and_cannot_report() -> None:
    runtime = _runtime("NEUTRAL", "CONTRADICTION", "CONTRADICTION")
    generation = _generation()
    review = runtime.review(generation, _arbitration(generation), _retrieval())
    assert review["verdict"] == "REVISION_REQUIRED"
    assert review["approved_claim_ids"] == []
    with pytest.raises(MediDiagError, match="REPORT_PROVENANCE_INVALID"):
        runtime.report("case-p5", generation, review)


def test_compliance_hit_escalates_even_when_nli_supports_claim() -> None:
    runtime = _runtime("ENTAILMENT")
    generation = _generation(["This evidence promises a guaranteed cure."])
    review = runtime.review(generation, _arbitration(generation), _retrieval())
    assert review["verdict"] == "ESCALATED"
    assert review["compliance_status"] == ComplianceStatus.BLOCKED.value
    assert any("absolute_term_blocked" in issue for issue in review["issues"])


def test_report_postcheck_rejects_claim_text_rewrite() -> None:
    runtime = _runtime("ENTAILMENT")
    generation = _generation(["Chest pain can require assessment by a qualified clinician."])
    review = runtime.review(generation, _arbitration(generation), _retrieval())
    review["claim_decisions"][0]["text"] = "Rewritten unreviewed claim."
    with pytest.raises(MediDiagError, match="REPORT_PROVENANCE_INVALID"):
        runtime.report("case-p5", generation, review)


class _P5WorkflowProvider(DeterministicWorkflowProvider):
    """把 P5 runtime 接入既有 worker 的无网络集成 provider。"""

    version = "p5-workflow-test-v1"

    def __init__(self, runtime: RuntimeMedicalReview) -> None:
        super().__init__()
        self.runtime = runtime

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        del normalized_query
        return _retrieval()

    def generate(
        self, question: str, retrieval: dict[str, Any], plan: dict[str, Any]
    ) -> dict[str, Any]:
        del question, retrieval, plan
        generation = _generation(
            ["Chest pain can require assessment by a qualified clinician."]
        )
        generation["agents"] = [
            {
                "agent_name": "p5_test_agent",
                "status": "SUCCEEDED",
                "claims": generation["claims"],
            }
        ]
        return generation

    def arbitrate(
        self, generation: dict[str, Any], retrieval: dict[str, Any]
    ) -> dict[str, Any]:
        del retrieval
        result = _arbitration(generation)
        result["verdict"] = "SINGLE_AGENT_BASELINE"
        return result

    def review(
        self,
        generation: dict[str, Any],
        arbitration: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        return self.runtime.review(generation, arbitration, retrieval)

    def report(
        self, case_id: str, generation: dict[str, Any], review: dict[str, Any]
    ) -> dict[str, Any]:
        return self.runtime.report(case_id, generation, review)


def test_worker_persists_nli_citation_and_strong_assistant_report(tmp_path) -> None:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'p5-worker.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session,
            "Deidentified simulated case for P5 runtime review tests.",
            "p5-case",
            "test-scope",
        )
        executor.start_workflow(
            session,
            case.case_id,
            "case_workflow",
            "p5-runtime-review",
            "input-hash",
        )
        case_id = case.case_id
    result = SingleMachineWorker(
        factory,
        _P5WorkflowProvider(_runtime("ENTAILMENT")),
        worker_id="p5-worker",
    ).run_once()

    assert result.final_state == "CLOSED_SUCCESS"
    with factory() as session:
        citation = session.execute(
            select(Citation).where(Citation.case_id == case_id)
        ).scalar_one()
        report = session.execute(
            select(CaseReport).where(CaseReport.case_id == case_id)
        ).scalar_one()
        assert citation.verdict == "SUPPORTED"
        assert citation.verifier_model == "fixed-nli-test@immutable-test-revision"
        assert report.structured_report["schema_version"] == "assistant-report-v1"
        assert report.structured_report["claims"][0]["citation_verdict"] == "SUPPORTED"
        assert report.compliance_status == ComplianceStatus.PASSED_FIXED_NLI.value
    engine.dispose()
