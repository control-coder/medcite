"""P4 运行时单/双专科 Agent 编排与受约束仲裁测试。"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from sqlalchemy import select

from medidiag.agents.runtime import AgentTopologyConfig, RuntimeMedicalAgents
from medidiag.db.models import AgentRun, StageArtifact
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.llm import LLMRequest, ProviderCapabilities, ProviderResult
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.openai_provider import OpenAICompatibleWorkflowProvider
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


def _retrieval() -> dict[str, Any]:
    chunks = [
        {
            "chunk_id": "ev_cardiac",
            "source": "public_fixture",
            "source_id": "article-1",
            "text": "Chest pain can require cardiovascular assessment and cautious differential review.",
            "evidence_level": "level_2_review",
            "metadata": {},
        },
        {
            "chunk_id": "ev_resp",
            "source": "public_fixture",
            "source_id": "article-2",
            "text": "Dyspnea can require respiratory assessment and uncertainty reporting.",
            "evidence_level": "level_1_guideline",
            "metadata": {},
        },
    ]
    return {
        "query": "chest pain and dyspnea",
        "evidence_bundle_id": "eb_test_v1",
        "evidence_bundle": {
            "schema_version": "evidence-bundle-v1",
            "bundle_id": "eb_test_v1",
            "corpus_version": "test-public-v1",
            "retrieval_config_hash": "retrieval-hash-v1",
            "evidence": chunks,
        },
        "chunks": chunks,
    }


class RecordingProvider:
    provider_id = "test-provider"
    profile_id = "test-profile"
    model = "test-model"
    is_configured = True

    def __init__(
        self,
        *,
        unknown_citation: bool = False,
        unknown_claim: bool = False,
        unknown_claim_once: bool = False,
        chinese_claim: bool = False,
    ) -> None:
        self.unknown_citation = unknown_citation
        self.unknown_claim = unknown_claim
        self.unknown_claim_once = unknown_claim_once
        self.chinese_claim = chinese_claim
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(structured_output=True, response_id=True)

    def generate(
        self,
        request: LLMRequest,
        *,
        timeout_s: float,
        idempotency_key: str,
    ) -> ProviderResult:
        del timeout_s
        with self._lock:
            self.requests.append(request)
            call_number = len(self.requests)
            arbitration_call_number = sum(
                item.prompt_version == "arbitrator-agent-v1" for item in self.requests
            )
        if request.prompt_version == "arbitrator-agent-v1":
            user = json.loads(str(request.messages[-1]["content"]))
            invalid_selection = self.unknown_claim or (
                self.unknown_claim_once and arbitration_call_number == 1
            )
            selected = ["unknown_claim"] if invalid_selection else user["allowed_claim_ids"][:1]
            payload = {
                "selected_claim_ids": selected,
                "conflicts": [{"type": "specialty_difference"}],
                "verdict": "REVISION_REQUIRED",
                "reason": "保留证据约束更强的既有 claim，仍需人工复核。",
            }
        else:
            specialty = _specialty_from_prompt(str(request.messages[-1]["content"]))
            citation = (
                "unknown_evidence"
                if self.unknown_citation
                else ("ev_resp" if specialty == "respiratory" else "ev_cardiac")
            )
            payload = {
                "specialty": specialty,
                "differential_diagnosis": [
                    {
                        "diagnosis": f"{specialty}_possibility",
                        "probability": 0.4,
                        "supporting_claim_indices": [0],
                    }
                ],
                "claims": [
                    {
                        "text": (
                            "经胃内镜检查提示需要谨慎评估。"
                            if self.chinese_claim
                            else f"{specialty} evidence indicates a cautious assessment is warranted."
                        ),
                        "citation_chunk_ids": [citation],
                        "confidence": 0.6,
                    }
                ],
                "risk_flags": ["qualified_clinician_review_required"],
                "missing_info": [],
                "recommended_tests": [],
                "uncertainty": "公开证据不足以形成确定性医学结论。",
                "abstain": False,
                "abstain_reason": "",
            }
        content = json.dumps(payload, ensure_ascii=False)
        return ProviderResult(
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            model=self.model,
            response_id=f"resp-{call_number}",
            system_fingerprint=None,
            content=content,
            parsed_json=payload,
            tool_calls=[],
            reasoning_content=None,
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
            raw_error_code=None,
            latency_ms=3,
            retry_count=0,
            provenance_mode="provider_response_id",
        )


def _specialty_from_prompt(prompt: str) -> str:
    for specialty in ("cardiology", "respiratory", "general_diagnosis", "evidence_skeptic"):
        if (
            f"专科: {specialty}" in prompt
            or f"专科标识: {specialty}" in prompt
            or f"specialty={specialty}" in prompt
            or f"你是 {specialty} 专科" in prompt
        ):
            return specialty
    return "general_diagnosis"


def test_single_topology_has_one_auditable_agent_and_no_arbitration_call() -> None:
    provider = RecordingProvider()
    runtime = RuntimeMedicalAgents(
        provider,
        config=AgentTopologyConfig(topology="single"),
    )
    generated = runtime.generate("chest pain", _retrieval(), {"objective": "test"})

    assert generated["topology"] == "single"
    assert len(generated["agents"]) == 1
    assert generated["agents"][0]["specialty"] == "general_diagnosis"
    assert generated["agents"][0]["provider_provenance"]["response_id"] == "resp-1"
    arbitration = runtime.arbitrate(generated, _retrieval())
    assert arbitration["arbitration_method"] == "not_applicable"
    assert arbitration["selected_claim_ids"] == arbitration["before_claim_ids"]
    assert len(provider.requests) == 1



def test_non_english_claim_is_discarded_as_auditable_abstention() -> None:
    provider = RecordingProvider(chinese_claim=True)
    runtime = RuntimeMedicalAgents(
        provider,
        config=AgentTopologyConfig(topology="single"),
    )

    generated = runtime.generate("chest pain", _retrieval(), {"objective": "test"})
    agent = generated["agents"][0]

    assert generated["claims"] == []
    assert generated["all_agents_abstained"] is True
    assert agent["output"]["abstain"] is True
    assert agent["output"]["abstain_reason"] == "claim_language_invalid"
    assert agent["provider_provenance"]["response_id"] == "resp-1"
    assert any(
        action.startswith("claims:discard_non_english_and_abstain")
        for action in agent["normalization_actions"]
    )
    arbitration = runtime.arbitrate(generated, _retrieval())
    assert arbitration["arbitration_method"] == "abstention_no_claims"
    assert len(provider.requests) == 1

def test_fixed_pair_runs_two_agents_with_shared_evidence_and_independent_inputs() -> None:
    provider = RecordingProvider()
    runtime = RuntimeMedicalAgents(
        provider,
        config=AgentTopologyConfig(topology="fixed_pair"),
    )
    generated = runtime.generate("chest pain and dyspnea", _retrieval(), {"objective": "test"})

    agents = generated["agents"]
    assert [item["specialty"] for item in agents] == ["cardiology", "respiratory"]
    assert len({item["agent_run_id"] for item in agents}) == 2
    assert len({item["input_hash"] for item in agents}) == 2
    assert {item["evidence_bundle_id"] for item in agents} == {"eb_test_v1"}
    assert {item["evidence_hash"] for item in agents} == {generated["evidence_hash"]}
    assert all(item["prompt_version"] == "specialist-agent-v1" for item in agents)

    arbitration = runtime.arbitrate(generated, _retrieval())
    assert arbitration["arbitration_method"] == "constrained_llm"
    assert set(arbitration["selected_claim_ids"]).issubset(arbitration["before_claim_ids"])
    assert arbitration["rule_baseline"]["verdict"] in {"APPROVED", "REVISION_REQUIRED", "ESCALATED"}


def test_fixed_pair_calls_reach_concurrency_barrier() -> None:
    barrier = threading.Barrier(2, timeout=2)

    class BarrierProvider(RecordingProvider):
        def generate(
            self, request: LLMRequest, *, timeout_s: float, idempotency_key: str
        ) -> ProviderResult:
            if request.prompt_version == "specialist-agent-v1":
                barrier.wait()
            return super().generate(request, timeout_s=timeout_s, idempotency_key=idempotency_key)

    runtime = RuntimeMedicalAgents(
        BarrierProvider(),
        config=AgentTopologyConfig(topology="fixed_pair"),
    )
    generated = runtime.generate("chest pain and dyspnea", _retrieval(), {})
    assert len(generated["agents"]) == 2


def test_dynamic_pair_records_routing_reason_scores_and_fallback() -> None:
    runtime = RuntimeMedicalAgents(
        RecordingProvider(),
        config=AgentTopologyConfig(topology="dynamic_pair"),
    )
    generated = runtime.generate("unmatched public simulation", _retrieval(), {})
    routing = generated["routing"]
    assert routing["reason"]
    assert "scores" in routing
    assert isinstance(routing["is_fallback"], bool)
    assert all(item["routing_reason"] == routing["reason"] for item in generated["agents"])


def test_unknown_evidence_id_becomes_auditable_agent_abstention() -> None:
    provider = RecordingProvider(unknown_citation=True)
    runtime = RuntimeMedicalAgents(
        provider,
        config=AgentTopologyConfig(topology="single"),
    )

    generated = runtime.generate("chest pain", _retrieval(), {})
    agent = generated["agents"][0]

    assert generated["claims"] == []
    assert generated["all_agents_abstained"] is True
    assert agent["output"]["abstain_reason"] == "claim_citation_invalid"
    assert agent["provider_provenance"]["response_id"] == "resp-1"
    assert any(
        action.startswith("claims:discard_invalid_citations_and_abstain")
        for action in agent["normalization_actions"]
    )
    arbitration = runtime.arbitrate(generated, _retrieval())
    assert arbitration["arbitration_method"] == "abstention_no_claims"
    assert len(provider.requests) == 1


def test_arbitration_contract_violation_gets_one_controlled_retry() -> None:
    provider = RecordingProvider(unknown_claim_once=True)
    runtime = RuntimeMedicalAgents(
        provider,
        config=AgentTopologyConfig(
            topology="fixed_pair",
            allow_arbitration_fallback=False,
        ),
    )

    generated = runtime.generate("chest pain and dyspnea", _retrieval(), {})
    arbitration = runtime.arbitrate(generated, _retrieval())

    assert arbitration["fallback_used"] is False
    assert arbitration["arbitration_method"] == "constrained_llm"
    assert arbitration["stage_provider_request_id"] == "resp-4"
    assert [
        item["response_id"] for item in arbitration["provider_attempt_provenance"]
    ] == ["resp-3", "resp-4"]
    assert arbitration["constrained_arbitration"]["normalization_actions"][0] == (
        "provider_contract_retry:1"
    )
    assert len(provider.requests) == 4


def test_unknown_claim_from_arbitrator_falls_back_to_rule_baseline() -> None:
    runtime = RuntimeMedicalAgents(
        RecordingProvider(unknown_claim=True),
        config=AgentTopologyConfig(topology="fixed_pair"),
    )
    generated = runtime.generate("chest pain and dyspnea", _retrieval(), {})
    arbitration = runtime.arbitrate(generated, _retrieval())
    assert arbitration["fallback_used"] is True
    assert arbitration["arbitration_method"] == "rule_fallback"
    assert arbitration["fallback_reason"]
    assert arbitration["selected_claim_ids"] == arbitration["before_claim_ids"]


def test_topology_change_keeps_same_evidence_bundle_hash() -> None:
    retrieval = _retrieval()
    single = RuntimeMedicalAgents(
        RecordingProvider(), config=AgentTopologyConfig(topology="single")
    ).generate("chest pain", retrieval, {})
    pair = RuntimeMedicalAgents(
        RecordingProvider(), config=AgentTopologyConfig(topology="fixed_pair")
    ).generate("chest pain", retrieval, {})
    assert single["evidence_bundle_id"] == pair["evidence_bundle_id"]
    assert single["evidence_hash"] == pair["evidence_hash"]
    assert len(single["agents"]) == 1
    assert len(pair["agents"]) == 2


class _FixtureReview:
    """P4 集成测试只验证 Agent 持久化，审核阶段使用确定性测试替身。"""

    def __init__(self) -> None:
        self.provider = DeterministicWorkflowProvider()

    def review(
        self,
        generation: dict[str, Any],
        arbitration: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        return self.provider.review(generation, arbitration, retrieval)

    def report(
        self,
        case_id: str,
        generation: dict[str, Any],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        return self.provider.report(case_id, generation, review)


class _StaticRAG:
    corpus_version = "test-public-v1"

    def normalize(self, question: str) -> dict[str, Any]:
        return {"normalized_query": question, "normalizer_version": "test-v1"}

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        del normalized_query
        return _retrieval()


@pytest.fixture
def worker_runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'p4-worker.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    yield engine, factory
    engine.dispose()


def test_worker_persists_two_agent_runs_and_arbitration_artifact(worker_runtime) -> None:
    _, factory = worker_runtime
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session,
            "Deidentified simulated chest pain and dyspnea case.",
            "p4-case-key",
            "test-scope",
        )
        executor.start_workflow(session, case.case_id, "case_workflow", "p4-runtime", "input-hash")
        case_id = case.case_id
    llm = RecordingProvider()
    agents = RuntimeMedicalAgents(llm, config=AgentTopologyConfig(topology="fixed_pair"))
    workflow = OpenAICompatibleWorkflowProvider(
        llm,
        rag_stage=_StaticRAG(),
        agent_stage=agents,
        review_stage=_FixtureReview(),  # type: ignore[arg-type]
    )
    result = SingleMachineWorker(factory, workflow, worker_id="p4-worker").run_once()
    assert result.final_state == "CLOSED_SUCCESS"

    with factory() as session:
        runs = (
            session.execute(
                select(AgentRun).where(AgentRun.case_id == case_id).order_by(AgentRun.agent_name)
            )
            .scalars()
            .all()
        )
        assert len(runs) == 2
        assert all(run.input_payload["evidence_bundle_id"] == "eb_test_v1" for run in runs)
        assert all(run.output_payload["provider_provenance"]["response_id"] for run in runs)
        arbitration = session.execute(
            select(StageArtifact).where(
                StageArtifact.case_id == case_id,
                StageArtifact.stage == "arbitration",
            )
        ).scalar_one()
        assert arbitration.payload["before_claim_ids"]
        assert arbitration.payload["arbitration_method"] == "constrained_llm"
