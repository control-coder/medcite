"""生成式回答（M12）的离线测试：假 LLM，不联网、不产生费用。"""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from medidiag.api.app import create_app
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.llm.budget import BudgetedTransport
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.llm.openai_compatible import OpenAICompatibleProvider
from medidiag.llm.profiles import get_provider_profile
from medidiag.workflow.application import build_application_provider, load_application_config
from medidiag.workflow.generative_answer import (
    ENTAILMENT_VERSION,
    PROMPT_VERSION,
    InvalidAnswer,
    validate_sentences,
)
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker

ROOT = Path(__file__).resolve().parents[1]
C1 = "戴口罩可以减少呼吸道传染病的传播。通风换气每天至少进行30分钟，有助于降低室内病原体浓度。"
C2 = "接种疫苗后少数人可能出现发热，一般不超过3天，不能因此认为疫苗无效。"
CHUNKS = {"c1": C1, "c2": C2}
RETRIEVAL = {"chunks": [{"chunk_id": k, "text": v} for k, v in CHUNKS.items()]}
Q1 = "通风换气每天至少进行30分钟"
Q2 = "戴口罩可以减少呼吸道传染病的传播"


def sentence(text, *cites):
    return {"text": text, "citations": [{"chunk_id": c, "quote": q} for c, q in cites]}


def generated(*sentences, status="sufficient"):
    return {"status": status, "sentences": list(sentences)}


GOOD = sentence("每天通风换气至少30分钟。", ("c1", Q1))


class FakeLLM:
    """按提示词版本分发脚本化回答；脚本用尽后重复最后一条。"""

    provider_id, profile_id = "fake", "fake"

    def __init__(self, **scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests = []

    def generate(self, request, *, timeout_s, idempotency_key):
        self.requests.append(request)
        queue = self.scripts[request.prompt_version]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(parsed_json=item, model=ACTIVE_MIMO_MODEL, finish_reason="stop", response_id="rid",
                               usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, retry_count=0)

    def calls(self, version):
        return [r for r in self.requests if r.prompt_version == version]


def verdicts(*values):
    return {"verdicts": [{"id": i, "entailment": v} for i, v in enumerate(values)]}


def excerpt(cid="c1"):
    return {"status": "sufficient", "claims": [{"text": CHUNKS[cid], "citation_chunk_ids": [cid]}]}


def make(llm, **kwargs):
    rag = SimpleNamespace(chunks=[SimpleNamespace(source_id="s1")])
    return MimoGroundedWorkflowProvider(rag, llm=llm, answer_mode="generative", **kwargs)  # type: ignore[arg-type]


def run(provider, question="通风多久合适？"):
    return provider.generate(question, RETRIEVAL, {})


def test_valid_output_is_shown_with_sentence_citations():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD, sentence("口罩有助于减少传播。", ("c1", Q2)))],
                     ENTAILMENT_VERSION: [verdicts("entailed", "partial")]})
    response = run(make(llm))
    payload = response.payload
    assert payload["answer_mode"] == "generative" and not payload["abstained"]
    first, second = payload["claims"]
    assert first["text"] == GOOD["text"] and first["citation_chunk_ids"] == ["c1"]
    assert first["citations"] == [{"chunk_id": "c1", "quote": Q1, "entailment": "entailed"}]
    assert second["citations"][0]["entailment"] == "partial"
    assert response.metadata["usage"]["input_tokens"] == 20
    request = llm.calls(ENTAILMENT_VERSION)[0]
    assert "通风多久合适" not in json.dumps(request.messages, ensure_ascii=False)
    assert request.reasoning_mode == "disabled" and request.response_format == {"type": "json_object"}


@pytest.mark.parametrize("bad,reason", [
    (sentence("口罩有用。", ("c1", "这句话不在证据原文里面啊啊")), "逐字"),
    (sentence("口罩有用。", ("c1", "戴口罩可以")), "少于"),
    (sentence("每天通风45分钟。", ("c1", Q1)), "45分钟"),
    (sentence("通风有帮助。", ("zzz", Q1)), "chunk_id"),
    (sentence("每次服用500毫克即可。", ("c1", Q1)), "剂量"),
    (sentence("每天通风换气30分钟，口服2片。", ("c1", Q1)), "剂量"),
    (sentence("你患有呼吸道传染病。", ("c1", Q2)), "诊断"),
    (sentence("建议你停药后再通风。", ("c1", Q1)), "诊断"),
])
def test_deterministic_validation_rejects(bad, reason):
    with pytest.raises(InvalidAnswer, match=reason):
        validate_sentences([(bad["text"], [(c["chunk_id"], c["quote"]) for c in bad["citations"]])], CHUNKS)


def test_validation_is_whitespace_insensitive_and_flags_negation():
    ok = [("每天通风 30 分钟。", [("c1", "通风换气 每天至少进行30 分钟")])]
    assert validate_sentences(ok, CHUNKS) == [False]
    negated = [("口罩不能减少传播。", [("c1", Q2)])]
    assert validate_sentences(negated, CHUNKS) == [True]
    assert validate_sentences([("不能因此认为疫苗无效。", [("c2", "不能因此认为疫苗无效")])], CHUNKS) == [False]


def test_dosage_rejected_even_when_source_contains_dose():
    chunks = {"d": "成人每次服用500毫克，每日不超过三次，具体请遵医嘱。"}
    with pytest.raises(InvalidAnswer, match="剂量"):
        validate_sentences([("成人每次服用500毫克。", [("d", "成人每次服用500毫克，每日不超过三次")])], chunks)


def test_suspicious_flag_is_passed_to_entailment_and_kept():
    negated = sentence("口罩不能减少传播。", ("c1", Q2))
    llm = FakeLLM(**{PROMPT_VERSION: [generated(negated)], ENTAILMENT_VERSION: [verdicts("partial")]})
    claim = run(make(llm)).payload["claims"][0]
    assert claim["suspicious"] is True
    items = json.loads(llm.calls(ENTAILMENT_VERSION)[0].messages[1]["content"])["items"]
    assert items[0]["check_negation"] is True


def test_invalid_output_retries_once_with_feedback():
    bad = generated(sentence("口罩有用。", ("c1", "这句话不在证据原文里面啊啊")))
    llm = FakeLLM(**{PROMPT_VERSION: [bad, generated(GOOD)], ENTAILMENT_VERSION: [verdicts("entailed")]})
    payload = run(make(llm, on_invalid="feedback")).payload
    assert payload["answer_mode"] == "generative" and len(payload["claims"]) == 1
    assert len(llm.calls(PROMPT_VERSION)) == 2
    feedback = llm.calls(PROMPT_VERSION)[1].messages[-1]["content"]
    assert "previous_output_rejected" in feedback and "逐字" in feedback
    assert "previous_output_rejected" not in json.dumps(llm.calls(PROMPT_VERSION)[0].messages, ensure_ascii=False)
    assert payload["generative_trace"]["invalid_reasons"]


def test_resample_retries_without_reason_and_none_does_not_retry():
    bad = generated(sentence("每天通风45分钟。", ("c1", Q1)))
    llm = FakeLLM(**{PROMPT_VERSION: [bad, generated(GOOD)], ENTAILMENT_VERSION: [verdicts("entailed")]})
    run(make(llm, on_invalid="resample"))
    assert "previous_output_rejected" not in json.dumps(llm.calls(PROMPT_VERSION)[1].messages, ensure_ascii=False)
    llm = FakeLLM(**{PROMPT_VERSION: [bad], "mimo-grounded-v1": [excerpt()]})
    payload = run(make(llm, on_invalid="none")).payload
    assert len(llm.calls(PROMPT_VERSION)) == 1 and payload["answer_mode"] == "extractive"


def test_retry_exhausted_degrades_to_extractive():
    bad = generated(sentence("每天通风45分钟。", ("c1", Q1)))
    llm = FakeLLM(**{PROMPT_VERSION: [bad], "mimo-grounded-v1": [excerpt()]})
    payload = run(make(llm, on_invalid="feedback")).payload
    assert len(llm.calls(PROMPT_VERSION)) == 2 and not llm.calls(ENTAILMENT_VERSION)
    assert payload["answer_mode"] == "extractive" and payload["degraded_from"] == "generative"
    assert payload["generative_trace"]["degraded"] == "invalid_output"


def test_all_not_entailed_degrades_to_extractive():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD, sentence("口罩有助于减少传播。", ("c1", Q2)))],
                     ENTAILMENT_VERSION: [verdicts("not_entailed", "not_entailed")], "mimo-grounded-v1": [excerpt()]})
    response = run(make(llm))
    payload = response.payload
    assert payload["answer_mode"] == "extractive" and payload["claims"][0]["text"] == C1
    assert payload["generative_trace"]["degraded"] == "all_not_entailed"
    assert len(payload["generative_trace"]["dropped"]) == 2
    assert response.metadata["usage"]["input_tokens"] == 30


def test_not_entailed_sentence_is_dropped_others_kept():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD, sentence("口罩有助于减少传播。", ("c1", Q2)))],
                     ENTAILMENT_VERSION: [verdicts("not_entailed", "entailed")]})
    payload = run(make(llm)).payload
    assert [c["text"] for c in payload["claims"]] == ["口罩有助于减少传播。"]
    assert payload["claims"][0]["claim_id"] == "sentence_0001"
    assert payload["generative_trace"]["dropped"][0]["entailment"] == "not_entailed"


@pytest.mark.parametrize("extractive", [{"status": "insufficient", "claims": []}, {"extra": "格式错误"}])
def test_extractive_also_fails_refuses(extractive):
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD)], ENTAILMENT_VERSION: [verdicts("not_entailed")],
                     "mimo-grounded-v1": [extractive]})
    payload = run(make(llm)).payload
    assert payload["abstained"] and payload["claims"] == [] and payload["answer_mode"] == "generative"
    assert payload["generative_trace"]["outcome"] == "refused"


def test_model_insufficient_refuses_without_entailment_or_fallback():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(status="insufficient")]})
    payload = run(make(llm)).payload
    assert payload["abstained"] and len(llm.requests) == 1


def test_entailment_failure_is_conservative_and_degrades():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD)], ENTAILMENT_VERSION: [MediDiagError("LLM_TIMEOUT")],
                     "mimo-grounded-v1": [excerpt()]})
    payload = run(make(llm)).payload
    assert payload["answer_mode"] == "extractive"
    assert payload["generative_trace"]["entailment_error"] == "LLM_TIMEOUT"
    assert payload["generative_trace"]["degraded"] == "entailment_unavailable"
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD)], ENTAILMENT_VERSION: [verdicts("entailed", "entailed")],
                     "mimo-grounded-v1": [excerpt()]})
    assert run(make(llm)).payload["generative_trace"]["entailment_error"] == "ENTAILMENT_SHAPE"


def test_empty_evidence_does_not_call_model():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD)]})
    payload = make(llm).generate("问题", {"chunks": []}, {}).payload
    assert payload["abstained"] and not llm.requests


def test_extractive_default_is_unchanged():
    llm = FakeLLM(**{"mimo-grounded-v1": [excerpt()]})
    provider = MimoGroundedWorkflowProvider(SimpleNamespace(chunks=[]), llm=llm)  # type: ignore[arg-type]
    payload = run(provider).payload
    assert provider.answer_mode == "extractive" and "answer_mode" not in payload
    assert [r.prompt_version for r in llm.requests] == ["mimo-grounded-v1"]


def approved_generation():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD, sentence("口罩有助于减少传播。", ("c1", Q2)))],
                     ENTAILMENT_VERSION: [verdicts("entailed", "partial")]})
    return make(llm), run(make(llm)).payload


def test_review_and_report_for_generative():
    provider, generation = approved_generation()
    review = provider.review(generation, provider.arbitrate(generation, RETRIEVAL), RETRIEVAL)
    assert review["verdict"] == "APPROVED"
    assert [v["verdict"] for v in review["citation_verdicts"]] == ["SUPPORTED", "PARTIAL"]
    report = provider.report("case", generation, review)
    assert report["title"] == "公开证据生成式回答" and report["answer_mode"] == "generative"
    text = "".join(report["limitations"])
    assert "不是医学审核" in text and "不提供个体诊断、剂量或停药建议" in text and "部分支持" in text
    assert report["claims"][0]["citations"][0]["quote"] == Q1


@pytest.mark.parametrize("tamper", ["quote", "number", "dose", "chunk", "entailment"])
def test_review_rechecks_rules(tamper):
    provider, generation = approved_generation()
    claim = generation["claims"][0]
    if tamper == "quote":
        claim["citations"][0]["quote"] = "不是原文的逐字片段哦哦哦"
    elif tamper == "number":
        claim["text"] = "每天通风换气至少60分钟。"
    elif tamper == "dose":
        claim["text"] = "每次服用2片。"
    elif tamper == "chunk":
        claim["citations"][0]["chunk_id"] = "other"
        claim["citation_chunk_ids"] = ["other"]
    else:
        claim["citations"][0]["entailment"] = "not_entailed"
    review = provider.review(generation, {}, RETRIEVAL)
    assert review["verdict"] == "ESCALATED"


def test_review_does_not_require_full_text_match():
    provider, generation = approved_generation()
    assert generation["claims"][0]["text"] != C1
    assert provider.review(generation, {}, RETRIEVAL)["verdict"] == "APPROVED"


def test_degraded_report_is_extractive_with_note():
    llm = FakeLLM(**{PROMPT_VERSION: [generated(GOOD)], ENTAILMENT_VERSION: [verdicts("not_entailed")],
                     "mimo-grounded-v1": [excerpt()]})
    provider = make(llm)
    generation = run(provider).payload
    review = provider.review(generation, {}, RETRIEVAL)
    report = provider.report("case", generation, review)
    assert review["verdict"] == "APPROVED" and report["answer_mode"] == "extractive"
    assert report["title"].startswith("公开证据短引") and "降级为抽取式" in "".join(report["limitations"])


def test_config_whitelist(tmp_path):
    base = yaml.safe_load((ROOT / "configs/application_v4_hybrid.yaml").read_text(encoding="utf-8"))
    config = load_application_config(ROOT / "configs/application_v4_generative.yaml")
    assert config["generation"] == {"answer_mode": "generative", "on_invalid": "feedback"}
    assert {**config, "generation": {}} == {**base, "generation": {}}
    assert "answer_mode" not in load_application_config(ROOT / "configs/application_v4_hybrid.yaml")["generation"]
    for value in ("abstractive", True, None):
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump({**base, "generation": {"answer_mode": value}}, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError):
            load_application_config(bad)
    with pytest.raises(ValueError, match="生成式"):
        build_application_provider("retrieval_mock", app_config="configs/application_v4_generative.yaml", root=ROOT)


def make_live_provider(tmp_path, monkeypatch, post):
    monkeypatch.setenv("MIMO_API_KEY", "test-placeholder-not-a-real-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", ACTIVE_MIMO_MODEL)
    llm = OpenAICompatibleProvider(get_provider_profile("mimo_v25"),
                                   post=BudgetedTransport(tmp_path / "calls.db", post=post),
                                   max_retries=0, max_structured_retries=0)
    rag = build_application_provider("retrieval_mock", root=ROOT).rag_stage
    return MimoGroundedWorkflowProvider(rag, llm=llm, answer_mode="generative", on_invalid="feedback")


def test_worker_and_api_expose_citations(tmp_path, monkeypatch):
    def reply(content):
        return httpx.Response(200, json={"id": "rid", "model": ACTIVE_MIMO_MODEL, "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(content, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})

    def post(url, **kwargs):
        messages = kwargs["json"]["messages"]
        data = json.loads(messages[1]["content"])
        if "items" in data:
            return reply(verdicts("partial"))
        cid, text = next(iter(data["evidence"].items()))
        quote = text[:12]
        return reply(generated(sentence(quote, (cid, quote))))

    provider = make_live_provider(tmp_path, monkeypatch, post)
    engine = create_db_engine("sqlite:///" + (tmp_path / "app.db").as_posix())
    init_db(engine)
    factory = get_session_factory(engine)
    try:
        with TestClient(create_app(session_factory=factory)) as client:
            created = client.post("/api/v1/consultations", headers={"Idempotency-Key": "create"}, json={
                "symptoms": "通风如何改善室内空气质量？", "duration": "模拟", "input_kind": "deidentified_simulation",
                "non_sensitive_confirmed": True})
            prefix = "/api/v1/cases/" + created.json()["case_id"]
            assert client.post(prefix + "/workflow", headers={"Idempotency-Key": "run"}).status_code == 200
            SingleMachineWorker(factory, provider).run_once()
            analysis = client.get(prefix + "/analysis").json()
    finally:
        engine.dispose()
    assert analysis["outcome"] == "ready", analysis
    assert analysis["answer_mode"] == "generative"
    claim = analysis["claims"][0]
    assert claim["citations"][0]["entailment"] == "partial"
    assert claim["citations"][0]["chunk_id"] in claim["evidence_ids"]
    assert analysis["observation"]["recorded_input_tokens"] == 20
    assert any("不是医学审核" in item for item in analysis["limitations"])
