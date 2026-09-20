"""公开正文模式：安全配置、中文检索、空证据与真实 worker/API 展示。"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from medidiag.api.app import create_app
from medidiag.cli import _build_provider
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.rag.retrieval import Retriever
from medidiag.workflow.application import build_application_provider, load_application_config
from medidiag.workflow.worker import SingleMachineWorker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def provider(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("本模式不得加载或调用向量、重排模型")
    monkeypatch.setattr(Retriever, "_build_embedding_index", forbidden)
    monkeypatch.setattr(Retriever, "_get_embedding_scores", forbidden)
    monkeypatch.setattr(Retriever, "rerank", forbidden)
    return build_application_provider("retrieval_mock", root=ROOT)


def test_distinct_chinese_evidence_and_source(provider):
    a = provider.retrieve("咳嗽纸巾遮住口鼻")
    b = provider.retrieve("天黑后打开窗户降温")
    assert a["chunks"][0]["chunk_id"] != b["chunks"][0]["chunk_id"]
    assert a["execution_mode"] == "retrieval_mock"
    assert all(c["source_url"].startswith("https://www.who.int/") for c in a["chunks"])
    assert provider.retrieve("量子纠缠计算芯片")["chunks"] == []
    tokens = provider.rag_stage.retriever.tokenize("中文输入 COVID-19")
    assert tokens == ["中文", "文输", "输入", "covid", "19"]


@pytest.mark.parametrize("change", ["use_embedding", "use_rerank", "research", "gate"])
def test_unsafe_application_config_rejected(tmp_path, change):
    config = copy.deepcopy(load_application_config(ROOT / "configs/application.yaml"))
    if change == "research":
        del config["application"]
    elif change == "gate":
        config["leakage_check"]["check_answer_key"] = False
    else:
        config["experiments"]["rag"]["rag_full"]["config"][change] = True
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_application_config(p)


def test_error_does_not_become_no_evidence(provider, monkeypatch):
    def broken(query):
        raise MediDiagError("RAG_INDEX_BUILD_FAILED")
    monkeypatch.setattr(provider.rag_stage, "retrieve", broken)
    with pytest.raises(MediDiagError, match="RAG_INDEX_BUILD_FAILED"):
        provider.retrieve("公开输入")


def test_cli_uses_application_config_not_research():
    p = _build_provider("retrieval_mock", "APPROVED", rag_config="不存在的研究配置.yaml",
                        app_config=str(ROOT / "configs/application.yaml"))
    assert p.version == "retrieval-mock-v1"


@pytest.mark.parametrize("query,outcome", [("通风如何改善室内空气质量？", "ready"),
                                          ("模拟提问：量子纠缠计算芯片", "insufficient_evidence")])
@pytest.mark.parametrize("runner", ["poll", "queue"])
def test_real_rag_worker_analysis(tmp_path, provider, query, outcome, runner, monkeypatch):
    engine = create_db_engine("sqlite:///" + (tmp_path / "public.db").as_posix())
    init_db(engine)
    factory = get_session_factory(engine)
    try:
        with TestClient(create_app(session_factory=factory)) as client:
            created = client.post("/api/v1/consultations", headers={"Idempotency-Key": "create"}, json={
                "symptoms": query, "duration": "模拟", "input_kind": "deidentified_simulation",
                "non_sensitive_confirmed": True})
            assert created.status_code == 201
            prefix = "/api/v1/cases/" + created.json()["case_id"]
            started = client.post(prefix + "/workflow", headers={"Idempotency-Key": "run"})
            assert started.status_code == 200
            if runner == "queue":
                from medidiag.workflow import queue
                monkeypatch.setattr(queue, "get_settings", lambda: SimpleNamespace(
                    database_url=str(engine.url), medidiag_app_provider="retrieval_mock",
                    medidiag_app_config=str(ROOT / "configs/application.yaml")))
                # 直接执行消费者函数，验证装配；不是 Redis/Celery 服务验收。
                queue.execute_task.run(started.json()["task_id"])
            else:
                SingleMachineWorker(factory, provider).run_once()
            result = client.get(prefix + "/analysis").json()
            assert result["execution_mode"] == "retrieval_mock"
            assert result["outcome"] == outcome
            if outcome == "ready":
                assert result["claims"] and result["evidence"][0]["source_url"]
                assert "模拟生成" in str(result["limitations"])
                assert all("不代表问题已获解答" in c["text"] for c in result["claims"])
            else:
                assert result["claims"] == [] and result["summary"] is None
    finally:
        engine.dispose()


def test_structural_review_never_claims_nli(provider):
    evidence = provider.retrieve("通风空气")
    generated = provider.generate("通风空气", evidence, {})
    review = provider.review(generated, {}, evidence)
    assert review["verdict"] == "APPROVED"
    assert all(c["verdict"] == "PARTIAL" and c["confidence"] is None for c in review["citation_verdicts"])
    generated["claims"][0]["citation_chunk_ids"] = ["unbound"]
    assert provider.review(generated, {}, evidence)["verdict"] == "ESCALATED"


def test_source_and_fixed_query_ids():
    sources = json.loads((ROOT / "examples/public_health/sources.json").read_text(encoding="utf-8"))["sources"]
    chunks = [json.loads(line) for line in (ROOT / "examples/public_health/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    known = {c["chunk_id"] for c in chunks}
    assert len(sources) == len(chunks) == 8
    assert {cid for source in sources for cid in source["chunk_ids"]} == known
    queries = [json.loads(line) for line in (ROOT / "examples/public_health/queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(queries) == 16
    assert all(set(q["gold_evidence_ids"]) <= known for q in queries)
    assert len([q for q in queries if q["split"] == "check"]) == 8


def test_form_labels_and_default_background_are_not_query(provider):
    normalized = provider.normalize("症状：量子纠缠计算芯片\n持续时间：模拟\n背景：未提供")
    assert normalized["normalized_query"] == "量子纠缠计算芯片 模拟"
    assert provider.retrieve(normalized["normalized_query"])["chunks"] == []
    # 真实填写的背景不按子串删除，避免误删用户提供的信息。
    assert "未提供具体阈值" in provider.normalize(
        "症状：疫苗问题\n持续时间：模拟\n背景：未提供具体阈值")["normalized_query"]


def test_model_pipeline_label_is_not_online_acceptance(provider):
    from medidiag.llm import FakeProvider
    from medidiag.workflow.openai_provider import OpenAICompatibleWorkflowProvider
    workflow = OpenAICompatibleWorkflowProvider(FakeProvider(), rag_stage=provider.rag_stage)
    assert workflow.retrieve("通风空气")["execution_mode"] == "model_pipeline"
