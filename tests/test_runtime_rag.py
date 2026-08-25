"""P3 运行时医学 RAG 与 EvidenceBundle 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from medidiag.errors import MediDiagError
from medidiag.llm import FakeProvider
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import SearchResult
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.openai_provider import OpenAICompatibleWorkflowProvider


class _FakeRetriever:
    """不加载模型的确定性检索替身。"""

    embedding_model_name = "fake-embedding"
    embedding_model_revision = "fake-rev"
    rerank_model_name = "fake-rerank"
    rerank_model_revision = "fake-rerank-rev"
    requested_device = "cpu"
    actual_device = "cpu"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.search_config: dict[str, bool] | None = None
        self.built = False

    def build_index(self, use_bm25: bool = True, use_embedding: bool = True) -> None:
        self.built = use_bm25 and use_embedding

    def search(
        self,
        query: str,
        top_k: int = 5,
        experiment_config: dict[str, bool] | None = None,
    ) -> list[SearchResult]:
        del query
        self.search_config = experiment_config
        return self.results[:top_k]

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        del query
        return list(reversed(candidates))[:top_k]


def _chunk(chunk_id: str = "kb_1") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source="公开指南",
        source_id="guideline-2026",
        text="Fever is a symptom that requires assessment in clinical context.",
        evidence_level="level_1_guideline",
        metadata={"section": "fever"},
    )


def _backend(results: list[SearchResult]) -> RuntimeMedicalRAG:
    retriever = _FakeRetriever(results)
    return RuntimeMedicalRAG(
        chunks=[_chunk()],
        corpus_version="medical-corpus-v1",
        retrieval_config={"top_k": 5, "candidate_k": 10},
        experiment_config={
            "use_bm25": True,
            "use_embedding": True,
            "use_evidence_weighting": True,
            "use_term_normalization": True,
            "use_rerank": True,
            "use_citation_review": True,
        },
        normalizer=TerminologyNormalizer(),
        retriever=retriever,
        corpus_path="corpus.jsonl",
        leakage_gate={"status": "passed", "rule_version": "test"},
    )


def test_runtime_rag_emits_versioned_evidence_bundle() -> None:
    chunk = _chunk()
    backend = _backend(
        [
            SearchResult(
                chunk_id=chunk.chunk_id,
                final_score=0.91,
                bm25_score=0.8,
                embedding_score=0.7,
                evidence_level_score=1.0,
                term_overlap=0.5,
                chunk=chunk,
            )
        ]
    )

    normalized = backend.normalize("Patient has pyrexia")
    payload = backend.retrieve(normalized["normalized_query"])
    bundle = payload["evidence_bundle"]
    evidence = bundle["evidence"][0]

    assert normalized["normalized_query"] == "Patient has fever"
    assert bundle["schema_version"] == "evidence-bundle-v1"
    assert payload["evidence_bundle_id"] == bundle["bundle_id"]
    assert bundle["corpus_version"] == "medical-corpus-v1"
    assert len(bundle["corpus_hash"]) == 64
    assert len(bundle["retrieval_config_hash"]) == 64
    assert evidence["source"] == "公开指南"
    assert evidence["source_id"] == "guideline-2026"
    assert len(evidence["chunk_hash"]) == 64
    assert evidence["bm25_score"] == 0.8
    assert payload["chunks"][0]["chunk_hash"] == evidence["chunk_hash"]
    assert bundle["retrieval_provenance"]["leakage_gate"]["status"] == "passed"


def test_runtime_rag_rejects_empty_evidence() -> None:
    backend = _backend([])
    with pytest.raises(MediDiagError) as caught:
        backend.retrieve("unknown condition")
    assert caught.value.code == "RAG_NO_EVIDENCE"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _minimal_config(tmp_path: Path, *, leaked: bool) -> dict[str, Any]:
    corpus = tmp_path / "corpus.jsonl"
    rag_eval = tmp_path / "rag.jsonl"
    agent_eval = tmp_path / "agent.jsonl"
    source_id = "rag_sample_1" if leaked else "pubmed-1"
    _write_jsonl(corpus, [{**_chunk().__dict__, "source_id": source_id}])
    record = {"sample_id": "rag_sample_1", "question": "short", "gold_answer": "yes"}
    _write_jsonl(rag_eval, [record])
    _write_jsonl(agent_eval, [{**record, "sample_id": "agent_sample_1"}])
    return {
        "dataset": {
            "version": "v1",
            "knowledge_base_path": str(corpus),
            "rag_eval_set_path": str(rag_eval),
            "agent_eval_set_path": str(agent_eval),
        },
        "leakage_check": {
            "chunk_fields_to_check": ["source", "source_id", "metadata.raw_id"],
            "check_question_text": True,
            "check_answer_key": True,
        },
        "runtime": {"device": "cpu", "batch_size": 2},
        "embedding": {"model": "fake", "revision": "r1", "batch_size": 2},
        "rerank": {"model": "fake", "revision": "r1"},
        "retrieval": {
            "top_k": 1,
            "candidate_k": 1,
            "weights": {
                "w1_bm25": 0.25,
                "w2_embedding": 0.45,
                "w3_evidence_level": 0.2,
                "w4_term_overlap": 0.1,
            },
            "evidence_levels": {"level_1_guideline": 1.0},
        },
        "experiments": {
            "rag": {
                "rag_full": {
                    "config": {
                        "use_bm25": True,
                        "use_embedding": True,
                        "use_evidence_weighting": True,
                        "use_term_normalization": True,
                        "use_rerank": True,
                        "use_citation_review": True,
                    }
                }
            }
        },
    }


def test_runtime_rag_runs_shared_leakage_gate_before_index(tmp_path: Path) -> None:
    with pytest.raises(MediDiagError) as caught:
        RuntimeMedicalRAG.from_config(
            _minimal_config(tmp_path, leaked=True), root=tmp_path, build_index=False
        )
    assert caught.value.code == "EVAL_DATA_LEAKAGE_DETECTED"


def test_live_workflow_requires_explicit_runtime_rag() -> None:
    workflow = OpenAICompatibleWorkflowProvider(FakeProvider())
    with pytest.raises(MediDiagError) as caught:
        workflow.retrieve("fever")
    assert caught.value.code == "RAG_CORPUS_INVALID"


def test_worker_persists_evidence_bundle_for_reverse_audit(tmp_path: Path) -> None:
    """Agent 后续阶段可从同一 case 的 retrieval artifact 反查证据包。"""
    from sqlalchemy import select

    from medidiag.db.models import StageArtifact
    from medidiag.db.session import create_db_engine, get_session_factory, init_db
    from medidiag.workflow.executor import WorkflowExecutor
    from medidiag.workflow.provider import DeterministicWorkflowProvider
    from medidiag.workflow.worker import SingleMachineWorker

    chunk = _chunk()
    backend = _backend(
        [
            SearchResult(
                chunk_id=chunk.chunk_id,
                final_score=0.9,
                bm25_score=0.7,
                embedding_score=0.8,
                evidence_level_score=1.0,
                term_overlap=0.5,
                chunk=chunk,
            )
        ]
    )

    class _RAGBackedFixture(DeterministicWorkflowProvider):
        def normalize(self, question: str) -> dict[str, Any]:
            return backend.normalize(question)

        def retrieve(self, normalized_query: str) -> dict[str, Any]:
            return backend.retrieve(normalized_query)

    engine = create_db_engine(f"sqlite:///{(tmp_path / 'rag-worker.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session,
            "Deidentified simulated fever case for evidence audit.",
            "runtime-rag-case",
            "test-scope",
        )
        executor.start_workflow(
            session, case.case_id, "case_workflow", "runtime-rag-task", "input-hash"
        )
        case_id = case.case_id

    result = SingleMachineWorker(
        factory, _RAGBackedFixture(), worker_id="runtime-rag-worker"
    ).run_once()
    assert result.final_state == "CLOSED_SUCCESS"
    with factory() as session:
        artifact = session.execute(
            select(StageArtifact).where(
                StageArtifact.case_id == case_id,
                StageArtifact.stage == "retrieval",
            )
        ).scalar_one()
        bundle = artifact.payload["evidence_bundle"]
        assert artifact.payload["evidence_bundle_id"] == bundle["bundle_id"]
        assert bundle["evidence"][0]["chunk_id"] == "kb_1"
        assert bundle["retrieval_provenance"]["leakage_gate"]["status"] == "passed"
    engine.dispose()
