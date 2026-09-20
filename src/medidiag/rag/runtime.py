"""运行时医学 RAG 与版本化 EvidenceBundle。

真实运行时与评测可复用同一个 ``Retriever``。本模块只负责 corpus、泄露门禁、
术语归一化、检索与可审计证据封装，不持有数据库会话，也不生成医学结论。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from medidiag.errors import MediDiagError
from medidiag.rag.leakage import LEAKAGE_FLAG, run_leakage_check_records
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever, SearchResult
from medidiag.schemas import KnowledgeChunk, load_knowledge_chunks, read_jsonl


class RetrieverContract(Protocol):
    """运行时检索器最小契约，便于离线测试注入。"""

    def build_index(self, use_bm25: bool = True, use_embedding: bool = True) -> None: ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        experiment_config: dict[str, bool] | None = None,
    ) -> list[SearchResult]: ...

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class EvidenceItem:
    """一条可追溯证据及其检索分数组成。"""

    chunk_id: str
    source: str
    source_id: str
    chunk_hash: str
    text: str
    evidence_level: str
    final_score: float
    bm25_score: float
    embedding_score: float
    evidence_level_score: float
    term_overlap: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvidenceBundle:
    """规划前冻结的版本化证据包。"""

    schema_version: str
    bundle_id: str
    query: str
    normalized_query: str
    corpus_version: str
    corpus_hash: str
    retrieval_config_hash: str
    evidence: tuple[EvidenceItem, ...]
    retrieval_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = [asdict(item) for item in self.evidence]
        return value


class RuntimeMedicalRAG:
    """版本化医学 corpus 的运行时检索 stage backend。"""

    schema_version = "evidence-bundle-v1"
    version = "runtime-medical-rag-v1"

    def __init__(
        self,
        *,
        chunks: list[KnowledgeChunk],
        corpus_version: str,
        retrieval_config: dict[str, Any],
        experiment_config: dict[str, bool],
        normalizer: TerminologyNormalizer,
        retriever: RetrieverContract,
        corpus_path: str,
        leakage_gate: dict[str, Any],
    ) -> None:
        if not chunks:
            raise MediDiagError("RAG_CORPUS_INVALID", detail="医学知识库为空")
        if not corpus_version.strip():
            raise MediDiagError("RAG_CORPUS_INVALID", detail="缺少 corpus version")
        self.chunks = chunks
        self.corpus_version = corpus_version
        self.retrieval_config = retrieval_config
        self.experiment_config = experiment_config
        self.normalizer = normalizer
        self.retriever = retriever
        self.corpus_path = corpus_path
        self.leakage_gate = leakage_gate
        self.corpus_hash = _stable_hash([asdict(chunk) for chunk in chunks])
        self.retrieval_config_hash = _stable_hash(
            {"retrieval": retrieval_config, "experiment": experiment_config}
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        root: str | Path = ".",
        build_index: bool = True,
        retriever_factory: Callable[..., RetrieverContract] = Retriever,
    ) -> RuntimeMedicalRAG:
        """从固定配置构建真实检索 backend，并在建索引前执行泄露门禁。"""
        base = Path(root).resolve()
        dataset = config["dataset"]
        corpus_path = _resolve(base, str(dataset["knowledge_base_path"]))
        try:
            raw_chunks = read_jsonl(corpus_path)
            chunks = load_knowledge_chunks(corpus_path)
            eval_paths = [
                _resolve(base, str(dataset["rag_eval_set_path"])),
                _resolve(base, str(dataset["agent_eval_set_path"])),
            ]
            hits: list[str] = []
            for eval_path in eval_paths:
                hits.extend(
                    run_leakage_check_records(
                        read_jsonl(eval_path), raw_chunks, config["leakage_check"]
                    )
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise MediDiagError(
                "RAG_CORPUS_INVALID", detail=f"医学 corpus 或门禁配置无效: {exc}"
            ) from exc
        if hits:
            raise MediDiagError(
                LEAKAGE_FLAG,
                detail="运行时医学 corpus 未通过泄露门禁: " + "; ".join(hits[:5]),
            )

        normalizer = TerminologyNormalizer(eval_set_path=eval_paths[0])
        runtime = config.get("runtime", {})
        batch_size = int(
            runtime.get("batch_size", config["embedding"].get("batch_size", 32))
        )
        retriever = retriever_factory(
            chunks,
            weights=config["retrieval"]["weights"],
            evidence_level_scores=config["retrieval"]["evidence_levels"],
            embedding_model=config["embedding"]["model"],
            rerank_model=config["rerank"]["model"],
            normalizer=normalizer,
            embedding_revision=config["embedding"]["revision"],
            rerank_revision=config["rerank"]["revision"],
            device=str(runtime.get("device", "auto")),
            embedding_batch_size=batch_size,
            rerank_batch_size=batch_size,
            bm25_tokenizer=str(config["retrieval"].get("bm25_tokenizer", "whitespace")),
        )
        backend = cls(
            chunks=chunks,
            corpus_version=str(dataset["version"]),
            retrieval_config=dict(config["retrieval"]),
            experiment_config=dict(config["experiments"]["rag"]["rag_full"]["config"]),
            normalizer=normalizer,
            retriever=retriever,
            corpus_path=str(corpus_path),
            leakage_gate={
                "status": "passed",
                "checked_eval_sets": [str(path) for path in eval_paths],
                "rule_version": "runtime-leakage-v1",
            },
        )
        if build_index:
            try:
                retriever.build_index(
                    use_bm25=backend.experiment_config.get("use_bm25", True),
                    use_embedding=backend.experiment_config.get("use_embedding", True),
                )
            except Exception as exc:
                raise MediDiagError(
                    "RAG_INDEX_BUILD_FAILED", detail=f"医学检索索引构建失败: {exc}"
                ) from exc
        return backend

    def normalize(self, question: str) -> dict[str, Any]:
        """输出版本化术语归一化结果。"""
        if not self.experiment_config.get("use_term_normalization", True):
            return {
                "normalized_query": " ".join(question.strip().split()),
                "normalizer_version": "whitespace-only-v1",
                "coverage": 0.0,
                "matched_terms": [],
            }
        normalized = self.normalizer.normalize(question)
        return {
            "normalized_query": normalized.normalized,
            "normalizer_version": "terminology-normalizer-v1",
            "coverage": normalized.coverage,
            "matched_terms": [asdict(item) for item in normalized.matched_terms],
        }

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        """检索证据并冻结 EvidenceBundle；空结果不会伪装为成功。"""
        candidate_k = int(self.retrieval_config.get("candidate_k", 10))
        top_k = int(self.retrieval_config.get("top_k", 5))
        candidates = self.retriever.search(
            normalized_query,
            top_k=max(candidate_k, top_k),
            experiment_config=self.experiment_config,
        )
        results = (
            self.retriever.rerank(normalized_query, candidates, top_k=top_k)
            if self.experiment_config.get("use_rerank", False)
            else candidates[:top_k]
        )
        evidence = tuple(self._evidence_item(item) for item in results if item.chunk)
        if not evidence:
            raise MediDiagError(
                "RAG_NO_EVIDENCE",
                detail="医学检索未返回可追溯证据，禁止进入普通报告链路",
            )
        bundle_seed = {
            "query": normalized_query,
            "corpus_hash": self.corpus_hash,
            "retrieval_config_hash": self.retrieval_config_hash,
            "evidence": [
                {"chunk_id": item.chunk_id, "chunk_hash": item.chunk_hash}
                for item in evidence
            ],
        }
        bundle = EvidenceBundle(
            schema_version=self.schema_version,
            bundle_id="eb_" + _stable_hash(bundle_seed)[:24],
            query=normalized_query,
            normalized_query=normalized_query,
            corpus_version=self.corpus_version,
            corpus_hash=self.corpus_hash,
            retrieval_config_hash=self.retrieval_config_hash,
            evidence=evidence,
            retrieval_provenance={
                "backend_version": self.version,
                "corpus_path": self.corpus_path,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "embedding_model": getattr(self.retriever, "embedding_model_name", None),
                "embedding_revision": getattr(
                    self.retriever, "embedding_model_revision", None
                ),
                "rerank_model": getattr(self.retriever, "rerank_model_name", None),
                "rerank_revision": getattr(
                    self.retriever, "rerank_model_revision", None
                ),
                "requested_device": getattr(self.retriever, "requested_device", None),
                "actual_device": getattr(self.retriever, "actual_device", None),
                "experiment_config": self.experiment_config,
                "leakage_gate": self.leakage_gate,
            },
        )
        return {
            "query": normalized_query,
            "top_k": len(evidence),
            "config_hash": self.retrieval_config_hash,
            "evidence_bundle_id": bundle.bundle_id,
            "evidence_bundle": bundle.to_dict(),
            "chunks": [self._legacy_chunk(item) for item in evidence],
        }

    @staticmethod
    def _evidence_item(result: SearchResult) -> EvidenceItem:
        chunk = result.chunk
        if chunk is None:
            raise ValueError("检索结果缺少 KnowledgeChunk")
        return EvidenceItem(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            source_id=chunk.source_id,
            chunk_hash=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            text=chunk.text,
            evidence_level=chunk.evidence_level,
            final_score=result.final_score,
            bm25_score=result.bm25_score,
            embedding_score=result.embedding_score,
            evidence_level_score=result.evidence_level_score,
            term_overlap=result.term_overlap,
            metadata=dict(chunk.metadata),
        )

    @staticmethod
    def _legacy_chunk(item: EvidenceItem) -> dict[str, Any]:
        """保留现有 generation/plan 所需字段，同时补齐审计信息。"""
        return {
            "chunk_id": item.chunk_id,
            "source": item.source,
            "source_id": item.source_id,
            "chunk_hash": item.chunk_hash,
            "evidence_level": item.evidence_level,
            "score": item.final_score,
            "final_score": item.final_score,
            "bm25_score": item.bm25_score,
            "embedding_score": item.embedding_score,
            "evidence_level_score": item.evidence_level_score,
            "term_overlap": item.term_overlap,
            "text": item.text,
            "source_url": item.metadata.get("source_url"),
            "metadata": item.metadata,
        }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
