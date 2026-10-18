"""运行时医学 RAG 与版本化 EvidenceBundle。

本模块只负责 corpus、泄露门禁、检索与可审计证据封装，检索器是可替换的
（BM25 或向量检索），可选交叉编码器重排；不持有数据库会话，也不生成医学结论。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from medidiag.errors import MediDiagError
from medidiag.rag.leakage import LEAKAGE_FLAG, run_leakage_check_records
from medidiag.rag.rerank import Reranker
from medidiag.rag.retrieval import Bm25Retriever, SearchResult
from medidiag.schemas import KnowledgeChunk, load_knowledge_chunks, read_jsonl


class RetrieverContract(Protocol):
    """运行时检索器最小契约，便于离线测试注入。"""

    def build_index(self) -> None: ...

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


def _bm25_factory(chunks: list[KnowledgeChunk], config: dict[str, Any]) -> RetrieverContract:
    return Bm25Retriever(chunks, bm25_tokenizer=str(config["retrieval"].get("bm25_tokenizer", "cjk_bigram")))


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
        embedding_config: dict[str, Any] | None = None,
        retriever: RetrieverContract,
        corpus_path: str,
        leakage_gate: dict[str, Any],
        reranker: Reranker | None = None,
    ) -> None:
        if not chunks:
            raise MediDiagError("RAG_CORPUS_INVALID", detail="医学知识库为空")
        if not corpus_version.strip():
            raise MediDiagError("RAG_CORPUS_INVALID", detail="缺少 corpus version")
        self.chunks = chunks
        self.corpus_version = corpus_version
        self.retrieval_config = retrieval_config
        self.embedding_config = embedding_config
        self.retriever = retriever
        self.corpus_path = corpus_path
        self.leakage_gate = leakage_gate
        # 配置了重排时，固定流程先取前 reranker.candidates 段，再按交叉编码器分数取前 top_k 段
        self.reranker = reranker
        self.corpus_hash = _stable_hash([asdict(chunk) for chunk in chunks])
        self.retrieval_config_hash = _stable_hash(
            {"retrieval": retrieval_config, "embedding": embedding_config}
        )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        root: str | Path = ".",
        build_index: bool = True,
        retriever_factory: Callable[[list[KnowledgeChunk], dict[str, Any]], RetrieverContract] = _bm25_factory,
        reranker_factory: Callable[[dict[str, Any]], Reranker] | None = None,
    ) -> RuntimeMedicalRAG:
        """从固定配置构建真实检索 backend，并在建索引前执行泄露门禁。

        配置里有 ``retrieval.rerank`` 时必须提供 ``reranker_factory``；模型加载由调用方负责，本模块不读模型缓存。
        """
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

        retriever = retriever_factory(chunks, config)
        reranker = None
        if config["retrieval"].get("rerank"):
            if reranker_factory is None:
                raise MediDiagError("RAG_CORPUS_INVALID", detail="配置了重排但没有提供重排模型")
            reranker = reranker_factory(config)
        backend = cls(
            chunks=chunks,
            corpus_version=str(dataset["version"]),
            retrieval_config=dict(config["retrieval"]),
            embedding_config=dict(config["embedding"]) if "embedding" in config else None,
            retriever=retriever,
            corpus_path=str(corpus_path),
            leakage_gate={
                "status": "passed",
                "checked_eval_sets": [str(path) for path in eval_paths],
                "rule_version": "runtime-leakage-v1",
            },
            reranker=reranker,
        )
        if build_index:
            try:
                retriever.build_index()
            except Exception as exc:
                raise MediDiagError(
                    "RAG_INDEX_BUILD_FAILED", detail=f"医学检索索引构建失败: {exc}"
                ) from exc
        return backend

    def normalize(self, question: str) -> dict[str, Any]:
        """只合并空白，不做术语替换（术语归一化代码已删除）。"""
        return {
            "normalized_query": " ".join(question.strip().split()),
            "normalizer_version": "whitespace-only-v1",
            "coverage": 0.0,
            "matched_terms": [],
        }

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        """检索证据并冻结 EvidenceBundle；空结果不会伪装为成功。"""
        evidence = self.search_evidence(normalized_query)
        if not evidence:
            raise MediDiagError(
                "RAG_NO_EVIDENCE",
                detail="医学检索未返回可追溯证据，禁止进入普通报告链路",
            )
        return self.freeze_evidence(normalized_query, evidence)

    def search_evidence(self, query: str, top_k: int | None = None) -> tuple[EvidenceItem, ...]:
        """只检索、不冻结；可能返回空元组。检索智能体用它做多次检索。

        配置了重排时返回的是重排后的前 top_k 段；``top_k`` 只供检索智能体取更多候选，不重排。
        """
        if top_k is not None:
            results = self.retriever.search(query, top_k=top_k)
            return tuple(self._evidence_item(item) for item in results if item.chunk)
        top_k = int(self.retrieval_config.get("top_k", 5))
        if self.reranker is None:
            results = self.retriever.search(query, top_k=top_k)
            return tuple(self._evidence_item(item) for item in results if item.chunk)
        candidates = self.search_evidence(query, top_k=self.reranker.candidates)
        return self.rerank_evidence(query, candidates, top_k)

    def rerank_evidence(self, question: str, items: Sequence[EvidenceItem], top_k: int) -> tuple[EvidenceItem, ...]:
        """按交叉编码器分数重排候选并去重；final_score 换成重排分数，原检索分数保留在对应字段。"""
        if self.reranker is None:
            raise MediDiagError("RAG_CORPUS_INVALID", detail="没有配置重排模型")
        return tuple(replace(item, final_score=score) for item, score in self.reranker.rerank(question, items, top_k))

    def freeze_evidence(self, normalized_query: str, evidence: tuple[EvidenceItem, ...]) -> dict[str, Any]:
        """把给定证据冻结成带哈希的 EvidenceBundle，并返回流程各阶段使用的检索结果。"""
        top_k = int(self.retrieval_config.get("top_k", 5))
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
                "method": self.retrieval_config.get("method"),
                "embedding_model": getattr(self.retriever, "embedding_model_name", None),
                "embedding_revision": getattr(
                    self.retriever, "embedding_model_revision", None
                ),
                "leakage_gate": self.leakage_gate,
                **({} if self.reranker is None else {"reranker": {
                    "model": self.reranker.model_name, "revision": self.reranker.model_revision,
                    "candidates": self.reranker.candidates}}),
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
