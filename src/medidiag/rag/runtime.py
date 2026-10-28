"""运行时医学 RAG 与版本化 EvidenceBundle。

本模块只负责 corpus、泄露门禁、检索与可审计证据封装，检索器是可替换的
（BM25、向量或混合检索），可选交叉编码器重排与 MMR 去冗余；
不持有数据库会话，也不生成医学结论。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from medidiag.errors import MediDiagError
from medidiag.rag.cache import DEFAULT_MAX_ENTRIES, DEFAULT_TTL_S, RetrievalCache, cache_key
from medidiag.rag.leakage import LEAKAGE_FLAG, run_leakage_check_records
from medidiag.rag.metadata import FilterTrace, MetadataIndex, search_with_relaxation
from medidiag.rag.rerank import Reranker
from medidiag.rag.retrieval import Bm25Retriever, SearchResult
from medidiag.schemas import KnowledgeChunk, load_knowledge_chunks, read_jsonl


class RetrieverContract(Protocol):
    """运行时检索器最小契约，便于离线测试注入。"""

    def build_index(self) -> None: ...

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...


def _bm25_factory(chunks: list[KnowledgeChunk], config: dict[str, Any]) -> RetrieverContract:
    return Bm25Retriever(chunks, bm25_tokenizer=str(config["retrieval"].get("bm25_tokenizer", "cjk_bigram")))


def _hybrid_factory(chunks: list[KnowledgeChunk], config: dict[str, Any]) -> RetrieverContract:
    """混合检索工厂：按配置建 BM25（可选 jieba 自定义词典）与向量检索器，再融合。"""
    from medidiag.rag.dense import DenseRetriever, load_local_encoder, store_options
    from medidiag.rag.hybrid import DEFAULT_CANDIDATES, DEFAULT_RRF_K, HybridRetriever
    from medidiag.rag.lexicon import build_lexicon

    retrieval = config["retrieval"]
    tokenizer = str(retrieval.get("bm25_tokenizer", "cjk_bigram"))
    lexicon = build_lexicon(chunks) if tokenizer == "jieba" else None
    bm25 = Bm25Retriever(chunks, bm25_tokenizer=tokenizer, lexicon=lexicon)
    name, revision = config["embedding"]["model"], config["embedding"]["revision"]
    runtime = config.get("runtime", {})
    encoder = load_local_encoder(name, revision, device=str(runtime.get("device", "cpu")),
                                 batch_size=int(runtime.get("batch_size", 8)))
    dense = DenseRetriever(chunks, encoder=encoder, model_name=name, model_revision=revision,
                           **store_options(chunks, config))
    hybrid = retrieval.get("hybrid", {})
    return HybridRetriever(
        chunks, bm25=bm25, dense=dense,
        candidates=int(hybrid.get("candidates", DEFAULT_CANDIDATES)),
        rrf_k=int(hybrid.get("rrf_k", DEFAULT_RRF_K)),
    )


def _mmr_lambda(retrieval_config: dict[str, Any]) -> float | None:
    """MMR 只在混合检索的 retrieval.hybrid.mmr_lambda 显式给出时开启。"""
    hybrid = retrieval_config.get("hybrid")
    if not isinstance(hybrid, dict) or "mmr_lambda" not in hybrid:
        return None
    return float(hybrid["mmr_lambda"])


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
        self._metadata: MetadataIndex | None = None
        self.last_filter_trace: FilterTrace | None = None
        self.corpus_hash = _stable_hash([asdict(chunk) for chunk in chunks])
        # 缓存开关不影响检索结果，不计入配置哈希，开关前后证据包标识保持一致
        self.retrieval_config_hash = _stable_hash(
            {"retrieval": {k: v for k, v in retrieval_config.items() if k != "cache"},
             "embedding": embedding_config}
        )
        cache_config = retrieval_config.get("cache") or {}
        self.cache: RetrievalCache | None = (
            RetrievalCache(int(cache_config.get("max_entries", DEFAULT_MAX_ENTRIES)),
                           float(cache_config.get("ttl_s", DEFAULT_TTL_S)))
            if cache_config.get("enabled") else None)
        self.last_cache_hit: bool | None = None  # 最近一次检索是否命中缓存；未开启缓存时为 None（非线程安全）

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

    @property
    def metadata(self) -> MetadataIndex:
        """元数据索引（主题、章节类型、来源类型、统计周期），首次使用时从块派生。"""
        if self._metadata is None:
            self._metadata = MetadataIndex(self.chunks)
        return self._metadata

    def retrieve(self, normalized_query: str, filters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """检索证据并冻结 EvidenceBundle；空结果不会伪装为成功。

        ``filters`` 非空时带元数据过滤检索，过滤与放宽轨迹写入结果的 ``metadata_filter``。
        """
        evidence, trace, hit = self._search_cached(normalized_query, None, filters)
        if not evidence:
            raise MediDiagError(
                "RAG_NO_EVIDENCE",
                detail="医学检索未返回可追溯证据，禁止进入普通报告链路",
            )
        result = self.freeze_evidence(normalized_query, evidence)
        if trace is not None:
            result["metadata_filter"] = trace.to_dict()
        if hit is not None:
            result["retrieval_cache"] = {"hit": hit}
        return result

    def _search_cached(self, query: str, top_k: int | None, filters: Mapping[str, Any] | None
                       ) -> tuple[tuple[EvidenceItem, ...], FilterTrace | None, bool | None]:
        """查缓存，未命中再检索并写回；未开启缓存时 hit 为 None。只缓存检索结果，不缓存模型回答。"""
        if self.cache is None:
            evidence, trace = self._search_uncached(query, top_k, filters)
            self.last_cache_hit = None
            return evidence, trace, None
        key = cache_key(query, top_k=top_k, filters=filters, corpus_hash=self.corpus_hash,
                        corpus_version=self.corpus_version, config_hash=self.retrieval_config_hash)
        hit, value = self.cache.get(key)
        if not hit:
            value = self._search_uncached(query, top_k, filters)
            self.cache.put(key, value)
        self.last_cache_hit = hit
        return value[0], value[1], hit

    def search_filtered(self, query: str, top_k: int | None = None,
                        filters: Mapping[str, Any] | None = None) -> tuple[tuple[EvidenceItem, ...], FilterTrace | None]:
        """带元数据过滤的检索：返回证据与过滤轨迹（没有过滤条件时轨迹为 None）。

        过滤值先校验（未知值丢弃）；过滤后不足最终段数（配置的 top_k）时按
        section_type → topic → source_type → period 的顺序逐个放宽重查。
        配置了 retrieval.cache 时先查缓存，命中与否见 ``last_cache_hit``。
        """
        evidence, trace, _ = self._search_cached(query, top_k, filters)
        return evidence, trace

    def _search_uncached(self, query: str, top_k: int | None, filters: Mapping[str, Any] | None
                         ) -> tuple[tuple[EvidenceItem, ...], FilterTrace | None]:
        if not filters:
            return self._search(query, top_k, None), None
        min_results = int(self.retrieval_config.get("top_k", 5))
        evidence, trace = search_with_relaxation(
            lambda allowed: self._search(query, top_k, allowed), self.metadata, filters, min_results)
        return tuple(evidence), trace

    def search_evidence(self, query: str, top_k: int | None = None,
                        filters: Mapping[str, Any] | None = None) -> tuple[EvidenceItem, ...]:
        """只检索、不冻结；可能返回空元组。检索智能体用它做多次检索。

        配置了重排时返回的是重排后的前 top_k 段；``top_k`` 只供检索智能体取更多候选，不重排。
        配置了 MMR（retrieval.hybrid.mmr_lambda）时，去冗余作用于重排之后；没有重排时直接作用于 RRF 结果。
        ``filters`` 见 ``search_filtered``；轨迹留在 ``last_filter_trace``（非线程安全，需要轨迹时用 ``search_filtered``）。
        """
        evidence, self.last_filter_trace = self.search_filtered(query, top_k, filters)
        return evidence

    def _search(self, query: str, top_k: int | None, allowed_ids: set[str] | None) -> tuple[EvidenceItem, ...]:
        # 只有真正过滤时才给检索器传 allowed_ids，保持不支持该参数的检索器可用
        extra = {} if allowed_ids is None else {"allowed_ids": allowed_ids}
        if top_k is not None:
            results = self.retriever.search(query, top_k=top_k, **extra)
            return tuple(self._evidence_item(item) for item in results if item.chunk)
        top_k = int(self.retrieval_config.get("top_k", 5))
        if self.reranker is None:
            results = self.retriever.search(query, top_k=top_k, **extra)
            return self._apply_mmr(tuple(self._evidence_item(item) for item in results if item.chunk), top_k)
        candidates = self._search(query, self.reranker.candidates, allowed_ids)
        return self.rerank_evidence(query, candidates, top_k)

    def rerank_evidence(self, question: str, items: Sequence[EvidenceItem], top_k: int) -> tuple[EvidenceItem, ...]:
        """按交叉编码器分数重排候选并去重；final_score 换成重排分数，原检索分数保留在对应字段。

        重排前多取 top_k 个候选供 MMR 挑选，保证去冗余后仍能凑满 top_k 段。
        """
        if self.reranker is None:
            raise MediDiagError("RAG_CORPUS_INVALID", detail="没有配置重排模型")
        pool = top_k * 2 if _mmr_lambda(self.retrieval_config) is not None else top_k
        reranked = tuple(
            replace(item, final_score=score) for item, score in self.reranker.rerank(question, items, pool)
        )
        return self._apply_mmr(reranked, top_k)

    def _apply_mmr(self, items: tuple[EvidenceItem, ...], top_k: int) -> tuple[EvidenceItem, ...]:
        """混合检索开启 MMR 时按子块向量去冗余；向量复用检索阶段的编码结果。"""
        mmr_lambda = _mmr_lambda(self.retrieval_config)
        if mmr_lambda is None or not items:
            return items
        vector_for = getattr(self.retriever, "vector_for", None)
        if vector_for is None:
            return items
        from medidiag.rag.hybrid import mmr_select

        return tuple(mmr_select(items, vector_for=vector_for, top_k=top_k, mmr_lambda=mmr_lambda))

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
                "bm25_tokenizer": self.retrieval_config.get("bm25_tokenizer", "cjk_bigram"),
                "embedding_model": getattr(self.retriever, "embedding_model_name", None),
                "embedding_revision": getattr(
                    self.retriever, "embedding_model_revision", None
                ),
                "leakage_gate": self.leakage_gate,
                **({"hybrid": dict(self.retrieval_config["hybrid"])}
                   if isinstance(self.retrieval_config.get("hybrid"), dict) else {}),
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
