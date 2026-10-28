"""应用安全模式工厂：配置校验先于索引构建，拒绝隐式联网与模型下载。"""

from pathlib import Path
from typing import Any

import yaml

from medidiag.rag.rerank import RERANK_MODEL
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.generative_answer import ANSWER_MODES
from medidiag.workflow.provider import DeterministicWorkflowProvider, WorkflowProvider
from medidiag.workflow.retrieval_mock import RetrievalMockWorkflowProvider

DENSE_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_RERANK_CANDIDATES = 50
ON_INVALID_CHOICES = ("none", "resample", "feedback")
BM25_TOKENIZERS = ("cjk_bigram", "jieba")
AGENT_VERSIONS = (1, 2)
VECTOR_STORES = ("numpy", "pgvector")
_HYBRID_KEYS = {"candidates", "rrf_k", "mmr_lambda"}
_ALLOWED_KEYS = {"application", "runtime", "embedding", "dataset", "retrieval", "generation", "leakage_check"}


def _is_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def _embedding_fixed(embedding: dict[str, Any]) -> bool:
    """固定版本的向量模型、只读本地缓存；dense 与 hybrid 使用同一校验。"""
    return (
        embedding.get("model") == DENSE_MODEL and _is_revision(embedding.get("revision"))
        and embedding.get("local_files_only") is True
    )


def load_application_config(path: str | Path) -> dict[str, Any]:
    """只接受三种显式检索方案：单路 BM25、单路向量检索、BM25 加向量的混合检索（均需固定版本、只读本地缓存）；都可加固定版本的重排。"""
    config: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        generation = config.get("generation", {})
        common = (
            set(config) <= _ALLOWED_KEYS  # 研究配置里的重排、权重、实验开关等键一律拒绝
            and config["application"] == {"mode": "retrieval_mock", "topology": "single"}
            and config["leakage_check"]["check_question_text"] is True
            and config["leakage_check"]["check_answer_key"] is True
            and set(config["leakage_check"]["chunk_fields_to_check"]) >= {"source", "source_id", "metadata.raw_id"}
            and generation.get("on_invalid", "none") in ON_INVALID_CHOICES
            and generation.get("answer_mode", "extractive") in ANSWER_MODES
            and isinstance(generation.get("rewrite", False), bool)
            and isinstance(generation.get("agent", False), bool)
            and isinstance(generation.get("context_rewrite", False), bool)
            and isinstance(generation.get("query_router", False), bool)
            and config["retrieval"].get("bm25_tokenizer", "cjk_bigram") in BM25_TOKENIZERS
            and config["retrieval"].get("vector_store", "numpy") in VECTOR_STORES
        )
        method = config["retrieval"]["method"]
        bm25 = method == "bm25" and "embedding" not in config
        embedding = config.get("embedding", {})
        dense = method == "dense" and _embedding_fixed(embedding)
        hybrid = method == "hybrid" and _embedding_fixed(embedding)
        # 改写只对字面匹配检索有用；向量检索（含混合检索的向量一路）下几乎没有收益，却要多一次调用，所以不接受这种组合
        rewrite_ok = not ((dense or hybrid) and generation.get("rewrite", False))
        # 检索智能体自己决定要不要换说法，不能再叠加固定的问题改写
        agent_ok = not (generation.get("agent", False) and generation.get("rewrite", False))
        # agent_version 只在开启检索智能体时才有意义：1 是 v1（默认），2 是 M11 的多工具检索 Agent
        agent_version = generation.get("agent_version", 1)
        agent_ok = agent_ok and type(agent_version) is int and agent_version in AGENT_VERSIONS and (
            agent_version == 1 or generation.get("agent", False) is True)
        # 向量存储只对用到向量的检索方案有意义，BM25 单路不接受
        store_ok = "vector_store" not in config["retrieval"] or dense or hybrid
        safe = (
            common and store_ok and (bm25 or dense or hybrid) and rewrite_ok and agent_ok
            and _rerank_ok(config["retrieval"]) and _hybrid_ok(config["retrieval"], method)
            and _cache_ok(config["retrieval"])
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("应用配置缺少明确的安全检索参数。") from exc
    if not safe:
        raise ValueError("应用模式仅允许显式的单路 BM25、单路固定版本向量检索或混合检索配置，禁止套用研究/在线配置或禁用泄露检查。")
    return config


def _hybrid_ok(retrieval: dict[str, Any], method: str) -> bool:
    """混合检索参数只在 method=hybrid 时给出；键集合固定、取值范围有限。"""
    hybrid = retrieval.get("hybrid")
    if hybrid is None:
        return True
    if method != "hybrid" or not isinstance(hybrid, dict) or not set(hybrid) <= _HYBRID_KEYS:
        return False
    candidates = hybrid.get("candidates", 20)
    rrf_k = hybrid.get("rrf_k", 60)
    mmr_lambda = hybrid.get("mmr_lambda", 0.7)
    return (
        type(candidates) is int and 5 <= candidates <= MAX_RERANK_CANDIDATES
        and type(rrf_k) is int and 1 <= rrf_k <= 1000
        and isinstance(mmr_lambda, (int, float)) and not isinstance(mmr_lambda, bool)
        and 0.0 <= float(mmr_lambda) <= 1.0
    )


def _cache_ok(retrieval: dict[str, Any]) -> bool:
    """检索缓存可选（M14）：键集合固定，容量与 TTL 有上限；默认不配置即关闭。"""
    cache = retrieval.get("cache")
    if cache is None:
        return True
    if not isinstance(cache, dict) or not set(cache) <= {"enabled", "max_entries", "ttl_s"}:
        return False
    max_entries, ttl_s = cache.get("max_entries", 256), cache.get("ttl_s", 3600)
    return (
        type(cache.get("enabled", False)) is bool
        and type(max_entries) is int and 1 <= max_entries <= 10000
        and isinstance(ttl_s, (int, float)) and not isinstance(ttl_s, bool) and 0 < ttl_s <= 86400
    )


def _rerank_ok(retrieval: dict[str, Any]) -> bool:
    """重排可选；开启时只接受固定的交叉编码器、固定版本、只读本地缓存，候选数不少于最终段数。"""
    rerank = retrieval.get("rerank")
    if rerank is None:
        return True
    candidates = rerank.get("candidates") if isinstance(rerank, dict) else None
    return (
        isinstance(rerank, dict) and set(rerank) == {"model", "revision", "local_files_only", "candidates"}
        and rerank["model"] == RERANK_MODEL and _is_revision(rerank["revision"])
        and rerank["local_files_only"] is True
        and type(candidates) is int and int(retrieval.get("top_k", 5)) <= candidates <= MAX_RERANK_CANDIDATES
    )


def retrieval_profile(config: dict[str, Any]) -> str:
    return str(config["retrieval"]["method"])


def _dense_factory(chunks: Any, config: dict[str, Any]) -> Any:
    from medidiag.rag.dense import DenseRetriever, load_local_encoder, store_options

    name, revision = config["embedding"]["model"], config["embedding"]["revision"]
    runtime = config.get("runtime", {})
    encoder = load_local_encoder(name, revision, device=str(runtime.get("device", "cpu")),
                                 batch_size=int(runtime.get("batch_size", 8)))
    return DenseRetriever(chunks, encoder=encoder, model_name=name, model_revision=revision,
                          **store_options(chunks, config))


def _rerank_factory(config: dict[str, Any]) -> Any:
    from medidiag.rag.rerank import Reranker, load_local_reranker

    rerank = config["retrieval"]["rerank"]
    device = str(config.get("runtime", {}).get("device", "cpu"))
    return Reranker(load_local_reranker(rerank["model"], rerank["revision"], device=device),
                    model_name=rerank["model"], model_revision=rerank["revision"], candidates=rerank["candidates"])


def build_application_provider(mode: str = "fake_offline", *,
                               app_config: str = "configs/application.yaml",
                               root: str | Path = ".", review_verdict: str = "APPROVED",
                               live_budget: str | Path | None = None) -> WorkflowProvider:
    """CLI 和 Celery 共享相同安全模式，不根据密钥是否存在切换在线服务。"""
    if mode == "fake_offline":
        return DeterministicWorkflowProvider(review_verdict=review_verdict)
    if mode not in {"retrieval_mock", "mimo_grounded"}:
        raise ValueError("应用模式只能是 fake_offline、retrieval_mock 或 mimo_grounded。")
    if mode == "mimo_grounded" and not live_budget:
        raise ValueError("真实应用必须显式指定 --live-budget 持久账本；选择该模式表示已授权调用。")
    path = Path(app_config)
    config = load_application_config(path if path.is_absolute() else Path(root) / path)
    if mode == "retrieval_mock" and config.get("generation", {}).get("answer_mode", "extractive") != "extractive":
        raise ValueError("生成式回答需要真实模型，离线 retrieval_mock 模式只支持抽取式。")
    reranker_factory = _rerank_factory if config["retrieval"].get("rerank") else None
    method = retrieval_profile(config)
    if method == "dense":
        rag = RuntimeMedicalRAG.from_config(config, root=root, retriever_factory=_dense_factory,
                                            reranker_factory=reranker_factory)
    elif method == "hybrid":
        from medidiag.rag.runtime import _hybrid_factory

        rag = RuntimeMedicalRAG.from_config(config, root=root, retriever_factory=_hybrid_factory,
                                            reranker_factory=reranker_factory)
    else:
        rag = RuntimeMedicalRAG.from_config(config, root=root, reranker_factory=reranker_factory)
    query_router = bool(config.get("generation", {}).get("query_router", False))
    answer_mode = config.get("generation", {}).get("answer_mode", "extractive")
    if mode == "retrieval_mock":
        # 离线模式没有 LLM，路由只用规则级
        return RetrievalMockWorkflowProvider(rag, query_router=query_router)
    from medidiag.llm.budget import BudgetedTransport
    from medidiag.llm.openai_compatible import OpenAICompatibleProvider
    from medidiag.llm.profiles import get_provider_profile
    from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
    from medidiag.workflow.retrieval_agent import RetrievalAgentWorkflowProvider

    assert live_budget is not None  # 上方已拒绝缺少账本的真实模式，此处仅供类型收窄
    llm = OpenAICompatibleProvider(get_provider_profile("mimo_v25"),
        post=BudgetedTransport(live_budget), max_retries=0, max_structured_retries=0)
    if not llm.is_configured:
        raise ValueError("mimo_v25 配置不完整，请通过安全环境配置凭据。")
    generation = config.get("generation", {})
    # 多轮追问检索前先补全问题；只影响带上一轮的病例，单轮问题的请求不变
    context_rewrite = bool(generation.get("context_rewrite", False))
    if generation.get("agent", False):
        if generation.get("agent_version", 1) == 2:
            from medidiag.workflow.retrieval_agent_v2 import RetrievalAgentV2WorkflowProvider

            return RetrievalAgentV2WorkflowProvider(
                rag, llm=llm, on_invalid=generation.get("on_invalid", "none"),
                context_rewrite=context_rewrite, query_router=query_router, answer_mode=answer_mode)
        return RetrievalAgentWorkflowProvider(rag, llm=llm, on_invalid=generation.get("on_invalid", "none"),
                                              context_rewrite=context_rewrite, query_router=query_router,
                                              answer_mode=answer_mode)
    return MimoGroundedWorkflowProvider(rag, llm=llm, on_invalid=generation.get("on_invalid", "none"),
                                        rewrite=bool(generation.get("rewrite", False)), context_rewrite=context_rewrite,
                                        query_router=query_router, answer_mode=answer_mode)
