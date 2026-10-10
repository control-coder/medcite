"""应用安全模式工厂：配置校验先于索引构建，拒绝隐式联网与模型下载。"""

from pathlib import Path
from typing import Any

import yaml

from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.provider import DeterministicWorkflowProvider, WorkflowProvider
from medidiag.workflow.retrieval_mock import RetrievalMockWorkflowProvider

DENSE_MODEL = "BAAI/bge-small-zh-v1.5"
ON_INVALID_CHOICES = ("none", "resample", "feedback")
_SAFE_FLAGS_OFF = ("use_rerank", "use_term_normalization", "use_evidence_weighting", "use_citation_review")
_BM25_WEIGHTS = {"w1_bm25": 1.0, "w2_embedding": 0.0, "w3_evidence_level": 0.0, "w4_term_overlap": 0.0}
_DENSE_WEIGHTS = {"w1_bm25": 0.0, "w2_embedding": 1.0, "w3_evidence_level": 0.0, "w4_term_overlap": 0.0}


def _is_revision(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def load_application_config(path: str | Path) -> dict[str, Any]:
    """只接受两种显式检索方案：单路 BM25，或单路向量检索（固定版本、只读本地缓存）。"""
    config: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        flags = config["experiments"]["rag"]["rag_full"]["config"]
        common = (
            config["application"] == {"mode": "retrieval_mock", "topology": "single"}
            and all(flags.get(key) is False for key in _SAFE_FLAGS_OFF)
            and config["leakage_check"]["check_question_text"] is True
            and config["leakage_check"]["check_answer_key"] is True
            and set(config["leakage_check"]["chunk_fields_to_check"]) >= {"source", "source_id", "metadata.raw_id"}
            and config.get("generation", {}).get("on_invalid", "none") in ON_INVALID_CHOICES
            and isinstance(config.get("generation", {}).get("rewrite", False), bool)
            and isinstance(config.get("generation", {}).get("agent", False), bool)
        )
        weights = config["retrieval"]["weights"]
        bm25 = (flags.get("use_bm25") is True and flags.get("use_embedding") is False and weights == _BM25_WEIGHTS)
        embedding = config["embedding"]
        dense = (
            flags.get("use_bm25") is False and flags.get("use_embedding") is True and weights == _DENSE_WEIGHTS
            and embedding.get("model") == DENSE_MODEL and _is_revision(embedding.get("revision"))
            and embedding.get("local_files_only") is True
            and config["rerank"].get("model") == "disabled"
        )
        # 改写只对字面匹配检索有用；向量检索下几乎没有收益，却要多一次调用，所以不接受这种组合
        rewrite_ok = not (dense and config.get("generation", {}).get("rewrite", False))
        # 检索智能体自己决定要不要换说法，不能再叠加固定的问题改写
        agent_ok = not (config.get("generation", {}).get("agent", False) and config.get("generation", {}).get("rewrite", False))
        safe = common and (bm25 or dense) and rewrite_ok and agent_ok
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("应用配置缺少明确的安全检索参数。") from exc
    if not safe:
        raise ValueError("应用模式仅允许显式的单路 BM25 或单路固定版本向量检索配置，禁止套用研究/在线配置或禁用泄露检查。")
    return config


def retrieval_profile(config: dict[str, Any]) -> str:
    return "dense" if config["experiments"]["rag"]["rag_full"]["config"].get("use_embedding") else "bm25"


def _dense_factory(chunks: Any, **kwargs: Any) -> Any:
    from medidiag.rag.dense import DenseRetriever, load_local_encoder

    name, revision = kwargs["embedding_model"], kwargs["embedding_revision"]
    encoder = load_local_encoder(name, revision, device=kwargs.get("device", "cpu"),
                                 batch_size=int(kwargs.get("embedding_batch_size", 8)))
    return DenseRetriever(chunks, encoder=encoder, model_name=name, model_revision=revision)


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
    if retrieval_profile(config) == "dense":
        rag = RuntimeMedicalRAG.from_config(config, root=root, retriever_factory=_dense_factory)
    else:
        rag = RuntimeMedicalRAG.from_config(config, root=root)
    if mode == "retrieval_mock":
        return RetrievalMockWorkflowProvider(rag)
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
    if generation.get("agent", False):
        return RetrievalAgentWorkflowProvider(rag, llm=llm, on_invalid=generation.get("on_invalid", "none"))
    return MimoGroundedWorkflowProvider(rag, llm=llm, on_invalid=generation.get("on_invalid", "none"),
                                        rewrite=bool(generation.get("rewrite", False)))
