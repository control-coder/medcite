"""应用安全模式工厂：配置校验先于索引构建，拒绝隐式联网与模型下载。"""

from pathlib import Path
from typing import Any

import yaml

from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.provider import DeterministicWorkflowProvider, WorkflowProvider
from medidiag.workflow.retrieval_mock import RetrievalMockWorkflowProvider


def load_application_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        flags = config["experiments"]["rag"]["rag_full"]["config"]
        safe = (
            config["application"] == {"mode": "retrieval_mock", "topology": "single"}
            and flags.get("use_bm25") is True
            and all(flags.get(key) is False for key in (
                "use_embedding", "use_rerank", "use_term_normalization", "use_evidence_weighting", "use_citation_review"))
            and config["retrieval"]["weights"] == {
                "w1_bm25": 1.0, "w2_embedding": 0.0, "w3_evidence_level": 0.0, "w4_term_overlap": 0.0}
            and config["leakage_check"]["check_question_text"] is True
            and config["leakage_check"]["check_answer_key"] is True
            and set(config["leakage_check"]["chunk_fields_to_check"]) >= {"source", "source_id", "metadata.raw_id"}
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("应用配置缺少明确的安全检索参数。") from exc
    if not safe:
        raise ValueError("应用模式仅允许显式的单路纯 BM25 配置，禁止套用研究/在线配置或禁用泄露检查。")
    return config


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
    rag = RuntimeMedicalRAG.from_config(config, root=root)
    if mode == "retrieval_mock":
        return RetrievalMockWorkflowProvider(rag)
    from medidiag.llm.budget import BudgetedTransport
    from medidiag.llm.openai_compatible import OpenAICompatibleProvider
    from medidiag.llm.profiles import get_provider_profile
    from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider

    llm = OpenAICompatibleProvider(get_provider_profile("mimo_v25"),
        post=BudgetedTransport(live_budget), max_retries=0, max_structured_retries=0)
    if not llm.is_configured:
        raise ValueError("mimo_v25 配置不完整，请通过安全环境配置凭据。")
    return MimoGroundedWorkflowProvider(rag, llm=llm)
