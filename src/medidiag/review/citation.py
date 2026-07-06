"""引用校验模块占位。

阶段 5/6 实现 CitationVerifier：
- 输入 claim 和 evidence chunk
- 使用固定 NLI / cross-encoder 模型判定 SUPPORTED / PARTIAL / UNSUPPORTED
- LLM judge 只作为辅助解释，不作为唯一真值
- 抽样人工复核 + Cohen's Kappa 一致性统计
"""

from __future__ import annotations

from enum import Enum


class CitationVerdict(str, Enum):
    """引用校验判定结果。"""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class CitationVerifier:
    """引用校验器骨架。

    TODO[阶段5]: 接入 microsoft/deberta-v3-base-mnli 进行 NLI 判定。
    """

    def __init__(self) -> None:
        raise NotImplementedError("CitationVerifier 将在阶段 5 实现")
