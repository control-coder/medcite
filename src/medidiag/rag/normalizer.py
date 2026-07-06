"""术语归一化模块占位。

阶段 4 实现三层归一化：
1. 轻量词典层：medical_terms.json 维护常见症状、疾病、检查项、缩写
2. 数据集派生层：从 MedQA / PubMedQA 的 question/answer/context 抽取高频医学短语，
   人工审核后加入 synonym map
3. 外部标准映射层：接入 MeSH descriptor 公开词表

不宣称 UMLS 级能力。
"""

from __future__ import annotations


class TerminologyNormalizer:
    """术语归一化器骨架。

    TODO[阶段4]: 实现三层归一化与 synonym map。
    """

    def __init__(self) -> None:
        raise NotImplementedError("TerminologyNormalizer 将在阶段 4 实现")
