"""诊断逻辑审核模块占位。

阶段 5 实现 ClinicalLogicReviewer：
- 检查诊断建议是否包含证据、风险提示、检查建议和不确定性说明
- 规则检查为主，必要时调用 LLM 生成审核意见
- 审核驳回闭环：连续失败超过阈值进入人工升级，不进死状态
"""

from __future__ import annotations


class ClinicalLogicReviewer:
    """诊断逻辑审核器骨架。

    TODO[阶段5]: 实现规则检查 + LLM 审核意见生成。
    """

    def __init__(self) -> None:
        raise NotImplementedError("ClinicalLogicReviewer 将在阶段 5 实现")
