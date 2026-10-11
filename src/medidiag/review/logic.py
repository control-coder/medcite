"""复核轮次上限判定。

复核被驳回后，连续失败达到上限就升级人工，不进入死状态。
"""

from __future__ import annotations

# 连续失败阈值（达到则进入人工升级）
MAX_CONSECUTIVE_FAILURES = 3


class ClinicalLogicReviewer:
    """复核轮次判定。"""

    @staticmethod
    def should_escalate(
        consecutive_failures: int,
        max_rounds: int = MAX_CONSECUTIVE_FAILURES,
    ) -> bool:
        """判断是否应该升级人工。

        SingleMachineWorker 用运行时配置的 max_review_rounds 调用本函数，
        使复核轮次上限只有这一处判定逻辑。

        Args:
            consecutive_failures: 连续审核失败次数。
            max_rounds: 上限轮次，默认为 MAX_CONSECUTIVE_FAILURES。

        Returns:
            True 如果应该升级。
        """
        return consecutive_failures >= max_rounds
