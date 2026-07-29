"""`eval/threshold_sensitivity.py` 的纯函数测试（不加载模型、不访问网络）。"""

from __future__ import annotations

from typing import Any

import pytest

from eval.threshold_sensitivity import (
    CANDIDATE_THRESHOLDS,
    KEYWORD_HIT_VALUE,
    ThresholdSensitivityError,
    classify,
    run_sensitivity,
)
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialty_data import MATCH_SATURATION_COUNT, THRESHOLDS


def _sample(
    total: float,
    top2_total: float = 0.0,
    keyword: float = 0.0,
    normalized_term: float = 0.0,
    specialty: str = "cardiology",
) -> dict[str, Any]:
    return {
        "sample_id": "s",
        "top1": {
            "specialty": specialty,
            "keyword_score": keyword,
            "normalized_term_score": normalized_term,
            "evidence_score": 0.0,
            "total": total,
        },
        "top2": {"specialty": "respiratory", "total": top2_total},
    }


def _artifact(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_set": "agent_manifest_v1",
        "branch_counts": {
            "rule_1_primary_below_threshold": 0,
            "rule_2_top1_plus_skeptic": 0,
            "rule_3_dynamic_top2": 0,
        },
        "samples": samples,
    }


class TestClassifyMatchesProductionRouter:
    """`classify` 必须与 `SpecialistRouter.route` 的规则顺序一致。

    这是本工具唯一的正确性依赖：如果两者漂移，反事实数字就不可与实测产物比较。
    """

    @pytest.mark.parametrize(
        ("top1", "top2"),
        [
            (0.0, 0.0),
            (1.9, 0.5),
            (2.0, 0.5),
            (2.0, 1.2),
            (5.0, 1.5),
            (5.0, 1.9),
            (3.0, 2.5),
            (6.0, 1.2),
        ],
    )
    def test_agrees_with_router_on_fallback_and_branch(
        self, top1: float, top2: float
    ) -> None:
        branch = classify(top1, top2, THRESHOLDS["MIN_PRIMARY_SCORE"])
        # 复现 router.route 的判定，只用它的阈值常量。
        if top1 < THRESHOLDS["MIN_PRIMARY_SCORE"]:
            expected = "rule_1_primary_below_threshold"
        elif (
            top2 < THRESHOLDS["MIN_SECONDARY_SCORE"]
            or (top1 - top2) >= THRESHOLDS["SCORE_GAP"]
        ):
            expected = "rule_2_top1_plus_skeptic"
        else:
            expected = "rule_3_dynamic_top2"
        assert branch == expected

    def test_router_still_has_exactly_three_rules(self) -> None:
        # 规则 4 已删除（DD-025）。若有人加回第四条规则，本工具的反事实会静默失真。
        source = SpecialistRouter.route.__doc__ or ""
        assert "LOW_CONFIDENCE" not in source
        assert "LOW_CONFIDENCE" not in THRESHOLDS

    def test_threshold_of_zero_never_falls_back(self) -> None:
        assert classify(0.0, 0.0, 0.0) != "rule_1_primary_below_threshold"


class TestKeywordHitTranslation:
    def test_one_hit_is_weight_over_saturation(self) -> None:
        # 关键词权重 3.0 / 饱和数 4.0 = 0.75 分每命中。这个换算是把抽象阈值
        # 翻译成"几个关键词"的全部依据。
        assert KEYWORD_HIT_VALUE == pytest.approx(0.75)
        assert MATCH_SATURATION_COUNT == 4.0

    def test_current_threshold_is_exactly_two_keyword_hits(self) -> None:
        """`MIN_PRIMARY_SCORE=1.5` 恰好等于 2 次关键词命中（DD-027）。

        旧值 2.0 落在 2 次（1.5）与 3 次（2.25）之间，因此是「2 次命中 + 一点
        其他信号」；新值正好落在 2 次命中上，即关键词证据本身就足够放行。
        """
        assert THRESHOLDS["MIN_PRIMARY_SCORE"] == pytest.approx(
            2 * KEYWORD_HIT_VALUE
        )
        assert 3 * KEYWORD_HIT_VALUE > THRESHOLDS["MIN_PRIMARY_SCORE"]

    def test_one_keyword_hit_alone_cannot_pass(self) -> None:
        # 单次命中（0.75）仍需另外 0.75 分才够，因此「一个词就放行」不成立。
        assert 1 * KEYWORD_HIT_VALUE < THRESHOLDS["MIN_PRIMARY_SCORE"]

    def test_hit_counts_are_recovered_from_scores(self) -> None:
        report = run_sensitivity(
            _artifact(
                [
                    _sample(2.5, keyword=0.5),
                    _sample(2.5, keyword=0.5),
                    _sample(3.0, keyword=0.75),
                    _sample(0.5, keyword=0.0),
                ]
            )
        )
        assert report["top1_keyword_hit_counts"] == {"0": 1, "2": 2, "3": 1}


class TestSensitivitySweep:
    def test_fallback_is_monotonic_in_the_threshold(self) -> None:
        samples = [_sample(float(i) / 10, top2_total=0.1) for i in range(0, 70)]
        report = run_sensitivity(_artifact(samples))
        series = [
            report["by_threshold"][f"min_primary_{t}"]["fallback_count"]
            for t in CANDIDATE_THRESHOLDS
        ]
        assert series == sorted(series)

    def test_branches_always_sum_to_sample_count(self) -> None:
        samples = [_sample(float(i) / 5, top2_total=float(i) / 8) for i in range(40)]
        report = run_sensitivity(_artifact(samples))
        for entry in report["by_threshold"].values():
            assert (
                entry["rule_1_primary_below_threshold"]
                + entry["rule_2_top1_plus_skeptic"]
                + entry["rule_3_dynamic_top2"]
                == report["sample_count"]
            )

    def test_current_production_value_is_flagged_exactly_once(self) -> None:
        report = run_sensitivity(_artifact([_sample(2.5)]))
        flagged = [
            entry
            for entry in report["by_threshold"].values()
            if entry["is_current_production_value"]
        ]
        assert len(flagged) == 1
        assert flagged[0]["min_primary_score"] == THRESHOLDS["MIN_PRIMARY_SCORE"]

    def test_signal_profile_separates_evidence_only_samples(self) -> None:
        report = run_sensitivity(
            _artifact(
                [
                    # 问题正文无任何专科信号，分数只来自检索证据。
                    _sample(0.6, keyword=0.0, normalized_term=0.0),
                    _sample(1.2, keyword=0.25, normalized_term=0.0),
                ]
            )
        )
        low = report["signal_profile_by_score_band"]["[0.0,1.0)"]
        assert low["count"] == 1
        assert low["evidence_only_no_query_signal"] == 1
        mid = report["signal_profile_by_score_band"]["[1.0,1.5)"]
        assert mid["evidence_only_no_query_signal"] == 0
        assert mid["keyword_hits_ge_1"] == 1


class TestInputValidation:
    def test_rejects_artifact_without_samples(self) -> None:
        with pytest.raises(ThresholdSensitivityError, match="samples"):
            run_sensitivity({"branch_counts": {}})

    def test_rejects_pre_rule4_removal_artifact(self) -> None:
        # 规则 4 删除前的产物按四条规则归类，反事实与它不可比（DD-025）。
        stale = _artifact([_sample(2.5)])
        stale["branch_counts"]["rule_4_low_confidence"] = 7
        with pytest.raises(ThresholdSensitivityError, match="rule-4"):
            run_sensitivity(stale)
