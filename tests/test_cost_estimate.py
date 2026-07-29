"""`eval/cost_estimate.py` 的纯函数测试（不加载模型、不访问网络、不调用 provider）。"""

from __future__ import annotations

import pytest

from eval.cost_estimate import (
    CACHE_BLOCK_TOKENS,
    CHARS_PER_TOKEN,
    PRICE_INPUT_CACHE_HIT,
    PRICE_INPUT_CACHE_MISS,
    PRICE_OUTPUT,
    CostEstimateError,
    actual_cost_from_manifest,
    build_cost_report,
    cache_hit_tokens_ceiling,
    estimate_scope,
)

PRICES = (PRICE_INPUT_CACHE_HIT, PRICE_INPUT_CACHE_MISS, PRICE_OUTPUT)


def _diagnostics(**overrides: object) -> dict[str, object]:
    """最小的 prompt 体积产物，字段名与 eval/prompt_size_diagnostics.py 一致。"""
    payload: dict[str, object] = {
        "system_prompt_chars": 655,
        "max_tokens_per_call": 2048,
        "knowledge_base_chunk_count": 12728,
        "experiments": {
            "agent_single": {"provider_calls": 100, "prompt_chars_total": 499_861},
        },
        "agent_total": {"provider_calls": 500, "prompt_chars_total": 2_497_517},
        "all_generation_total": {
            "provider_calls": 1100,
            "prompt_chars_total": 4_624_836,
        },
    }
    payload.update(overrides)
    return payload


class TestPricingArithmetic:
    def test_official_prices_are_the_documented_values(self) -> None:
        # 定价是外部事实，改动它必须是有意的（DD-026）。
        assert (PRICE_INPUT_CACHE_HIT, PRICE_INPUT_CACHE_MISS, PRICE_OUTPUT) == (
            0.02,
            1.0,
            2.0,
        )

    def test_one_million_tokens_costs_exactly_the_unit_price(self) -> None:
        scope = estimate_scope(1, 4_000_000, 0, 0, *PRICES)
        band = scope["by_chars_per_token"]["chars_per_token_4.0"]
        assert band["input_tokens_estimated"] == 1_000_000
        # 无 system prompt 则无缓存前缀，全部按 cache-miss 计价。
        assert band["cost_input_only"]["total"] == pytest.approx(PRICE_INPUT_CACHE_MISS)

    def test_output_priced_at_max_tokens_times_calls(self) -> None:
        scope = estimate_scope(1000, 0, 0, 1000, *PRICES)
        band = scope["by_chars_per_token"]["chars_per_token_4.0"]
        assert band["output_tokens_at_max"] == 1_000_000
        assert band["cost_at_max_output"]["output"] == pytest.approx(PRICE_OUTPUT)

    def test_total_is_the_sum_of_its_parts(self) -> None:
        scope = estimate_scope(500, 2_497_517, 655, 2048, *PRICES)
        for band in scope["by_chars_per_token"].values():
            for key in ("cost_at_max_output", "cost_input_only", "cost_at_max_output_no_cache"):
                cost = band[key]
                assert cost["total"] == pytest.approx(
                    cost["input_cache_hit"] + cost["input_cache_miss"] + cost["output"],
                    abs=1e-4,
                )

    def test_cheaper_cache_hit_price_never_raises_cost(self) -> None:
        # cache-hit 单价远低于 cache-miss，因此计入缓存不可能比全 miss 更贵。
        scope = estimate_scope(500, 2_497_517, 655, 2048, *PRICES)
        for band in scope["by_chars_per_token"].values():
            assert (
                band["cost_at_max_output"]["total"]
                <= band["cost_at_max_output_no_cache"]["total"]
            )

    def test_input_only_is_a_lower_bound_of_max_output(self) -> None:
        scope = estimate_scope(500, 2_497_517, 655, 2048, *PRICES)
        for band in scope["by_chars_per_token"].values():
            assert band["cost_input_only"]["total"] <= band["cost_at_max_output"]["total"]

    def test_lower_chars_per_token_never_estimates_fewer_input_tokens(self) -> None:
        scope = estimate_scope(500, 2_497_517, 655, 2048, *PRICES)
        series = [
            scope["by_chars_per_token"][f"chars_per_token_{ratio}"][
                "input_tokens_estimated"
            ]
            for ratio in sorted(CHARS_PER_TOKEN)
        ]
        assert series == sorted(series, reverse=True)

    def test_covers_every_ratio(self) -> None:
        scope = estimate_scope(10, 1000, 655, 2048, *PRICES)
        assert set(scope["by_chars_per_token"]) == {
            f"chars_per_token_{ratio}" for ratio in CHARS_PER_TOKEN
        }


class TestCacheCeiling:
    def test_first_call_can_never_hit(self) -> None:
        assert cache_hit_tokens_ceiling(1, 655, 4.0) == 0
        assert cache_hit_tokens_ceiling(0, 655, 4.0) == 0

    def test_rounds_down_to_whole_cache_blocks(self) -> None:
        # 655 字符 / 4.0 约 164 token；64 粒度下每次调用最多命中 128。
        assert cache_hit_tokens_ceiling(2, 655, 4.0) == 128
        assert cache_hit_tokens_ceiling(11, 655, 4.0) == 1280

    def test_prefix_shorter_than_one_block_hits_nothing(self) -> None:
        assert cache_hit_tokens_ceiling(1000, CACHE_BLOCK_TOKENS * 2 - 2, 1.0) == (
            CACHE_BLOCK_TOKENS * 999
        )
        assert cache_hit_tokens_ceiling(1000, 10, 4.0) == 0

    def test_hit_ceiling_never_exceeds_total_input(self) -> None:
        # 极端情形：调用多、prompt 短。命中量不得超过实际输入 token 数。
        scope = estimate_scope(1000, 1000, 100_000, 0, *PRICES)
        for band in scope["by_chars_per_token"].values():
            assert (
                band["input_cache_hit_tokens_ceiling"] <= band["input_tokens_estimated"]
            )
            assert band["input_cache_miss_tokens"] >= 0


class TestActualCostFromManifest:
    def test_retrieval_only_run_returns_none(self) -> None:
        # 纯检索 run 不调用 generation，provider_usage 为空 —— 不是错误。
        manifest = {"run_id": "r1", "generation": {"provider_usage": {}}}
        assert actual_cost_from_manifest(manifest, *PRICES) is None

    def test_missing_generation_block_returns_none(self) -> None:
        assert actual_cost_from_manifest({"run_id": "r1"}, *PRICES) is None

    def test_uses_provider_reported_cache_split(self) -> None:
        manifest = {
            "run_id": "r1",
            "generation": {
                "provider_usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 500_000,
                    "prompt_cache_hit_tokens": 400_000,
                    "prompt_cache_miss_tokens": 600_000,
                }
            },
        }
        actual = actual_cost_from_manifest(manifest, *PRICES)
        assert actual is not None
        assert actual["measured"] is True
        assert actual["cache_split_source"] == "provider_reported"
        assert actual["cache_split_reconciles_prompt_tokens"] is True
        # 0.4M×0.02 + 0.6M×1.0 + 0.5M×2.0 = 0.008 + 0.6 + 1.0
        assert actual["cost"]["total"] == pytest.approx(1.608)

    def test_absent_cache_split_is_charged_as_all_miss(self) -> None:
        manifest = {
            "run_id": "r1",
            "generation": {
                "provider_usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}
            },
        }
        actual = actual_cost_from_manifest(manifest, *PRICES)
        assert actual is not None
        assert actual["cache_split_source"] == "absent_assumed_all_miss"
        assert actual["prompt_cache_hit_tokens"] == 0
        assert actual["cost"]["total"] == pytest.approx(PRICE_INPUT_CACHE_MISS)

    def test_inconsistent_split_is_reported_not_silently_aligned(self) -> None:
        manifest = {
            "run_id": "r1",
            "generation": {
                "provider_usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                    "prompt_cache_hit_tokens": 1,
                    "prompt_cache_miss_tokens": 2,
                }
            },
        }
        actual = actual_cost_from_manifest(manifest, *PRICES)
        assert actual is not None
        assert actual["cache_split_reconciles_prompt_tokens"] is False


class TestBuildCostReport:
    def test_reports_measurement_basis_of_every_layer(self) -> None:
        report = build_cost_report(_diagnostics())
        basis = report["measurement_basis"]
        assert basis["prompt_chars"].startswith("measured")
        assert basis["input_tokens"].startswith("estimated")
        assert basis["output_tokens"].startswith("ceiling")
        assert basis["cache_hit_tokens"].startswith("ceiling")

    def test_missing_required_block_raises(self) -> None:
        payload = _diagnostics()
        del payload["agent_total"]
        with pytest.raises(CostEstimateError, match="agent_total"):
            build_cost_report(payload)

    def test_price_override_is_recorded(self) -> None:
        report = build_cost_report(_diagnostics(), 0.05, 2.0, 8.0)
        assert report["pricing"]["overridden"] is True
        assert report["pricing"]["output"] == 8.0

    def test_default_prices_are_not_flagged_as_overridden(self) -> None:
        assert build_cost_report(_diagnostics())["pricing"]["overridden"] is False

    def test_no_actual_section_without_a_manifest(self) -> None:
        assert "actual" not in build_cost_report(_diagnostics())

    def test_retrieval_only_manifest_records_why_actual_is_absent(self) -> None:
        report = build_cost_report(
            _diagnostics(), *PRICES, manifest={"run_id": "r1", "generation": {}}
        )
        assert report["actual"]["measured"] is False
        assert "retrieval-only" in report["actual"]["reason"]

    def test_scope_totals_are_consistent_with_measured_chars(self) -> None:
        report = build_cost_report(_diagnostics())
        assert report["all_generation_total"]["prompt_chars_total_measured"] == 4_624_836
        assert report["all_generation_total"]["provider_calls"] == 1100
        assert set(report["experiments"]) == {"agent_single"}
