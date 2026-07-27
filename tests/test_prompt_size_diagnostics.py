"""`eval/prompt_size_diagnostics.py` 的纯函数测试（不加载模型、不访问网络）。"""

from __future__ import annotations

from eval.prompt_size_diagnostics import (
    CHARS_PER_TOKEN,
    CONTEXT_WINDOWS,
    _overflow_counts,
    _percentiles,
)


class TestPercentiles:
    def test_empty_input_is_all_zero(self) -> None:
        assert _percentiles([]) == {
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p99": 0,
            "max": 0,
        }

    def test_single_value_collapses_to_that_value(self) -> None:
        assert _percentiles([7]) == {
            "min": 7,
            "p50": 7,
            "p90": 7,
            "p99": 7,
            "max": 7,
        }

    def test_input_order_does_not_matter(self) -> None:
        forward = _percentiles(list(range(101)))
        backward = _percentiles(list(reversed(range(101))))
        assert forward == backward
        assert forward["min"] == 0
        assert forward["p50"] == 50
        assert forward["max"] == 100


class TestOverflowCounts:
    def test_covers_every_ratio_and_window(self) -> None:
        counts = _overflow_counts([1000])
        assert set(counts) == {f"chars_per_token_{r}" for r in CHARS_PER_TOKEN}
        for window_counts in counts.values():
            assert set(window_counts) == {
                f"over_{w // 1024}k_tokens" for w in CONTEXT_WINDOWS
            }

    def test_small_prompts_never_overflow(self) -> None:
        counts = _overflow_counts([5000, 6000, 7000])
        assert all(
            value == 0
            for window_counts in counts.values()
            for value in window_counts.values()
        )

    def test_counts_calls_not_samples(self) -> None:
        # 500,000 字符：/3.0 约 167K token，超过 128K；/4.0 为 125K，未超过。
        counts = _overflow_counts([500_000] * 3)
        assert counts["chars_per_token_3.0"]["over_128k_tokens"] == 3
        assert counts["chars_per_token_4.0"]["over_128k_tokens"] == 0

    def test_looser_ratio_never_reports_more_overflow(self) -> None:
        values = [10, 100_000, 400_000, 900_000]
        counts = _overflow_counts(values)
        for window in CONTEXT_WINDOWS:
            key = f"over_{window // 1024}k_tokens"
            series = [
                counts[f"chars_per_token_{ratio}"][key] for ratio in sorted(CHARS_PER_TOKEN)
            ]
            assert series == sorted(series, reverse=True)
