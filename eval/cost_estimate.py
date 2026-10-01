"""按 DeepSeek 官方定价把实测 prompt 字符量折算成费用，不调用任何 provider。

这是**费用估算工具**，不产生任何评测指标，也不加载模型、不访问网络：它只读
`eval/prompt_size_diagnostics.py` 写出的 JSON（实测字符数），以及可选的一次已完成
run 的 `manifest.json`（实测 token 数）。

三条口径必须分清，输出 JSON 也按这三条分层：

1. **字符数是实测值** —— 真实检索 + 真实 `build_prompt` 的结果，来自
   `artifacts/reports/prompt_size_diagnostics.json`。
2. **token 数是估算值** —— 由字符数除以 3.0–4.5 的字符/token 假设得到，未经
   tokenizer 核实。DeepSeek 未公开可离线复现的 tokenizer，因此这一层无法消除。
3. **输出 token 用的是上限** —— `max_tokens × 调用次数`。实际输出长度**从未被
   测量过**，所以这里给的是天花板而不是期望值。

只有 `--run-dir` 指向一次真实付费 run 时，`actual` 一节才是实测费用；其余全部是估算。

用法::

    python -m eval.cost_estimate
    python -m eval.cost_estimate --run-dir artifacts/reports/raw/<run_id>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

ROOT = Path(__file__).resolve().parent.parent

# DeepSeek 官方定价（人民币 / 每百万 token），2026-07-29 记入，决策见 DD-026。
# 价格随模型与时间变化，三个 `--price-*` 选项可覆盖；覆盖后的值会写进输出 JSON。
PRICE_INPUT_CACHE_HIT = 0.02
PRICE_INPUT_CACHE_MISS = 1.0
PRICE_OUTPUT = 2.0
CURRENCY = "CNY"

# 与 eval/prompt_size_diagnostics.py 保持同一组假设，使两份产物可逐档对照。
CHARS_PER_TOKEN = (3.0, 3.5, 4.0, 4.5)

# DeepSeek 自动上下文缓存的命中粒度（token）。不足一整块的尾部不计入命中。
CACHE_BLOCK_TOKENS = 64


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class CostEstimateError(Exception):
    """输入产物缺少估算所需的实测字段。"""


def _price_block(
    price_hit: float, price_miss: float, price_output: float
) -> dict[str, Any]:
    return {
        "currency": CURRENCY,
        "unit": "per_1m_tokens",
        "input_cache_hit": price_hit,
        "input_cache_miss": price_miss,
        "output": price_output,
        "source": "DeepSeek official pricing, recorded 2026-07-29",
        "overridden": (
            price_hit != PRICE_INPUT_CACHE_HIT
            or price_miss != PRICE_INPUT_CACHE_MISS
            or price_output != PRICE_OUTPUT
        ),
    }


def _cost(
    input_hit_tokens: float,
    input_miss_tokens: float,
    output_tokens: float,
    price_hit: float,
    price_miss: float,
    price_output: float,
) -> dict[str, float]:
    """把 token 数按每百万 token 单价折算成费用，分项与合计一起返回。"""
    hit = input_hit_tokens / 1_000_000 * price_hit
    miss = input_miss_tokens / 1_000_000 * price_miss
    out = output_tokens / 1_000_000 * price_output
    return {
        "input_cache_hit": round(hit, 4),
        "input_cache_miss": round(miss, 4),
        "output": round(out, 4),
        "total": round(hit + miss + out, 4),
    }


def cache_hit_tokens_ceiling(calls: int, system_prompt_chars: int, ratio: float) -> int:
    """本工作负载**最多**能命中的缓存 token 数（上限，不是实测）。

    唯一的稳定公共前缀是 `AGENT_SYSTEM_PROMPT`；用户消息从第一句起就逐样本不同
    。DeepSeek 自动缓存按
    `CACHE_BLOCK_TOKENS` 粒度命中，因此每次调用的命中量向下取整到整块。

    这是上限而非期望值，两个原因：首次调用必然全 miss，且缓存条目有存活期，
    本项目从未观测过一次真实的 `prompt_cache_hit_tokens`。
    """
    prefix_tokens = system_prompt_chars / ratio
    per_call = int(prefix_tokens // CACHE_BLOCK_TOKENS) * CACHE_BLOCK_TOKENS
    return per_call * max(0, calls - 1)


def estimate_scope(
    calls: int,
    prompt_chars_total: int,
    system_prompt_chars: int,
    max_tokens: int,
    price_hit: float,
    price_miss: float,
    price_output: float,
) -> dict[str, Any]:
    """对一个范围（单个实验 / agent 三组 / 全部五组）逐档估算费用。

    每个字符/token 档位给两条边界，因为实际输出长度未知：

    - `output_at_max_tokens`：输出打满 `max_tokens`，费用上界。
    - `input_only`：输出为 0，费用下界（实际不可能达到，但界定了输入侧的份额）。

    缓存按 `cache_hit_tokens_ceiling` 的上限计入，因此这里报的是**最乐观的缓存
    情形**；真实费用不会低于 `no_cache` 一列。
    """
    by_ratio: dict[str, Any] = {}
    for ratio in CHARS_PER_TOKEN:
        input_tokens = prompt_chars_total / ratio
        hit_tokens = min(
            float(cache_hit_tokens_ceiling(calls, system_prompt_chars, ratio)),
            input_tokens,
        )
        miss_tokens = input_tokens - hit_tokens
        output_ceiling = float(max_tokens * calls)
        by_ratio[f"chars_per_token_{ratio}"] = {
            "input_tokens_estimated": round(input_tokens),
            "input_cache_hit_tokens_ceiling": round(hit_tokens),
            "input_cache_miss_tokens": round(miss_tokens),
            "output_tokens_at_max": round(output_ceiling),
            "cost_at_max_output": _cost(
                hit_tokens, miss_tokens, output_ceiling, price_hit, price_miss, price_output
            ),
            "cost_input_only": _cost(
                hit_tokens, miss_tokens, 0.0, price_hit, price_miss, price_output
            ),
            # 缓存全不命中的对照，是唯一不依赖缓存假设的输入侧数字。
            "cost_at_max_output_no_cache": _cost(
                0.0, input_tokens, output_ceiling, price_hit, price_miss, price_output
            ),
        }
    return {
        "provider_calls": calls,
        "prompt_chars_total_measured": prompt_chars_total,
        "max_tokens_per_call": max_tokens,
        "by_chars_per_token": by_ratio,
    }


def actual_cost_from_manifest(
    manifest: dict[str, Any],
    price_hit: float,
    price_miss: float,
    price_output: float,
) -> dict[str, Any] | None:
    """从一次已完成 run 的 manifest 读取实测 usage 并折算实际费用。

    返回 `None` 表示该 run 没有 provider usage —— 纯检索 run 不调用 generation，
    `provider_usage` 为空字典。这不是错误，调用方据此跳过 `actual` 一节。

    只有这一节是实测费用；`estimate` 全部是估算。
    """
    usage = manifest.get("generation", {}).get("provider_usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None

    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if isinstance(hit, int) and isinstance(miss, int) and hit + miss > 0:
        # provider 自报的命中/未命中拆分优先，它是唯一的实测缓存证据。
        cache_source = "provider_reported"
        hit_tokens, miss_tokens = float(hit), float(miss)
        # hit + miss 应等于 prompt_tokens；不等时如实记录而不静默对齐。
        reconciled = hit + miss == prompt_tokens
    else:
        cache_source = "absent_assumed_all_miss"
        hit_tokens, miss_tokens = 0.0, float(prompt_tokens)
        reconciled = True

    return {
        "run_id": manifest.get("run_id", ""),
        "measured": True,
        "cache_split_source": cache_source,
        "cache_split_reconciles_prompt_tokens": reconciled,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cache_hit_tokens": round(hit_tokens),
        "prompt_cache_miss_tokens": round(miss_tokens),
        "cost": _cost(
            hit_tokens, miss_tokens, float(completion_tokens),
            price_hit, price_miss, price_output,
        ),
    }


def build_cost_report(
    diagnostics: dict[str, Any],
    price_hit: float = PRICE_INPUT_CACHE_HIT,
    price_miss: float = PRICE_INPUT_CACHE_MISS,
    price_output: float = PRICE_OUTPUT,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """由 prompt 体积产物构建完整费用报告。

    `diagnostics` 必须是 `eval/prompt_size_diagnostics.py` 写出的 JSON。
    """
    for key in ("experiments", "agent_total", "all_generation_total"):
        if key not in diagnostics:
            raise CostEstimateError(
                f"prompt size diagnostics missing '{key}'; "
                "regenerate with `python -m eval.prompt_size_diagnostics`"
            )
    system_chars = int(diagnostics["system_prompt_chars"])
    max_tokens = int(diagnostics["max_tokens_per_call"])

    def scope(block: dict[str, Any]) -> dict[str, Any]:
        return estimate_scope(
            int(block["provider_calls"]),
            int(block["prompt_chars_total"]),
            system_chars,
            max_tokens,
            price_hit,
            price_miss,
            price_output,
        )

    report: dict[str, Any] = {
        "pricing": _price_block(price_hit, price_miss, price_output),
        "measurement_basis": {
            "prompt_chars": "measured (real retrieval + real build_prompt)",
            "input_tokens": "estimated (chars / chars-per-token, no tokenizer)",
            "output_tokens": "ceiling only (max_tokens x calls; never measured)",
            "cache_hit_tokens": "ceiling only (never observed on this workload)",
            "knowledge_base_chunk_count": diagnostics.get(
                "knowledge_base_chunk_count"
            ),
            "system_prompt_chars": system_chars,
            "source": "artifacts/reports/prompt_size_diagnostics.json",
        },
        "experiments": {
            name: scope(block) for name, block in diagnostics["experiments"].items()
        },
        "agent_total": scope(diagnostics["agent_total"]),
        "all_generation_total": scope(diagnostics["all_generation_total"]),
    }
    if manifest is not None:
        actual = actual_cost_from_manifest(
            manifest, price_hit, price_miss, price_output
        )
        report["actual"] = actual if actual is not None else {
            "measured": False,
            "run_id": manifest.get("run_id", ""),
            "reason": "run has no provider usage (retrieval-only run calls no generation)",
        }
    return report


@click.command()
@click.option(
    "--diagnostics",
    default="artifacts/reports/prompt_size_diagnostics.json",
    show_default=True,
    help="eval.prompt_size_diagnostics 写出的 JSON（提供实测 prompt 字符数）",
)
@click.option(
    "--run-dir",
    default=None,
    help="可选：已完成 run 的目录，读取其 manifest.json 的实测 usage 折算实际费用",
)
@click.option("--price-input-cache-hit", type=float, default=PRICE_INPUT_CACHE_HIT, show_default=True)
@click.option("--price-input-cache-miss", type=float, default=PRICE_INPUT_CACHE_MISS, show_default=True)
@click.option("--price-output", type=float, default=PRICE_OUTPUT, show_default=True)
@click.option("--output", default="artifacts/reports/cost_estimate.json", show_default=True)
def cli(
    diagnostics: str,
    run_dir: str | None,
    price_input_cache_hit: float,
    price_input_cache_miss: float,
    price_output: float,
    output: str,
) -> None:
    diagnostics_path = _resolve(diagnostics)
    if not diagnostics_path.exists():
        raise click.UsageError(
            f"{diagnostics_path} not found; run "
            "`python -m eval.prompt_size_diagnostics --config eval/config.formal.yaml` first"
        )
    loaded = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    manifest: dict[str, Any] | None = None
    if run_dir is not None:
        manifest_path = _resolve(run_dir) / "manifest.json"
        if not manifest_path.exists():
            raise click.UsageError(f"{manifest_path} not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        report = build_cost_report(
            loaded,
            price_input_cache_hit,
            price_input_cache_miss,
            price_output,
            manifest,
        )
    except CostEstimateError as exc:
        raise click.UsageError(str(exc)) from exc

    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    click.echo(
        f"pricing ({CURRENCY}/1M tokens): "
        f"cache_hit={price_input_cache_hit} cache_miss={price_input_cache_miss} "
        f"output={price_output}"
    )
    click.echo("estimates below are NOT measured cost; token counts are estimated")
    for key in ("agent_total", "all_generation_total"):
        scope = report[key]
        click.echo(f"  {key} (calls={scope['provider_calls']}):")
        for ratio in (3.0, 4.0):
            band = scope["by_chars_per_token"][f"chars_per_token_{ratio}"]
            click.echo(
                f"    @{ratio} chars/token: "
                f"in={band['input_tokens_estimated']} "
                f"out_max={band['output_tokens_at_max']} "
                f"total<={band['cost_at_max_output']['total']} {CURRENCY} "
                f"(no-cache {band['cost_at_max_output_no_cache']['total']}, "
                f"input-only {band['cost_input_only']['total']})"
            )
    if "actual" in report:
        actual = report["actual"]
        if actual.get("measured"):
            click.echo(
                f"  actual run {actual['run_id']}: "
                f"{actual['cost']['total']} {CURRENCY} (measured)"
            )
        else:
            click.echo(f"  actual: not available — {actual['reason']}")
    click.echo(f"output={output_path}")


if __name__ == "__main__":
    cli()
