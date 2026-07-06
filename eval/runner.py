"""评测 runner。

阶段 0：CLI 骨架 + 配置加载验证。
后续阶段：实现消融分组执行、指标计算、报告生成。

用法：
    python -m eval.runner --help
    python -m eval.runner --config eval/config.yaml --show-config
    python -m eval.runner --config eval/config.yaml --group A --output reports/raw/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml

# 支持从项目根目录或 src 布局运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


VALID_GROUPS = ("A", "B", "C", "D", "E", "F", "all")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """加载评测配置 YAML。"""
    path = Path(config_path)
    if not path.exists():
        raise click.FileError(str(path), hint="eval config not found")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise click.UsageError(f"eval config must be a mapping, got {type(cfg)}")
    return cfg


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """校验配置完整性，返回缺失字段列表。"""
    issues: list[str] = []
    required_top_keys = (
        "generation", "embedding", "rerank", "judge",
        "dataset", "retrieval", "ablation", "workflow",
        "metrics", "reproduction",
    )
    for key in required_top_keys:
        if key not in cfg:
            issues.append(f"missing top-level key: {key}")

    # 锁定的模型字段
    if "generation" in cfg and not cfg["generation"].get("model"):
        issues.append("generation.model must be locked")
    if "embedding" in cfg and not cfg["embedding"].get("model"):
        issues.append("embedding.model must be locked")
    if "rerank" in cfg and not cfg["rerank"].get("model"):
        issues.append("rerank.model must be locked")
    if "judge" in cfg and not cfg["judge"].get("model"):
        issues.append("judge.model must be locked")

    # 消融组别必须齐全 A-F
    if "ablation" in cfg and "groups" in cfg["ablation"]:
        groups = cfg["ablation"]["groups"]
        for g in ("A", "B", "C", "D", "E", "F"):
            if g not in groups:
                issues.append(f"missing ablation group: {g}")

    return issues


def show_config_summary(cfg: dict[str, Any]) -> None:
    """打印配置摘要（敏感字段打码）。"""
    click.echo("=" * 70)
    click.echo("MediDiag Eval Configuration Summary")
    click.echo("=" * 70)
    click.echo(f"  generation_model : {cfg['generation']['model']}")
    click.echo(f"  embedding_model  : {cfg['embedding']['model']}")
    click.echo(f"  rerank_model     : {cfg['rerank']['model']}")
    click.echo(f"  judge_model      : {cfg['judge']['model']}")
    click.echo(f"  temperature      : {cfg['generation']['temperature']}")
    click.echo(f"  seed             : {cfg['generation']['seed']}")
    click.echo(f"  dataset_version  : {cfg['dataset']['version']}")
    click.echo(
        f"  retrieval weights: "
        f"w1={cfg['retrieval']['weights']['w1_bm25']} "
        f"w2={cfg['retrieval']['weights']['w2_embedding']} "
        f"w3={cfg['retrieval']['weights']['w3_evidence_level']} "
        f"w4={cfg['retrieval']['weights']['w4_term_overlap']}"
    )
    click.echo(f"  ablation groups  : {', '.join(cfg['ablation']['groups'].keys())}")
    click.echo(f"  metrics count    : {len(cfg['metrics'])}")
    click.echo("=" * 70)


@click.command()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=False),
    default="eval/config.yaml",
    show_default=True,
    help="评测配置 YAML 路径。",
)
@click.option(
    "--group",
    type=click.Choice(VALID_GROUPS, case_sensitive=False),
    default="all",
    show_default=True,
    help="执行的消融组别。all 表示 A-F 全部。",
)
@click.option(
    "--output", "output_dir",
    type=click.Path(exists=False),
    default="reports/raw/",
    show_default=True,
    help="原始结果输出目录。",
)
@click.option(
    "--show-config", is_flag=True,
    help="仅打印配置摘要并退出，不执行评测。",
)
@click.option(
    "--validate", is_flag=True,
    help="仅校验配置完整性并退出。",
)
def cli(
    config_path: str,
    group: str,
    output_dir: str,
    show_config: bool,
    validate: bool,
) -> None:
    """MediDiag 评测 runner。

    锁定的配置在 eval/config.yaml。换任何模型/数据集/权重必须重跑全部评测。
    """
    cfg = load_config(config_path)

    if show_config:
        show_config_summary(cfg)
        return

    issues = validate_config(cfg)
    if issues:
        click.echo("Configuration validation FAILED:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(2)

    if validate:
        click.echo("Configuration validation: OK")
        return

    # ----- 阶段 0：仅打印执行计划，不真正跑评测 -----
    click.echo("MediDiag Eval Runner (stage 0 skeleton)")
    click.echo(f"  config : {config_path}")
    click.echo(f"  group  : {group}")
    click.echo(f"  output : {output_dir}")
    click.echo("")
    click.echo("NOTE: 实际评测执行在阶段 6 实现。")
    click.echo("      阶段 0 仅验证配置加载与 CLI 可用性。")

    # 写一个占位结果文件，证明 output 路径可写
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "stage": "stage_0_skeleton",
        "group": group,
        "config_path": str(config_path),
        "generation_model": cfg["generation"]["model"],
        "note": "placeholder; real evaluation starts at stage 6",
    }
    placeholder_path = out / f"stage0_group_{group}.json"
    with placeholder_path.open("w", encoding="utf-8") as f:
        json.dump(placeholder, f, ensure_ascii=False, indent=2)
    click.echo(f"  placeholder written: {placeholder_path}")


if __name__ == "__main__":
    cli()
