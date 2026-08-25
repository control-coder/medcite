"""数据泄露校验 CLI。

纯检查逻辑位于 ``medidiag.rag.leakage``，因此运行时 RAG 与评测 runner 使用同一
规则；本模块只保留 YAML/Click 适配和历史导入兼容。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import yaml

from medidiag.rag.leakage import (
    LEAKAGE_FLAG,
    extract_eval_sample_ids,
    load_jsonl_records,
    run_leakage_check_paths,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """兼容旧调用方，并把文件错误转换为 Click 错误。"""
    try:
        return load_jsonl_records(path)
    except FileNotFoundError as exc:
        raise click.FileError(str(path), hint="file not found") from exc
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def load_leakage_config(config_path: str | Path) -> dict[str, Any]:
    """从评测配置读取 leakage_check 段。"""
    path = Path(config_path)
    if not path.exists():
        raise click.FileError(str(path), hint="config not found")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise click.UsageError("评测配置必须是 mapping")
    value = config.get("leakage_check", {})
    if not isinstance(value, dict):
        raise click.UsageError("leakage_check 必须是 mapping")
    return value


def run_leakage_check(
    eval_set_path: str | Path,
    kb_path: str | Path,
    leak_config: dict[str, Any],
) -> list[str]:
    """执行共享泄露门禁；缺失输入不能被当作成功空操作。"""
    try:
        return run_leakage_check_paths(eval_set_path, kb_path, leak_config)
    except FileNotFoundError as exc:
        raise click.FileError(str(exc), hint="file not found") from exc
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False),
    default="eval/config.yaml",
    show_default=True,
    help="评测配置 YAML（读取 leakage_check 段）。",
)
@click.option(
    "--eval-set",
    "eval_set_path",
    type=click.Path(exists=False),
    required=True,
    help="评测集 JSONL 路径。",
)
@click.option(
    "--kb",
    "kb_path",
    type=click.Path(exists=False),
    required=True,
    help="知识库 chunk JSONL 路径。",
)
@click.option(
    "--strict",
    is_flag=True,
    default=True,
    help="命中泄露时立即以非零码退出（默认开启）。",
)
def cli(config_path: str, eval_set_path: str, kb_path: str, strict: bool) -> None:
    """检查评测信息是否泄露到知识库可检索字段。"""
    leak_config = load_leakage_config(config_path)
    fields = leak_config.get(
        "chunk_fields_to_check", ["source", "source_id", "metadata.raw_id"]
    )
    click.echo("MediDiag Data Leakage Check")
    click.echo(f"  config        : {config_path}")
    click.echo(f"  eval_set      : {eval_set_path}")
    click.echo(f"  knowledge_base: {kb_path}")
    click.echo(f"  fields_check  : {fields}")
    click.echo("")

    if not Path(eval_set_path).exists() or not Path(kb_path).exists():
        click.echo(f"!!! {LEAKAGE_FLAG} !!!", err=True)
        click.echo("评测集或知识库文件不存在；泄露门禁无法执行。", err=True)
        raise click.exceptions.Exit(2)

    eval_records = load_jsonl(eval_set_path)
    kb_records = load_jsonl(kb_path)
    click.echo(f"  eval samples  : {len(eval_records)}")
    click.echo(f"  eval ids      : {len(extract_eval_sample_ids(eval_records))}")
    click.echo(f"  kb chunks     : {len(kb_records)}")
    click.echo("")

    hits = run_leakage_check(eval_set_path, kb_path, leak_config)
    if hits:
        click.echo(f"!!! {LEAKAGE_FLAG} !!!")
        click.echo(f"Found {len(hits)} leakage hit(s):")
        for hit in hits[:20]:
            click.echo(f"  - {hit}")
        if len(hits) > 20:
            click.echo(f"  ... and {len(hits) - 20} more")
        if strict:
            sys.exit(1)
    else:
        click.echo("OK: no data leakage detected.")


if __name__ == "__main__":
    cli()
