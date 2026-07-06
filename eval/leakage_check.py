"""数据泄露校验脚本。

防止测试集样本 ID、问题原文或 answer key 出现在知识库 chunk 的可检索字段中。
命中时输出 ``EVAL_DATA_LEAKAGE_DETECTED`` 并以非零码退出。

PLAN.md 要求：
- 测试样本 ID 不得出现在 chunk 的 source / source_id / metadata.raw_id
- chunk 不得直接保留测试样本 ID、问题原文或 answer key 作为可检索 metadata
- 公开数据转 chunk 时，允许保留文献来源 ID，但不得保留评测样本 ID
- 映射关系只能保存在 eval label 文件中，不进入检索索引

阶段 0：实现 CLI + 文件读取骨架 + 校验逻辑框架。
阶段 1：接入真实数据后真正执行校验。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import click
import yaml


LEAKAGE_FLAG = "EVAL_DATA_LEAKAGE_DETECTED"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """加载 JSONL 文件。"""
    p = Path(path)
    if not p.exists():
        raise click.FileError(str(p), hint="file not found")
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise click.UsageError(f"{p}:{line_no} invalid JSON: {e}") from e
    return records


def extract_eval_sample_ids(eval_records: Iterable[dict[str, Any]]) -> set[str]:
    """从评测集中提取所有 sample_id。"""
    ids: set[str] = set()
    for rec in eval_records:
        sid = rec.get("sample_id")
        if sid:
            ids.add(str(sid))
    return ids


def extract_eval_questions(eval_records: Iterable[dict[str, Any]]) -> set[str]:
    """从评测集中提取所有问题原文（用于检查 chunk 是否泄露问题文本）。"""
    questions: set[str] = set()
    for rec in eval_records:
        q = rec.get("question")
        if q and isinstance(q, str):
            # 归一化：去掉首尾空白，取完整字符串
            questions.add(q.strip())
    return questions


def extract_eval_answer_keys(eval_records: Iterable[dict[str, Any]]) -> set[str]:
    """从评测集中提取所有 gold_answer（用于检查 chunk 是否泄露答案）。"""
    answers: set[str] = set()
    for rec in eval_records:
        a = rec.get("gold_answer")
        if a and isinstance(a, str):
            answers.add(a.strip())
    return answers


def check_chunk_for_leakage(
    chunk: dict[str, Any],
    eval_sample_ids: set[str],
    eval_questions: set[str],
    eval_answers: set[str],
    fields_to_check: list[str],
    check_question_text: bool,
    check_answer_key: bool,
) -> list[str]:
    """检查单个 chunk 是否泄露评测信息，返回命中的泄露描述列表。"""
    hits: list[str] = []
    chunk_id = chunk.get("chunk_id", chunk.get("id", "<unknown>"))

    # 1. 检查 sample_id 是否出现在指定字段
    for field in fields_to_check:
        value = _get_nested(chunk, field)
        if value is None:
            continue
        value_str = str(value)
        for sid in eval_sample_ids:
            if sid == value_str or sid in value_str:
                hits.append(
                    f"chunk {chunk_id}: field '{field}' contains eval sample_id '{sid}'"
                )

    # 2. 检查 chunk 文本是否包含完整问题原文
    if check_question_text:
        chunk_text = chunk.get("text", "") or chunk.get("content", "")
        if isinstance(chunk_text, str):
            for q in eval_questions:
                if q and len(q) > 20 and q in chunk_text:
                    hits.append(
                        f"chunk {chunk_id}: text contains eval question verbatim"
                    )

    # 3. 检查 chunk 文本是否包含完整 answer key
    if check_answer_key:
        chunk_text = chunk.get("text", "") or chunk.get("content", "")
        if isinstance(chunk_text, str):
            for a in eval_answers:
                if a and len(a) > 5 and a in chunk_text:
                    hits.append(
                        f"chunk {chunk_id}: text contains eval answer key verbatim"
                    )

    return hits


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    """支持 a.b.c 形式的嵌套字段访问。"""
    current: Any = d
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def load_leakage_config(config_path: str | Path) -> dict[str, Any]:
    """从 eval/config.yaml 读取 leakage_check 段。"""
    p = Path(config_path)
    if not p.exists():
        raise click.FileError(str(p), hint="config not found")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("leakage_check", {})


@click.command()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=False),
    default="eval/config.yaml",
    show_default=True,
    help="评测配置 YAML（读取 leakage_check 段）。",
)
@click.option(
    "--eval-set", "eval_set_path",
    type=click.Path(exists=False),
    required=True,
    help="评测集 JSONL 路径。",
)
@click.option(
    "--kb", "kb_path",
    type=click.Path(exists=False),
    required=True,
    help="知识库 chunk JSONL 路径。",
)
@click.option(
    "--strict", is_flag=True, default=True,
    help="命中泄露时立即以非零码退出（默认开启）。",
)
def cli(
    config_path: str,
    eval_set_path: str,
    kb_path: str,
    strict: bool,
) -> None:
    """数据泄露校验。

    检查测试集样本 ID、问题原文、answer key 是否泄露到知识库 chunk 的可检索字段。
    命中时输出 ``EVAL_DATA_LEAKAGE_DETECTED`` 并以非零码退出。
    """
    leak_cfg = load_leakage_config(config_path)
    fields_to_check = leak_cfg.get("chunk_fields_to_check", ["source", "source_id", "metadata.raw_id"])
    check_question_text = leak_cfg.get("check_question_text", True)
    check_answer_key = leak_cfg.get("check_answer_key", True)

    click.echo("MediDiag Data Leakage Check")
    click.echo(f"  config        : {config_path}")
    click.echo(f"  eval_set      : {eval_set_path}")
    click.echo(f"  knowledge_base: {kb_path}")
    click.echo(f"  fields_check  : {fields_to_check}")
    click.echo(f"  check_question: {check_question_text}")
    click.echo(f"  check_answer  : {check_answer_key}")
    click.echo("")

    # 阶段 0：如果数据文件不存在，仅打印说明并退出（不视为失败）
    if not Path(eval_set_path).exists() or not Path(kb_path).exists():
        click.echo("NOTE: 评测集或知识库文件尚未就绪（阶段 1 生成）。")
        click.echo("      阶段 0 仅验证 CLI 与配置加载可用性。")
        click.echo("      阶段 1 数据就绪后，此命令将真正执行泄露校验。")
        return

    eval_records = load_jsonl(eval_set_path)
    kb_records = load_jsonl(kb_path)

    eval_sample_ids = extract_eval_sample_ids(eval_records)
    eval_questions = extract_eval_questions(eval_records)
    eval_answers = extract_eval_answer_keys(eval_records)

    click.echo(f"  eval samples  : {len(eval_records)}")
    click.echo(f"  eval ids      : {len(eval_sample_ids)}")
    click.echo(f"  kb chunks     : {len(kb_records)}")
    click.echo("")

    all_hits: list[str] = []
    for chunk in kb_records:
        hits = check_chunk_for_leakage(
            chunk,
            eval_sample_ids,
            eval_questions,
            eval_answers,
            fields_to_check,
            check_question_text,
            check_answer_key,
        )
        all_hits.extend(hits)

    if all_hits:
        click.echo(f"!!! {LEAKAGE_FLAG} !!!")
        click.echo(f"Found {len(all_hits)} leakage hit(s):")
        for hit in all_hits[:20]:
            click.echo(f"  - {hit}")
        if len(all_hits) > 20:
            click.echo(f"  ... and {len(all_hits) - 20} more")
        if strict:
            sys.exit(1)
    else:
        click.echo("OK: no data leakage detected.")


if __name__ == "__main__":
    cli()
