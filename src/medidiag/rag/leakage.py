"""可复用的数据泄露检查核心。

本模块不依赖 Click 或评测 runner，使运行时医学 RAG 与离线评测能够执行同一套
泄露规则。命中规则时由调用方决定是终止评测还是将工作流升级为失败。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

LEAKAGE_FLAG = "EVAL_DATA_LEAKAGE_DETECTED"


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """严格加载 JSONL；缺失文件或非法行均抛出异常。"""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {target}")
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{target}:{line_number} 不是合法 JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{target}:{line_number} 必须是 JSON object")
            records.append(value)
    return records


def extract_eval_sample_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item["sample_id"]) for item in records if item.get("sample_id")}


def extract_eval_questions(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(item["question"]).strip()
        for item in records
        if isinstance(item.get("question"), str) and str(item["question"]).strip()
    }


def extract_eval_answer_keys(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(item["gold_answer"]).strip()
        for item in records
        if isinstance(item.get("gold_answer"), str)
        and str(item["gold_answer"]).strip()
    }


def get_nested(value: dict[str, Any], dotted_key: str) -> Any:
    """读取 ``metadata.raw_id`` 一类嵌套字段。"""
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def check_chunk_for_leakage(
    chunk: dict[str, Any],
    eval_sample_ids: set[str],
    eval_questions: set[str],
    eval_answers: set[str],
    fields_to_check: list[str],
    check_question_text: bool,
    check_answer_key: bool,
) -> list[str]:
    """检查一个知识 chunk，返回全部泄露命中描述。"""
    hits: list[str] = []
    chunk_id = chunk.get("chunk_id", chunk.get("id", "<unknown>"))
    for field in fields_to_check:
        field_value = get_nested(chunk, field)
        if field_value is None:
            continue
        rendered = str(field_value)
        for sample_id in eval_sample_ids:
            if sample_id == rendered or sample_id in rendered:
                hits.append(
                    f"chunk {chunk_id}: field '{field}' contains eval sample_id "
                    f"'{sample_id}'"
                )

    if check_question_text:
        text = chunk.get("text", "") or chunk.get("content", "")
        if isinstance(text, str):
            for question in eval_questions:
                if len(question) > 50 and question in text:
                    hits.append(f"chunk {chunk_id}: text contains eval question verbatim")

    if check_answer_key:
        for field in fields_to_check:
            field_value = get_nested(chunk, field)
            if field_value is None:
                continue
            rendered = str(field_value)
            for answer in eval_answers:
                if len(answer) > 5 and answer in rendered:
                    hits.append(
                        f"chunk {chunk_id}: field '{field}' contains eval answer key"
                    )
    return hits


def run_leakage_check_records(
    eval_records: Iterable[dict[str, Any]],
    kb_records: Iterable[dict[str, Any]],
    leak_config: dict[str, Any],
) -> list[str]:
    """对已加载记录执行泄露门禁。"""
    eval_values = list(eval_records)
    fields = list(
        leak_config.get(
            "chunk_fields_to_check", ["source", "source_id", "metadata.raw_id"]
        )
    )
    sample_ids = extract_eval_sample_ids(eval_values)
    questions = extract_eval_questions(eval_values)
    answers = extract_eval_answer_keys(eval_values)
    hits: list[str] = []
    for chunk in kb_records:
        hits.extend(
            check_chunk_for_leakage(
                chunk,
                sample_ids,
                questions,
                answers,
                fields,
                bool(leak_config.get("check_question_text", True)),
                bool(leak_config.get("check_answer_key", True)),
            )
        )
    return hits


def run_leakage_check_paths(
    eval_set_path: str | Path,
    kb_path: str | Path,
    leak_config: dict[str, Any],
) -> list[str]:
    """加载文件并执行泄露门禁。"""
    return run_leakage_check_records(
        load_jsonl_records(eval_set_path),
        load_jsonl_records(kb_path),
        leak_config,
    )
