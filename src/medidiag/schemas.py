"""评测集与知识库的数据 schema。

阶段 1 核心数据结构。用 dataclass 定义，无第三方依赖。

数据泄露防护约束（PLAN.md）：
- EvalSample.sample_id 不得出现在 KnowledgeChunk 的 source / source_id / metadata.raw_id 中
- chunk 可保留文献来源 ID（如 PubMed ID），但不得保留评测样本 ID
- chunk 文本不得完整包含评测问题原文或 answer key
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class EvalSample:
    """评测集单个样本。

    对应 eval/datasets/eval_set.jsonl 的一行。
    """

    sample_id: str
    """评测样本唯一 ID，格式: ``{source}_{native_id}``，如 ``pubmedqa_21645374``。"""

    source: str
    """公开数据集来源: ``PubMedQA`` | ``MedQA`` | ``manual``。"""

    question: str
    """问题文本。"""

    gold_answer: str
    """标准答案。PubMedQA 为 yes/no/maybe；MedQA 为正确选项文本。"""

    gold_evidence_ids: list[str] = field(default_factory=list)
    """支撑答案的证据 chunk_id 列表。无证据时为空。"""

    label_source: str = "dataset"
    """标注来源:
    - ``dataset``: 直接来自公开数据集原始 context
    - ``dataset_no_evidence``: 数据集无 explanation，gold_evidence 为空
    - ``manual_review``: 人工新增标注
    """

    labeler: str = "dataset"
    """标注人: ``dataset`` | 标注人姓名 | ``manual_review``。"""

    review_status: str = "single_checked"
    """复核状态: ``single_checked`` | ``double_checked``。"""

    options: dict[str, str] | None = None
    """选项（MedQA 有 A/B/C/D），PubMedQA 无选项则为 None。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加元数据（如 meta_info=step1、YEAR、MESHES 等）。"""


@dataclass
class KnowledgeChunk:
    """知识库单个 chunk。

    对应 eval/datasets/knowledge_chunks.jsonl 的一行。

    数据泄露约束:
    - ``source_id`` 为文献来源 ID（如 PubMed ID、教材章节名），
      不得为评测样本 ID（如 ``pubmedqa_21645374``）。
    - ``text`` 不得完整包含评测问题原文或 answer key。
    """

    chunk_id: str
    """chunk 唯一 ID，格式: ``kb_{source}_{seq}``，如 ``kb_pubmedqa_00001``。"""

    source: str
    """chunk 来源类型: ``PubMedQA_context`` | ``MedQA_textbook`` | ``manual``。"""

    source_id: str
    """文献来源 ID（PubMed ID / 教材章节名），非评测样本 ID。"""

    text: str
    """chunk 文本内容。"""

    evidence_level: str = "level_5_other"
    """证据等级:
    - ``level_1_guideline``: 指南
    - ``level_2_review``: 综述
    - ``level_3_primary_study``: 原始研究
    - ``level_4_case_report``: 病例报告
    - ``level_5_other``: 其他
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加元数据（如 context_label=RESULTS、textbook_name 等）。"""


# ===== 序列化辅助 =====


def sample_to_dict(sample: EvalSample) -> dict[str, Any]:
    """EvalSample -> dict。"""
    return asdict(sample)


def chunk_to_dict(chunk: KnowledgeChunk) -> dict[str, Any]:
    """KnowledgeChunk -> dict。"""
    return asdict(chunk)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """写入 JSONL 文件，返回记录数。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""
    p = Path(path)
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval_samples(path: str | Path) -> list[EvalSample]:
    """从 JSONL 加载评测样本。"""
    records = read_jsonl(path)
    return [EvalSample(**rec) for rec in records]


def load_knowledge_chunks(path: str | Path) -> list[KnowledgeChunk]:
    """从 JSONL 加载知识库 chunks。"""
    records = read_jsonl(path)
    return [KnowledgeChunk(**rec) for rec in records]
