"""查询改写：把口语问题改写成一句规范的书面表述，再和原问题一起交给 BM25 检索。

只用于字面匹配检索：评测里它把口语换说法类问题的 Hit@3 从 6/21 提到 17/21；
向量检索下几乎没有额外收益，所以向量方案不使用（见 docs/evaluation.md 第六、七节）。
提示词与评测使用同一份，改动它会使评测的录制文件请求哈希失配。
"""

from __future__ import annotations

import json
from typing import Any

from medidiag.llm.contracts import LLMRequest
from medidiag.llm.models import ACTIVE_MIMO_MODEL

REWRITE_PROMPT = (
    "你是公共卫生科普检索的查询改写助手。把用户的口语化问题改写成一句规范的书面表述，"
    "使用世界卫生组织等科普页面常用的术语，保留原意。只做改写：不回答问题，"
    "不添加问题中没有的事实、数字或建议。"
    '返回 JSON：{"query":"改写后的一句话，不超过 60 字"}。'
)
MAX_REWRITE_CHARS = 120


def build_rewrite_request(question: str, model: str = ACTIVE_MIMO_MODEL) -> LLMRequest:
    return LLMRequest(
        messages=[{"role": "system", "content": REWRITE_PROMPT},
                  {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)}],
        model=model, response_format={"type": "json_object"}, max_tokens=256,
        reasoning_mode="disabled", prompt_version="query-rewrite-v1")


def parse_rewrite(parsed_json: dict[str, Any] | None) -> str | None:
    """取出改写句；缺失、为空或过长（超过 120 字）时返回 None，调用方回退到原问题。"""
    text = (parsed_json or {}).get("query")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_REWRITE_CHARS:
        return None
    return text.strip()
