"""幂等键工具。

三层幂等键（PLAN.md）：
1. 创建病例: Idempotency-Key + user_scope
2. 启动工作流: case_id + workflow_type
3. Agent 执行: case_id + agent_name + input_hash + attempt_group
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_input_hash(payload: dict[str, Any]) -> str:
    """计算输入 payload 的哈希（sha256），用于 Agent 幂等。

    序列化时按 key 排序，确保相同内容产生相同哈希。
    """
    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def make_case_key(idempotency_key: str, user_scope: str) -> str:
    """创建病例幂等键: idempotency_key + user_scope。"""
    return f"case:{idempotency_key}:{user_scope}"


def make_workflow_key(case_id: str, workflow_type: str) -> str:
    """启动工作流幂等键: case_id + workflow_type。"""
    return f"workflow:{case_id}:{workflow_type}"


def make_agent_key(
    case_id: str,
    agent_name: str,
    input_hash: str,
    attempt_group: str,
) -> str:
    """Agent 执行幂等键: case_id + agent_name + input_hash + attempt_group。"""
    return f"agent:{case_id}:{agent_name}:{input_hash[:16]}:{attempt_group}"
