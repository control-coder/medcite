"""Agent 基类占位。

阶段 5 实现：
- 病例归一化 worker
- 证据检索 worker
- 诊断生成 Agent（JSON schema 约束）
- 双专科并行 Agent（如心内科 / 呼吸科）
- 仲裁 Agent
- 单 Agent baseline（用于对比）

多 Agent 边界：只在"双专科并行 + 仲裁"场景体现；
其他环节诚实描述为流水线式 Agent 编排。
"""

from __future__ import annotations


class BaseAgent:
    """Agent 基类骨架。

    TODO[阶段5]: 实现 Agent 执行接口、JSON schema 约束、trace 记录。
    """

    def __init__(self) -> None:
        raise NotImplementedError("BaseAgent 将在阶段 5 实现")
