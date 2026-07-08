"""诊断生成 Agent + 单 Agent baseline。

用于 PLAN_GENERATED 阶段，基于证据生成候选诊断。
单 Agent baseline 时 specialty="general_diagnosis"。
双专科并行时由 SpecialistAgent 替代。
"""

from __future__ import annotations

from medidiag.agents.base import BaseAgent
from medidiag.agents.llm_client import LLMClient


class DiagnosisAgent(BaseAgent):
    """诊断生成 Agent（流水线节点）。

    单 Agent baseline: 不做专科路由，直接基于全部证据生成诊断。
    用于消融实验 A 组（单 Agent 基线）对比 B/C 组（双专科）。

    用法:
        agent = DiagnosisAgent(llm_client)
        output = agent.generate(case_question, evidence)
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__(
            specialty="general_diagnosis", llm_client=llm_client
        )
