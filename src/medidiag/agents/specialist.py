"""专科诊断 Agent：双专科并行诊断。

统一入参: 固定专科标识 + 完整病例 + 证据 + 路由说明 + 输出约束。
统一输出: AgentOutput（鉴别诊断 / 带引用claim / 风险 / 缺失 / 检查 / 不确定性 / 弃权）。

多 Agent 边界: 只在"双专科并行 + 仲裁"场景体现。
"""

from __future__ import annotations

from medidiag.agents.base import BaseAgent
from medidiag.agents.llm_client import LLMClient


class SpecialistAgent(BaseAgent):
    """专科诊断 Agent。

    继承 BaseAgent，统一入参和输出。
    每个实例绑定一个专科标识（如 cardiology / respiratory）。

    用法:
        agent = SpecialistAgent("cardiology", llm_client)
        output = agent.generate(case_question, evidence, routing_note)
    """

    def __init__(
        self,
        specialty: str,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__(specialty=specialty, llm_client=llm_client)
