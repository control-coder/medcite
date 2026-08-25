"""Agent 基类与统一输出模板。

所有专科 Agent 统一入参: 固定专科标识 + 完整病例 + 证据 + 路由说明 + 输出约束。
所有专科统一固定 JSON 输出模板: 鉴别诊断 / 带引用claim / 风险标记 /
缺失信息 / 建议检查 / 不确定性说明 / 弃权选项。

输出约束:
- 强制绑定引用（每个 claim 必须有 citation_chunk_ids）
- 标注不确定性
- 禁用绝对诊断
- 强制医疗免责声明
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from medidiag.agents.llm_client import LLMCompletion
from medidiag.schemas import KnowledgeChunk

# 稳定系统前缀：将策略和指令与动态病例 prompt 分离，便于复用 DeepSeek 自动上下文缓存。
AGENT_SYSTEM_PROMPT = """MediDiag EvidenceFlow 证据约束起草 Agent。
你只能使用后续用户消息中提供的病例、选项和检索证据，不得补充未给出的事实。
每个医学 claim 必须绑定用户提供的 citation_chunk_ids；证据不足、引用不存在或无法判断时必须 abstain。
输出必须是严格 JSON 对象，不得输出 Markdown、解释文字或 JSON 之外的前后缀。固定结构如下：
{
  "differential_diagnosis": [
    {"diagnosis": "诊断名", "probability": 0.0, "supporting_claim_indices": [0]}
  ],
  "claims": [
    {"text": "claim 文本", "citation_chunk_ids": ["chunk_id"], "confidence": 0.0}
  ],
  "risk_flags": ["风险标记"],
  "missing_info": ["缺失信息"],
  "recommended_tests": ["建议检查"],
  "uncertainty": "不确定性说明",
  "abstain": false,
  "abstain_reason": ""
}
必须标注不确定性、风险和缺失信息；禁止绝对化诊断、处方、剂量和急救分诊。
这是软件工程与公开数据评测演示，不是医疗建议。上述固定策略和 JSON 契约必须始终优先执行。"""


@dataclass
class Claim:
    """带引用的 claim。

    每个 claim 必须绑定 citation（chunk_id 列表）。
    """

    text: str
    """claim 文本。"""

    citation_chunk_ids: list[str] = field(default_factory=list)
    """引用的知识库 chunk_id 列表。"""

    confidence: float = 0.0
    """置信度 0~1。"""


@dataclass
class DiagnosisItem:
    """鉴别诊断项。"""

    diagnosis: str
    """诊断名称。"""

    probability: float = 0.0
    """可能性 0~1。"""

    supporting_claim_indices: list[int] = field(default_factory=list)
    """支撑 claim 的索引（指向 AgentOutput.claims）。"""


@dataclass
class AgentOutput:
    """Agent 统一输出模板。

    所有专科 Agent 必须输出此结构。
    """

    specialty: str
    """专科标识。"""

    differential_diagnosis: list[DiagnosisItem] = field(default_factory=list)
    """鉴别诊断列表。"""

    claims: list[Claim] = field(default_factory=list)
    """带引用的 claim 列表。"""

    risk_flags: list[str] = field(default_factory=list)
    """风险标记。"""

    missing_info: list[str] = field(default_factory=list)
    """缺失信息。"""

    recommended_tests: list[str] = field(default_factory=list)
    """建议检查。"""

    uncertainty: str = ""
    """不确定性说明。"""

    abstain: bool = False
    """是否弃权。"""

    abstain_reason: str = ""
    """弃权原因。"""

    raw_response: str = ""
    """原始 LLM 响应（调试用）。"""

    provider_request_id: str | None = None
    """供应商返回的非敏感请求关联 ID。"""

    provider_usage: dict[str, int] = field(default_factory=dict)
    """单次调用的非敏感 token/cache usage，不包含 prompt 或响应正文。"""

    def to_dict(self) -> dict[str, Any]:
        """转为字典。"""
        return {
            "specialty": self.specialty,
            "differential_diagnosis": [
                {
                    "diagnosis": d.diagnosis,
                    "probability": d.probability,
                    "supporting_claim_indices": d.supporting_claim_indices,
                }
                for d in self.differential_diagnosis
            ],
            "claims": [
                {
                    "text": c.text,
                    "citation_chunk_ids": c.citation_chunk_ids,
                    "confidence": c.confidence,
                }
                for c in self.claims
            ],
            "risk_flags": self.risk_flags,
            "missing_info": self.missing_info,
            "recommended_tests": self.recommended_tests,
            "uncertainty": self.uncertainty,
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason,
            "provider_request_id": self.provider_request_id,
            "provider_usage": dict(self.provider_usage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], specialty: str = "") -> AgentOutput:
        """从字典构建（解析 LLM JSON 输出）。"""
        claims = [
            Claim(
                text=c.get("text", ""),
                citation_chunk_ids=c.get("citation_chunk_ids", []),
                confidence=c.get("confidence", 0.0),
            )
            for c in data.get("claims", [])
        ]
        diagnosis = [
            DiagnosisItem(
                diagnosis=d.get("diagnosis", ""),
                probability=d.get("probability", 0.0),
                supporting_claim_indices=d.get("supporting_claim_indices", []),
            )
            for d in data.get("differential_diagnosis", [])
        ]
        return cls(
            specialty=data.get("specialty", specialty),
            differential_diagnosis=diagnosis,
            claims=claims,
            risk_flags=data.get("risk_flags", []),
            missing_info=data.get("missing_info", []),
            recommended_tests=data.get("recommended_tests", []),
            uncertainty=data.get("uncertainty", ""),
            abstain=data.get("abstain", False),
            abstain_reason=data.get("abstain_reason", ""),
            provider_request_id=data.get("provider_request_id"),
            provider_usage={
                str(key): int(value)
                for key, value in data.get("provider_usage", {}).items()
                if isinstance(value, int) and not isinstance(value, bool)
            },
        )


class AgentClient(Protocol):
    """既有 Agent 所需的最小同步生成接口。"""

    @property
    def is_configured(self) -> bool: ...

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion: ...


# 强制医疗免责声明
MANDATORY_DISCLAIMER = "仅供学习和工程演示，不构成医疗建议。"


class AgentProviderError(RuntimeError):
    """Provider 基础设施故障，不是模型主动弃权。

    fail-closed 的调用方（正式评测）必须让该异常向上传播并终止运行：
    把 API key 失效或 provider 不可达折算成 ``abstain=True``，会让一次
    形式上成功的 run 产出完全由故障构成的指标。
    """


class BaseAgent:
    """Agent 基类。

    所有专科 Agent 继承此类，统一入参和输出。

    输出约束:
        - 强制绑定引用（每个 claim 必须有 citation_chunk_ids）
        - 标注不确定性
        - 禁用绝对诊断
        - 强制医疗免责声明
    """

    def __init__(
        self,
        specialty: str,
        llm_client: AgentClient | None = None,
        fail_closed: bool = False,
    ) -> None:
        self.specialty = specialty
        self.llm = llm_client
        # fail_closed=True 时 provider 故障向上抛 AgentProviderError；默认的
        # 宽松路径供开发、demo 和 dry-run 使用。
        self.fail_closed = fail_closed

    def generate(
        self,
        case_question: str,
        evidence: list[KnowledgeChunk] | list[dict[str, Any]],
        routing_note: str = "",
        case_options: dict[str, str] | None = None,
        claim_language: str | None = None,
    ) -> AgentOutput:
        """生成诊断输出。

        Args:
            case_question: 病例问题。
            evidence: 检索证据 chunks。
            routing_note: 路由说明（专科选择理由）。
            case_options: 病例选项（MedQA 的 A/B/C/D）。
            claim_language: ``claims[].text`` 的强制语言；formal NLI 评测传入 ``en``。

        Returns:
            AgentOutput 统一输出。
        """
        prompt = self.build_prompt(
            case_question, evidence, routing_note, case_options, claim_language
        )

        if self.llm and self.llm.is_configured:
            try:
                try:
                    completion = self.llm.complete(prompt, system_prompt=AGENT_SYSTEM_PROMPT)
                except TypeError as exc:
                    # 兼容不支持 system_prompt 的测试 double；生产客户端使用稳定前缀。
                    if "system_prompt" not in str(exc):
                        raise
                    completion = self.llm.complete(prompt)
            except Exception as e:
                if self.fail_closed:
                    raise AgentProviderError(
                        f"provider call failed for specialty={self.specialty}: "
                        f"{type(e).__name__}: {e}"
                    ) from e
                return AgentOutput(
                    specialty=self.specialty,
                    uncertainty=f"LLM error: {e}",
                    abstain=True,
                    abstain_reason=f"llm_error: {e}",
                    raw_response=str(e),
                )
            try:
                output = self.parse_output(completion.content)
            except Exception as e:
                # 解析失败时保留 provider request ID，便于 trace 关联。
                return AgentOutput(
                    specialty=self.specialty,
                    uncertainty=f"LLM output parse error: {e}",
                    abstain=True,
                    abstain_reason=f"llm_output_parse_error: {e}",
                    raw_response=completion.content,
                    provider_request_id=completion.request_id,
                )
            output.provider_request_id = completion.request_id
            output.provider_usage = dict(getattr(completion, "usage", {}) or {})
            return output
        else:
            # 无 LLM 时弃权（测试用）
            return AgentOutput(
                specialty=self.specialty,
                uncertainty="LLM client not configured",
                abstain=True,
                abstain_reason="no_llm_client",
            )

    def build_prompt(
        self,
        case_question: str,
        evidence: list[Any],
        routing_note: str,
        case_options: dict[str, str] | None = None,
        claim_language: str | None = None,
    ) -> str:
        """构建 LLM prompt。

        包含输出约束: 强制绑定引用、标注不确定性、禁用绝对诊断、强制免责。
        """
        if claim_language not in {None, "en"}:
            raise ValueError("claim_language 目前只支持 None 或 'en'")
        # formal NLI 使用英语 judge；只约束用于判定的 claim，其他展示字段仍可为中文。
        claim_example = "English evidence-backed claim" if claim_language == "en" else "claim 文本"
        claim_language_constraint = (
            "\n6. claims[].text 必须使用英文完整陈述，且不得包含中文汉字；"
            "该字段将直接作为英文 NLI judge 的输入。"
            if claim_language == "en"
            else ""
        )

        # 证据文本
        evidence_text = ""
        for i, chunk in enumerate(evidence[:10]):  # Top10 证据
            if isinstance(chunk, dict):
                cid = chunk.get("chunk_id", f"chunk_{i}")
                text = chunk.get("text", "")
            else:
                cid = chunk.chunk_id
                text = chunk.text
            evidence_text += f"[{cid}] {text}\n\n"

        # 选项
        options_text = ""
        if case_options:
            for key, val in case_options.items():
                options_text += f"{key}. {val}\n"

        prompt = f"""你是 {self.specialty} 专科医学诊断 Agent。

病例问题:
{case_question}
{options_text}

检索证据:
{evidence_text}

路由说明: {routing_note}

请基于上述证据生成诊断建议。输出必须是合法 JSON，格式如下：
{{
  "differential_diagnosis": [
    {{"diagnosis": "诊断名", "probability": 0.0, "supporting_claim_indices": [0]}}
  ],
  "claims": [
    {{"text": "{claim_example}", "citation_chunk_ids": ["chunk_id"], "confidence": 0.8}}
  ],
  "risk_flags": ["风险标记1"],
  "missing_info": ["缺失信息1"],
  "recommended_tests": ["建议检查1"],
  "uncertainty": "不确定性说明",
  "abstain": false,
  "abstain_reason": ""
}}

输出约束:
1. 每个 claim 必须绑定 citation_chunk_ids（引用上述证据的 chunk_id）
2. 标注不确定性
3. 禁用绝对诊断措辞（如"确诊""保证治愈"）
4. 如证据不足可弃权（abstain=true）
5. {MANDATORY_DISCLAIMER}{claim_language_constraint}

只输出 JSON，不要其他文字。"""
        return prompt

    def parse_output(self, raw_response: str) -> AgentOutput:
        """解析 LLM JSON 输出为 AgentOutput。

        如果 JSON 解析失败，返回弃权输出。
        """
        try:
            # 尝试提取 JSON（LLM 可能包裹在 markdown 代码块中）
            text = raw_response.strip()
            if text.startswith("```"):
                # 去掉 markdown 代码块
                lines = text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                text = "\n".join(lines)
            data = json.loads(text)
            output = AgentOutput.from_dict(data, specialty=self.specialty)
            output.raw_response = raw_response
            return output
        except (json.JSONDecodeError, KeyError) as e:
            return AgentOutput(
                specialty=self.specialty,
                uncertainty=f"JSON parse error: {e}",
                abstain=True,
                abstain_reason=f"json_parse_error: {e}",
                raw_response=raw_response,
            )
