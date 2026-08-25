"""运行时单/双专科 Agent 编排与受约束仲裁。

本模块把既有 ``DiagnosisAgent`` / ``SpecialistAgent`` 接入统一 ``LLMProvider``，
并把每次调用封装为可持久化 AgentRun artifact。双专科只并行生成，不把流水线
节点夸大为通用多 Agent 协作；仲裁先保留确定性规则基线，再执行只能选择既有
claim 的受约束 LLM 仲裁。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from medidiag.agents.arbitration import ArbitrationAgent
from medidiag.agents.base import AgentOutput
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.llm_client import LLMCompletion
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialist import SpecialistAgent
from medidiag.errors import MediDiagError
from medidiag.llm import LLMProvider, LLMRequest, ProviderResult
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.schemas import KnowledgeChunk

AgentTopology = Literal["single", "fixed_pair", "dynamic_pair"]


@dataclass(frozen=True)
class AgentTopologyConfig:
    """运行时 Agent topology 与 prompt 契约。"""

    topology: AgentTopology = "dynamic_pair"
    fixed_pair: tuple[str, str] = ("cardiology", "respiratory")
    specialist_prompt_version: str = "specialist-agent-v1"
    arbitration_prompt_version: str = "arbitrator-agent-v1"
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.topology not in {"single", "fixed_pair", "dynamic_pair"}:
            raise ValueError(f"不支持的 Agent topology: {self.topology}")
        if len(self.fixed_pair) != 2 or self.fixed_pair[0] == self.fixed_pair[1]:
            raise ValueError("fixed_pair 必须包含两个不同专科")
        if self.timeout_s <= 0:
            raise ValueError("Agent timeout_s 必须大于 0")


class ProviderAgentClient:
    """把统一 LLMProvider 适配为既有 BaseAgent 的同步 complete 接口。"""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        prompt_version: str,
        timeout_s: float,
        idempotency_key: str,
    ) -> None:
        self.provider = provider
        self.prompt_version = prompt_version
        self.timeout_s = timeout_s
        self.idempotency_key = idempotency_key
        self.last_result: ProviderResult | None = None

    @property
    def is_configured(self) -> bool:
        return bool(getattr(self.provider, "is_configured", True))

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = self.provider.generate(
            LLMRequest(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0 if temperature is None else temperature,
                max_tokens=max_tokens or 1200,
                prompt_version=self.prompt_version,
            ),
            timeout_s=self.timeout_s,
            idempotency_key=self.idempotency_key,
        )
        self.last_result = result
        return LLMCompletion(
            content=result.content,
            request_id=result.response_id,
            model=result.model,
            usage=dict(result.usage),
        )


class _ConstrainedArbitrationPayload(BaseModel):
    """LLM 仲裁只能选择已有 claim，并给出受限流程 verdict。"""

    selected_claim_ids: list[str] = Field(min_length=1)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    verdict: Literal["APPROVED", "REVISION_REQUIRED", "ESCALATED"]
    reason: str = Field(min_length=1, max_length=1200)


class RuntimeMedicalAgents:
    """复用同一 EvidenceBundle 的可审计 Agent 编排器。"""

    version = "runtime-medical-agents-v1"

    def __init__(
        self,
        provider: LLMProvider,
        *,
        config: AgentTopologyConfig | None = None,
        normalizer: TerminologyNormalizer | None = None,
        evidence_level_scores: dict[str, float] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or AgentTopologyConfig()
        self.router = SpecialistRouter(
            normalizer=normalizer,
            evidence_level_scores=evidence_level_scores,
        )

    def generate(
        self,
        question: str,
        retrieval: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        evidence = self._evidence(retrieval)
        evidence_bundle = retrieval.get("evidence_bundle")
        if not isinstance(evidence_bundle, dict):
            raise MediDiagError(
                "AGENT_RUNTIME_INVALID", detail="Agent runtime 缺少版本化 EvidenceBundle"
            )
        bundle_id = str(
            retrieval.get("evidence_bundle_id") or evidence_bundle.get("bundle_id") or ""
        )
        if not bundle_id:
            raise MediDiagError("AGENT_RUNTIME_INVALID", detail="EvidenceBundle 缺少 bundle_id")
        evidence_hash = _stable_hash(evidence_bundle)
        routing = self._routing(question, evidence)
        specialties = routing["specialty_pair"]

        if self.config.topology == "single":
            artifacts = [
                self._run_agent(
                    specialty="general_diagnosis",
                    question=question,
                    evidence=evidence,
                    plan=plan,
                    bundle_id=bundle_id,
                    evidence_hash=evidence_hash,
                    routing=routing,
                )
            ]
        else:
            # 两个调用使用独立 adapter，避免并发时覆盖 provider provenance。
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="medidiag-specialist"
            ) as pool:
                futures = [
                    pool.submit(
                        self._run_agent,
                        specialty=specialty,
                        question=question,
                        evidence=evidence,
                        plan=plan,
                        bundle_id=bundle_id,
                        evidence_hash=evidence_hash,
                        routing=routing,
                    )
                    for specialty in specialties
                ]
                artifacts = [future.result() for future in futures]

        claims: list[dict[str, Any]] = []
        for agent_index, artifact in enumerate(artifacts, start=1):
            for claim_index, claim in enumerate(artifact["output"].get("claims", []), start=1):
                value = dict(claim)
                value["claim_id"] = f"claim_{agent_index:02d}_{claim_index:04d}"
                value["agent_run_id"] = artifact["agent_run_id"]
                value["specialty"] = artifact["specialty"]
                claims.append(value)
        if not claims:
            raise MediDiagError("AGENT_RUNTIME_INVALID", detail="Agent 未生成可仲裁 claim")

        payload: dict[str, Any] = {
            "topology": self.config.topology,
            "routing": routing,
            "evidence_bundle_id": bundle_id,
            "evidence_hash": evidence_hash,
            "agents": artifacts,
            "claims": claims,
            "claims_before_arbitration": claims,
            "risk_flags": sorted(
                {
                    flag
                    for artifact in artifacts
                    for flag in artifact["output"].get("risk_flags", [])
                }
            ),
            "uncertainty": "；".join(
                text
                for artifact in artifacts
                if (text := str(artifact["output"].get("uncertainty", "")).strip())
            ),
        }
        request_id = next(
            (
                str(item["provider_provenance"]["response_id"])
                for item in artifacts
                if item["provider_provenance"].get("response_id")
            ),
            None,
        )
        payload["stage_provider_request_id"] = request_id
        return payload

    def arbitrate(
        self,
        generation: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        before = list(generation.get("claims_before_arbitration") or generation.get("claims") or [])
        before_ids = [str(item.get("claim_id")) for item in before if item.get("claim_id")]
        if not before_ids:
            raise MediDiagError("AGENT_RUNTIME_INVALID", detail="仲裁前 claim 集合为空")
        topology = str(generation.get("topology", self.config.topology))
        if topology == "single":
            payload = {
                "verdict": "SINGLE_AGENT_BASELINE",
                "arbitration_method": "not_applicable",
                "before_claim_ids": before_ids,
                "selected_claim_ids": before_ids,
                "excluded_claim_ids": [],
                "conflicts": [],
                "routing": generation.get("routing", {}),
                "fallback_used": False,
                "fallback_reason": None,
                "rule_baseline": None,
                "constrained_arbitration": None,
            }
            return payload

        outputs = [
            AgentOutput.from_dict(item.get("output", {}), specialty=str(item.get("specialty", "")))
            for item in generation.get("agents", [])
        ]
        if len(outputs) != 2:
            raise MediDiagError("AGENT_RUNTIME_INVALID", detail="双专科仲裁要求恰好两个 Agent 输出")
        rule = ArbitrationAgent().arbitrate(outputs[0], outputs[1]).to_dict()
        fallback_reason: str | None = None
        provider_result: ProviderResult | None = None
        constrained: dict[str, Any] | None = None
        try:
            provider_result = self.provider.generate(
                LLMRequest(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是受约束的医疗助手工程仲裁组件。只能从输入的 claim_id 中选择，"
                                "不得新增、改写或补充医学 claim。verdict 只能是 APPROVED、"
                                "REVISION_REQUIRED 或 ESCALATED。只返回 JSON。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "claims": before,
                                    "rule_baseline": rule,
                                    "routing": generation.get("routing", {}),
                                    "evidence_bundle_id": generation.get("evidence_bundle_id"),
                                    "allowed_claim_ids": before_ids,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=900,
                    prompt_version=self.config.arbitration_prompt_version,
                ),
                timeout_s=self.config.timeout_s,
                idempotency_key="agent-arbitration:"
                + _stable_hash(
                    {"claims": before, "evidence_hash": generation.get("evidence_hash")}
                ),
            )
            raw = provider_result.parsed_json
            if raw is None:
                raise ValueError("provider 未返回 JSON object")
            parsed = _ConstrainedArbitrationPayload.model_validate(raw)
            if not set(parsed.selected_claim_ids).issubset(set(before_ids)):
                raise ValueError("仲裁选择了未知 claim_id")
            constrained = parsed.model_dump(mode="json")
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"

        if constrained is None:
            selected_ids = before_ids
            verdict = str(rule["verdict"])
            conflicts = list(rule.get("conflicts", []))
            method = "rule_fallback"
        else:
            selected_ids = list(dict.fromkeys(constrained["selected_claim_ids"]))
            verdict = str(constrained["verdict"])
            conflicts = list(constrained.get("conflicts", []))
            method = "constrained_llm"
        excluded_ids = [claim_id for claim_id in before_ids if claim_id not in selected_ids]
        provenance = _provider_provenance(provider_result) if provider_result else None
        payload = {
            "verdict": verdict,
            "arbitration_method": method,
            "before_claim_ids": before_ids,
            "selected_claim_ids": selected_ids,
            "excluded_claim_ids": excluded_ids,
            "claims_after_arbitration": [
                item for item in before if item.get("claim_id") in selected_ids
            ],
            "conflicts": conflicts,
            "routing": generation.get("routing", {}),
            "fallback_used": constrained is None,
            "fallback_reason": fallback_reason,
            "rule_baseline": rule,
            "constrained_arbitration": constrained,
            "provider_provenance": provenance,
            "evidence_bundle_id": generation.get("evidence_bundle_id"),
            "evidence_hash": generation.get("evidence_hash"),
        }
        payload["stage_provider_request_id"] = (
            provider_result.response_id if provider_result else None
        )
        return payload

    def _routing(self, question: str, evidence: list[KnowledgeChunk]) -> dict[str, Any]:
        if self.config.topology == "single":
            return {
                "specialty_pair": ["general_diagnosis"],
                "confidence": 1.0,
                "is_fallback": False,
                "reason": "single topology baseline",
                "scores": {},
            }
        if self.config.topology == "fixed_pair":
            return {
                "specialty_pair": list(self.config.fixed_pair),
                "confidence": 1.0,
                "is_fallback": False,
                "reason": "fixed_pair topology 使用配置锁定的专科组合",
                "scores": {},
            }
        return self.router.route(question, evidence).to_dict()

    def _run_agent(
        self,
        *,
        specialty: str,
        question: str,
        evidence: list[KnowledgeChunk],
        plan: dict[str, Any],
        bundle_id: str,
        evidence_hash: str,
        routing: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_version = self.config.specialist_prompt_version
        input_payload = {
            "question": question,
            "plan": plan,
            "specialty": specialty,
            "routing_reason": routing.get("reason"),
            "evidence_bundle_id": bundle_id,
            "evidence_hash": evidence_hash,
            "prompt_version": prompt_version,
        }
        input_hash = _stable_hash(input_payload)
        client = ProviderAgentClient(
            self.provider,
            prompt_version=prompt_version,
            timeout_s=self.config.timeout_s,
            idempotency_key=f"agent:{specialty}:{input_hash}",
        )
        agent = (
            DiagnosisAgent(client, fail_closed=True)
            if specialty == "general_diagnosis"
            else SpecialistAgent(specialty, client, fail_closed=True)
        )
        started = time.perf_counter()
        try:
            output = agent.generate(
                question,
                evidence,
                routing_note=str(routing.get("reason", "")),
            )
        except Exception as exc:
            raise MediDiagError(
                "AGENT_RUNTIME_INVALID",
                detail=f"{specialty} Agent 调用失败: {type(exc).__name__}",
            ) from exc
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        allowed_ids = {item.chunk_id for item in evidence}
        for claim in output.claims:
            if not claim.citation_chunk_ids or not set(claim.citation_chunk_ids).issubset(
                allowed_ids
            ):
                raise MediDiagError(
                    "AGENT_RUNTIME_INVALID",
                    detail=f"{specialty} Agent 引用了未知 EvidenceBundle chunk_id",
                )
        if output.abstain:
            raise MediDiagError(
                "AGENT_RUNTIME_INVALID",
                detail=f"{specialty} Agent 输出弃权: {output.abstain_reason}",
            )
        result = client.last_result
        if result is None:
            raise MediDiagError(
                "AGENT_RUNTIME_INVALID", detail=f"{specialty} 缺少 provider provenance"
            )
        run_id = f"agent_{uuid.uuid4().hex}"
        return {
            "agent_run_id": run_id,
            "agent_name": (
                "diagnosis_agent"
                if specialty == "general_diagnosis"
                else f"specialist_agent:{specialty}"
            ),
            "specialty": specialty,
            "status": "SUCCEEDED",
            "topology": self.config.topology,
            "prompt_version": prompt_version,
            "input_hash": input_hash,
            "input": input_payload,
            "evidence_bundle_id": bundle_id,
            "evidence_hash": evidence_hash,
            "routing_reason": routing.get("reason"),
            "latency_ms": latency_ms,
            "output": output.to_dict(),
            "provider_provenance": _provider_provenance(result),
        }

    @staticmethod
    def _evidence(retrieval: dict[str, Any]) -> list[KnowledgeChunk]:
        raw = retrieval.get("chunks", [])
        if not isinstance(raw, list) or not raw:
            raise MediDiagError("AGENT_RUNTIME_INVALID", detail="Agent runtime 没有检索证据")
        return [
            KnowledgeChunk(
                chunk_id=str(item["chunk_id"]),
                source=str(item.get("source", "")),
                source_id=str(item.get("source_id", "")),
                text=str(item.get("text", "")),
                evidence_level=str(item.get("evidence_level", "level_5_other")),
                metadata=dict(item.get("metadata", {})),
            )
            for item in raw
        ]


def _provider_provenance(result: ProviderResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "provider_id": result.provider_id,
        "profile_id": result.profile_id,
        "model": result.model,
        "response_id": result.response_id,
        "system_fingerprint": result.system_fingerprint,
        "provenance_mode": result.provenance_mode,
        "retry_count": result.retry_count,
        "usage": dict(result.usage),
        "latency_ms": result.latency_ms,
        "filtered_parameters": list(result.filtered_parameters),
    }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()
