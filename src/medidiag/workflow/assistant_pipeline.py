"""医疗助手 Agent 的共享阶段执行契约。

本模块只负责阶段调用、schema 校验、哈希与可审计 artifact 封装；数据库事务、
任务租约和状态迁移仍由 workflow runtime 负责。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from medidiag.workflow.idempotency import compute_input_hash
from medidiag.workflow.provider_runtime import (
    ProviderAttempt,
    ProviderCallOutcome,
    ProviderCallRunner,
    ProviderResponse,
)

PipelineMode = Literal["worker", "evaluation", "development"]


@dataclass(frozen=True)
class StageContext:
    """一次阶段执行的非敏感运行上下文。"""

    run_id: str
    case_id: str
    mode: PipelineMode
    task_id: str | None = None
    sample_id: str | None = None
    experiment: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True)
class StageArtifactEnvelope:
    """worker 与评测共享的版本化阶段 artifact。"""

    schema_version: str
    pipeline_version: str
    stage: str
    input_hash: str
    output_hash: str
    payload: dict[str, Any]
    latency_ms: int
    component_version: str
    provider_profile: str
    provider_request_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "stage": self.stage,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "payload": self.payload,
            "latency_ms": self.latency_ms,
            "component_version": self.component_version,
            "provider_profile": self.provider_profile,
            "provider_request_ids": list(self.provider_request_ids),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class StageExecution:
    """阶段 artifact 与底层 provider 尝试信息。"""

    artifact: StageArtifactEnvelope
    provider_outcome: ProviderCallOutcome

    @property
    def payload(self) -> dict[str, Any]:
        return self.artifact.payload

    @property
    def request_id(self) -> str | None:
        return self.provider_outcome.request_id

    @property
    def retry_count(self) -> int:
        return self.provider_outcome.retry_count

    @property
    def elapsed_ms(self) -> int:
        return self.artifact.latency_ms

    @property
    def metadata(self) -> dict[str, Any]:
        return self.provider_outcome.metadata


class AssistantPipeline:
    """共享的医疗助手阶段执行入口，不持有数据库会话。"""

    schema_version = "assistant-stage-artifact-v1"
    version = "assistant-pipeline-v1"

    def __init__(
        self,
        *,
        component_version: str,
        provider_profile: str = "unspecified",
        call_runner: ProviderCallRunner | None = None,
        stage_timeout_seconds: dict[str, float] | None = None,
    ) -> None:
        self.component_version = component_version
        self.provider_profile = provider_profile
        self.call_runner = call_runner or ProviderCallRunner()
        self.stage_timeout_seconds = dict(stage_timeout_seconds or {})

    def run_stage(
        self,
        *,
        stage: str,
        input_payload: dict[str, Any],
        operation: Callable[[], dict[str, Any] | ProviderResponse],
        context: StageContext,
        on_attempt: Callable[[ProviderAttempt], None] | None = None,
    ) -> StageExecution:
        """执行并校验单个阶段，生成跨入口一致的 artifact envelope。"""

        outcome = self.call_runner.call(stage, operation, on_attempt=on_attempt)
        request_ids = (outcome.request_id,) if outcome.request_id else ()
        artifact = StageArtifactEnvelope(
            schema_version=self.schema_version,
            pipeline_version=self.version,
            stage=stage,
            input_hash=compute_input_hash(input_payload),
            output_hash=compute_input_hash(outcome.payload),
            payload=outcome.payload,
            latency_ms=outcome.elapsed_ms,
            component_version=self.component_version,
            provider_profile=self.provider_profile,
            provider_request_ids=request_ids,
            metadata={
                "run_id": context.run_id,
                "case_id": context.case_id,
                "task_id": context.task_id,
                "sample_id": context.sample_id,
                "experiment": context.experiment,
                "mode": context.mode,
                "prompt_version": context.prompt_version,
                "stage_timeout_seconds": self.stage_timeout_seconds.get(stage),
                "provider_retry_count": outcome.retry_count,
                **outcome.metadata,
            },
        )
        return StageExecution(artifact=artifact, provider_outcome=outcome)


__all__ = [
    "AssistantPipeline",
    "PipelineMode",
    "StageArtifactEnvelope",
    "StageContext",
    "StageExecution",
    "ProviderAttempt",
    "ProviderResponse",
]

