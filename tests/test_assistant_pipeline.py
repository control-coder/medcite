"""共享 AssistantPipeline 契约测试。"""

from __future__ import annotations

from medidiag.workflow.assistant_pipeline import AssistantPipeline, StageContext
from medidiag.workflow.provider_runtime import ProviderResponse


def _context(mode: str = "development") -> StageContext:
    return StageContext(
        run_id="run_shared_001",
        case_id="case_shared_001",
        sample_id="sample_shared_001" if mode != "worker" else None,
        task_id="task_shared_001" if mode == "worker" else None,
        experiment="shared_contract",
        mode=mode,  # type: ignore[arg-type]
        prompt_version="prompt-v1",
    )


def test_same_input_and_output_generate_comparable_artifacts() -> None:
    """worker 与 evaluation 只允许运行上下文不同，事实哈希必须一致。"""
    worker_pipeline = AssistantPipeline(
        component_version="deterministic-provider-v1",
        provider_profile="fake_offline",
    )
    eval_pipeline = AssistantPipeline(
        component_version="deterministic-provider-v1",
        provider_profile="fake_offline",
    )
    def operation() -> dict[str, str]:
        return {"normalized_query": "deidentified simulated case"}

    worker = worker_pipeline.run_stage(
        stage="normalize",
        input_payload={"question": "Deidentified simulated case"},
        operation=operation,
        context=_context("worker"),
    ).artifact
    evaluation = eval_pipeline.run_stage(
        stage="normalize",
        input_payload={"question": "Deidentified simulated case"},
        operation=operation,
        context=_context("evaluation"),
    ).artifact

    assert worker.schema_version == evaluation.schema_version
    assert worker.pipeline_version == evaluation.pipeline_version
    assert worker.stage == evaluation.stage == "normalize"
    assert worker.input_hash == evaluation.input_hash
    assert worker.output_hash == evaluation.output_hash
    assert worker.payload == evaluation.payload
    assert worker.metadata["mode"] == "worker"
    assert evaluation.metadata["mode"] == "evaluation"


def test_pipeline_preserves_provider_provenance_and_timeout_policy() -> None:
    pipeline = AssistantPipeline(
        component_version="provider-v2",
        provider_profile="mimo_v25",
        stage_timeout_seconds={"generation": 45},
    )

    execution = pipeline.run_stage(
        stage="generation",
        input_payload={"question": "公开数据构造的模拟问题"},
        operation=lambda: ProviderResponse(
            payload={"agents": [], "claims": []},
            request_id="resp_pipeline_001",
            metadata={"model": "mimo-v2.5"},
        ),
        context=_context(),
    )

    artifact = execution.artifact
    assert artifact.provider_profile == "mimo_v25"
    assert artifact.provider_request_ids == ("resp_pipeline_001",)
    assert artifact.metadata["stage_timeout_seconds"] == 45
    assert artifact.metadata["model"] == "mimo-v2.5"


def test_pipeline_reuses_shared_stage_schema_validation() -> None:
    pipeline = AssistantPipeline(component_version="test-v1")

    try:
        pipeline.run_stage(
            stage="normalize",
            input_payload={"question": "模拟问题"},
            operation=lambda: {"normalized_query": ""},
            context=_context(),
        )
    except Exception as exc:
        assert getattr(exc, "code", None) == "PROVIDER_SCHEMA_INVALID"
    else:
        raise AssertionError("空 normalized_query 必须被共享 schema 拒绝")
