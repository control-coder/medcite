"""结构化日志与脱敏契约测试。

核心约束：worker、provider、租约和 API 边界的日志不得包含病例问题正文、
证据文本、prompt、完整 provider 响应或 API key。
"""

from __future__ import annotations

import io
import json

import httpx
import pytest
import structlog

from medidiag.api.app import create_app
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.observability import logging as medidiag_logging
from medidiag.observability.logging import configure_logging, get_logger
from medidiag.observability.redaction import sanitize
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker

# 出现在日志里就算失败的字面量。
_CASE_QUESTION = (
    "68 岁脱敏模拟病例：反复胸骨后压榨性疼痛伴放射至左臂，联系方式 "
    "patient@example.com / 13800138000，证件号 11010119800101001X。"
)
_API_KEY = "sk-super-secret-deepseek-key-value"
_EVIDENCE_TEXT = "Fixture evidence body that must never appear in a log line."


@pytest.fixture
def log_stream():
    """把 structlog 输出重定向到内存缓冲区，测试结束后恢复默认配置。"""
    buffer = io.StringIO()
    medidiag_logging._configured = False
    configure_logging(stream=buffer, force=True)
    try:
        yield buffer
    finally:
        structlog.reset_defaults()
        medidiag_logging._configured = False


def test_sensitive_keys_and_patterns_are_redacted(log_stream) -> None:
    logger = get_logger("test")
    logger.warning(
        "test.event",
        question=_CASE_QUESTION,
        deepseek_api_key=_API_KEY,
        authorization=f"Bearer {_API_KEY}",
        contact="reach me at patient@example.com or 13800138000",
        case_id="case_abc123",
    )
    output = log_stream.getvalue()

    assert "case_abc123" in output
    assert _CASE_QUESTION not in output
    assert _API_KEY not in output
    assert "patient@example.com" not in output
    assert "13800138000" not in output
    assert "11010119800101001X" not in output


def test_oversized_values_are_truncated_not_emitted(log_stream) -> None:
    """误传进来的 prompt 或 provider 响应以长度标记暴露，而不是完整落盘。"""
    body = "provider response body " * 40
    get_logger("test").info("test.event", detail=body)
    output = log_stream.getvalue()
    assert body not in output
    assert f"[TRUNCATED:{len(body)}]" in output


def test_redaction_contract_is_shared_with_trace_exporter() -> None:
    from medidiag.observability.trace_exporter import TraceExporter

    payload = {"question": _CASE_QUESTION, "deepseek_api_key": _API_KEY}
    assert TraceExporter._sanitize(payload) == sanitize(payload)


def test_worker_run_emits_stage_logs_without_case_content(log_stream, tmp_path) -> None:
    """完整跑一遍 worker：状态跳转有日志，病例正文没有。"""
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'log.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)

    from medidiag.workflow.executor import WorkflowExecutor

    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session, _CASE_QUESTION, "idem-logging", "log-test"
        )
        executor.start_workflow(
            session, case.case_id, "case_workflow", "wf-logging", "input-hash"
        )
        case_id = case.case_id
    SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="log-worker"
    ).run_once()
    engine.dispose()

    output = log_stream.getvalue()
    assert "lease.acquired" in output
    assert "workflow.stage_committed" in output
    assert "worker.task_claimed" in output
    assert case_id in output
    _assert_no_sensitive_literal(output)


def test_api_error_log_omits_rejected_input(log_stream, tmp_path) -> None:
    """API 错误日志只记录错误码与路由，不记录被拒绝的输入。

    `MediDiagError.detail` 在 CASE_INVALID_INPUT 等分支上来自 pydantic
    ValidationError，会带上被拒绝的输入值，因此不进日志。
    """
    from fastapi.testclient import TestClient

    engine = create_db_engine(f"sqlite:///{(tmp_path / 'api-log.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    client = TestClient(create_app(session_factory=factory))

    response = client.post(
        "/api/v1/cases",
        json={"question": _CASE_QUESTION, "input_kind": "deidentified_simulation"},
        headers={"Idempotency-Key": "idem-api-log"},
    )
    engine.dispose()

    # 输入包含邮箱、手机号和身份证号，被脱敏门禁拒绝。
    assert response.status_code == 400
    assert response.json()["code"] == "CASE_INPUT_NOT_DEIDENTIFIED"
    output = log_stream.getvalue()
    assert "api.error" in output
    assert "CASE_INPUT_NOT_DEIDENTIFIED" in output
    assert "/api/v1/cases" in output
    _assert_no_sensitive_literal(output)


def test_provider_retry_decision_is_logged_without_response_body(log_stream) -> None:
    from medidiag.agents.llm_client import LLMClient
    from medidiag.errors import MediDiagError
    from medidiag.workflow.deepseek_provider import DeepSeekWorkflowProvider
    from medidiag.workflow.provider_runtime import ProviderCallRunner

    secret_body = "internal provider failure narrative that must not be logged"

    def fake_post(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            503,
            json={"error": {"message": secret_body}},
            headers={"x-request-id": "req-retry-1"},
            request=request,
        )

    provider = DeepSeekWorkflowProvider(
        LLMClient(
            api_key=_API_KEY, model="demo-model", max_retries=0, post=fake_post
        )
    )
    retrieval = {
        "chunks": [
            {"chunk_id": "c1", "source": "fixture", "text": _EVIDENCE_TEXT}
        ]
    }
    with pytest.raises(MediDiagError):
        ProviderCallRunner(sleep=lambda _: None).call(
            "generation",
            lambda: provider.generate(
                _CASE_QUESTION, retrieval, provider.plan("q", retrieval)
            ),
        )

    output = log_stream.getvalue()
    assert "provider.attempt_failed" in output
    assert "PROVIDER_UNAVAILABLE" in output
    assert "retry_decision" in output
    assert secret_body not in output
    _assert_no_sensitive_literal(output)


def test_json_renderer_emits_parsable_lines(monkeypatch, tmp_path) -> None:
    from medidiag.config import Settings, get_settings

    buffer = io.StringIO()
    get_settings.cache_clear()
    monkeypatch.setenv("STRUCTLOG_DEV", "0")
    monkeypatch.setattr(
        "medidiag.observability.logging.get_settings",
        lambda: Settings(structlog_dev=0, log_level="INFO"),
    )
    medidiag_logging._configured = False
    configure_logging(stream=buffer, force=True)
    try:
        get_logger("test").info("test.event", case_id="case_1", stage="review")
        payload = json.loads(buffer.getvalue().strip())
        assert payload["event"] == "test.event"
        assert payload["case_id"] == "case_1"
        assert payload["level"] == "info"
    finally:
        structlog.reset_defaults()
        medidiag_logging._configured = False
        get_settings.cache_clear()


def _assert_no_sensitive_literal(output: str) -> None:
    for literal in (
        _CASE_QUESTION,
        _API_KEY,
        _EVIDENCE_TEXT,
        "patient@example.com",
        "13800138000",
        "11010119800101001X",
    ):
        assert literal not in output, f"sensitive literal reached the log: {literal!r}"
