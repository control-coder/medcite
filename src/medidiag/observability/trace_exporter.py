"""Export database events as redacted, reproducible trace artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from medidiag.db.models import (
    AgentRun,
    Case,
    CaseEventLog,
    CaseReport,
    StageArtifact,
    WorkflowTask,
)
from medidiag.errors import MediDiagError, get_error_spec

TRACE_SCHEMA_VERSION = "1.0"

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "idempotency_key",
    "password",
    "prompt",
    "question",
    "secret",
    "token",
}
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CN_ID = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


@dataclass(frozen=True)
class TraceExportResult:
    case_id: str
    trace_id: str
    raw_path: Path
    summary_path: Path
    event_count: int


class TraceExporter:
    """Join append-only events with stage metadata and write redacted traces."""

    def export_case(
        self,
        session: Session,
        case_id: str,
        *,
        raw_dir: str | Path,
        summary_dir: str | Path,
    ) -> TraceExportResult:
        case = session.execute(
            select(Case).where(Case.case_id == case_id)
        ).scalar_one_or_none()
        if case is None:
            raise MediDiagError("CASE_NOT_FOUND", detail=f"case {case_id} not found")

        events = list(
            session.execute(
                select(CaseEventLog)
                .where(CaseEventLog.case_id == case_id)
                .order_by(CaseEventLog.id)
            ).scalars()
        )
        tasks = list(
            session.execute(
                select(WorkflowTask)
                .where(WorkflowTask.case_id == case_id)
                .order_by(WorkflowTask.id)
            ).scalars()
        )
        artifacts = list(
            session.execute(
                select(StageArtifact)
                .where(StageArtifact.case_id == case_id)
                .order_by(StageArtifact.id)
            ).scalars()
        )
        agent_runs = list(
            session.execute(
                select(AgentRun)
                .where(AgentRun.case_id == case_id)
                .order_by(AgentRun.id)
            ).scalars()
        )
        report = session.execute(
            select(CaseReport)
            .where(CaseReport.case_id == case_id)
            .order_by(CaseReport.version.desc())
            .limit(1)
        ).scalar_one_or_none()

        raw_events = [
            self._event_record(case, event, tasks, artifacts, agent_runs)
            for event in events
        ]
        raw_path = Path(raw_dir) / f"{case.trace_id}.jsonl"
        summary_path = Path(summary_dir) / f"{case.trace_id}.json"
        self._write_jsonl(raw_path, raw_events)
        summary = self._summary(case, raw_path, raw_events, artifacts, report)
        self._write_json(summary_path, summary)
        return TraceExportResult(
            case_id=case.case_id,
            trace_id=case.trace_id,
            raw_path=raw_path,
            summary_path=summary_path,
            event_count=len(raw_events),
        )

    def _event_record(
        self,
        case: Case,
        event: CaseEventLog,
        tasks: list[WorkflowTask],
        artifacts: list[StageArtifact],
        agent_runs: list[AgentRun],
    ) -> dict[str, Any]:
        detail = event.detail or {}
        stage = detail.get("stage")
        attempt = detail.get("attempt", detail.get("task_attempt"))
        task_id = detail.get("task_id")
        artifact = self._find_artifact(artifacts, task_id, stage, attempt)
        if task_id is None and artifact is not None:
            task_id = artifact.task_id
        if task_id is None and len(tasks) == 1:
            task_id = tasks[0].task_id

        latency_ms = detail.get("latency_ms")
        if latency_ms is None and artifact is not None:
            latency_ms = artifact.latency_ms
        ended_at = event.created_at
        started_at = (
            ended_at - timedelta(milliseconds=latency_ms)
            if latency_ms is not None
            else ended_at
        )
        error_code = detail.get("error_code")
        if error_code is None and detail.get("reason") in {
            "TASK_LEASE_LOST",
            "TASK_LEASE_EXPIRED",
        }:
            error_code = detail["reason"]
        error_meta = self._error_metadata(error_code)

        matching_runs = []
        if stage == "generation" and task_id is not None and attempt is not None:
            attempt_group = f"{task_id}:{attempt}"
            matching_runs = [
                item.run_id for item in agent_runs if item.attempt_group == attempt_group
            ]

        review_payload = artifact.payload if artifact and stage == "review" else {}
        compliance_status = review_payload.get("compliance_status")
        citation_verdicts = review_payload.get("citation_verdicts", [])
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event_id": event.id,
            "trace_id": case.trace_id,
            "case_id": case.case_id,
            "task_id": task_id,
            "agent_run_id": matching_runs[0] if len(matching_runs) == 1 else None,
            "agent_run_ids": matching_runs,
            "event_type": event.event_type,
            "stage": stage,
            "from_state": event.from_status,
            "to_state": event.to_status,
            "trigger_subject": event.trigger_subject,
            "worker_id": event.trigger_entity,
            "attempt": attempt,
            "input_hash": artifact.input_hash if artifact else None,
            "output_hash": artifact.output_hash if artifact else None,
            "component_version": (
                artifact.component_version if artifact else detail.get("provider_version")
            ),
            "started_at": self._iso_utc(started_at),
            "ended_at": self._iso_utc(ended_at),
            "latency_ms": latency_ms,
            "retry_count": self._retry_count(event.event_type, detail),
            "error_code": error_code,
            **error_meta,
            "compliance_hit": (
                compliance_status is not None
                and not str(compliance_status).startswith("PASSED")
            ),
            "compliance_status": compliance_status,
            "citation_verdicts": citation_verdicts,
            "human_decision": (
                {
                    "decision": detail.get("decision"),
                    "reason": detail.get("reason"),
                }
                if event.event_type == "human_decision"
                else None
            ),
            "detail": detail,
        }
        return self._sanitize(record)

    @staticmethod
    def _find_artifact(
        artifacts: list[StageArtifact],
        task_id: str | None,
        stage: str | None,
        attempt: int | None,
    ) -> StageArtifact | None:
        if stage is None:
            return None
        candidates = [item for item in artifacts if item.stage == stage]
        if task_id is not None:
            candidates = [item for item in candidates if item.task_id == task_id]
        if attempt is not None:
            candidates = [item for item in candidates if item.attempt == attempt]
        return candidates[-1] if candidates else None

    def _summary(
        self,
        case: Case,
        raw_path: Path,
        raw_events: list[dict[str, Any]],
        artifacts: list[StageArtifact],
        report: CaseReport | None,
    ) -> dict[str, Any]:
        latest_by_stage: dict[str, StageArtifact] = {}
        for artifact in artifacts:
            latest_by_stage[artifact.stage] = artifact
        retrieval = latest_by_stage.get("retrieval")
        generation = latest_by_stage.get("generation")
        review = latest_by_stage.get("review")

        evidence = []
        if retrieval:
            for item in retrieval.payload.get("chunks", []):
                evidence.append(
                    {
                        "chunk_id": item.get("chunk_id"),
                        "source": item.get("source"),
                        "source_id": item.get("source_id"),
                        "evidence_level": item.get("evidence_level"),
                        "score": item.get("score"),
                    }
                )
        claims = []
        if generation:
            for item in generation.payload.get("claims", []):
                claims.append(
                    {
                        "claim_id": item.get("claim_id"),
                        "claim_hash": self._hash_text(item.get("text", "")),
                        "citation_chunk_ids": item.get("citation_chunk_ids", []),
                    }
                )

        stage_latencies: dict[str, list[int]] = {}
        for item in artifacts:
            stage_latencies.setdefault(item.stage, []).append(item.latency_ms)
        summary = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": case.trace_id,
            "case_id": case.case_id,
            "final_state": case.status,
            "raw_file": raw_path.name,
            "raw_event_ids": [item["event_id"] for item in raw_events],
            "event_count": len(raw_events),
            "stage_latencies_ms": stage_latencies,
            "timeline": [
                {
                    "event_id": item["event_id"],
                    "event_type": item["event_type"],
                    "stage": item["stage"],
                    "from_state": item["from_state"],
                    "to_state": item["to_state"],
                    "error_code": item["error_code"],
                }
                for item in raw_events
            ],
            "evidence": evidence,
            "claims": claims,
            "citation_verdicts": (
                review.payload.get("citation_verdicts", []) if review else []
            ),
            "compliance_status": (
                review.payload.get("compliance_status") if review else None
            ),
            "recovery": {
                "lease_reclaimed_event_ids": [
                    item["event_id"]
                    for item in raw_events
                    if item["event_type"] == "lease_reclaimed"
                ],
                "lease_lost_event_ids": [
                    item["event_id"]
                    for item in raw_events
                    if item["error_code"] == "TASK_LEASE_LOST"
                ],
            },
            "human_decisions": [
                {"event_id": item["event_id"], **item["human_decision"]}
                for item in raw_events
                if item["human_decision"] is not None
            ],
            "report": (
                {
                    "report_id": report.report_id,
                    "version": report.version,
                    "risk_warnings": report.risk_warnings,
                    "compliance_status": report.compliance_status,
                    "generation_version": report.generation_version,
                }
                if report
                else None
            ),
        }
        return self._sanitize(summary)

    @staticmethod
    def _error_metadata(error_code: str | None) -> dict[str, Any]:
        if error_code is None:
            return {"retryable": False, "action": None, "alert": False}
        try:
            spec = get_error_spec(error_code)
        except KeyError:
            return {"retryable": False, "action": None, "alert": True}
        return {
            "retryable": spec.retryable,
            "action": spec.default_action,
            "alert": spec.alert,
        }

    @staticmethod
    def _retry_count(event_type: str, detail: dict[str, Any]) -> int | None:
        if detail.get("provider_retry_count") is not None:
            return int(detail["provider_retry_count"])
        if event_type == "provider_call" and detail.get("provider_attempt") is not None:
            return max(0, int(detail["provider_attempt"]) - 1)
        return None

    @classmethod
    def _sanitize(cls, value: Any, key: str | None = None) -> Any:
        if key and cls._is_sensitive_key(key):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {item_key: cls._sanitize(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, str):
            value = _BEARER.sub("Bearer [REDACTED]", value)
            value = _EMAIL.sub("[REDACTED_EMAIL]", value)
            value = _PHONE.sub("[REDACTED_PHONE]", value)
            return _CN_ID.sub("[REDACTED_ID]", value)
        return value

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        return (
            normalized in _SENSITIVE_KEYS
            or normalized.endswith(("_api_key", "_password", "_secret", "_token"))
            or "authorization" in normalized
            or normalized.endswith("question")
            or normalized == "prompt"
        )

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _iso_utc(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        )
        TraceExporter._atomic_write(path, content)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        TraceExporter._atomic_write(path, content)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
