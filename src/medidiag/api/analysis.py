"""将研究报告收敛为用户可读契约，不暴露内部推理或上游异常。"""

from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from medidiag.api.schemas import AnalysisResponse, ClaimResponse, EvidenceResponse
from medidiag.db.models import Case, CaseReport, StageArtifact, WorkflowTask


def build_analysis(session: Session, case: Case) -> AnalysisResponse:
    task = session.scalar(select(WorkflowTask).where(WorkflowTask.case_id == case.case_id)
                          .order_by(WorkflowTask.id.desc()).limit(1))
    artifact = session.scalar(
        select(StageArtifact).where(StageArtifact.case_id == case.case_id,
                                    StageArtifact.stage == "retrieval")
        .order_by(StageArtifact.id.desc()).limit(1)
    )
    evidence = []
    for chunk in artifact.payload.get("chunks", []) if artifact else []:
        url = str(chunk.get("source_url") or chunk.get("url") or "")
        parsed = urlparse(url)
        evidence.append(EvidenceResponse(
            chunk_id=chunk["chunk_id"], text=chunk.get("text", ""),
            source=chunk.get("source", "未标注来源"), source_id=chunk.get("source_id", ""),
            source_url=url if parsed.scheme in {"http", "https"} and parsed.netloc else None,
            evidence_level=chunk.get("evidence_level", "未标注证据级别"),
        ))
    result = AnalysisResponse(case_id=case.case_id, status=case.status,
                              outcome="processing", message="正在处理，请稍后查看。", evidence=evidence)
    if case.status == "CLOSED_CANCELLED":
        result.outcome, result.message = "cancelled", "任务已取消，没有可用报告。"
        result.retry_action = "resubmit"
    elif case.status in {"ESCALATED", "CLOSED_FAILED", "CLOSED_ESCALATED"}:
        result.outcome, result.message = "failed", "本次处理未完成，未交付报告；可重新提交，持续失败时联系维护者。"
        result.failure_code = task.error_code if task else None
        result.retry_action = "resubmit"
    elif case.status == "CLOSED_SUCCESS":
        report = session.scalar(select(CaseReport).where(CaseReport.case_id == case.case_id)
                                .order_by(CaseReport.version.desc()).limit(1))
        data = report.structured_report if report else {}
        if not report or data.get("schema_version") != "assistant-report-v1":
            result.outcome, result.message = "failed", "未找到兼容的辅助分析报告。"
            return result
        ids = {item.chunk_id for item in evidence}
        for claim in data.get("claims", []):
            links = claim.get("citation_chunk_ids", [])
            if links and all(item in ids for item in links):
                result.claims.append(ClaimResponse(claim_id=claim["claim_id"], text=claim["text"],
                                                   evidence_ids=links))
        result.limitations = list(data.get("limitations", []))
        if data.get("uncertainty"):
            result.limitations.append(str(data["uncertainty"]))
        result.risk_warnings = list(data.get("risk_warnings", []))
        result.next_steps = list(data.get("next_steps", []))
        if not evidence or not result.claims:
            result.outcome = "insufficient_evidence"
            result.message = "证据不足：没有可追踪的支持引用，不输出确定性分析。"
            result.limitations.append("检索为空或引用无法关联到本任务的证据片段。")
        else:
            result.outcome, result.message = "ready", "辅助分析已完成；引用关联不等于医学正确性核验。"
            result.summary = data.get("summary")
    return result
