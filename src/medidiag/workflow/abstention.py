"""无证据时只记录拒答，不让生成模型补造证据。"""

from typing import Any


def abstention_payload(stage: str, case_id: str, retrieval: dict[str, Any] | None = None) -> dict[str, Any]:
    """``retrieval`` 带查询路由的拒答决定（``query_router.refused``）时，说明改用路由给出的固定提示。"""
    route = (retrieval or {}).get("query_router")
    routed = route if isinstance(route, dict) and route.get("refused") else None
    payloads: dict[str, dict[str, Any]] = {
        "plan": {"objective": "无证据时拒答", "evidence_ids": [], "requires_uncertainty": True},
        "generation": {"agents": [], "claims": [], "risk_flags": [], "uncertainty": "未检索到证据。"},
        "arbitration": {"selected_claim_ids": [], "conflicts": [], "verdict": "NO_EVIDENCE"},
        "review": {"verdict": "APPROVED", "issues": [], "citation_verdicts": [],
                   "compliance_status": "PASS_WITH_DEMO_LIMITATION"},
        "report": {
            "schema_version": "assistant-report-v1", "case_id": case_id,
            "title": "证据不足说明", "summary": "证据不足，无法形成辅助分析。", "claims": [],
            "filtered_claim_count": 0, "disclaimer": "仅作工程演示，不构成医疗建议。",
            "risk_warnings": ["不能用于真实医疗决策。"],
            "limitations": ["未检索到支持证据；未调用生成模型。"],
            "next_steps": ["请补充公开来源或模拟背景；真实健康问题请咨询专业人员。"],
            "provenance": {"guard": "no-evidence-v1"},
        },
    }
    if routed is not None:
        message = str(routed.get("message") or "")
        intent = str(routed.get("intent"))
        payloads["generation"]["uncertainty"] = message
        payloads["report"].update(
            title="无法作答说明", summary=message,
            limitations=[f"查询路由判定为 {intent}（来源：{routed.get('source')}），未检索、未调用生成模型。"],
            next_steps=(["请立即拨打 120 或前往就近急诊。"] if intent == "emergency"
                        else ["真实健康问题请咨询专业人员。"]),
            provenance={"guard": "query-router-v1", "intent": intent, "source": routed.get("source"),
                        "rule": routed.get("rule")})
    return payloads[stage]
