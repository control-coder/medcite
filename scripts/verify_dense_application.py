"""向量检索 + 真实模型的应用路径验收：同一套工作流（状态机、租约、审核、报告），只换检索方案。

需要显式授权（--allow-live）和一个本轮账本（BudgetedTransport 上限 8 次请求，含补救请求）。
用 6 个问题覆盖：字面问法、口语换说法、有答案、主题相关但没答案、范围外。结果写入
artifacts/reports/application/，不改动任何现有数据库。

    python -I scripts/verify_dense_application.py --allow-live --ledger .cache/dense-live-ledger.db
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

QUESTIONS = [
    ("字面问法", "咳嗽时如何用纸巾遮住口鼻并洗手？"),
    ("口语换说法", "屋里太闷怎样透透气？"),
    ("有答案", "高温时可以把孩子留在停放的车辆中吗？"),
    ("主题相关但没答案", "疫苗抗体滴度的具体阈值"),
    ("范围外", "量子纠缠计算芯片是什么原理？"),
    ("口语换说法", "天气特别热的时候出门要注意什么？"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--app-config", default="configs/application_dense.yaml")
    parser.add_argument("--cases", type=int, nargs="*", help="只运行这些序号（从 1 起）的问题；默认全部")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("必须取得真实调用授权并显式传 --allow-live")

    from medidiag.db.models import Case, CaseReport, StageArtifact
    from medidiag.db.session import create_db_engine, get_session_factory, init_db
    from medidiag.llm.budget import BudgetedTransport
    from medidiag.llm.models import ACTIVE_MIMO_MODEL
    from medidiag.workflow.application import build_application_provider
    from medidiag.workflow.executor import WorkflowExecutor
    from medidiag.workflow.worker import SingleMachineWorker

    ledger = BudgetedTransport(args.ledger)
    before = len(ledger.records())
    provider = build_application_provider("mimo_grounded", app_config=args.app_config, root=ROOT,
                                          live_budget=args.ledger)
    with tempfile.TemporaryDirectory(prefix="medidiag-dense-live-") as tmp:
        engine = create_db_engine("sqlite:///" + (Path(tmp) / "verify.db").as_posix())
        init_db(engine)
        factory = get_session_factory(engine)
        executor = WorkflowExecutor()
        worker = SingleMachineWorker(factory, provider, worker_id="dense-live")
        rows = []
        selected = [q for i, q in enumerate(QUESTIONS, 1) if not args.cases or i in args.cases]
        for label, question in selected:
            with factory() as session:
                case = executor.create_case(session, question, uuid.uuid4().hex, "dense-live")
                executor.start_workflow(session, case.case_id, "case_workflow", "run", "hash")
                case_id = case.case_id
            started = time.perf_counter()
            for _ in range(20):
                if not worker.run_once().processed:
                    break
            elapsed = round(time.perf_counter() - started, 2)
            with factory() as session:
                status = session.query(Case.status).filter(Case.case_id == case_id).scalar()
                report = session.query(CaseReport).filter(CaseReport.case_id == case_id).first()
                structured = report.structured_report if report else {}
                retrieved = session.query(StageArtifact.payload).filter(
                    StageArtifact.case_id == case_id, StageArtifact.stage == "retrieval").first()
                rewrite = (retrieved[0] or {}).get("query_rewrite") if retrieved else None
                search_agent = (retrieved[0] or {}).get("search_agent") if retrieved else None
            rows.append({"kind": label, "question": question, "status": status, "seconds": elapsed,
                         "abstained": not structured.get("claims"), "query_rewrite": rewrite, "search_agent": search_agent,
                         "retrieved_chunks": [c["chunk_id"] for c in (retrieved[0] or {}).get("chunks", [])] if retrieved else [],
                         "claims": [{"chunk_id": c["citation_chunk_ids"][0], "text": c["text"]}
                                    for c in structured.get("claims", [])]})
            print(label, question, status, "拒答" if rows[-1]["abstained"] else f"{len(rows[-1]['claims'])} 条摘录")
        engine.dispose()
    records = ledger.records()
    report_doc = {
        "schema_version": "dense-application-live-v1", "created_at": datetime.now(UTC).isoformat(),
        "app_config": args.app_config, "on_invalid": "feedback", "cases_selected": args.cases or "all", "model": ACTIVE_MIMO_MODEL,
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "calls": len(records) - before, "ledger_total": len(records),
        "transport_records": records, "cases": rows,
        "boundary": "少量问题的真实调用冒烟，证明所选检索方案能走完整个应用工作流；不是准确率评测。"}
    out = ROOT / "artifacts/reports/application" / f"dense-live-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("调用次数：", report_doc["calls"], "报告：", out)


if __name__ == "__main__":
    main()
