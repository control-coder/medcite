"""显式授权后，在既有环境、新库与独立进程中验收 MiMo；沿用同一调用账本。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from medidiag.llm.budget import BudgetedTransport
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from scripts.verify_offline import ROOT, run_browser


def collect_case_metadata(work: Path, cases: list[dict]) -> list[dict]:
    """从本次独立库补齐终态已清空的活动任务及生成事件，不读取患者库。"""
    with sqlite3.connect("file:" + (work / "verification.db").as_posix() + "?mode=ro", uri=True) as db:
        for case in cases:
            row = db.execute("SELECT task_id FROM workflow_tasks WHERE case_id=? ORDER BY id DESC LIMIT 1",
                             (case["case_id"],)).fetchone()
            if not row:
                continue
            case["task_id"] = row[0]
            details = [json.loads(item[0]) for item in db.execute(
                "SELECT detail FROM case_event_log WHERE case_id=? AND event_type='stage_completed'",
                (case["case_id"],))]
            case["generation_audit"] = [{k: v for k, v in detail.items() if k in {
                "task_id", "stage", "provider_request_id", "provider_retry_count", "usage", "model",
                "retry_count", "prompt_version", "finish_reason"}}
                for detail in details if detail.get("stage") == "generation" and detail.get("task_id") == row[0]]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--ledger", type=Path, required=True, help="已建立的本轮账本，不自动重置")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("必须取得用户真实调用授权并显式传 --allow-live")
    if Path(sys.prefix).name.lower() != "medidiag" or sys.version_info[:2] != (3, 11):
        parser.error("只允许 conda medidiag / Python 3.11")
    if not args.ledger.is_file():
        parser.error("请指定本轮已建立的账本，不自动创建新的付费预算")
    ledger = BudgetedTransport(args.ledger)
    before = ledger.records()
    if len(before) > 5:
        parser.error("当前余额不足三次代表性请求；停止，不重置账本")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    work = ROOT / ".cache/implementation" / ("mimo-live-" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True)
    env = {**os.environ, "DATABASE_URL": "sqlite:///" + (work / "verification.db").as_posix(),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "MEDIDIAG_ALLOWED_HOSTS": '["127.0.0.1", "localhost"]'}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    processes = []
    logs = []
    success = False
    report_path = ROOT / "artifacts/reports/application" / ("mimo-live-" + stamp + ".json")
    try:
        for name, command in [
            ("api", [sys.executable, "-m", "uvicorn", "medidiag.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)]),
            ("worker", [sys.executable, "-m", "medidiag.cli", "worker", "--loop", "--provider", "mimo_grounded", "--live-budget", str(ledger.path)])]:
            log = (work / (name + ".log")).open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=flags))
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get("/app/").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError("验收服务提前退出，请查看临时日志")
                time.sleep(0.2)
            else:
                raise RuntimeError("API 启动超时")
        run_browser("mimo-live.spec.ts", {**env, "MEDIDIAG_WEB_URL": base,
            "MEDIDIAG_ALLOW_LIVE": ACTIVE_MIMO_MODEL, "MEDIDIAG_LIVE_CASES": str(work / "cases.json")}, work, flags)
        cases = json.loads((work / "cases.json").read_text(encoding="utf-8"))
        after = ledger.records()
        assert len(cases) == 4 and len(after) - len(before) == 3
        assert all(c["outcome"] == "ready" for c in [cases[0]["analysis"], cases[1]["analysis"]])
        success = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        case_file = work / "cases.json"
        records = ledger.records()
        files = ["configs/application.yaml", "examples/public_health/sources.json", "examples/public_health/chunks.jsonl",
                 "src/medidiag/workflow/mimo_grounded.py", "src/medidiag/llm/budget.py"]
        report = {"schema_version": "mimo-application-acceptance-v1", "created_at": stamp,
            "success": success, "mode": "mimo_grounded", "model": ACTIVE_MIMO_MODEL, "provider": "mimo_v25",
            "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "implementation_state": "本轮源码在基线上未提交；随本报告提交，文件摘要用于定位验收实现。",
            "file_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
            "limits": {"max_calls_total": 8, "request_timeout_s": 45, "automatic_retries": 0,
                       "max_output_tokens_per_call": 2048, "max_input_characters": 12000},
            "timeout_boundary": "httpx 各网络操作超时45秒；不是完整工作流的45秒硬截止。",
            "calls_before": len(before), "calls_after": len(records), "transport_records": records,
            "cases": collect_case_metadata(work, json.loads(case_file.read_text(encoding="utf-8"))) if case_file.exists() else [],
            "cost_usd": None, "work_directory": work.relative_to(ROOT).as_posix(),
            "boundary": "既有环境、新SQLite、独立API/worker和Edge；不是新机安装、临床有效性或生产部署验收。",
            "fault_evidence": "超时、429、错误JSON、未知引用和改写由 tests/test_mimo_grounded.py 离线注入，不是真实服务故障。"}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("真实验收记录：", report_path)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
