"""指定 conda 环境内，以独立数据库、API 和离线 worker 进程复现完整流程。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def run_browser(spec: str, env: dict[str, str], work: Path, flags: int) -> None:
    """显式保存浏览器输出，避免 Windows 隐藏子进程吞掉失败原因。"""
    path = work / (spec + ".log")
    with path.open("w", encoding="utf-8") as log:
        result = subprocess.run([shutil.which("node"), "node_modules/@playwright/test/cli.js", "test", spec],
                                cwd=ROOT / "frontend", env=env, stdout=log, stderr=log, creationflags=flags)
    if result.returncode:
        raise RuntimeError(f"浏览器验收失败，请查看本次日志：{path}")
    print(f"浏览器验收通过：{spec}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["fake_offline", "retrieval_mock"], default="fake_offline")
    parser.add_argument("--browser", action="store_true", help="同时运行现有 Edge 浏览器验收，不下载浏览器")
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "medidiag" or sys.version_info[:2] != (3, 11):
        raise SystemExit("请先激活 conda medidiag（Python 3.11），禁止使用其他环境。")
    if not (ROOT / "frontend" / "dist" / "index.html").is_file():
        raise SystemExit("请先执行 npm ci --prefix frontend 与 npm run build --prefix frontend。")
    work = ROOT / ".cache" / "implementation" / ("offline-" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True)
    env = {**os.environ, "DATABASE_URL": "sqlite:///" + (work / "verification.db").as_posix(),
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONUTF8": "1", "MEDIDIAG_ALLOWED_HOSTS": '["127.0.0.1", "localhost"]'}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], env=env, cwd=ROOT, check=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = []
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with (work / "processes.log").open("w", encoding="utf-8") as log:
        try:
            for command in (
                ["-m", "uvicorn", "medidiag.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
                ["-m", "medidiag.cli", "worker", "--loop", "--provider", args.provider],
            ):
                processes.append(subprocess.Popen([sys.executable, *command], env=env, cwd=ROOT, stdout=log, stderr=log, creationflags=flags))
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        client.get("/api/v1/session").raise_for_status()
                        break
                    except httpx.HTTPError:
                        if time.monotonic() >= deadline or any(p.poll() is not None for p in processes):
                            raise RuntimeError("服务启动失败，请检查本次独立日志。") from None
                        time.sleep(0.25)
                for path in ("/app/", "/assistant", "/demo"):
                    client.get(path).raise_for_status()
                payload = {"symptoms": ("通风如何改善室内空气质量？" if args.provider == "retrieval_mock"
                                       else "模拟验证输入：轻微不适，不涉及真实患者。"), "duration": "模拟两天",
                           "input_kind": "deidentified_simulation", "non_sensitive_confirmed": True}
                created = client.post("/api/v1/consultations", json=payload, headers={"Idempotency-Key": "offline-create"})
                created.raise_for_status()
                case_id = created.json()["case_id"]
                prefix = "/api/v1/cases/" + case_id
                duplicate = client.post("/api/v1/consultations", json=payload, headers={"Idempotency-Key": "offline-create"})
                duplicate.raise_for_status()
                assert duplicate.json()["case_id"] == case_id
                client.post(prefix + "/workflow", headers={"Idempotency-Key": "offline-run"}).raise_for_status()
                deadline = time.monotonic() + 30
                while True:
                    response = client.get(prefix + "/analysis")
                    response.raise_for_status()
                    analysis = response.json()
                    if analysis["outcome"] != "processing":
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("离线 worker 未在限定时间内完成。")
                    time.sleep(0.25)
                assert analysis["outcome"] == "ready", analysis["outcome"]
                assert analysis["execution_mode"] == args.provider
                evidence_ids = {item["chunk_id"] for item in analysis["evidence"]}
                assert analysis["claims"] and all(set(item["evidence_ids"]) <= evidence_ids for item in analysis["claims"])
                assert analysis["observation"]["cost_usd"] is None
                history = client.get("/api/v1/cases").json()
                assert [item["case_id"] for item in history["items"]] == [case_id]
                with httpx.Client(base_url=str(client.base_url), trust_env=False) as other:
                    assert other.get(prefix).status_code == 404
                    assert other.get("/api/v1/cases").json()["items"] == []
                if args.provider == "retrieval_mock":
                    assert all(e["source_url"] for e in analysis["evidence"])
                    empty = client.post("/api/v1/consultations", json={**payload, "symptoms": "模拟提问：量子纠缠计算芯片"},
                                        headers={"Idempotency-Key": "empty-create"})
                    empty.raise_for_status()
                    empty_prefix = "/api/v1/cases/" + empty.json()["case_id"]
                    client.post(empty_prefix + "/workflow", headers={"Idempotency-Key": "empty-run"}).raise_for_status()
                    deadline = time.monotonic() + 30
                    while True:
                        data = client.get(empty_prefix + "/analysis").json()
                        if data["outcome"] != "processing":
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError("空证据验证超时。")
                        time.sleep(0.25)
                    assert data["outcome"] == "insufficient_evidence" and data["claims"] == []
                    assert data["execution_mode"] == "retrieval_mock"
                if args.browser:
                    browser_env = {**env, "MEDIDIAG_WEB_URL": str(client.base_url), "MEDIDIAG_TEST_PROVIDER": args.provider}
                    spec = "public-rag.spec.ts" if args.provider == "retrieval_mock" else "flow.spec.ts"
                    run_browser(spec, browser_env, work, flags)
                    # 正常闭环结束后停止本次 worker，避免快速离线处理与取消按钮竞争。
                    # 此处只验证待领取任务的真实 API/浏览器取消，不伪称慢模型或运行中中断。
                    processes[1].terminate()
                    processes[1].wait(timeout=10)
                    browser_env["MEDIDIAG_TEST_WORKER_PAUSED"] = "1"
                    run_browser("lifecycle.spec.ts", browser_env, work, flags)
                summary = {"case_id": case_id, "provider": args.provider, "status": analysis["status"], "evidence_count": len(evidence_ids),
                           "browser_verified": args.browser, "browser_lifecycle_verified": args.browser,
                           "cancellation_boundary": ("浏览器待领取取消；运行中迟到写入另由单元测试覆盖。"
                                                     if args.browser else "本次未运行浏览器取消验收。"),
                           "separate_api_and_worker": True, "new_database": True, "cross_owner_denied": True,
                           "conda_environment": "medidiag", "python_version": sys.version.split()[0],
                           "boundary": "现有指定环境、新数据库与独立进程；不是全新机器或临床验收。"}
                (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print("离线闭环通过；记录目录：", work)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
