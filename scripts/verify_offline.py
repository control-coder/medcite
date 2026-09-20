"""指定 conda 环境内，以独立数据库、API 和离线 worker 进程复现完整流程。"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
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
            for args in (
                ["-m", "uvicorn", "medidiag.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
                ["-m", "medidiag.cli", "worker", "--loop", "--provider", "fake_offline"],
            ):
                processes.append(subprocess.Popen([sys.executable, *args], env=env, cwd=ROOT, stdout=log, stderr=log, creationflags=flags))
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
                payload = {"symptoms": "模拟验证输入：轻微不适，不涉及真实患者。", "duration": "模拟两天",
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
                evidence_ids = {item["chunk_id"] for item in analysis["evidence"]}
                assert analysis["claims"] and all(set(item["evidence_ids"]) <= evidence_ids for item in analysis["claims"])
                assert analysis["observation"]["cost_usd"] is None
                history = client.get("/api/v1/cases").json()
                assert [item["case_id"] for item in history["items"]] == [case_id]
                with httpx.Client(base_url=str(client.base_url), trust_env=False) as other:
                    assert other.get(prefix).status_code == 404
                    assert other.get("/api/v1/cases").json()["items"] == []
                summary = {"case_id": case_id, "status": analysis["status"], "evidence_count": len(evidence_ids),
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
