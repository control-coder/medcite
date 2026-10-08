"""多进程可靠性压测：随机杀死/冻结 worker，核对任务不丢失、结果不重复。

编排进程提交 N 个病例，启动若干独立 worker 进程（每个进程循环领取任务，空闲时兼做租约扫描），
然后在运行期间：

- ``--kills``：对正在持有任务的 worker 做硬终止（Windows TerminateProcess / POSIX SIGKILL），再起新进程补位；
- ``--freezes``：把持有任务的 worker 整个进程挂起超过租约时长（心跳线程一并停止），等其他 worker 接管后再恢复，
  检验“僵尸写入”会被租约条件拒绝而不是产生双写。

结束后只从数据库和调用日志核对不变量，不信任 worker 自己的输出。Provider 是带固定延迟的确定性假实现，
不产生任何付费调用。SQLite 文件库默认；``--spawn-postgres`` 会启动一个一次性的 postgres 容器（容器名、端口、
数据目录均独立，结束即删除），不会碰已有容器与卷。

用法示例::

    python -I scripts/stress_reliability.py --backend sqlite --kills 8 --freezes 2
    python -I scripts/stress_reliability.py --backend postgres --kills 8 --freezes 2
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import os
import random
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from medidiag.config import get_settings
from medidiag.db.models import Case, CaseEventLog, CaseReport, StageArtifact, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker

PROVIDER_STAGES = ("normalize", "retrieve", "plan", "generate", "arbitrate", "review", "report")
REPORT_DIR = Path("artifacts/reports/reliability")


class DelayedProvider(DeterministicWorkflowProvider):
    """每次 provider 调用先追加一行调用日志，再睡眠固定时长，使 kill 落在调用中途。"""

    def __init__(self, delay_s: float, log_path: str) -> None:
        super().__init__()
        self._delay_s = delay_s
        self._log_path = log_path

    def _enter(self, stage: str) -> None:
        # 每个进程独立文件：Windows 上跨进程 O_APPEND 并非原子，共用一个文件会丢行或交错。
        fd = os.open(f"{self._log_path}.{os.getpid()}", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            os.write(fd, f"{stage}\n".encode())
        finally:
            os.close(fd)
        time.sleep(self._delay_s)

    def normalize(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("normalize")
        return super().normalize(*a, **k)

    def retrieve(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("retrieve")
        return super().retrieve(*a, **k)

    def plan(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("plan")
        return super().plan(*a, **k)

    def generate(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("generate")
        return super().generate(*a, **k)

    def arbitrate(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("arbitrate")
        return super().arbitrate(*a, **k)

    def review(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("review")
        return super().review(*a, **k)

    def report(self, *a: Any, **k: Any) -> dict[str, Any]:
        self._enter("report")
        return super().report(*a, **k)


# ---------------------------------------------------------------- 子进程：worker


def child_main(worker_id: str, delay_s: float, log_path: str, stop_file: str, lease_log: str) -> None:
    engine = create_db_engine(get_settings().database_url)
    factory = get_session_factory(engine)
    worker = SingleMachineWorker(factory, DelayedProvider(delay_s, log_path), worker_id=worker_id)
    scanner = LeaseScanner(factory, recovery_worker_id=worker_id)
    while not os.path.exists(stop_file):
        try:
            result = worker.run_once()
            if not result.processed:
                # 只有空闲时才扫描，被接管的任务立刻由本进程执行，避免接管后排队再次过期。
                scanner.scan_once(limit=1)
                result = worker.run_once()
        except Exception as exc:  # 被冻结后恢复的僵尸写入会走到这里
            code = getattr(exc, "code", None) or type(exc).__name__
            with open(f"{lease_log}.{worker_id}", "a", encoding="utf-8") as handle:
                handle.write(f"{worker_id} {code}\n")
            result = None
        if result is None or not result.processed:
            time.sleep(0.1)


# ---------------------------------------------------------------- 进程挂起/恢复


def _suspend(pid: int) -> None:
    if sys.platform == "win32":
        handle = ctypes.windll.kernel32.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
        ctypes.windll.ntdll.NtSuspendProcess(handle)
        ctypes.windll.kernel32.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGSTOP)


def _resume(pid: int) -> None:
    if sys.platform == "win32":
        handle = ctypes.windll.kernel32.OpenProcess(0x0800, False, pid)
        ctypes.windll.ntdll.NtResumeProcess(handle)
        ctypes.windll.kernel32.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGCONT)


# ---------------------------------------------------------------- 一次性 Postgres


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def throwaway_postgres() -> Any:
    name = f"medidiag-stress-pg-{uuid.uuid4().hex[:8]}"
    port = _free_port()
    password = secrets.token_hex(8)
    subprocess.run(
        ["docker", "run", "--rm", "-d", "--name", name, "-p", f"127.0.0.1:{port}:5432",
         "--tmpfs", "/var/lib/postgresql/data", "-e", f"POSTGRES_PASSWORD={password}",
         "-e", "POSTGRES_DB=medidiag_stress", "postgres:17-alpine"],
        check=True, capture_output=True,
    )
    try:
        for _ in range(60):
            probe = subprocess.run(["docker", "exec", name, "pg_isready", "-U", "postgres", "-d", "medidiag_stress"],
                                   capture_output=True)
            if probe.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("postgres 容器未在 60 秒内就绪")
        time.sleep(1)  # pg_isready 在初始化临时实例阶段也可能返回成功
        yield f"postgresql+psycopg://postgres:{password}@127.0.0.1:{port}/medidiag_stress"
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ---------------------------------------------------------------- 编排


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


class Fleet:
    def __init__(self, database_url: str, args: argparse.Namespace, workdir: Path) -> None:
        self.args = args
        self.workdir = workdir
        self.env = dict(
            os.environ,
            DATABASE_URL=database_url,
            MEDIDIAG_LEASE_SECONDS=str(args.lease_seconds),
            MEDIDIAG_HEARTBEAT_SECONDS=str(max(1, args.lease_seconds // 3)),
            PYTHONIOENCODING="utf-8",
        )
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.spawned = 0
        self.invocation_log = str(workdir / "invocations.log")
        self.lease_log = str(workdir / "lease_errors.log")
        self.stop_file = str(workdir / "STOP")
        self.flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.started_wall = datetime.now(UTC).replace(tzinfo=None)

    def spawn(self) -> str:
        self.spawned += 1
        worker_id = f"stress-w{self.spawned}"
        self.procs[worker_id] = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--child", worker_id,
             "--stage-delay-ms", str(self.args.stage_delay_ms), "--invocation-log", self.invocation_log,
             "--stop-file", self.stop_file, "--lease-log", self.lease_log],
            env=self.env, creationflags=self.flags, stdout=subprocess.DEVNULL,
            stderr=open(self.workdir / f"{worker_id}.err", "wb"),
        )
        return worker_id

    def stop_all(self) -> None:
        Path(self.stop_file).touch()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def run_scenario(database_url: str, args: argparse.Namespace) -> dict[str, Any]:
    os.environ["DATABASE_URL"] = database_url
    os.environ["MEDIDIAG_LEASE_SECONDS"] = str(args.lease_seconds)
    os.environ["MEDIDIAG_HEARTBEAT_SECONDS"] = str(max(1, args.lease_seconds // 3))
    get_settings.cache_clear()
    engine = create_db_engine(database_url)
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    rng = random.Random(args.seed)

    workdir = Path(tempfile.mkdtemp(prefix="medidiag-stress-"))
    fleet = Fleet(database_url, args, workdir)
    try:
        case_ids: list[str] = []
        with factory() as session:
            for i in range(args.cases):
                case = executor.create_case(session, f"公开模拟输入 {i}，用于可靠性压测。", uuid.uuid4().hex, "stress")
                executor.start_workflow(session, case.case_id, "case_workflow", "run", "hash")
                case_ids.append(case.case_id)

        started = time.monotonic()
        fleet.started_wall = datetime.now(UTC).replace(tzinfo=None)
        for _ in range(args.workers):
            fleet.spawn()

        kills_done = frozen_done = 0
        kill_skipped = 0
        events: list[dict[str, Any]] = []
        next_event = started + args.first_event_after_s
        plan = ["kill"] * args.kills + ["freeze"] * args.freezes
        rng.shuffle(plan)
        deadline = started + args.timeout_s

        def holders() -> dict[str, str]:
            with factory() as session:
                rows = session.execute(
                    select(WorkflowTask.lease_owner, WorkflowTask.task_id).where(WorkflowTask.status == "RUNNING")
                ).all()
            return {owner: task for owner, task in rows if owner in fleet.procs and fleet.procs[owner].poll() is None}

        def unfinished() -> int:
            with factory() as session:
                return int(session.scalar(select(func.count()).select_from(Case).where(
                    Case.case_id.in_(case_ids), Case.status.not_in(["CLOSED_SUCCESS", "ESCALATED", "CLOSED_FAILURE"]))) or 0)

        frozen: list[tuple[str, float]] = []
        while time.monotonic() < deadline:
            now = time.monotonic()
            for worker_id, resume_at in list(frozen):
                if now >= resume_at:
                    _resume(fleet.procs[worker_id].pid)
                    frozen.remove((worker_id, resume_at))
            if plan and now >= next_event:
                held = holders()
                busy = [w for w in held if all(w != f[0] for f in frozen)]
                if busy:
                    victim = rng.choice(sorted(busy))
                    action = plan.pop()
                    if action == "kill":
                        fleet.procs[victim].kill()
                        fleet.procs[victim].wait()
                        fleet.spawn()
                        kills_done += 1
                    else:
                        _suspend(fleet.procs[victim].pid)
                        frozen.append((victim, now + args.lease_seconds + args.freeze_extra_s))
                        frozen_done += 1
                    events.append({"t": round(now - started, 2), "action": action, "worker": victim, "task": held[victim]})
                else:
                    kill_skipped += 1
                next_event = now + rng.uniform(0.5, 1.5) * args.event_interval_s
            if unfinished() == 0 and not frozen:
                break
            time.sleep(0.1)
        elapsed = time.monotonic() - started
        for worker_id, _ in frozen:
            _resume(fleet.procs[worker_id].pid)
        fleet.stop_all()
        return collect(factory, case_ids, fleet, events, elapsed, kills_done, frozen_done, kill_skipped, args, rng)
    finally:
        for proc in fleet.procs.values():
            if proc.poll() is None:
                proc.kill()
        engine.dispose()
        shutil.rmtree(workdir, ignore_errors=True)


def collect(factory: Any, case_ids: list[str], fleet: Fleet, events: list[dict[str, Any]], elapsed: float,
            kills_done: int, frozen_done: int, kill_skipped: int, args: argparse.Namespace,
            rng: random.Random) -> dict[str, Any]:
    del rng
    with factory() as session:
        cases = session.execute(select(Case.case_id, Case.status, Case.created_at, Case.updated_at)
                                .where(Case.case_id.in_(case_ids))).all()
        status = Counter(row.status for row in cases)
        reports = Counter(session.scalars(select(CaseReport.case_id).where(CaseReport.case_id.in_(case_ids))))
        artifact_rows = session.execute(select(StageArtifact.case_id, StageArtifact.stage)
                                        .where(StageArtifact.case_id.in_(case_ids))).all()
        artifacts = Counter((row.case_id, row.stage) for row in artifact_rows)
        stages_per_case: dict[str, set[str]] = {}
        for case_id, stage in artifacts:
            stages_per_case.setdefault(case_id, set()).add(stage)
        tasks = session.execute(select(WorkflowTask.status, WorkflowTask.attempt).where(
            WorkflowTask.case_id.in_(case_ids))).all()
        reclaim_rows = session.execute(select(CaseEventLog.case_id, CaseEventLog.created_at, CaseEventLog.detail)
                                       .where(CaseEventLog.case_id.in_(case_ids),
                                              CaseEventLog.event_type == "lease_reclaimed")
                                       .order_by(CaseEventLog.created_at, CaseEventLog.id)).all()
    reclaimed = len(reclaim_rows)
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in reclaim_rows:
        detail = row.detail or {}
        chains.setdefault(row.case_id, []).append({
            "t": round((row.created_at - fleet.started_wall).total_seconds(), 2),
            "old_owner": detail.get("old_owner"), "new_owner": detail.get("new_owner"),
            "old_attempt": detail.get("old_attempt"),
        })
    reclaim_chains = {cid: chain for cid, chain in chains.items() if len(chain) > 1}

    succeeded = [row for row in cases if row.status == "CLOSED_SUCCESS"]
    latencies = [(row.updated_at - row.created_at).total_seconds() for row in succeeded]
    full_stages = max((len(s) for s in stages_per_case.values()), default=0)
    invocations = Counter(
        line.strip()
        for path in fleet.workdir.glob("invocations.log.*")
        for line in path.read_text().splitlines()
        if line.strip()
    )
    provider_calls = sum(invocations.values())
    committed_provider_artifacts = sum(
        count for (_, stage), count in artifacts.items() if stage in {"normalize", "retrieval", "plan", "generation",
                                                                      "arbitration", "review", "report"})
    lease_errors = Counter(
        line.split(maxsplit=1)[1]
        for path in fleet.workdir.glob("lease_errors.log.*")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    attempts = Counter(attempt for _, attempt in tasks)

    violations = {
        "cases_not_terminal_success": len(case_ids) - len(succeeded),
        "cases_without_exactly_one_report": sum(1 for cid in case_ids if reports.get(cid, 0) != 1),
        "duplicate_stage_artifacts": sum(1 for count in artifacts.values() if count > 1),
        "cases_missing_stage_artifacts": sum(1 for cid in case_ids if len(stages_per_case.get(cid, ())) != full_stages),
        "tasks_left_running_or_pending": sum(1 for status_, _ in tasks if status_ in ("RUNNING", "PENDING")),
    }
    return {
        "config": {
            "backend": args.backend, "workers": args.workers, "cases": args.cases, "kills_planned": args.kills,
            "freezes_planned": args.freezes, "stage_delay_ms": args.stage_delay_ms,
            "lease_seconds": args.lease_seconds, "seed": args.seed,
            "provider": "DelayedProvider(无网络、无付费调用)",
        },
        "injected": {"kills": kills_done, "freezes": frozen_done, "ticks_without_busy_worker": kill_skipped,
                     "events": events},
        "outcome": {
            "cases": len(case_ids), "status": dict(status),
            "terminal_success_rate": round(len(succeeded) / len(case_ids), 4),
            "lease_reclaims": reclaimed, "task_attempt_histogram": {str(k): v for k, v in sorted(attempts.items())},
            "stale_writers_rejected": dict(lease_errors),
            "multi_reclaim_chains": reclaim_chains,
        },
        "side_effects": {
            "violations": violations,
            "provider_calls": provider_calls,
            "committed_provider_artifacts": committed_provider_artifacts,
            "reexecuted_provider_calls": provider_calls - committed_provider_artifacts,
        },
        "performance": {
            "wall_seconds": round(elapsed, 2),
            "throughput_cases_per_s": round(len(succeeded) / elapsed, 3) if elapsed else None,
            "latency_s": {
                "p50": round(statistics.median(latencies), 2) if latencies else None,
                "p95": round(percentile(latencies, 0.95), 2) if latencies else None,
                "max": round(max(latencies), 2) if latencies else None,
            },
            "note": "延迟 = 病例创建到最终写入；所有病例在 t0 一次性提交，因此包含排队时间。",
        },
        "passed": all(v == 0 for v in violations.values()),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--database-url", help="显式数据库 URL（覆盖 --backend 的默认行为）；不可指向已有数据")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--kills", type=int, default=8)
    parser.add_argument("--freezes", type=int, default=2)
    parser.add_argument("--stage-delay-ms", type=int, default=150)
    parser.add_argument("--lease-seconds", type=int, default=3)
    parser.add_argument("--freeze-extra-s", type=float, default=1.5)
    parser.add_argument("--event-interval-s", type=float, default=1.5)
    parser.add_argument("--first-event-after-s", type=float, default=1.5)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", help="报告文件名（写入 artifacts/reports/reliability/，不覆盖已有文件）")
    parser.add_argument("--child")
    parser.add_argument("--invocation-log")
    parser.add_argument("--stop-file")
    parser.add_argument("--lease-log")
    args = parser.parse_args()

    if args.child:
        child_main(args.child, args.stage_delay_ms / 1000, args.invocation_log, args.stop_file, args.lease_log)
        return 0

    with contextlib.ExitStack() as stack:
        temp_db: Path | None = None
        if args.database_url:
            url = args.database_url
        elif args.backend == "postgres":
            url = stack.enter_context(throwaway_postgres())
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="medidiag-stress-db-"))
            stack.callback(shutil.rmtree, temp_dir, True)
            temp_db = temp_dir / "stress.db"
            url = f"sqlite:///{temp_db.as_posix()}"
        report = run_scenario(url, args)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with (REPORT_DIR / args.out).open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
