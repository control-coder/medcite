"""MediDiag command-line entry points."""

from __future__ import annotations

import time

import click

from medidiag.config import get_settings
from medidiag.db.session import create_db_engine, get_session_factory
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker


@click.group()
def main() -> None:
    """MediDiag-Agent EvidenceFlow CLI."""


@main.command()
def version() -> None:
    """显示版本号。"""
    from medidiag import __version__

    click.echo(__version__)


@main.command()
def errors() -> None:
    """列出所有已注册的错误码。"""
    from medidiag.errors import all_error_codes

    for code, spec in all_error_codes().items():
        click.echo(
            f"{code:35s} [{spec.category.value}] "
            f"retryable={spec.retryable!s:5s} "
            f"escalate={spec.requires_human_escalation!s:5s} "
            f"alert={spec.alert!s:5s} "
            f"http={spec.http_status}"
        )


def _mode(once: bool, loop: bool) -> None:
    if once == loop:
        raise click.UsageError("choose exactly one of --once or --loop")


def _session_factory():
    engine = create_db_engine(get_settings().database_url)
    return engine, get_session_factory(engine)


@main.command()
@click.option("--once", is_flag=True, help="Process at most one task.")
@click.option("--loop", is_flag=True, help="Continuously poll for tasks.")
@click.option("--worker-id", default="local-worker", show_default=True)
def worker(once: bool, loop: bool, worker_id: str) -> None:
    """Run the single-machine deterministic development worker."""
    _mode(once, loop)
    engine, factory = _session_factory()
    runner = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id=worker_id
    )
    try:
        while True:
            result = runner.run_once()
            click.echo(
                f"processed={result.processed} task_id={result.task_id} "
                f"state={result.final_state}"
            )
            if once:
                return
            time.sleep(1.0 if not result.processed else 0.05)
    finally:
        engine.dispose()


@main.command("lease-scan")
@click.option("--once", is_flag=True, help="Scan once.")
@click.option("--loop", is_flag=True, help="Continuously scan expired tasks.")
@click.option("--worker-id", default="local-worker", show_default=True)
def lease_scan(once: bool, loop: bool, worker_id: str) -> None:
    """Reclaim expired attempts for the local recovery worker."""
    _mode(once, loop)
    engine, factory = _session_factory()
    scanner = LeaseScanner(factory, recovery_worker_id=worker_id)
    try:
        while True:
            reclaimed = scanner.scan_once()
            click.echo(f"reclaimed={len(reclaimed)} task_ids={reclaimed}")
            if once:
                return
            time.sleep(get_settings().medidiag_lease_scan_seconds)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
