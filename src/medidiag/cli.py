"""MediDiag command-line entry points."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import click

from medidiag.config import get_settings
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.observability.logging import configure_logging
from medidiag.workflow.deepseek_provider import DeepSeekWorkflowProvider
from medidiag.workflow.provider import DeterministicWorkflowProvider, WorkflowProvider
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker

_PROVIDER_CHOICES = click.Choice(["deepseek", "deterministic"], case_sensitive=False)


@click.group()
def main() -> None:
    """MediDiag-Agent EvidenceFlow CLI."""
    # 尽早配置，使子命令的第一条日志也经过脱敏处理器。
    configure_logging()


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


def _build_provider(provider_name: str, review_verdict: str) -> WorkflowProvider:
    if provider_name == "deterministic":
        return DeterministicWorkflowProvider(review_verdict=review_verdict)
    provider = DeepSeekWorkflowProvider()
    if not provider.client.is_configured:
        raise click.UsageError(
            "DEEPSEEK_API_KEY is required for --provider deepseek; "
            "use --provider deterministic for the network-free fixture."
        )
    return provider


@main.command()
@click.option("--once", is_flag=True, help="Process at most one task.")
@click.option("--loop", is_flag=True, help="Continuously poll for tasks.")
@click.option("--worker-id", default="local-worker", show_default=True)
@click.option(
    "--provider",
    "provider_name",
    type=_PROVIDER_CHOICES,
    default="deepseek",
    show_default=True,
    help="Live DeepSeek drafts one constrained generation stage; deterministic is fixture-only.",
)
@click.option(
    "--review-verdict",
    type=click.Choice(["APPROVED", "REVISION_REQUIRED", "ESCALATED"]),
    default="APPROVED",
    show_default=True,
    help="Only used by the deterministic development provider.",
)
def worker(
    once: bool, loop: bool, worker_id: str, provider_name: str, review_verdict: str
) -> None:
    """Run one local worker with a live or deterministic provider."""
    _mode(once, loop)
    engine, factory = _session_factory()
    runner = SingleMachineWorker(
        factory,
        _build_provider(provider_name, review_verdict),
        worker_id=worker_id,
        max_review_rounds=get_settings().medidiag_max_review_rounds,
    )
    try:
        while True:
            try:
                result = runner.run_once()
            except Exception as exc:  # a single bad task must not kill the worker
                click.echo(f"worker error: {exc}", err=True)
                if once:
                    raise click.exceptions.Exit(1) from exc
                time.sleep(1.0)
                continue
            click.echo(
                f"processed={result.processed} task_id={result.task_id} "
                f"state={result.final_state}"
            )
            if once:
                return
            time.sleep(1.0 if not result.processed else 0.05)
    finally:
        engine.dispose()


@main.command("demo")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8400, type=click.IntRange(1, 65535), show_default=True)
@click.option(
    "--provider",
    "provider_name",
    type=_PROVIDER_CHOICES,
    default="deepseek",
    show_default=True,
)
def demo(host: str, port: int, provider_name: str) -> None:
    """Run the server-rendered demo and its local worker in one process.

    Open ``/demo``, submit only public/deidentified text, and the page will poll the
    task until the worker persists its final report. This is a local engineering demo;
    it does not expose a production worker service or real patient workflow.
    """
    import uvicorn

    from medidiag.api.app import create_app

    engine = create_db_engine(get_settings().database_url)
    init_db(engine)
    factory = get_session_factory(engine)
    provider = _build_provider(provider_name, "APPROVED")
    runner = SingleMachineWorker(
        factory,
        provider,
        worker_id="demo-worker",
        max_review_rounds=get_settings().medidiag_max_review_rounds,
    )
    stop = threading.Event()
    worker_thread = threading.Thread(
        target=_demo_worker_loop,
        args=(stop, runner),
        name="medidiag-demo-worker",
        daemon=True,
    )
    app = create_app(session_factory=factory)
    app.state.demo_runtime = {
        "label": (
            "DeepSeek 实时起草"
            if provider_name == "deepseek"
            else "确定性本地 fixture"
        ),
        "detail": (
            f"生成阶段使用 {provider.version}；检索仍为本地演示 fixture，"
            "不构成医学 RAG 或正式评测。"
        ),
    }
    click.echo(
        f"MediDiag demo provider={provider.version}; open http://{host}:{port}/demo"
    )
    worker_thread.start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        stop.set()
        worker_thread.join(timeout=2)
        engine.dispose()


def _demo_worker_loop(stop: threading.Event, runner: SingleMachineWorker) -> None:
    """Bounded polling loop for the single-process presentation command."""
    while not stop.is_set():
        try:
            result = runner.run_once()
        except Exception as exc:  # demo must stay available for an operator to inspect events
            click.echo(f"demo-worker error: {exc}", err=True)
            stop.wait(1.0)
            continue
        stop.wait(0.15 if result.processed else 0.5)


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


@main.command("trace-export")
@click.option("--case-id", required=True, help="Case ID to export.")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("traces"),
    show_default=True,
)
def trace_export(case_id: str, output_root: Path) -> None:
    """Export one persisted case as raw JSONL and a redacted JSON summary."""
    from medidiag.observability.trace_exporter import TraceExporter

    engine, factory = _session_factory()
    try:
        with factory() as session:
            result = TraceExporter().export_case(
                session,
                case_id,
                raw_dir=output_root / "raw",
                summary_dir=output_root / "summary",
            )
        click.echo(
            f"trace_id={result.trace_id} events={result.event_count} "
            f"raw={result.raw_path} summary={result.summary_path}"
        )
    finally:
        engine.dispose()


@main.command("trace-examples")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=Path("traces"),
    show_default=True,
)
def trace_examples(output_root: Path) -> None:
    """Generate three deterministic P1-A trace scenarios."""
    from medidiag.observability.scenarios import generate_trace_examples

    for item in generate_trace_examples(output_root):
        click.echo(
            f"scenario={item.scenario} state={item.final_state} "
            f"trace_id={item.export.trace_id}"
        )


if __name__ == "__main__":
    main()
