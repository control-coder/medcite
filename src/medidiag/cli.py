"""MediDiag command-line entry points."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import click
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from medidiag.agents.runtime import AgentTopologyConfig, RuntimeMedicalAgents
from medidiag.config import get_settings, load_eval_config
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.observability.logging import configure_logging
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.review.runtime import RuntimeMedicalReview
from medidiag.workflow.openai_provider import OpenAICompatibleWorkflowProvider
from medidiag.workflow.provider import DeterministicWorkflowProvider, WorkflowProvider
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker

_AGENT_TOPOLOGY_CHOICES = click.Choice(["single", "fixed_pair", "dynamic_pair"])

_PROVIDER_CHOICES = click.Choice(
    ["fake_offline", "deepseek_default", "mimo_v25", "openai_compatible_custom"],
    case_sensitive=False,
)


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


def _session_factory() -> tuple[Engine, sessionmaker[Session]]:
    engine = create_db_engine(get_settings().database_url)
    return engine, get_session_factory(engine)


def _build_provider(
    provider_name: str,
    review_verdict: str,
    *,
    rag_config: str = "eval/config.yaml",
    agent_topology: str = "dynamic_pair",
    specialist_pair: str = "cardiology,respiratory",
) -> WorkflowProvider:
    if provider_name == "fake_offline":
        return DeterministicWorkflowProvider(review_verdict=review_verdict)
    provider = OpenAICompatibleWorkflowProvider(profile_id=provider_name)
    if not provider.is_configured:
        raise click.UsageError(
            f"LLM profile {provider_name} 缺少 base URL、model 或 api_key_env 对应密钥；"
            "可使用 --provider fake_offline 运行无网络 fixture。"
        )
    pair = tuple(item.strip() for item in specialist_pair.split(",") if item.strip())
    if len(pair) != 2:
        raise click.UsageError("--specialist-pair 必须是两个逗号分隔的专科标识")
    config = load_eval_config(rag_config)
    rag_stage = RuntimeMedicalRAG.from_config(config, root=Path.cwd())
    provider.rag_stage = rag_stage
    provider.agent_stage = RuntimeMedicalAgents(
        provider.llm,
        config=AgentTopologyConfig(
            topology=agent_topology,  # type: ignore[arg-type]
            fixed_pair=(pair[0], pair[1]),
            timeout_s=float(config["generation"]["timeout_seconds"]),
        ),
        normalizer=rag_stage.normalizer,
        evidence_level_scores=dict(config["retrieval"]["evidence_levels"]),
    )
    provider.review_stage = RuntimeMedicalReview.from_config(config)
    return provider


@main.command()
@click.option("--once", is_flag=True, help="Process at most one task.")
@click.option("--loop", is_flag=True, help="Continuously poll for tasks.")
@click.option("--worker-id", default="local-worker", show_default=True)
@click.option(
    "--provider",
    "provider_name",
    type=_PROVIDER_CHOICES,
    default="deepseek_default",
    show_default=True,
    help="选择 LLM profile；fake_offline 为无网络 fixture。",
)
@click.option(
    "--rag-config",
    default="eval/config.yaml",
    show_default=True,
    type=click.Path(exists=False, dir_okay=False),
    help="版本化医学 corpus、模型 revision、检索权重和 leakage gate 配置。",
)
@click.option(
    "--agent-topology",
    type=_AGENT_TOPOLOGY_CHOICES,
    default="dynamic_pair",
    show_default=True,
    help="live provider 的 Agent topology；fake_offline 不使用该选项。",
)
@click.option(
    "--specialist-pair",
    default="cardiology,respiratory",
    show_default=True,
    help="fixed_pair 使用的两个逗号分隔专科标识。",
)
@click.option(
    "--review-verdict",
    type=click.Choice(["APPROVED", "REVISION_REQUIRED", "ESCALATED"]),
    default="APPROVED",
    show_default=True,
    help="仅供 fake_offline 开发 fixture 使用。",
)
def worker(
    once: bool,
    loop: bool,
    worker_id: str,
    provider_name: str,
    rag_config: str,
    agent_topology: str,
    specialist_pair: str,
    review_verdict: str,
) -> None:
    """Run one local worker with a live or deterministic provider."""
    _mode(once, loop)
    engine, factory = _session_factory()
    runner = SingleMachineWorker(
        factory,
        _build_provider(
            provider_name,
            review_verdict,
            rag_config=rag_config,
            agent_topology=agent_topology,
            specialist_pair=specialist_pair,
        ),
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
                f"processed={result.processed} task_id={result.task_id} state={result.final_state}"
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
    default="deepseek_default",
    show_default=True,
)
@click.option(
    "--rag-config",
    default="eval/config.yaml",
    show_default=True,
    type=click.Path(exists=False, dir_okay=False),
    help="版本化医学 corpus、模型 revision、检索权重和 leakage gate 配置。",
)
@click.option(
    "--agent-topology",
    type=_AGENT_TOPOLOGY_CHOICES,
    default="dynamic_pair",
    show_default=True,
    help="live provider 的 Agent topology。",
)
@click.option(
    "--specialist-pair",
    default="cardiology,respiratory",
    show_default=True,
    help="fixed_pair 使用的两个逗号分隔专科标识。",
)
def demo(
    host: str,
    port: int,
    provider_name: str,
    rag_config: str,
    agent_topology: str,
    specialist_pair: str,
) -> None:
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
    provider = _build_provider(
        provider_name,
        "APPROVED",
        rag_config=rag_config,
        agent_topology=agent_topology,
        specialist_pair=specialist_pair,
    )
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
    if provider_name == "fake_offline":
        runtime_label = "确定性本地 fixture"
        runtime_detail = (
            f"阶段执行使用 {provider.version}；检索与生成均为确定性测试 fixture，"
            "不构成医学 RAG 或正式评测。"
        )
    else:
        assert isinstance(provider, OpenAICompatibleWorkflowProvider)
        assert provider.rag_stage is not None
        runtime_label = f"{provider_name} {agent_topology} Agent + 版本化医学 RAG"
        runtime_detail = (
            f"生成阶段使用 {provider.version} 与 {agent_topology} topology；检索使用 corpus "
            f"{provider.rag_stage.corpus_version} 并保存 EvidenceBundle。"
            "审核使用固定 revision 的英文 NLI judge；该判定不代表临床有效性或正式评测结论。"
        )
    app.state.demo_runtime = {
        "label": runtime_label,
        "detail": runtime_detail,
    }
    click.echo(f"MediDiag 演示已启动：provider={provider.version}；访问 http://{host}:{port}/demo")
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
            f"scenario={item.scenario} state={item.final_state} trace_id={item.export.trace_id}"
        )


if __name__ == "__main__":
    main()
