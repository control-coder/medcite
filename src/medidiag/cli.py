"""MediDiag 命令行入口。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import click
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from medidiag.config import get_settings
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.llm import LLMRequest, build_llm_provider
from medidiag.observability.logging import configure_logging
from medidiag.workflow.application import build_application_provider
from medidiag.workflow.provider import WorkflowProvider
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker

# 应用可用的运行方式：确定性 fixture、真实检索加模拟生成、真实检索加模型生成。
_PROVIDER_CHOICES = click.Choice(["fake_offline", "retrieval_mock", "mimo_grounded"], case_sensitive=False)

# 连接探测可用的模型配置档案，见 llm/profiles.yaml。
_PROFILE_CHOICES = click.Choice(
    ["fake_offline", "mimo_v25", "openai_compatible_custom"], case_sensitive=False
)


@click.group()
def main() -> None:
    """MediDiag-Agent EvidenceFlow 命令行接口。"""
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
    app_config: str = "configs/application.yaml",
    live_budget: str | None = None,
) -> WorkflowProvider:
    try:
        return build_application_provider(provider_name, app_config=app_config,
                                          review_verdict=review_verdict, live_budget=live_budget)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@main.command()
@click.option("--once", is_flag=True, help="Process at most one task.")
@click.option("--loop", is_flag=True, help="Continuously poll for tasks.")
@click.option("--worker-id", default="local-worker", show_default=True)
@click.option(
    "--provider",
    "provider_name",
    type=_PROVIDER_CHOICES,
    default="retrieval_mock",
    show_default=True,
    help="显式选择模式；fake_offline 为 fixture，retrieval_mock 为真实检索/模拟生成，mimo_grounded 调用模型。",
)
@click.option("--live-budget", type=click.Path(dir_okay=False), default=None,
              help="mimo_grounded 必填：已授权的持久调用账本；整轮最多8次，45秒，零自动重试。")
@click.option("--app-config", default="configs/application.yaml", show_default=True,
              help="retrieval_mock 专用安全应用配置；不读取研究模型配置。")
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
    app_config: str,
    live_budget: str | None,
    review_verdict: str,
) -> None:
    """运行一个使用在线或确定性 Provider 的本地 worker。"""
    _mode(once, loop)
    engine, factory = _session_factory()
    runner = SingleMachineWorker(
        factory,
        _build_provider(
            provider_name,
            review_verdict,
            app_config=app_config, live_budget=live_budget,
        ),
        worker_id=worker_id,
        max_review_rounds=get_settings().medidiag_max_review_rounds,
    )
    try:
        while True:
            try:
                result = runner.run_once()
            except Exception as exc:  # 单个异常任务不能终止 worker
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
    default="retrieval_mock",
    show_default=True,
)
@click.option("--live-budget", type=click.Path(dir_okay=False), default=None,
              help="mimo_grounded 必填：已授权的持久调用账本；整轮最多8次，45秒，零自动重试。")
@click.option("--app-config", default="configs/application.yaml", show_default=True,
              help="retrieval_mock 专用安全应用配置；不读取研究模型配置。")
def demo(
    host: str,
    port: int,
    provider_name: str,
    app_config: str,
    live_budget: str | None,
) -> None:
    """在同一进程中运行服务端渲染的演示和本地 worker。

    打开 ``/demo``，只提交公开或已脱敏文本；页面会轮询任务，直到 worker 持久化最终报告。
    这是本地工程演示，不提供生产级 worker 服务或真实患者工作流。
    """
    import uvicorn

    from medidiag.api.app import create_app

    engine = create_db_engine(get_settings().database_url)
    init_db(engine)
    factory = get_session_factory(engine)
    provider = _build_provider(
        provider_name,
        "APPROVED",
        app_config=app_config, live_budget=live_budget,
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
        runtime_label = ("真实检索 / 模拟生成" if provider_name == "retrieval_mock"
                         else "真实检索 / MiMo 单路受约束摘录")
        runtime_detail = f"{provider.version}；仅完整短引与引用关联检查，未运行 NLI，不证明医学正确。"
    app.state.demo_runtime = {
        "label": runtime_label,
        "detail": runtime_detail,
    }
    click.echo(
        f"MediDiag 演示已启动：provider={provider.version}；"
        f"用户页 http://{host}:{port}/assistant；工程页 http://{host}:{port}/demo"
    )
    worker_thread.start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        stop.set()
        worker_thread.join(timeout=2)
        engine.dispose()


def _demo_worker_loop(stop: threading.Event, runner: SingleMachineWorker) -> None:
    """单进程演示命令使用的有界轮询循环。"""
    while not stop.is_set():
        try:
            result = runner.run_once()
        except Exception as exc:  # 演示必须保持可用，便于操作人员检查事件
            click.echo(f"demo-worker error: {exc}", err=True)
            stop.wait(1.0)
            continue
        stop.wait(0.15 if result.processed else 0.5)


@main.command("provider-smoke")
@click.option(
    "--provider",
    "provider_name",
    type=_PROFILE_CHOICES,
    required=True,
    help="显式选择要探测的 profile；命令不会由 demo 或测试默认触发。",
)
@click.option("--timeout", "timeout_s", default=20.0, show_default=True, type=float)
def provider_smoke(provider_name: str, timeout_s: float) -> None:
    """发送一次不含医疗数据的最小 Provider 探测请求。"""
    provider = build_llm_provider(provider_name, max_retries=0)
    if not getattr(provider, "is_configured", True):
        raise click.UsageError(
            f"LLM profile {provider_name} 缺少 base URL、model 或 api_key_env 对应密钥。"
        )
    result = provider.generate(
        LLMRequest(
            messages=[
                {
                    "role": "system",
                    "content": "这是连接探测。不要提供医疗信息，只返回 JSON status=ok。",
                },
                {"role": "user", "content": "connection smoke test"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=32,
            prompt_version="provider-smoke-v1",
        ),
        timeout_s=timeout_s,
        idempotency_key=f"provider-smoke-{uuid.uuid4().hex}",
    )
    safe_summary = {
        "profile": result.profile_id,
        "model": result.model,
        "response_id_present": bool(result.response_id),
        "system_fingerprint_present": bool(result.system_fingerprint),
        "latency_ms": result.latency_ms,
        "usage": result.usage,
        "provenance_mode": result.provenance_mode,
        "provider_snapshot_verifiable": bool(result.system_fingerprint),
    }
    click.echo(json.dumps(safe_summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        "探测成功仅说明最小 Provider 调用可用，不代表医疗工作流、临床有效性或 formal run 成功。"
    )


@main.command("lease-scan")
@click.option("--once", is_flag=True, help="Scan once.")
@click.option("--loop", is_flag=True, help="Continuously scan expired tasks.")
@click.option("--worker-id", default="local-worker", show_default=True)
def lease_scan(once: bool, loop: bool, worker_id: str) -> None:
    """回收本地恢复 worker 中已过期的任务尝试。"""
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
    default=Path("artifacts/traces"),
    show_default=True,
)
def trace_export(case_id: str, output_root: Path) -> None:
    """将一个已持久化病例导出为原始 JSONL 和脱敏 JSON 摘要。"""
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
    default=Path("artifacts/traces"),
    show_default=True,
)
def trace_examples(output_root: Path) -> None:
    """生成包含 P7 可靠性故障的确定性 Trace 示例。"""
    from medidiag.observability.scenarios import generate_trace_examples

    for item in generate_trace_examples(output_root):
        click.echo(
            f"scenario={item.scenario} state={item.final_state} trace_id={item.export.trace_id}"
        )


@main.group()
def corpus() -> None:
    """语料处理子命令。"""


@corpus.command("ingest")
@click.option(
    "--sources",
    "sources_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="来源清单 JSON（examples/public_health_v4/sources.json）。",
)
@click.option(
    "--cache",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(".cache/corpus"),
    show_default=True,
    help="语料缓存目录；快照读 raw/，解析结果写 parsed/。",
)
def corpus_ingest(sources_path: Path, cache_dir: Path) -> None:
    """按来源清单解析、清洗与去重，输出统一文档模型 JSONL 与解析报告。"""
    from medidiag.rag.ingest.pipeline import run_ingest

    report = run_ingest(sources_path, cache_dir)
    ok = sum(1 for f in report.files if f.status == "ok")
    failed = [f for f in report.files if f.status == "failed"]
    skipped = [f for f in report.files if f.status == "skipped"]
    click.echo(
        f"解析完成：成功 {ok}，失败 {len(failed)}，跳过 {len(skipped)}；"
        f"文档级去重剔除 {len(report.dropped_doc_ids)} 篇；"
        f"报告见 {cache_dir / 'parsed' / 'report.json'}"
    )
    for item in failed:
        click.echo(f"  失败 {item.doc_id}（{item.format}）：{item.failure_reason}", err=True)
    for item in skipped:
        click.echo(f"  跳过 {item.doc_id}（{item.format}）：{item.failure_reason}")


if __name__ == "__main__":
    main()
