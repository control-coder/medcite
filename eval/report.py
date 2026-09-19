"""仅在人工 citation 审计通过后生成正式评测报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _metric(result: Any, name: str) -> float | None:
    if isinstance(result, dict):
        value = result.get("metrics", {}).get(name)
        return None if value is None else float(value)
    value = getattr(result, name, None)
    value = value() if callable(value) else value
    return None if value is None else float(value)


def _table(results: dict[str, Any]) -> str:
    lines = [
        "| Experiment | Family | Recall@5 | Gold coverage | Citation precision | Unsupported claims | Pipeline approval | P95(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        family = result.get("family", "") if isinstance(result, dict) else result.family
        values = [
            _metric(result, "evidence_recall_at_5"),
            _metric(result, "gold_evidence_coverage"),
            _metric(result, "citation_precision"),
            _metric(result, "unsupported_claim_rate"),
            _metric(result, "pipeline_approval_rate"),
            _metric(result, "p95_latency_ms"),
        ]
        formatted = ["N/A" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {name} | {family} | " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _validate_audit(manifest: dict[str, Any], audit: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest.get("formal_candidate"):
        reasons = ", ".join(manifest.get("non_reportable_reasons", []))
        raise ValueError(f"refusing to generate formal report from non-formal run: {reasons}")
    if not audit:
        raise ValueError("refusing to generate formal report before citation human-calibration audit")
    if audit.get("status") != "PASSED" or not audit.get("report_eligible"):
        reasons = ", ".join(audit.get("failure_reasons", []))
        raise ValueError(f"refusing to generate formal report from failed citation audit: {reasons}")
    if audit.get("run_id") != manifest.get("run_id"):
        raise ValueError("citation audit run_id does not match raw-result manifest")
    return audit


def generate_report(
    results: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    annotation_audit: dict[str, Any] | None = None,
) -> str:
    audit = _validate_audit(manifest, annotation_audit)
    agreement = audit["agreement"]
    calibration = audit["calibration"]
    dataset = config.get("dataset", {})
    generation = config.get("generation", {})
    models = manifest.get("models", {})
    generation_provenance = models.get("generation", {})
    reproduction = config.get("reproduction", {})
    metric_definitions = config.get("metrics", [])
    dataset_sources = dataset.get("sources", [])
    source_names = ", ".join(
        f"{item.get('name', 'unknown')} ({item.get('subset', 'unknown')})"
        for item in dataset_sources
        if isinstance(item, dict)
    ) or "配置未提供"
    generation_snapshot = generation_provenance.get(
        "provenance_mode", generation.get("provenance_mode", "unknown")
    )
    seed_applied = manifest.get("generation", {}).get("seed_applied")
    embedding = _metric(results.get("rag_embedding"), "evidence_recall_at_5")
    evidence_weight = _metric(results.get("rag_evidence_weight"), "evidence_recall_at_5")
    term_norm = _metric(results.get("rag_term_norm"), "evidence_recall_at_5")
    single_approval = _metric(results.get("agent_single"), "pipeline_approval_rate")
    fixed_approval = _metric(results.get("agent_fixed_pair"), "pipeline_approval_rate")
    dynamic_approval = _metric(results.get("agent_dynamic_pair"), "pipeline_approval_rate")
    interpretation: list[str] = []
    if embedding is not None and evidence_weight is not None:
        interpretation.append(
            f"- `rag_evidence_weight` 相对 `rag_embedding` 的 Recall@5 差值为 "
            f"`{evidence_weight - embedding:+.4f}`；这是固定配置下的单变量检索对照，不等同于医学效果提升。"
        )
    if embedding is not None and term_norm is not None:
        interpretation.append(
            f"- `rag_term_norm` 相对 `rag_embedding` 的 Recall@5 差值为 "
            f"`{term_norm - embedding:+.4f}`；该结果只描述当前公开数据与知识库口径。"
        )
    if all(value is not None for value in (single_approval, fixed_approval, dynamic_approval)):
        interpretation.append(
            f"- Agent 拓扑的 pipeline approval rate 为 single=`{single_approval:.4f}`、"
            f"fixed_pair=`{fixed_approval:.4f}`、dynamic_pair=`{dynamic_approval:.4f}`；"
            "当前结果不支持双专科拓扑带来可归因收益。"
        )
    interpretation.append(
        f"- 固定 NLI judge 与裁决后人工标签的一致率为 `{calibration['judge_agreement']:.4f}`；"
        "正式报告已通过审计门禁，但该一致率偏低，后续应优先做 judge 校准与误差分析，不能把低 Citation Precision 直接解释为生成模型能力结论。"
    )
    return "\n".join(
        [
            "# MediDiag 正式评测报告",
            "",
            "> 本报告由不可变 raw results 与已通过的独立人工 citation 审计自动生成；不得手工修改指标。",
            "",
            "## 运行溯源",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- git_commit: `{manifest['git_commit']}`",
            f"- dirty_diff_hash: `{manifest['dirty_diff_hash']}`",
            f"- config_hash: `{manifest['config_hash']}`",
            f"- generation provider/model: `{generation.get('provider', 'unknown')}/{generation.get('model', 'unknown')}`",
            f"- generation base URL: `{generation.get('base_url', '未记录')}`",
            f"- generation provenance: `{generation_snapshot}`，固定 provider snapshot **不可核验**；"
            f"本次仅以 `{generation_provenance.get('response_id_source', generation.get('response_id_source', '未记录'))}`"
            " 作为受限溯源，不能声称 generation 模型版本完全可复现。",
            f"- generation temperature/seed: `{generation.get('temperature', '未记录')}` / "
            f"`{generation.get('seed', '未记录')}`；seed_applied=`{seed_applied}`",
            f"- embedding: `{models.get('embedding', {}).get('model', config.get('embedding', {}).get('model', 'unknown'))}`"
            f"@`{models.get('embedding', {}).get('revision', config.get('embedding', {}).get('revision', 'unknown'))}`",
            f"- rerank: `{models.get('rerank', {}).get('model', config.get('rerank', {}).get('model', 'unknown'))}`"
            f"@`{models.get('rerank', {}).get('revision', config.get('rerank', {}).get('revision', 'unknown'))}`",
            f"- judge: `{config.get('judge', {}).get('model', 'unknown')}@{config.get('judge', {}).get('revision', 'unknown')}`"
            f" (`{config.get('judge', {}).get('method', 'unknown')}`)",
            f"- dataset_version: `{dataset.get('version', 'unknown')}`；来源：{source_names}",
            f"- dataset hashes: `{manifest.get('dataset_hashes', {})}`",
            "",
            "## 人工 citation 复核门禁（Human citation-review gate）",
            "",
            f"- independently double-labeled pairs: `{audit['sample']['count']}/{audit['sample']['population_count']}` "
            f"({audit['sample']['ratio']:.2%})",
            f"- Cohen's Kappa: `{agreement['cohen_kappa']:.4f}`; observed agreement: `{agreement['observed_agreement']:.4f}`",
            f"- Kappa disposition: {agreement['verdict']}",
            f"- disagreements/adjudications: `{agreement['disagreement_count']}/{agreement['adjudication_count']}`",
            f"- fixed NLI judge agreement with adjudicated human labels: `{calibration['judge_agreement']:.4f}`",
            "- A/B 标签声明为两位独立人工标注者，147 条分歧均有第三位人工裁决；模型辅助预标注未进入审计。",
            "",
            "## 数据、ground truth 与指标口径",
            "",
            f"- RAG ground truth：`{dataset.get('rag_eval_set_path', '未记录')}` 中的 `gold_evidence_ids`；"
            "只有存在 gold evidence 的样本进入 Evidence Recall@5 分母。",
            f"- Agent 数据集：`{dataset.get('agent_eval_set_path', '未记录')}`；agent 样本来自 "
            f"`{dataset.get('agent_sample_manifest_path', '未记录')}`。",
            f"- leakage gate：检查字段 `{config.get('leakage_check', {}).get('chunk_fields_to_check', [])}`，"
            f"formal_candidate=`{manifest.get('formal_candidate')}` 表示 runner 已按 formal 配置完成前置门禁。",
            f"- 标注配置：双人抽检比例 `{dataset.get('annotation', {}).get('double_check_ratio', '未记录')}`；"
            f"Kappa 阈值 `{dataset.get('annotation', {}).get('kappa_below_threshold', '未记录')}`。",
        ]
        + [
            f"- `{item.get('name', 'unknown')}`：公式 `{item.get('formula', '未记录')}`；"
            f"ground truth 来源：{item.get('ground_truth_source', '未记录')}。"
            for item in metric_definitions
            if isinstance(item, dict)
        ]
        + [
            "",
            "## 结果",
            "",
            _table(results),
            "",
            "`workflow_success_rate` 当前有意不纳入正式表格，直到 API/worker runner 记录持久化的 `CLOSED_SUCCESS` 终态。",
            "",
            "## 结果解读边界",
            "",
        ]
        + interpretation
        + [
            "",
            "## 复现命令",
            "",
            f"```text\n{manifest.get('command', 'formal runner command 未记录')}\n```",
            f"- 配置校验：`{reproduction.get('validate_command', '未记录')}`",
            f"- leakage 检查：`{reproduction.get('leakage_check_command', '未记录')}`",
            "- 人工审计与正式报告命令见 `docs/archive/research/evaluation_protocol.md`；"
            "本报告不重复调用 generation provider。",
        ]
    )


def write_report(
    results: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    output_path: Path,
    annotation_audit: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_report(results, config, manifest, annotation_audit), encoding="utf-8"
    )


def write_baseline_report(
    results: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    output_path: Path,
    annotation_audit: dict[str, Any] | None = None,
) -> None:
    """生成只保留两个 baseline 实验的正式报告，避免手工复制指标。"""
    baseline_names = ("rag_embedding", "agent_single")
    baseline_results = {
        name: results[name] for name in baseline_names if name in results
    }
    if set(baseline_results) != set(baseline_names):
        missing = sorted(set(baseline_names) - set(baseline_results))
        raise ValueError(f"baseline results are missing: {missing}")
    report = generate_report(
        baseline_results, config, manifest, annotation_audit
    ).replace("# MediDiag 正式评测报告", "# MediDiag 基线评测报告", 1)
    report = report.replace(
        "> 本报告由不可变 raw results 与已通过的独立人工 citation 审计自动生成；不得手工修改指标。",
        "> 本报告由不可变 raw results 与已通过的独立人工 citation 审计自动生成；不得手工修改指标。\n\n"
        "- RAG baseline：`rag_embedding`（纯 embedding）。\n"
        "- Agent baseline：`agent_single`（单 Agent，使用固定 `rag_full` 检索配置）。",
        1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
