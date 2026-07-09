"""评测报告生成。

生成 reports/baseline.md + reports/final_eval.md。
每个指标含: 公式 / ground truth 来源 / baseline / 消融结果 / 复现命令。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_GROUP_CONFIGS = {
    "A": "纯 embedding（基线）",
    "B": "A + BM25",
    "C": "A + 证据等级加权",
    "D": "A + 术语归一化",
    "E": "A + 引用审核",
    "F": "全量组合",
}


def _get_metric(group_result: Any, metric: str) -> float:
    """从 GroupResult 或 dict 获取指标值。"""
    if hasattr(group_result, metric):
        return getattr(group_result, metric)
    if isinstance(group_result, dict):
        return group_result.get("metrics", {}).get(metric, 0.0)
    return 0.0


def _format_metrics_table(results: dict[str, Any]) -> str:
    """格式化消融指标对比表。"""
    lines = [
        "| 组别 | 配置 | Recall@5 | Citation Precision | Unsupported Rate | Workflow Success | P95(ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in ("A", "B", "C", "D", "E", "F"):
        if g not in results:
            continue
        gr = results[g]
        lines.append(
            f"| {g} | {_GROUP_CONFIGS.get(g, '')} | "
            f"{_get_metric(gr, 'recall_at_5'):.4f} | "
            f"{_get_metric(gr, 'citation_precision'):.4f} | "
            f"{_get_metric(gr, 'unsupported_claim_rate'):.4f} | "
            f"{_get_metric(gr, 'workflow_success_rate'):.4f} | "
            f"{_get_metric(gr, 'p95_latency'):.1f} |"
        )
    return "\n".join(lines)


def generate_baseline_report(
    results: dict[str, Any], cfg: dict[str, Any]
) -> str:
    """生成 baseline 报告。"""
    gen = cfg["generation"]
    lines = [
        "# Baseline Report",
        "",
        "> 自动生成，请勿手动编辑。换模型/数据集必须重跑。",
        "",
        "## 锁定配置",
        f"- generation_model: `{gen['model']}`",
        f"- embedding_model: `{cfg['embedding']['model']}`",
        f"- rerank_model: `{cfg['rerank']['model']}`",
        f"- judge_model: `{cfg['judge']['model']}`",
        f"- temperature: {gen['temperature']}, seed: {gen['seed']}",
        f"- dataset_version: {cfg['dataset']['version']}",
        f"- retrieval weights: w1={cfg['retrieval']['weights']['w1_bm25']}, "
        f"w2={cfg['retrieval']['weights']['w2_embedding']}, "
        f"w3={cfg['retrieval']['weights']['w3_evidence_level']}, "
        f"w4={cfg['retrieval']['weights']['w4_term_overlap']}",
        "",
        "## 消融实验结果",
        "",
        _format_metrics_table(results),
        "",
        "## 指标说明",
        "",
        "### Evidence Recall@5",
        "- 公式: 至少命中 1 条 gold_evidence 的样本数 / 总样本数",
        "- ground truth: eval_set.gold_evidence_ids",
        "- 数据集来源: PubMedQA (300 样本) + MedQA (200 样本)",
        "",
        "### Citation Precision",
        "- 公式: SUPPORTED citation 数 / 系统输出 citation 总数",
        "- ground truth: judge_model (deberta-v3-base-mnli) + 人工抽样",
        "- 标注规则: SUPPORTED/PARTIAL/UNSUPPORTED，NLI 模型判定 + 规则降级",
        "",
        "### Unsupported Claim Rate",
        "- 公式: UNSUPPORTED claims / total claims",
        "- ground truth: judge_model",
        "",
        "### Workflow Success Rate",
        "- 公式: CLOSED_SUCCESS case 数 / 总 case 数",
        "- ground truth: case 终态统计（APPROVED = success）",
        "",
        "### Terminology Normalization Gain",
        "- 公式: Recall@5(D组, with norm) - Recall@5(A组, without norm)",
        "- ground truth: ablation group A vs D",
        "",
        "### P95 Latency",
        "- 公式: 端到端 P95 延迟 (ms)",
        "- ground truth: 分阶段 latency 埋点",
        "",
        "## Cohen's Kappa 标注一致性",
        "- 抽样 20% 做双人复核",
        "- Kappa < 0.6: 剔除或重新标注",
        "- Kappa 0.6-0.8: 进入分歧讨论",
        "- Kappa > 0.8: 视为稳定标注",
        "",
        "## 数据泄露校验",
        "- leakage_check: 测试样本 ID 不出现在知识库 chunk source/source_id/metadata.raw_id",
        "- 命中则输出 EVAL_DATA_LEAKAGE_DETECTED",
        "",
        "## 复现命令",
        "```bash",
        "python -m eval.runner --config eval/config.yaml --group all --output reports/raw/",
        "python -m eval.leakage_check --config eval/config.yaml --eval-set eval/datasets/eval_set.jsonl --kb eval/datasets/knowledge_chunks.jsonl",
        "```",
    ]
    return "\n".join(lines)


def generate_final_report(
    results: dict[str, Any], cfg: dict[str, Any]
) -> str:
    """生成最终报告。"""
    gen = cfg["generation"]

    recall_a = _get_metric(results.get("A", {}), "recall_at_5")
    recall_d = _get_metric(results.get("D", {}), "recall_at_5")
    term_gain = recall_d - recall_a

    success_a = _get_metric(results.get("A", {}), "workflow_success_rate")
    success_c = _get_metric(results.get("C", {}), "workflow_success_rate")
    multi_agent_gain = success_c - success_a

    routing_coverage = _get_metric(results.get("C", {}), "routing_coverage")

    lines = [
        "# Final Evaluation Report",
        "",
        "> 自动生成，请勿手动编辑。",
        "",
        "## 锁定配置",
        f"- generation_model: `{gen['model']}`",
        f"- temperature: {gen['temperature']}, seed: {gen['seed']}",
        f"- dataset_version: {cfg['dataset']['version']}",
        "",
        "## 消融实验结果",
        "",
        _format_metrics_table(results),
        "",
        "## 关键发现",
        "",
        "### Terminology Normalization Gain",
        f"- Recall@5(A组, 无归一化): {recall_a:.4f}",
        f"- Recall@5(D组, 有归一化): {recall_d:.4f}",
        f"- **Gain: {term_gain:+.4f}**",
        "",
        "### 多 Agent 增益",
        f"- 单 Agent (A组) Workflow Success: {success_a:.4f}",
        f"- 双专科动态路由 (C组) Workflow Success: {success_c:.4f}",
        f"- **增益: {multi_agent_gain:+.4f}**",
        "",
        "### 路由覆盖率（C组动态路由）",
        f"- 非兜底路由比例: {routing_coverage:.4f}",
        "",
        "### 消融三组对比",
        "| 组 | 配置 | Recall@5 | Workflow Success |",
        "|---|---|---|---|",
        f"| A | 单 Agent baseline | {recall_a:.4f} | {success_a:.4f} |",
        f"| B | 固定心内科+呼吸科 | {_get_metric(results.get('B', {}), 'recall_at_5'):.4f} | {_get_metric(results.get('B', {}), 'workflow_success_rate'):.4f} |",
        f"| C | 动态路由 Top2 | {_get_metric(results.get('C', {}), 'recall_at_5'):.4f} | {success_c:.4f} |",
        "",
        "## API 调用优化",
        "- **AgentOutputCache**: 基于 input_hash 缓存，避免重复 LLM 调用（temperature=0 可复现）",
        "- **批量 embedding**: 检索索引构建一次，所有消融组复用",
        "- **检索结果复用**: embedding_scores 跨消融组复用（只权重不同）",
        "- **--dry-run**: 只计算检索指标，不调 LLM（快速验证管线）",
        "",
        "## 贡献边界",
        "- **Upstream Reference**: edict（工程模式）、MedQA、PubMedQA、MeSH",
        "- **Third-party Components**: DeepSeek API、sentence-transformers、cross-encoder、deberta-v3-base-mnli、FAISS、BM25",
        "- **My Contributions**: 状态机执行器、任务租约、医学 RAG 消融、术语归一化三层、引用校验+合规管控、双专科仲裁实验",
        "",
        "## 复现命令",
        "```bash",
        "python -m eval.runner --config eval/config.yaml --group all --output reports/raw/",
        "```",
    ]
    return "\n".join(lines)


def write_reports(
    results: dict[str, Any],
    cfg: dict[str, Any],
    reports_dir: Path,
) -> None:
    """写入报告文件。"""
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline = generate_baseline_report(results, cfg)
    final = generate_final_report(results, cfg)

    (reports_dir / "baseline.md").write_text(baseline, encoding="utf-8")
    (reports_dir / "final_eval.md").write_text(final, encoding="utf-8")
