# Baseline Report

> 阶段 0 占位。实际 baseline 数据将在阶段 6 由 `eval/runner.py` 自动生成。

## Locked Configuration

详见 `eval/config.yaml`。

## Baseline Metrics

| 指标 | 公式 | Baseline 值 | 备注 |
|---|---|---|---|
| Evidence Recall@5 | 至少命中 1 条 gold_evidence 的样本数 / 总样本数 | TBD | A 组（纯 embedding） |
| Citation Precision | SUPPORTED citation 数 / 系统输出 citation 总数 | TBD | |
| Judge Agreement | judge 判定与人工抽样复核一致样本数 / 抽样复核样本数 | TBD | Cohen's Kappa |
| Unsupported Claim Rate | UNSUPPORTED claims / total claims | TBD | |
| Terminology Normalization Gain | Recall@5(with norm) - Recall@5(without norm) | TBD | |
| Workflow Success Rate | CLOSED_SUCCESS case 数 / 总 case 数 | TBD | |
| P95 latency | 端到端 P95 | TBD | |

## Ablation Groups

| 组别 | 配置 | 目的 | 状态 |
|---|---|---|---|
| A | 纯 embedding 检索 | 基线 | pending |
| B | A + BM25 | 单变量：混合检索收益 | pending |
| C | A + evidence level weighting | 单变量：证据等级收益 | pending |
| D | A + terminology normalization | 单变量：术语归一化收益 | pending |
| E | A + citation verifier | 单变量：引用审核对 unsupported claim 的影响 | pending |
| F | A + B + C + D + E 全量组合 | 全量效果，不用于单独归因 | pending |

## Reproduction Command

```bash
# 锁定在 eval/config.yaml
python -m eval.runner --config eval/config.yaml --group all --output reports/raw/
```
