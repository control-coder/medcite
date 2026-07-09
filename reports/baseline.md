# Baseline Report

> 自动生成，请勿手动编辑。换模型/数据集必须重跑。

## 锁定配置
- generation_model: `deepseek-v4-flash-free`
- embedding_model: `sentence-transformers/all-MiniLM-L6-v2`
- rerank_model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- judge_model: `microsoft/deberta-v3-base-mnli`
- temperature: 0, seed: 42
- dataset_version: v1
- retrieval weights: w1=0.25, w2=0.45, w3=0.2, w4=0.1

## 消融实验结果

| 组别 | 配置 | Recall@5 | Citation Precision | Unsupported Rate | Workflow Success | P95(ms) |
|---|---|---|---|---|---|---|
| A | 纯 embedding（基线） | 0.0400 | 0.0000 | 0.0000 | 0.0400 | 0.0 |
| B | A + BM25 | 0.3000 | 0.0000 | 0.0000 | 0.3000 | 0.0 |
| C | A + 证据等级加权 | 0.0600 | 0.0000 | 0.0000 | 0.0600 | 0.0 |
| D | A + 术语归一化 | 0.0800 | 0.0000 | 0.0000 | 0.0800 | 0.0 |
| E | A + 引用审核 | 0.0400 | 0.0000 | 0.0000 | 0.0400 | 0.0 |
| F | 全量组合 | 0.5200 | 0.0000 | 0.0000 | 0.5200 | 0.0 |

## 指标说明

### Evidence Recall@5
- 公式: 至少命中 1 条 gold_evidence 的样本数 / 总样本数
- ground truth: eval_set.gold_evidence_ids
- 数据集来源: PubMedQA (300 样本) + MedQA (200 样本)

### Citation Precision
- 公式: SUPPORTED citation 数 / 系统输出 citation 总数
- ground truth: judge_model (deberta-v3-base-mnli) + 人工抽样
- 标注规则: SUPPORTED/PARTIAL/UNSUPPORTED，NLI 模型判定 + 规则降级

### Unsupported Claim Rate
- 公式: UNSUPPORTED claims / total claims
- ground truth: judge_model

### Workflow Success Rate
- 公式: CLOSED_SUCCESS case 数 / 总 case 数
- ground truth: case 终态统计（APPROVED = success）

### Terminology Normalization Gain
- 公式: Recall@5(D组, with norm) - Recall@5(A组, without norm)
- ground truth: ablation group A vs D

### P95 Latency
- 公式: 端到端 P95 延迟 (ms)
- ground truth: 分阶段 latency 埋点

## Cohen's Kappa 标注一致性
- 抽样 20% 做双人复核
- Kappa < 0.6: 剔除或重新标注
- Kappa 0.6-0.8: 进入分歧讨论
- Kappa > 0.8: 视为稳定标注

## 数据泄露校验
- leakage_check: 测试样本 ID 不出现在知识库 chunk source/source_id/metadata.raw_id
- 命中则输出 EVAL_DATA_LEAKAGE_DETECTED

## 复现命令
```bash
python -m eval.runner --config eval/config.yaml --group all --output reports/raw/
python -m eval.leakage_check --config eval/config.yaml --eval-set eval/datasets/eval_set.jsonl --kb eval/datasets/knowledge_chunks.jsonl
```