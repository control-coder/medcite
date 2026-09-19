# MediDiag 基线评测报告

> 本报告由不可变 raw results 与已通过的独立人工 citation 审计自动生成；不得手工修改指标。

- RAG baseline：`rag_embedding`（纯 embedding）。
- Agent baseline：`agent_single`（单 Agent，使用固定 `rag_full` 检索配置）。

## 运行溯源

- run_id: `20260729T130801479412Z_ded91c0c061e`
- git_commit: `b6021d2853716882e98a5e5854fe938aaeafbd48`
- dirty_diff_hash: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- config_hash: `ded91c0c061e520675bc186b065d5a0d19c028ed3b99b638660ef2723179683d`
- generation provider/model: `deepseek/deepseek-v4-flash`
- generation base URL: `https://api.deepseek.com`
- generation provenance: `provider_response_id`，固定 provider snapshot **不可核验**；本次仅以 `response.id` 作为受限溯源，不能声称 generation 模型版本完全可复现。
- generation temperature/seed: `0` / `42`；seed_applied=`False`
- embedding: `sentence-transformers/all-MiniLM-L6-v2`@`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- rerank: `cross-encoder/ms-marco-MiniLM-L-6-v2`@`c5ee24cb16019beea0893ab7796b1df96625c6b8`
- judge: `cross-encoder/nli-MiniLM2-L6-H768@b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` (`nli`)
- dataset_version: `v1`；来源：MedQA (train), PubMedQA (pqa_labeled)
- dataset hashes: `{'rag_eval_set_path': 'cb99d4f2327db6cc0f1085042ce926099e8a538b4fc34099edd500c44655b82c', 'agent_eval_set_path': 'd99c3448ea8bf71512b0f3aa93461ec29c6a2f6a76bca134d127f752bbf03d6e', 'agent_sample_manifest_path': '393cc546008221145a986ea2348e891ec6bd07d03aac738c59d6e25a7ab1f6a6', 'knowledge_base_path': '07b9de2c07f062233e753e4623c9e84f4bfba2bc2f36726c43f0a2d2b591b43f'}`

## 人工 citation 复核门禁（Human citation-review gate）

- independently double-labeled pairs: `562/2809` (20.01%)
- Cohen's Kappa: `0.6049`; observed agreement: `0.7384`
- Kappa disposition: DISCUSS: 0.6-0.8，进入分歧讨论（记录裁决人、理由、修改字段）
- disagreements/adjudications: `147/147`
- fixed NLI judge agreement with adjudicated human labels: `0.3754`
- A/B 标签声明为两位独立人工标注者，147 条分歧均有第三位人工裁决；模型辅助预标注未进入审计。

## 数据、ground truth 与指标口径

- RAG ground truth：`eval/datasets/eval_set_pubmedqa.jsonl` 中的 `gold_evidence_ids`；只有存在 gold evidence 的样本进入 Evidence Recall@5 分母。
- Agent 数据集：`eval/datasets/eval_set_medqa.jsonl`；agent 样本来自 `eval/datasets/agent_eval_manifest_v1.jsonl`。
- leakage gate：检查字段 `['source', 'source_id', 'metadata.raw_id']`，formal_candidate=`True` 表示 runner 已按 formal 配置完成前置门禁。
- 标注配置：双人抽检比例 `0.2`；Kappa 阈值 `0.6`。
- `evidence_recall_at_5`：公式 `eligible samples with >=1 gold evidence hit / samples with non-empty gold_evidence_ids`；ground truth 来源：PubMedQA eval_set.gold_evidence_ids。
- `gold_evidence_coverage`：公式 `samples with non-empty gold_evidence_ids / all samples`；ground truth 来源：eval_set.gold_evidence_ids。
- `citation_precision`：公式 `SUPPORTED claim-citation pairs / all emitted claim-citation pairs`；ground truth 来源：configured judge + human calibration sample。
- `judge_agreement`：公式 `judge labels matching adjudicated human labels / adjudicated sample`；ground truth 来源：double annotation + adjudication。
- `unsupported_claim_rate`：公式 `claims whose best citation verdict is UNSUPPORTED / all claims`；ground truth 来源：configured judge。
- `terminology_normalization_gain`：公式 `Recall@5(rag_term_norm) - Recall@5(rag_embedding)`；ground truth 来源：single-variable RAG experiments。
- `workflow_success_rate`：公式 `cases reaching CLOSED_SUCCESS / all workflow cases`；ground truth 来源：persisted case terminal state。
- `p95_latency_ms`：公式 `cold and warm end-to-end P95 latency`；ground truth 来源：stage timing events。
- `stage_latency_ms`：公式 `normalize/retrieval/rerank/generation/judge/review/state_transition latency`；ground truth 来源：stage timing events。

## 结果

| Experiment | Family | Recall@5 | Gold coverage | Citation precision | Unsupported claims | Pipeline approval | P95(ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| rag_embedding | rag | 0.6536 | 0.9333 | N/A | N/A | N/A | 15.1100 |
| agent_single | agent | N/A | 0.0000 | 0.0331 | 0.2909 | 0.6000 | 8071.2700 |

`workflow_success_rate` 当前有意不纳入正式表格，直到 API/worker runner 记录持久化的 `CLOSED_SUCCESS` 终态。

## 结果解读边界

- 固定 NLI judge 与裁决后人工标签的一致率为 `0.3754`；正式报告已通过审计门禁，但该一致率偏低，后续应优先做 judge 校准与误差分析，不能把低 Citation Precision 直接解释为生成模型能力结论。

## 复现命令

```text
D:\resume_project\medidiag\eval\runner.py --config eval/config.formal.yaml --experiment all --output artifacts/reports/raw
```
- 配置校验：`python -m eval.runner --config eval/config.yaml --validate`
- leakage 检查：`python -m eval.leakage_check --config eval/config.yaml --eval-set eval/datasets/eval_set_pubmedqa.jsonl --kb eval/datasets/knowledge_chunks.jsonl`
- 人工审计与正式报告命令见 `docs/archive/research/evaluation_protocol.md`；本报告不重复调用 generation provider。