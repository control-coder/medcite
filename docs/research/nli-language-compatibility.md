# NLI 输入语言兼容性复盘

> 复盘日期：2026-08-18。本文只分析 2026-07-29 历史 formal run 的已冻结 raw、抽检、人工双标和裁决产物；不改写历史标签、NLI verdict 或正式报告。

## 结论

历史 run 中存在大量“含中文 claim + 英文 evidence”的实际 NLI 输入。它与较低的 judge agreement 明确相关：在有文本 evidence 的抽检 pair 中，此类输入的 agreement 为 28.3%，英文 claim + 英文 evidence 为 38.9%，相差 10.6 个百分点。

因此，语言不匹配是历史 `judge_agreement=0.3754` 的重要混杂因素；但它不是唯一解释。英文-英文子组的一致率仍只有 38.9%，且 NLI 对 `PARTIAL` 有明显偏置。不能把历史 Citation Precision 的低值直接解释为 generation 医学能力，也不能预先声称修复语言后指标会提升。

## 数据与关联口径

分析使用以下已版本化产物：

- `eval/annotations/citation_sample_v1.jsonl`：562 条抽检 pair；
- `citation_labels_a_v1.jsonl`、`citation_labels_b_v1.jsonl`：两名独立标注者的完整标签；
- `citation_adjudication_v1.jsonl`：147 条第三人裁决；
- `artifacts/reports/citation_annotation_audit.json`：审计汇总；
- `artifacts/reports/raw/20260729T130801479412Z_ded91c0c061e/`：2,809 条原始 citation 结果；
- `eval/datasets/knowledge_chunks.jsonl`：按 `evidence_chunk_id` 回连的 evidence 文本（仓库只含 PubMedQA 部分，完整知识库需按 [eval/README](../../eval/README.md) 本地重建）。

关联键为 `(experiment, sample_id, claim_id, evidence_chunk_id)`。562/562 条抽检记录均能唯一回连 raw 结果，且其 `judge_verdict` 与 raw 中的 verdict 一致。524/524 条非空 evidence 都能回连至同一知识库文本；其余 38 条是无 citation 的 `UNSUPPORTED`，不属于自然语言 NLI 输入，不能与有文本 pair 混在语言能力分母中。

语言类型采用可复现的字符级规则：含汉字且无 ASCII 拉丁字母为“中文”；两者都有为“中英文混用”；只含 ASCII 拉丁字母且无汉字为“英文”。这只是格式分类，不等于语义语言识别。

## 历史抽检统计

| claim-evidence 组合 | 抽检数 | 占全部抽检 | judge 与裁决后人工标签一致 | agreement |
| --- | ---: | ---: | ---: | ---: |
| 中文 claim + 英文 evidence | 176 | 31.3% | 54 | 30.7% |
| 中英文混用 claim + 英文 evidence | 114 | 20.3% | 28 | 24.6% |
| 含中文 claim + 英文 evidence 合计 | 290 | 51.6% | 82 | 28.3% |
| 英文 claim + 英文 evidence | 234 | 41.6% | 91 | 38.9% |
| 无文本 evidence | 38 | 6.8% | 38 | 100.0% |

在真实文本 NLI pair 分母中，含中文 claim + 英文 evidence 为 290/524（55.3%）。全量 raw 也呈现同一问题：1,445/2,809（51.4%）为该组合。

最主要的不一致单元是 `NLI=PARTIAL`、人工最终=`SUPPORTED`：含中文 claim + 英文 evidence 为 120/290（41.4%），英文-英文为 70/234（29.9%）。同时，跨语言子组的 A/B 分歧率为 35.9%，英文-英文为 18.4%，说明该子组本身也更难，不能把全部误差归咎于 judge。

## 工程处置

从本次修复开始，正式评测采用“英文 NLI 输入契约”，而不是在历史结果上补译或改判：

1. `eval/config*.yaml` 明确声明 `judge.input_language: en`；配置校验拒绝缺失、`any` 或其他语言值。
2. formal NLI runner 向 Agent prompt 传入 `claim_language=en`，只要求 `claims[].text` 使用英文完整陈述；诊断展示、风险提示等其他字段仍可为中文。
3. `CitationVerifier` 在实际 NLI 推理前检查 claim 与 evidence 是否含汉字。命中即以 `NLI_JUDGE_LANGUAGE_MISMATCH` fail-closed，不能静默作为 `PARTIAL` 或复用旧输出。
4. Agent 输出缓存键包含 `claim_language`，禁止复用未受该契约约束的历史生成结果。
5. 无 citation 或 chunk 缺失仍沿用既有 `UNSUPPORTED` 路径，不触发语言门禁，因为没有发生 NLI 文本推理。

此处选择“前置英语生成 + fail-closed”而不是隐式翻译：固定翻译器会引入新的 provider、版本、提示词、失败处理和翻译质量变量。若后续必须评测中文 claim，应另行实现可溯源的语言路由，并锁定翻译模型/版本、保存原文与译文、路由模式、错误码和独立人工复核；不得静默补译。

## 对历史产物的影响与下一步

- 保留历史 raw、抽检、A/B 标签、裁决和 `citation_annotation_audit.json`，它们仍可复现“当时输入下”的测量及人工标注过程。
- 历史 `artifacts/reports/final_eval.md` 的 citation 指标和 `judge_agreement` 从 2026-08-18 起只可称为“受语言混杂影响的历史测量”，不得作为当前 judge 校准完成、医学效果或简历指标依据。
- 必须创建新的 formal run ID，重新执行 raw run、20% 分层抽样、双人独立标注、第三人裁决和 audit；不得改写或覆盖历史 run。
- 新审计至少分开报告：全体 agreement、英文-英文子组 agreement、无文本 evidence 数，以及 `PARTIAL -> SUPPORTED` 与 `PARTIAL -> UNSUPPORTED` 的误差变化。
- 如果新的英文-英文 agreement 仍接近历史 38.9%，下一项工作是独立分析 `PARTIAL` 映射、claim 粒度与 evidence 选择；不得把语言问题当作唯一根因。