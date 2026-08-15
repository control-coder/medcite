# 模型辅助 citation 预标注（非正式）

本目录的内容是 2026-08-15 对冻结样本 `../citation_sample_v1.jsonl` 进行的**模型辅助预标注**。它们只可用于离线数据质量排查或安排人工复核的工作量估计，绝不是人工复核，也不能作为独立标注者、裁决者、Kappa、`judge_agreement`、`report_eligible`、Citation Precision 或正式报告的输入。

## 当前产物

- `citation_model_prelabels_gpt54mini_v1.jsonl`：`gpt-5.4-mini` 的预标注。
- `citation_model_prelabels_gpt56terra_v1.jsonl`：`gpt-5.6-terra` 的预标注。
- `manifest_v1.json`：对两个文件、源样本哈希、条目覆盖、标签分布与相互一致率的固定记录。

两份预标注均带有 `annotation_method: model_assisted_prelabel`，因此不能满足 `eval.annotation_audit audit` 所需的 `annotation_method: human_independent`。即使后续人为添加其他字段，审计器也会拒绝该方法值。

## 重要限制

1. 两个模型的标签在 562 条上完全一致（100%）。这只能说明当前两份机器输出未形成分歧，**不**证明标签正确、模型独立、NLI judge 被校准或 Citation Precision 有效；共同提示、相同上下文和共同偏差都可能造成一致。
2. 为避免锚定偏差，正式 A/B 人工标注者在独立提交前不得查看本目录中的标签或 rationale。人工作业仍只应查看冻结 sample 与自己的空白标签表。
3. 任何本目录之外的引用都必须明确写为“模型辅助预标注（非正式）”，不得缩写为“人工标注”或“人工抽检结果”。
4. 这些文件不包含真实患者数据、API key 或完整 provider response；仍只允许处理公开数据或脱敏模拟输入。