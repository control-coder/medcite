# Citation 人工复核产物目录

此目录只保存**实际人工完成**的正式评测标注；仓库不提供伪造标签、示例标签或可直接当作正式结果的模板。

正式 NLI run 完成后，按以下顺序产生文件：

1. `citation_sample_v1.jsonl`：由 `python -m eval.annotation_audit prepare` 从该 run 实际输出的 claim-citation pairs 生成。默认按固定 seed 做 verdict 分层抽样，覆盖不少于 20%。同时生成不可分离的 `citation_sample_v1.jsonl.manifest.json`。
2. `citation_labels_a_v1.jsonl` 与 `citation_labels_b_v1.jsonl`：两位不同标注者独立填写。每一行必须包含：
   - `annotation_id`
   - `label`：`SUPPORTED`、`PARTIAL` 或 `UNSUPPORTED`
   - `annotator_id`
   - `annotated_at`：ISO-8601 时间
   - `rationale`：判定依据
3. `citation_adjudication_v1.jsonl`：仅对所有 A/B 不一致的 `annotation_id` 填写；文件可为空，但仍必须存在。每行必须包含：
   - `annotation_id`、`final_label`
   - `adjudicator_id`、`adjudicated_at`
   - `reason`：裁决理由
   - `modified_fields`：非空数组，记录改动字段

审计器会拒绝：少于 20% 的样本、样本与 raw run 不匹配、标注者相同、漏标/多标、非法标签、无判定依据、未裁决的分歧以及 `Cohen's Kappa < 0.60`。`0.60 <= Kappa < 0.80` 的结果只有在每个分歧都有裁决记录时才可通过门禁；`Kappa >= 0.80` 仍要求所有实际分歧具有裁决记录。

不要提交任何真实患者数据、密钥或可识别个人信息。当前项目只允许公开数据或明确脱敏模拟输入。
