# Citation 人工复核产物目录

此目录保存 formal run 的 citation 抽样模板与**实际人工完成**的正式评测标注。截至 2026-08-16，`citation_labels_a_v1.jsonl`、`citation_labels_b_v1.jsonl` 已覆盖 562 条样本，`citation_adjudication_v1.jsonl` 已覆盖全部 147 条分歧；审计结果为 `PASSED`、`report_eligible: true`。不得把模型输出、示例数据、脚本生成数据或同一人重复填写的数据伪装为人工复核。

## 文件与隔离边界

1. `citation_sample_v1.jsonl`：由 `python -m eval.annotation_audit prepare` 从指定 formal run 实际发出的 claim-citation pairs 生成。默认固定 seed 做 judge verdict 分层抽样，覆盖不少于 20%。同名 `.manifest.json` 与该模板不可分离。
2. `citation_labels_a_v1.jsonl`、`citation_labels_b_v1.jsonl`：两位不同的人在互不可见对方结论的前提下独立填写。只有这两个文件可作为 `eval.annotation_audit audit` 的输入。
3. `citation_adjudication_v1.jsonl`：第三位人工裁决者只处理 A/B 不一致的 `annotation_id`；即使没有分歧，空文件也必须存在。
4. `model_assisted/`（如本地需要）：只允许保存明确标有 `model_assisted_prelabel` 的模型辅助预标注，供人工组织标注工作时做隔离质检。它不是人工标签，不得改名、复制或转换为上述 A/B/裁决文件，也不得进入 Kappa、`judge_agreement`、`report_eligible` 或正式报告。为防止锚定偏差，实际 A/B 标注者在独立完成前不应查看这些文件。

## A/B 人工标签 schema

每一行必须包含以下字段：

- `annotation_id`：与抽样模板逐条匹配的 ID。
- `label`：`SUPPORTED`、`PARTIAL` 或 `UNSUPPORTED`。
- `annotator_id`：不含真实患者信息的稳定匿名标识；同一文件只能有一个 ID，A/B 必须不同。
- `annotated_at`：ISO-8601 时间。
- `rationale`：本条 claim 与 evidence 的中文判定依据。
- `annotation_method`：固定为 `human_independent`。模型辅助文件使用的 `model_assisted_prelabel` 会被正式审计器拒绝。
- `reviewer_type`：固定为 `human`。
- `assistance_disclosure`：固定为 `none`。使用 LLM、自动规则、脚本批量推断或已有模型预标注辅助时不得填写为 `none`，该文件不能进入正式审计。
- `independence_attestation`：JSON 布尔值 `true`，表示该标注者已独立完成该条判断，且未查看另一位标注者的标签。

示例（仅说明 schema，**不是**可提交的正式标签）：

```json
{
  "annotation_id": "<来自 citation_sample_v1.jsonl>",
  "label": "PARTIAL",
  "annotator_id": "reviewer_a_anon",
  "annotated_at": "2026-08-15T10:30:00+08:00",
  "rationale": "证据支持症状改善，但没有覆盖 claim 中的长期疗效结论。",
  "annotation_method": "human_independent",
  "reviewer_type": "human",
  "assistance_disclosure": "none",
  "independence_attestation": true
}
```

## 裁决 schema

每个实际分歧恰好一条裁决记录。除 `annotation_id`、`final_label`、`adjudicator_id`、`adjudicated_at`、`reason`、非空数组 `modified_fields` 外，还必须填写：

- `reviewer_type`：固定为 `human`。
- `assistance_disclosure`：固定为 `none`。

审计器会拒绝：样本低于 20%、样本与 raw run 不匹配、漏标/多标、非法标签、相同标注者、非人工或使用模型辅助的正式标签、未声明独立性、未裁决的分歧，以及 `Cohen's Kappa < 0.60`。`0.60 <= Kappa < 0.80` 时仍必须逐条完成裁决；`Kappa >= 0.80` 也不免除实际分歧的裁决。

`annotation_method`、`reviewer_type`、`assistance_disclosure` 与独立性字段是可审计的声明与文件级溯源，不替代线下的身份核验、盲法组织记录或研究伦理审查。不要提交真实患者数据、密钥或可识别个人信息；本项目只允许公开数据或明确脱敏模拟输入。
