# 正式评测与人工 citation 复核协议

> 本协议面向 P2 正式评测，定义工程门禁。它不构成医学研究结论，也不以开发态结果代替人工标注。
> 首次 formal run 已于 2026-07-29 完成，结果见 [final_eval.md](../../artifacts/reports/final_eval.md)，解读边界见 [NLI 语言兼容性复盘](nli-language-compatibility.md)。本文件只保留可执行的协议本身。

## 目标与边界

正式评测的 Citation Precision 由固定 NLI judge 对**实际输出**的 claim-citation pair 计算；同时必须以不少于 20% 的独立双人标注样本校准 judge。开发态 `rule_fallback`、`--limit`、`--dry-run`、未固定模型 revision、旧 A-F exploratory artifacts 均不能进入正式报告。

该流程不自动生成任何人工标签，不把模型判断伪装成人工复核，也不把 Kappa 工具本身视为已完成标注。模型如参与预标注，只能写入与正式 A/B 标签隔离的 `model_assisted/` 目录，并显式标记为 `model_assisted_prelabel`；其结果不得输入正式 audit。当前 formal run 已完成真实人工审计，审计结果见 `artifacts/reports/citation_annotation_audit.json`，正式报告见 `artifacts/reports/final_eval.md`。

## 模型版本锁定（formal）

`eval/config.formal.template.yaml` 是可提交的正式配置模板；它故意包含 `REPLACE_*` 占位符，因此**不能**直接通过校验或启动正式 run。开始正式评测时，先复制为本地 `eval/config.formal.yaml`，再补齐经过核验的版本信息：

1. `embedding`、`rerank`、`judge` 的 `revision` 必须是完整 40 位 Hugging Face commit SHA；分支名、`latest`、短 SHA 和描述性字符串都会被拒绝。
2. generation 除 `model` 外必须填写 provider 模型标识、`provenance_mode` 与溯源来源。供应商可核验 snapshot 存在时使用 `provider_snapshot`；如不提供，使用 `provider_response_id` 与 `response.id`。
3. 若 generation provider 不能返回或公开可核验的 snapshot/version，必须停止正式 run；不能把供应商模型名或自行编造的字符串当作不可变版本。
4. runner 会把 generation 的 `model`、`revision`、`provenance_mode`、`response_id_source` 以及其他模型 revision 一并写入 `manifest.json`，并在 Agent 样本中写入实际 `provider_request_ids`。

```powershell
Copy-Item eval/config.formal.template.yaml eval/config.formal.yaml
python -m eval.runner --config eval/config.formal.yaml --validate
```

## NLI 输入语言契约（formal）

当前固定 `cross-encoder/nli-MiniLM2-L6-H768` judge 的 formal 输入契约为 `judge.input_language: en`。正式 runner 仅约束会进入 judge 的 `claims[].text`：它必须是英文完整陈述且不含中文汉字；诊断展示、风险提示和其他用户可见字段仍可使用中文。`CitationVerifier` 在实际 NLI 推理前检查 claim 与 evidence，发现汉字即以 `NLI_JUDGE_LANGUAGE_MISMATCH` fail-closed；不得将该错误折算为 `PARTIAL`、弃权或继续产出报告。

无 citation 或 citation chunk 缺失时没有文本 NLI 推理，仍按既有 `UNSUPPORTED` 规则处理。缓存键包含 `claim_language`，不得复用未受该契约约束的旧 agent 输出。若将来需要评测中文 claim，必须新增可溯源的翻译/跨语言 judge 路由并锁定其模型、版本、提示词和失败处理；不得将翻译作为隐式步骤。

2026-07-29 历史 formal run 的抽检中，290/524（55.3%）实际文本 pair 为含中文 claim + 英文 evidence。因此其 citation 指标和 `judge_agreement=0.3754` 只作为受语言混杂影响的历史测量保留；新的 formal run 必须重新执行抽样、双标、裁决和 audit。完整复盘见 [NLI 语言兼容性复盘](nli-language-compatibility.md)。

## 运行顺序

1. 从模板创建本地 `eval/config.formal.yaml`，填入完整 HF commit SHA、generation 模型标识与受限溯源配置；保留 `evaluation.mode: formal`、`judge.method: nli` 与 `judge.input_language: en`。
2. 执行 `--validate`。若出现模型锁定、数据集、leakage 或 NLI 配置错误，先修正配置；不得降级为 development 结果后继续声明正式结论。
3. 不传 `--limit` 或 `--dry-run`，运行 evaluation。runner 在模型加载和生成前执行 leakage gate；NLI 加载、推理或输入语言不兼容都会 fail-closed。
4. 从刚生成的 `artifacts/reports/raw/<run_id>/` 创建实际 claim-citation pair 的人工复核模板。
5. 两名不同的人在盲法条件下独立完成全部模板，逐条填写判定依据、日期以及 `annotation_method: human_independent`、`reviewer_type: human`、`assistance_disclosure: none`、`independence_attestation: true`。这些是正式门禁的必填声明；`model_assisted_prelabel` 会被审计器拒绝，不能由模型代填。
6. 对每条不一致记录第三位人工裁决者、理由、最终标签和修改字段；裁决记录同样声明 `reviewer_type: human` 与 `assistance_disclosure: none`。Kappa 低于 0.60 时重新标注或剔除，不能出正式报告。
7. 运行 audit 生成与 raw manifest 绑定的 `citation_annotation_audit.json`；只有 audit 为 `PASSED` 且 `report_eligible: true` 时才可以渲染 `artifacts/reports/final_eval.md`。

## 可复现命令

```powershell
# 1. 检查正式配置；不要以 development config 的结果作为正式结论
python -m eval.runner --config eval/config.formal.yaml --validate
python -m eval.runner --config eval/config.formal.yaml --experiment all --output artifacts/reports/raw

# 2. 将 <run_id> 替换为上一步实际输出的 ID；这一步不生成标签
python -m eval.annotation_audit prepare `
  --run-dir artifacts/reports/raw/<run_id> `
  --output eval/annotations/citation_sample_v1.jsonl `
  --ratio 0.20 --seed 42

# 3. 两位独立标注者人工填写 label 文件，裁决人填写分歧记录后进行审计
python -m eval.annotation_audit audit `
  --run-dir artifacts/reports/raw/<run_id> `
  --sample eval/annotations/citation_sample_v1.jsonl `
  --annotator-a eval/annotations/citation_labels_a_v1.jsonl `
  --annotator-b eval/annotations/citation_labels_b_v1.jsonl `
  --adjudication eval/annotations/citation_adjudication_v1.jsonl `
  --output artifacts/reports/citation_annotation_audit.json

# 4. audit 成功后才允许渲染正式报告
python -m eval.annotation_audit report `
  --run-dir artifacts/reports/raw/<run_id> `
  --audit artifacts/reports/citation_annotation_audit.json `
  --output artifacts/reports/final_eval.md
```

## 运行期间必须盯的信号

1. **第一个样本返回后立刻核对 `provider_usage.prompt_tokens`**，与费用估算比对。偏差超过 ±30% 说明字符/token 假设不成立，应中止并重算预算。
2. 控制台逐实验输出 `Recall@5` 与 `GoldCoverage`。agent 族的 `Recall@5` 应为 `N/A`（MedQA 按设计没有 gold evidence）；出现 `0.0000` 说明有别的问题。
3. 任何 `AgentProviderError` 都会终止 run。formal 模式下这是有意的 fail-closed 行为，不要改成宽松路径重跑——那样得到的指标由故障构成。
4. 余额与速率限制。429 会由 `LLMClient` 有限重试吸收，但连续 429 意味着预算或配额不足，应主动中止。
5. 挂钟时间。仅检索链路 100 样本 × 3 组约 5.5 分钟；加上 generation 与 NLI judge 后主要由 provider 延迟决定（`timeout_seconds: 60`）。首次 formal run 全程约 85 分钟。

## 结束后必须核对的项

1. `artifacts/reports/raw/<run_id>/` 下有 `manifest.json`、`config.snapshot.json` 和每个实验一份 JSON；缺任何一个都说明 run 未正常结束。
2. runner 结束时 `manifest.json` 应为 `formal_candidate: true`；`report_eligible` 初始为 `false`，只有通过的人工 audit 才能授予 `true`（DD-015）。
3. `non_reportable_reasons` 不含 `evaluation_mode_is_development` 或 `dry_run_*`。
4. **每个 agent 样本都有非空 `provider_request_ids`**。缺失会在 run 中直接抛 `FORMAL_GENERATION_RESPONSE_ID_MISSING`，但结束后仍应抽查确认。
5. `git_commit` 与运行时 HEAD 一致，`dirty_diff_hash` 为空。
6. `resolved_package_versions` 记录实际解析到的依赖版本（DD-018），`runtime` 记录实际设备。
7. `generation.seed_applied` 为 `false` 且带 `seed_not_applied_reason`（DD-019），正式报告不得声称按 seed 可复现。
8. 用 `python -m eval.cost_estimate --run-dir artifacts/reports/raw/<run_id>` 取得实测 token 与实际费用（输出 JSON 的 `actual` 一节）。核对：实际输入 token 与估算偏差、实际 `completion_tokens` 相对上限的比例、`cache_split_source` 是否为 `provider_reported`。
9. 立刻做 `artifacts/reports/raw/<run_id>/` 的离线备份：它是不可再生的付费产物。

## 审计验证项

- sample 的 `annotation_id`、claim、evidence、judge 元数据必须与 raw run 的原始 JSON 一致；模板 sidecar manifest 绑定 run 的 `manifest.json` hash。
- sample 覆盖率必须不低于 20%；脚本按 verdict 分层并使用固定 `seed` 选择，避免只抽到单一 verdict。
- A/B 文件必须覆盖完全相同的 sample，且各自只包含一位、彼此不同的 `annotator_id`。
- 所有 label 必须属于 `SUPPORTED`、`PARTIAL`、`UNSUPPORTED`；每项必须记录 ISO-8601 日期、文字判定依据、`annotation_method: human_independent`、`reviewer_type: human`、`assistance_disclosure: none` 和 `independence_attestation: true`。审计器拒绝模型、规则或脚本辅助产生的标签进入 formal gate。
- 所有实际分歧都必须有且只能有一条裁决记录；审计输出 Kappa、混淆矩阵、分歧数、裁决数、NLI judge 与裁决后人工标签的一致率。
- `Kappa < 0.60` 输出 `FAILED` 并阻断报告；`0.60 <= Kappa < 0.80` 输出讨论提示且要求完整裁决；`Kappa >= 0.80` 视为稳定，但不免除实际分歧的裁决记录。

## 报告解释

人工样本的 `judge_agreement` 是固定 NLI judge 相对于裁决后标签的校准值，不等于 Citation Precision。Citation Precision 的分母仍是正式 run 实际输出的所有 claim-citation pairs；人工复核只验证其 judge 行为是否具有可审计的抽样校准。

正式报告自动写入 run ID、Git/dirty hash、config hash、模型版本、样本覆盖、Kappa、分歧/裁决数与指标表。它不会从开发 run、无 audit run 或 audit 失败 run 生成。
