# 报告与证据索引

本目录按证据类型保存评测与诊断产物。原始报告（含失败和无效结论）保持生成时的内容，不重新生成或改写。

## 应用工程结果

- `application/`：模拟主题检索回归、公开中文语料检索对照、真实 MiMo 端到端验证的独立报告；方法与结果解读见 [评测与验证](../../docs/evaluation.md)。它们不是研究 formal 报告，不能相互混算。
- 验证过程中的数据库、日志与调用账本保存在本地 `.cache/implementation/`，不纳入版本库。

## 历史研究与诊断产物

以下按原研究记录解释，应用侧验证不重新认证历史标注或改变报告有效性。

- `raw/<run_id>/`：新 runner 的逐次原始结果、config snapshot 与 manifest。
- `archive/`：旧 A-F 与 5 样本探索性结果，以及被后续改动取代的原始结果，只用于审计历史，不得作为正式结论引用。见 [归档说明](archive/README.md)。
- `routing_diagnostics_*.json`、`prompt_size_diagnostics.json`：诊断工具（`eval/routing_diagnostics.py`、`eval/prompt_size_diagnostics.py`）的输出，是**代码行为测量**，不是评测结果，不得作为指标引用。
- `cost_estimate.json`：`eval/cost_estimate.py` 按 DeepSeek 官方单价对实测 prompt 字符数做的**费用折算**，不是评测结果。五层口径在产物的 `measurement_basis` 里标注：单价是外部事实，字符数实测，输入 token 估算，输出 token 与缓存命中**只有上限**。只有带 `--run-dir` 时的 `actual` 一节是实测费用；其余全部是估算，不得当作已发生的支出引用。
- `threshold_sensitivity.json`：`eval/threshold_sensitivity.py` 对 `routing_diagnostics_*_rule4_removed.json` 里已测得的分数做的**阈值反事实重算**（改阈值不改分数，故无需重跑检索）。它只回答「某个 `MIN_PRIMARY_SCORE` 会放行多少样本、这些样本的关键词信号有多强」，**不回答被放行的样本走对了哪个专科**——那需要本项目没有的人工标注。不是评测结果。
- `citation_annotation_audit.json`：562 条人工双标注、147 条第三人裁决的历史审计结果；`status=PASSED`、`report_eligible=true`、Cohen's Kappa=`0.6049`。它绑定具体 raw run、sample hash 和三个标注文件 hash，但不满足 2026-08-18 新增的英文 NLI 输入语言契约，不能替代新 run 的 audit。
- `final_eval.md`：由 `eval/report.py` 在历史人工审计通过后自动生成。历史 run 中 55.3% 的实际文本 NLI pair 为含中文 claim + 英文 evidence，因此其 citation 指标和 judge agreement=`0.3754` 仅作受语言混杂影响的历史测量；新 formal run 必须重新生成报告，详见 [语言兼容性复盘](../../docs/research/nli-language-compatibility.md)。
- `baseline.md`：由同一 formal run 自动生成的基线报告，只保留 `rag_embedding` 与 `agent_single`，用于后续单变量消融和 Agent 拓扑对照的共同参照，不得手工补写。

不要手工填入目标值，也不要把 development、rule fallback、`--limit` 或 `--dry-run` 结果改名为正式报告。`eval/annotations_ui/` 是人工审阅界面生成目录，不属于正式标注产物，不纳入版本提交。
