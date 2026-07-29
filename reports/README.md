# Reports

- `raw/<run_id>/`：新 runner 的逐次原始结果、config snapshot 与 manifest。
- `archive/`：旧 A-F 与 5 样本探索性结果，以及被后续改动取代的原始结果，只用于审计历史，不得用于简历。见 `archive/README.md`。
- `routing_diagnostics_*.json`、`prompt_size_diagnostics.json`：诊断工具（`eval/routing_diagnostics.py`、`eval/prompt_size_diagnostics.py`）的输出，是**代码行为测量**，不是评测结果，不得作为指标引用。
- `cost_estimate.json`：`eval/cost_estimate.py` 按 DeepSeek 官方单价对实测 prompt 字符数做的**费用折算**，不是评测结果。五层口径在产物的 `measurement_basis` 里标注：单价是外部事实，字符数实测，输入 token 估算，输出 token 与缓存命中**只有上限**。只有带 `--run-dir` 时的 `actual` 一节是实测费用；其余全部是估算，不得当作已发生的支出引用。
- `threshold_sensitivity.json`：`eval/threshold_sensitivity.py` 对 `routing_diagnostics_*_rule4_removed.json` 里已测得的分数做的**阈值反事实重算**（改阈值不改分数，故无需重跑检索）。它只回答「某个 `MIN_PRIMARY_SCORE` 会放行多少样本、这些样本的关键词信号有多强」，**不回答被放行的样本走对了哪个专科**——那需要本项目没有的人工标注。不是评测结果。
- `baseline.md` / `final_eval.md`：仅允许由 `eval/report.py` 从 `report_eligible: true` 的 formal run 自动生成。当前尚不存在合格 formal run，因此不提供这两个文件。

不要手工填入目标值，也不要把 development、rule fallback、`--limit` 或 `--dry-run` 结果改名为正式报告。
