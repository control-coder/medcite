# Reports

- `raw/<run_id>/`：新 runner 的逐次原始结果、config snapshot 与 manifest。
- `archive/`：旧 A-F 与 5 样本探索性结果，以及被后续改动取代的原始结果，只用于审计历史，不得用于简历。见 `archive/README.md`。
- `routing_diagnostics_*.json`、`prompt_size_diagnostics.json`：诊断工具（`eval/routing_diagnostics.py`、`eval/prompt_size_diagnostics.py`）的输出，是**代码行为测量**，不是评测结果，不得作为指标引用。
- `baseline.md` / `final_eval.md`：仅允许由 `eval/report.py` 从 `report_eligible: true` 的 formal run 自动生成。当前尚不存在合格 formal run，因此不提供这两个文件。

不要手工填入目标值，也不要把 development、rule fallback、`--limit` 或 `--dry-run` 结果改名为正式报告。
