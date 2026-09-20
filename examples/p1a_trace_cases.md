# P1-A 可复现 Trace 案例

这些案例只使用确定性、非诊断 provider 和临时 SQLite，不调用外部模型，也不接收真实患者数据。

## 生成命令

```powershell
conda run --no-capture-output -n medidiag python -m medidiag.cli trace-examples --output-root artifacts/traces
```

命令生成：

- `artifacts/traces/raw/<trace_id>.jsonl`：逐事件结构化 trace。
- `artifacts/traces/summary/<trace_id>.json`：不含病例原文和证据文本的脱敏摘要。

## 案例与验收点

| 案例 | 流程 | 必须可见的证据 |
|---|---|---|
| `success` | 创建 -> worker -> `CLOSED_SUCCESS` | evidence 元数据、claim hash、citation verdict、阶段 latency、最终报告元数据 |
| `lease_recovery` | 旧 worker 租约过期 -> 新 worker reclaim -> 旧写入被拒绝 -> `CLOSED_SUCCESS` | `lease_reclaimed`、`TASK_LEASE_LOST`、attempt 变化和恢复后的阶段事件 |
| `review_escalation` | reviewer -> `ESCALATED` -> human `APPROVED` -> 新任务恢复 -> `CLOSED_SUCCESS` | 升级状态、人工理由摘要、恢复任务和最终报告 |

双专科收益或噪声案例尚未形成真实模型实验，不包含在本轮生成器中，也不得据此声称多 Agent 收益。
