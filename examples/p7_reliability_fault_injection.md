# P7 可靠性故障注入与 Trace 审阅说明

## 定位

本组场景用于验证单机医疗助手 Agent 工作流的确定性故障处理，不是生产级 chaos engineering，也不用于证明真实 Provider SLA 或临床可靠性。

## 覆盖矩阵

| 场景 | 注入边界 | 预期错误码/结果 | 重试 | 业务写入约束 | Trace 证据 |
|---|---|---|---|---|---|
| timeout | generation | `LLM_TIMEOUT`，最终 `ESCALATED` | 耗尽 2 次 | 无 generation artifact/报告 | provider attempt + stage failure |
| 429 后恢复 | normalize | 最终 `CLOSED_SUCCESS` | 2 次失败后成功 | 只有一份 normalize artifact 和一份报告 | 三次 attempt 与 request ID |
| 5xx | generation | `PROVIDER_UNAVAILABLE` | 有限重试后升级 | 无失败阶段 artifact/报告 | HTTP 503、重试决策 |
| 空响应 | stage schema | `PROVIDER_SCHEMA_INVALID` | 不重试 | fail-closed | rejected attempt + stage failure |
| JSON 失败 | structured output | `STRUCTURED_OUTPUT_INVALID` | 不重试 | fail-closed | 稳定错误码 |
| 上下文超限 | provider adapter | `PROVIDER_CONTEXT_LIMIT` | 不盲目重试 | fail-closed | 稳定错误码 |
| judge timeout | review | `JUDGE_TIMEOUT` | 有限重试后升级 | 无 review artifact/报告 | review stage failure |
| worker crash/lease expiry | task lease | scanner 接管 | 新 attempt 重入 | 已提交 stage 复用，artifact 不重复 | lease reclaimed |
| 旧 worker late write | commit fence | `TASK_LEASE_LOST` | 不重试 | 不覆盖新 worker artifact/report | lease lost |
| 重复启动 | executor/API | 同键返回原 task；异键 `WORKFLOW_ALREADY_RUNNING` | 不适用 | 单 active task/单启动事件 | 数据库断言 |
| 病例 CAS conflict | `commit_stage` | 自动恢复 | 50/100/200ms，最多 3 次 | 失败事务不写 artifact/event | `optimistic_lock_retry_count` |

## 复现命令

```powershell
conda run -n medidiag python -m pytest -p no:cacheprovider tests/test_reliability_fault_injection.py -q
conda run -n medidiag python -m medidiag.cli trace-examples --output-root traces
```

`artifacts/traces/raw/*.jsonl` 与 `artifacts/traces/summary/*.json` 默认被 Git 忽略，应在本地按需生成。summary 只保存脱敏可靠性元数据；不得提交 API key、问题原文、完整 reasoning 或 provider 原始错误 body。

## 边界

- 所有 Provider 故障均由 fake/mock 离线注入，本步未发送真实网络请求。
- `CLOSED_SUCCESS` 只表示工程工作流成功，不代表医学结论正确。
- `ESCALATED` 是等待人工处理的中间态，不是普通成功，也不是 `CLOSED_ESCALATED`。
- P7 不产生 formal 指标；正式 RAG、citation、judge 和 topology 结论属于 P8。
