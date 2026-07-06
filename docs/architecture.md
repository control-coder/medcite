# Architecture

> 阶段 0 占位文档。完整架构说明将在阶段 6 补充。

## 高层组件

- **API 层**（FastAPI）：病例创建、工作流启动、状态查询、人工干预入口。
- **状态机执行器**（`workflow/state_machine.py`）：14 状态 + 合法跳转表 + 触发主体。
- **任务执行器**（worker + 租约）：本地任务表 + lease + 幂等键 + 乐观锁。
- **医学 RAG**（`rag/`）：术语归一化 + BM25 + embedding + rerank + 证据等级加权。
- **Agent 编排**（`agents/`）：病例归一化 / 证据检索 / 诊断生成 / 双专科并行 / 仲裁。
- **审核模块**（`review/` + `compliance/`）：CitationVerifier / ClinicalLogicReviewer / ComplianceGuard / ArbitrationAgent。
- **评测**（`eval/`）：config.yaml 锁定变量 + runner + leakage_check + 报告生成。
- **可观测性**：错误码分级 + 分阶段 latency 埋点 + event log。

## 事务边界

- 数据库事务只负责 `cases` / `workflow_tasks` / `case_event_log` 的状态推进与事件追加。
- 外部 IO（RAG / LLM / judge）在事务外执行，结果通过二次事务写入。
- 状态变更与事件日志写入必须在同一事务内完成。

## 数据流

```
API -> [事务: 写 cases + workflow_tasks + event_log]
    -> worker 领取任务 [事务: PENDING -> RUNNING + lease]
    -> 外部 IO (RAG / LLM / judge)
    -> [事务: 二次校验 lease + 写结果 + 推进状态 + 追加 event]
```

详细设计见 `design-decisions.md`（阶段 6 补充）。
