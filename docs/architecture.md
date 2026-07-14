# Architecture

> 更新日期：2026-07-14。本文区分当前已实现组件和一期目标链路。

## 当前实现边界

当前仓库包含：

- 状态机与执行器原型：14 状态、触发主体、非法跳转、状态+事件事务接口。
- 数据层：cases、workflow_tasks、case_event_log、agent_runs、citations、reviews、stage_artifacts、case_reports。
- 并发机制原型：幂等服务、乐观锁重试、租约 acquire/renew/reclaim 与迟到写入检查。
- RAG/Agent 组件：术语归一化、BM25、embedding、evidence weighting、rerank、单/双 Agent 和仲裁。
- 审核组件：citation、clinical logic、compliance 规则。
- 评测链路：配置验证、实验隔离、leakage gate、raw provenance 和指标聚合。
- P0-C 单机闭环：六个 FastAPI API、单机 worker、lease scanner、人工回流和结构化报告。
- Provider runtime：阶段 schema、timeout/HTTP 错误映射、有限退避和逐 attempt 事件审计。

当前 P0-C 只接入确定性非诊断 provider，用于验证事务、恢复和 API 契约。可靠性边界可包裹后续真实 adapter，但真实 RAG/LLM/judge adapter、成功调用缓存、完整 trace exporter 和演示页尚未交付。

## 一期目标链路

```text
FastAPI/CLI
  -> case service (idempotency + optimistic CAS)
  -> workflow_tasks (one active task per case)
  -> single-machine worker acquires lease
  -> external stages outside transaction
       normalize -> retrieve -> rerank -> generate -> judge -> review
  -> atomic result CAS (owner + attempt + RUNNING + lease_until)
  -> stage artifacts + state transition + append-only event
  -> report generation or ESCALATED human wait state
```

## 事务边界

1. 短事务创建/领取任务，提交状态与 event。
2. 事务外执行 RAG/LLM/judge；所有调用设置 timeout、错误码与有限重试。
   - timeout 按 stage 映射为 RAG/LLM/judge 错误；429 与瞬时 5xx 有限重试。
   - schema 与其他 4xx 不自动重试；未知异常保留 crash/reclaim 语义。
   - 每次 attempt 单独追加 request ID、latency、error code 与 retry decision 事件。
3. 单条条件 UPDATE 校验 task ID、owner、attempt、RUNNING 和有效 lease。
4. 命中后在同一事务写阶段产物、推进 case version/status 并追加 event。
5. 未命中则丢弃旧结果，并在独立事务追加 `TASK_LEASE_LOST`。

步骤 3-5 的数据库原子 CAS 已在 P0-B 落盘并由模型、executor、lease 与迁移测试覆盖。它只证明单机 SQLite 条件更新语义，不代表多 worker 生产部署能力。

## 评测数据流

```text
eval/config.yaml
  -> schema/semantic validation
  -> leakage gate
  -> select rag_* or agent_* experiment family
  -> explicit Retriever/LLM/Judge construction
  -> per-sample raw records
  -> run manifest + config snapshot + hashes
  -> report eligibility gate
  -> formal report (only when all gates pass)
```

RAG 和 Agent 实验不能复用同一标识：

- `rag_*` 只评估检索/审核组件，生成 topology 固定为 single。
- `agent_*` 固定 `rag_full`，只比较 single/fixed_pair/dynamic_pair。

## 当前 raw 语义

- `pipeline_approval_rate`：规则/审核流水线是否通过，仅用于组件实验。
- `workflow_success_rate`：必须来自数据库终态 `CLOSED_SUCCESS`；当前 eval runner 输出 `null`。
- development run：允许快速验证，但 manifest 明确不可报告。
- formal run：固定 NLI 和不可变版本，禁止 fallback、limit 与 dry-run。

## 当前持久化产物

- normalize artifact：归一化输入与词表版本。
- retrieval artifact：query、top-k、各分数、模型和配置 hash。
- agent_runs：输入 hash、attempt group 和结构化输出。
- citations：claim-citation pair 和 judge metadata。
- reviews：轮次、问题、判定与升级原因。
- case_reports：结构化报告、风险提示、合规状态和生成版本。
- case_event_log：append-only 事件，不作为业务结果的唯一存储。

`provider_call` 事件保存 trace/task 关联、provider version、provider attempt、request ID、latency、错误码、retryable、retry decision 和 HTTP status，不保存 API key 或原始病例文本。成功阶段的 `stage_completed` 事件额外保存最终 request ID 与 retry count。

每个 stage artifact 使用 `(case_id, task_id, stage, attempt)` 唯一约束，并保存 input/output hash、component version 和 latency。provider 调用发生在事务外；写入由 `commit_stage()` 将 lease fence、case CAS、artifact 和 event 合并进同一事务。

## 部署与隐私边界

- 一期单机、SQLite、本地 worker；不证明多 worker 生产扩展。
- 只处理公开或脱敏模拟数据，不接入医院系统和真实患者档案。
- API key、未脱敏文本和个人身份信息不得进入 raw result、event 或 trace。
