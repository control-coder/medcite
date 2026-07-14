# Architecture

> 更新日期：2026-07-13。本文区分当前已实现组件和一期目标链路。

## 当前实现边界

当前仓库包含：

- 状态机与执行器原型：14 状态、触发主体、非法跳转、状态+事件事务接口。
- 数据层原型：cases、workflow_tasks、case_event_log、agent_runs、citations、reviews。
- 并发机制原型：幂等服务、乐观锁重试、租约 acquire/renew/reclaim 与迟到写入检查。
- RAG/Agent 组件：术语归一化、BM25、embedding、evidence weighting、rerank、单/双 Agent 和仲裁。
- 审核组件：citation、clinical logic、compliance 规则。
- 评测链路：配置验证、实验隔离、leakage gate、raw provenance 和指标聚合。

当前不存在可运行 FastAPI app、真实单机 worker/lease scanner、case_reports 持久化、完整 trace exporter 或演示页。以下目标图不能解释为已经交付。

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

## 一期持久化目标

- normalize artifact：归一化输入与词表版本。
- retrieval artifact：query、top-k、各分数、模型和配置 hash。
- agent_runs：输入 hash、attempt group、结构化输出与 provider request ID。
- citations：claim-citation pair 和 judge metadata。
- reviews：轮次、问题、判定与升级原因。
- case_reports：结构化报告、风险提示、合规状态和生成版本。
- case_event_log：append-only 事件，不作为业务结果的唯一存储。

## 部署与隐私边界

- 一期单机、SQLite、本地 worker；不证明多 worker 生产扩展。
- 只处理公开或脱敏模拟数据，不接入医院系统和真实患者档案。
- API key、未脱敏文本和个人身份信息不得进入 raw result、event 或 trace。
