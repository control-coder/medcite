"""SQLAlchemy 数据模型占位。

阶段 2 实现：
- cases（含 version 乐观锁字段）
- workflow_tasks（含 lease_owner / lease_until / heartbeat_at / attempt）
- case_event_log（状态变更与事件同事务）
- agent_runs（Agent 执行记录，含 input_hash 幂等）
- citations（claim 与 evidence 绑定）
- reviews（审核记录）
- evidence_chunks（知识库 chunk，source/source_id/metadata.raw_id 不得含测试样本 ID）

事务边界：
- 状态推进 + 事件追加必须在同一事务内
- 外部 IO（RAG/LLM/judge）不进事务
"""

from __future__ import annotations


# TODO[阶段2]: 定义 Base = declarative_base()
# TODO[阶段2]: 实现 Case / WorkflowTask / CaseEventLog / AgentRun / Citation / Review / EvidenceChunk 模型
