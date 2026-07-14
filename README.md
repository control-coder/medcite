# MediDiag-Agent EvidenceFlow

## Scope 声明

本项目是医疗循证诊断工作流的工程原型，只用于公开数据评测、脱敏模拟输入和软件工程演示，不用于真实医疗诊断、治疗决策或患者服务。

第一阶段目标只承诺：单机 MVP、可复现评测、任务崩溃恢复、结构化日志，以及 FastAPI + Jinja2 + HTMX 最小演示页。多 worker 横向扩展、生产级告警平台、真实医疗合规认证、真实患者数据、容器级隔离、生产 Dashboard、WebSocket 实时推送和全量公开基准跑分均属于二期预留或明确不在范围内。

当前状态：**核心机制原型已存在，正在按 `SUPPLEMENT_PLAN.md` 补齐产品与评测闭环，不能表述为“第一阶段完成”。** 截至 2026-07-13，本轮已开始 P0-A 评测真实性改造；真实 API/worker 闭环、正式 NLI 评测、双人标注、Kappa、完整 trace 和演示页仍未验收。

## Current Status

| 范围 | 当前证据 | 状态 |
|---|---|---|
| 状态机、数据模型、乐观锁、幂等与租约单元机制 | `src/medidiag/workflow/`、`src/medidiag/db/`、对应测试 | P0-B 已补复合唯一约束、active task CAS、lease/result 条件 UPDATE；真实 worker 仍未实现 |
| RAG/Agent 实验拆分 | `eval/config.yaml` 中 `rag_*` 与 `agent_*` | 本轮已实现；Conda 全量测试通过 |
| 配置唯一事实源 | Retriever 必须显式接收 YAML 权重、模型和实验开关 | 本轮已实现；Conda 全量测试通过 |
| judge 行为 | development 明示 `rule_fallback`；formal 强制固定 NLI 且 fail-closed | 本轮已实现门禁；尚无正式 NLI raw result |
| 指标口径 | Recall 仅统计 evidence-eligible 样本；另报 Gold Evidence Coverage；citation pair 与 claim 分母分离 | 本轮已实现；Conda 全量测试通过 |
| raw provenance | run ID、config snapshot/hash、dataset hash、Git/dirty hash、模型 revision、非报告原因 | 本轮已实现；尚未生成正式可报告 run |
| Agent 固定比较集 | `eval/datasets/agent_eval_manifest_v1.jsonl` 固定 100 个 MedQA v1 样本 | 本轮已生成并通过 schema/引用完整性测试 |

当前验证基线（Conda `medidiag`，2026-07-14）：

- P0-A 提交后全量测试：`223 passed in 24.39s`。
- P0-B 代码与迁移加入后全量测试：`235 passed in 27.95s`。
- 最后两条 active-task mismatch/旧 worker 恢复测试加入后，executor 定向测试：`36 passed in 1.15s`。
- 配置 CLI：`Configuration validation: OK`。
- 实际 leakage gate：300 个 PubMedQA 样本、1927 个 chunks，`OK: no data leakage detected`。

这些结果只证明当前自动化开发门禁通过，不代表正式 NLI、人工标注、真实 worker 端到端或第一阶段完成。
| API、单机 worker、扫描器、人工升级闭环 | 尚无可运行入口 | 未完成 |
| 正式人工复核与报告 | 尚无真实双人标注、裁决和稳定 Kappa | 未完成 |
| 最小演示页与结构化 trace | 目录/模型基础存在，未形成可运行展示 | 未完成 |

历史 `reports/raw/real/group_*.json` 及归档报告仅是 exploratory artifacts。它们使用旧 A-F 耦合实验和规则 judge，不得用于简历或正式指标。

## Upstream Reference

- `../edict/`：只参考事件驱动工作流、事件日志和可回放 trace 的工程组织思想，未把其代码或已有能力列为个人贡献。
- MedQA / PubMedQA：公开评测数据来源。PubMedQA 用于 evidence retrieval/citation；没有 gold evidence 的 MedQA 样本不进入 Evidence Recall 分母。
- MeSH：公开医学术语来源。当前词表包含轻量词典和数据集派生条目，不宣称具备 UMLS 级覆盖。

## Third-party Components

| 类别 | 组件 | 边界 |
|---|---|---|
| Generation | DeepSeek OpenAI-compatible API | 第三方模型调用与集成 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 第三方向量模型 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 第三方 cross-encoder |
| Judge | `microsoft/deberta-v3-base-mnli` | 正式模式计划使用的固定 NLI；当前没有正式结果 |
| Retrieval | FAISS、`rank-bm25` | 第三方索引和检索库 |
| Backend | FastAPI、SQLAlchemy、Alembic | 第三方框架；API 尚未实现 |
| Test/config | pytest、PyYAML | 测试与配置工具 |

第三方模型、框架和数据集只算集成，不算核心创新。

## My Contributions

以下内容有当前源码或测试文件支撑，但完成度以本 README 的 Current Status 为准：

- 面向医疗诊断场景的 14 状态执行器、触发主体与非法跳转校验。
- SQLAlchemy 数据模型、状态与事件同事务的执行器原型、幂等服务、乐观锁重试和任务租约原型。
- P0-B 数据库一致性补强：三组复合唯一约束、`active_task_id + version` 启动 CAS、不可复活的 lease renew、原子 reclaim/result CAS 与 `TASK_LEASE_LOST` 独立事件事务。
- 医学术语归一化、BM25/embedding/证据等级组合排序和 cross-encoder rerank 接口。
- 单 Agent、固定双专科、动态双专科和仲裁的实验组件。
- CitationVerifier、ClinicalLogicReviewer 与 ComplianceGuard 规则组件。
- 本轮新增的配置驱动评测门禁：实验族隔离、正式 judge fail-closed、eligible 指标分母、leakage 前置检查、run provenance 与探索性结果隔离。

不把尚未接入真实 worker 的组件描述为完整多 Agent 协作系统；在独立实验支持收益前，只称为“流水线式 Agent 编排 + 双专科仲裁实验”。

## Evaluation Contract

实验分为两个互不耦合的族：

- RAG：`rag_embedding`、`rag_bm25`、`rag_evidence_weight`、`rag_term_norm`、`rag_citation_review`、`rag_full`。前四个单变量组相对 pure embedding baseline 只改变一个开关；`rag_full` 只报告组合效果。
- Agent：`agent_single`、`agent_fixed_pair`、`agent_dynamic_pair`。三组固定使用 `rag_full`，只改变 Agent topology。

当前 `eval/config.yaml` 默认为 `evaluation.mode: development`，模型 revision 明示为 `development-unpinned`，judge 为 `rule_fallback`。该配置可用于开发，但 raw manifest 会写入 `report_eligible: false`。正式模式必须满足：

- 所有模型 revision 为不可变版本；
- `judge.method: nli`，模型加载或推理失败立即终止；
- 不使用 `--limit` 或 `--dry-run`；
- 固定 100 样本 Agent manifest 存在且通过 schema/唯一性校验；
- leakage gate 通过；
- 后续双人标注、裁决和 Kappa 门禁通过。

核心指标口径：

| 指标 | 口径 |
|---|---|
| Evidence Recall@5 | eligible 样本中 top-5 至少命中一条 gold evidence 的样本数 / `gold_evidence_ids` 非空样本数 |
| Gold Evidence Coverage | `gold_evidence_ids` 非空样本数 / 全部样本数 |
| Citation Precision | `SUPPORTED` claim-citation pair 数 / 所有实际输出的 claim-citation pair 数 |
| Unsupported Claim Rate | 最佳有效 citation 仍为 `UNSUPPORTED` 的 claim 数 / 全部 claim 数 |
| Workflow Success Rate | 持久化终态为 `CLOSED_SUCCESS` 的病例数 / 全部工作流病例数；当前 eval runner 不生成该值 |

## Development Commands

```powershell
conda activate medidiag
pip install -e ".[dev]"

python -m pytest -q
python -m eval.runner --config eval/config.yaml --validate
python -m eval.runner --config eval/config.yaml --show-config

# 开发态检索检查：结果明确不可用于正式报告
python -m eval.runner --config eval/config.yaml --experiment rag_all --dry-run --limit 20

# 独立 leakage gate
python -m eval.leakage_check `
  --config eval/config.yaml `
  --eval-set eval/datasets/eval_set_pubmedqa.jsonl `
  --kb eval/datasets/knowledge_chunks.jsonl
```

评测输出写入 `reports/raw/<run_id>/`。每个 run 包含 config snapshot、manifest 和各实验原始结果；`eval/report.py` 会拒绝从 `report_eligible: false` 的 run 生成正式报告。

## Documents

- `SUPPLEMENT_PLAN.md`：当前直接实施与验收依据。
- `overview.md`：本轮实施记录、验证状态和下一步。
- `docs/architecture.md`：当前实现边界与目标数据流。
- `docs/design-decisions.md`：评测模式、实验隔离、judge 与结果资格决策。
- `docs/archive/overview_historical_2026-07-06.md`：历史阶段记录，不代表当前验收结论。
- `reports/archive/`：旧探索性报告，不得用于简历。

## License

MIT。仅用于工程演示与公开数据评测。
