# MediDiag-Agent EvidenceFlow

医疗循证诊断多 Agent 工作流平台（工程原型）。核心不是医疗 prompt 换皮，而是**证据前置检索、医学 RAG 单变量消融、状态机一致性、任务租约恢复、审核驳回闭环、合规输出管控和公开基准评测**。

> 模拟项目：只用于公开数据评测、脱敏模拟输入和软件工程演示，**不用于真实医疗诊断、治疗决策或患者服务**。

## Scope 声明

第一阶段只承诺：单机 MVP、可复现评测、任务崩溃恢复、结构化日志，以及 FastAPI + Jinja2 + HTMX 最小演示页。

二期预留或明确不在范围内：多 worker 横向扩展、生产级告警平台、真实医疗合规认证、真实患者数据、容器级隔离、生产 Dashboard、WebSocket 实时推送、全量公开基准跑分。

**当前状态**：工程门禁全绿（452 测试通过、ruff/mypy 阻断级通过）；首次 formal run 已完成（2026-07-29，9 组实验、1100 次付费调用、实测 1.7509 元），但 `report_eligible: false`——唯一阻断是缺真实双人 citation 人工校准。**项目尚不能表述为"第一阶段完成"，现有指标不得写入简历。**详见 [doc/status.md](doc/status.md)。

## 核心特性

- **证据前置工作流**：14 状态状态机（`CREATED → NORMALIZED → EVIDENCE_RETRIEVED → PLAN_GENERATED → SPECIALIST_REVIEWING → ARBITRATION_REVIEWING → APPROVED/REVISION_REQUIRED/ESCALATED → REPORT_GENERATED → CLOSED_*`），诊断规划前必须先检索证据。
- **一致性机制**：状态变更与事件日志同事务；外部 IO 在事务外；乐观锁 + 幂等键 + 任务租约（心跳续期、超时接管、迟到写入拒绝并记 `TASK_LEASE_LOST`）。
- **医学 RAG**：术语归一化、BM25 + embedding + 证据等级加权 + cross-encoder rerank，组件级单变量消融。
- **审核闭环**：每个 claim 绑定 citation，固定 NLI judge 判定 `SUPPORTED / PARTIAL / UNSUPPORTED`；连续审核失败进入人工升级，人工从 `ESCALATED` 三向回流。
- **合规管控**：超范围拒答、绝对化措辞拦截、强制风险提示，合规命中写入 trace/event log。
- **可复现评测**：配置唯一事实源、leakage 前置硬门禁、raw provenance（config/dataset/Git hash、模型 revision、response ID）、双人标注 + Cohen's Kappa + 分歧裁决通过后才允许生成正式报告。

## 快速开始

```powershell
conda activate medidiag
pip install -e ".[dev]"

# 测试与静态检查（与 CI 三步一一对应）
python -m ruff check .
python -m pytest -q --cov=medidiag --cov=eval --cov-report=term-missing
python -m mypy src

# 评测配置校验（development 模式，输出不可用于正式报告）
python -m eval.runner --config eval/config.yaml --validate
python -m eval.runner --config eval/config.yaml --show-config

# 本地工程闭环与演示页（http://127.0.0.1:8400/demo）
alembic upgrade head
medidiag demo            # 确定性 provider，无需联网
medidiag demo --provider deepseek   # 需 .env 配置 DEEPSEEK_API_KEY
```

注意：必须用 conda `medidiag` 环境而不是 base（base 缺 `sentence_transformers` 与 `faiss`）。复现 CI 离线行为时设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。

更多命令（leakage gate、prompt 体积、路由诊断、费用折算、trace 导出、人工审计）见 [doc/handoff.md](doc/handoff.md) 与 [doc/evaluation_protocol.md](doc/evaluation_protocol.md)。

## MVP API

| Method | Path | 作用 |
|---|---|---|
| POST | `/api/v1/cases` | 使用 `Idempotency-Key` 创建公开/脱敏模拟病例 |
| POST | `/api/v1/cases/{case_id}/workflow` | 幂等启动或恢复工作流 |
| GET | `/api/v1/cases/{case_id}` | 查询 version、active task 和人工动作 |
| GET | `/api/v1/cases/{case_id}/events` | 使用稳定 event ID cursor 分页 |
| GET | `/api/v1/cases/{case_id}/report` | 获取已生成的结构化报告 |
| POST | `/api/v1/cases/{case_id}/human-decisions` | 仅从 `ESCALATED` 执行三类人工决策 |

## 文档导航

| 文档 | 内容 |
|---|---|
| [doc/charter.md](doc/charter.md) | 项目章程：定位、范围、质量底线、评测与合规规则 |
| [doc/structure.md](doc/structure.md) | 项目结构详解：目录、架构图、状态机、事务边界 |
| [doc/progress.md](doc/progress.md) | 项目进度：阶段路线图与各切片完成状态 |
| [doc/status.md](doc/status.md) | 项目状态：当前验收快照、formal run 结果、未验收项 |
| [doc/log.md](doc/log.md) | 项目日志与决策记录：时间线 + DD-001~DD-027 |
| [doc/handoff.md](doc/handoff.md) | 续作交接：当前基线、环境、后续执行顺序 |
| [doc/evaluation_protocol.md](doc/evaluation_protocol.md) | 正式评测与人工 citation 复核协议 |
| [reports/README.md](reports/README.md) | 评测产物目录说明（raw/archive/诊断产物口径） |
| [eval/annotations/README.md](eval/annotations/README.md) | 人工标注文件 schema 与填写规则 |
| [examples/p1a_trace_cases.md](examples/p1a_trace_cases.md) | 可复现 trace 工程案例 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | vendored 前端资产版本、哈希与许可证 |

## Upstream Reference

- `../edict/`：只参考事件驱动工作流、事件日志和可回放 trace 的工程组织思想，未把其代码或已有能力列为个人贡献。
- MedQA / PubMedQA：公开评测数据来源。PubMedQA 用于 evidence retrieval/citation；没有 gold evidence 的 MedQA 样本不进入 Evidence Recall 分母。
- MeSH：公开医学术语来源。当前词表包含轻量词典和数据集派生条目，不宣称具备 UMLS 级覆盖。

## Third-party Components

| 类别 | 组件 | 边界 |
|---|---|---|
| Generation | DeepSeek OpenAI-compatible API（`deepseek-v4-flash`） | 第三方模型调用与集成；provider snapshot 不可核验，使用 `provider_response_id` 受限溯源 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 第三方向量模型（HF commit SHA 锁定） |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 第三方 cross-encoder（HF commit SHA 锁定） |
| Judge | `cross-encoder/nli-MiniLM2-L6-H768` | 固定 NLI judge（HF commit SHA 锁定） |
| Retrieval | FAISS、`rank-bm25` | 第三方索引和检索库 |
| Backend | FastAPI、SQLAlchemy、Alembic | 第三方框架 |
| Demo UI | Jinja2、HTMX 2.0.4 | 服务端模板与局部刷新；HTMX 以固定本地 BSD 2-Clause 资产集成 |
| Test/config | pytest、PyYAML | 测试与配置工具 |

第三方模型、框架和数据集只算集成，不算核心创新。

## My Contributions

以下每一项均可被源码、测试或评测产物验证（完成度以 [doc/status.md](doc/status.md) 为准）：

- 14 状态执行器、触发主体与非法跳转校验；状态与事件同事务、幂等、乐观锁重试、任务租约（含心跳与 `TASK_LEASE_LOST` 脑裂防护）。
- 数据库一致性补强：复合唯一约束、`active_task_id + version` 启动 CAS、原子 reclaim/result CAS。
- 单机闭环：六个 FastAPI API、配置化 worker/lease scanner、阶段产物、人工升级回流、结构化报告、崩溃接管恢复。
- Provider 调用可靠性边界：七类阶段 schema、timeout/429/瞬时 5xx 有限重试、provider request ID 与 retry decision 事件审计。
- Trace exporter：统一 trace schema、raw/summary 反向关联、脱敏；四类确定性工程案例（含双专科无明确收益负向 fixture）。
- 最小演示页：Jinja2 + 本地 HTMX、2 秒局部轮询、人工处置、证据/citation/审核/报告视图、无 JS fallback。
- 医学术语归一化、BM25/embedding/证据等级组合排序、cross-encoder rerank；单 Agent / 固定双专科 / 动态双专科与仲裁实验组件。
- 配置驱动评测门禁：实验族隔离、formal judge fail-closed、eligible 指标分母、leakage 前置检查、run provenance、双人标注/Kappa/裁决审计与报告阻断。

在独立实验支持收益前，只称为"流水线式 Agent 编排 + 双专科仲裁实验"，不称为多 Agent 协作系统。

## License

MIT，全文见 [LICENSE](LICENSE)。仅用于工程演示与公开数据评测。
