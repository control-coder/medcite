# MediDiag-Agent EvidenceFlow

## Scope 声明

本项目是医疗循证诊断工作流的工程原型，只用于公开数据评测、脱敏模拟输入和软件工程演示，不用于真实医疗诊断、治疗决策或患者服务。

第一阶段目标只承诺：单机 MVP、可复现评测、任务崩溃恢复、结构化日志，以及 FastAPI + Jinja2 + HTMX 最小演示页。多 worker 横向扩展、生产级告警平台、真实医疗合规认证、真实患者数据、容器级隔离、生产 Dashboard、WebSocket 实时推送和全量公开基准跑分均属于二期预留或明确不在范围内。

当前状态：**P0-A/P0-B 已完成开发门禁，P0-C 已形成确定性 provider 下的单机 API/worker 工程闭环，P1-A/P1-B 已交付结构化 trace 与最小演示页；P2 已补齐正式 raw run 后的双人 citation 标注、Kappa、裁决与自动报告门禁。项目仍不能表述为“第一阶段完成”。** generation 已切换到 DeepSeek 官方 API，并增加自动上下文缓存遥测；真实 RAG、完整正式 NLI raw result、真实双人标注、稳定 Kappa 和固定检索条件下的 Agent 拓扑正式对照仍未验收。


本轮已将 Hugging Face embedding、rerank 和 judge 锁定到可访问的完整 commit SHA；generation 固定为 `https://api.deepseek.com` 与 `deepseek-v4-flash`。DeepSeek 当前响应未提供可核验的不变 snapshot，因此 formal 配置使用 `provider_response_id` 受限溯源并要求逐样本保存非敏感 `response.id`；模板与本地 formal 配置均可通过配置校验，但尚未运行完整正式 NLI raw run，也没有真实人工标签。
> 代码级完成项、验收缺口和工作区清理策略见 [`docs/current-status.md`](docs/current-status.md)。

## Current Status

| 范围 | 当前证据 | 状态 |
|---|---|---|
| 状态机、数据模型、乐观锁、幂等与租约单元机制 | `src/medidiag/workflow/`、`src/medidiag/db/`、对应测试 | P0-B 条件更新已验证；P0-C worker 已复用同一 CAS 边界 |
| RAG/Agent 实验拆分 | `eval/config.yaml` 中 `rag_*` 与 `agent_*` | 本轮已实现；Conda 全量测试通过 |
| 配置唯一事实源 | Retriever 必须显式接收 YAML 权重、模型和实验开关 | 本轮已实现；Conda 全量测试通过 |
| judge 行为 | development 明示 `rule_fallback`；formal 强制固定 NLI 且 fail-closed | 本轮已实现门禁；尚无正式 NLI raw result |
| 指标口径 | Recall 仅统计 evidence-eligible 样本；另报 Gold Evidence Coverage；citation pair 与 claim 分母分离 | 本轮已实现；Conda 全量测试通过 |
| raw provenance | run ID、config snapshot/hash、dataset hash、Git/dirty hash、模型 revision、非报告原因 | 本轮已实现；尚未生成正式可报告 run |
| Agent 固定比较集 | `eval/datasets/agent_eval_manifest_v1.jsonl` 固定 100 个 MedQA v1 样本 | 本轮已生成并通过 schema/引用完整性测试 |
| API、单机 worker、扫描器、人工升级闭环 | 六个 FastAPI API、`worker`/`lease-scan` CLI、阶段产物和结构化报告 | 确定性闭环已覆盖；实时 DeepSeek 起草 adapter 已接入，正式 RAG/NLI 仍待验收 |
| Provider 调用可靠性 | Pydantic stage schema、timeout/429/5xx 分类、有限退避、request ID 与逐次调用事件 | 已通过定向和全量测试 |
| 正式人工复核与报告 | `eval.annotation_audit` 提供 20% 分层抽样、双人标注/Kappa/裁决审计及报告阻断 | 工程门禁已实现；尚无真实标注或正式报告 |
| 结构化 trace 与工程案例 | JSONL raw + 脱敏 summary、成功/租约恢复/审核升级/双专科无明确收益四类确定性案例 | 工程链路已通过测试；第四类是负向 fixture，不替代正式 Agent 拓扑实验 |
| 最小演示页 | FastAPI + Jinja2 + 本地 HTMX，含轮询、人工处置、证据/citation/报告视图；`medidiag demo` 同进程启动 worker | DeepSeek/确定性 provider 均可选择；实时路径只使用本地 fixture，不产生评测指标 |

当前验证基线（Conda `medidiag`，截至 2026-07-23）：

- P0-A 提交后全量测试：`223 passed in 24.39s`。
- P0-B 代码与迁移加入后全量测试：`235 passed in 27.95s`。
- 最后两条 active-task mismatch/旧 worker 恢复测试加入后，executor 定向测试：`36 passed in 1.15s`。
- P0-C 最终全量测试：`255 passed in 29.94s`；外部 IO 崩溃恢复 worker 定向测试：`5 passed in 1.98s`。
- Provider runtime + worker 定向测试：`13 passed in 2.63s`；加入该切片后全量测试：`264 passed, 1 warning in 29.64s`。
- Trace exporter + worker/executor 定向测试：`44 passed in 4.86s`；加入 P1-A 首个切片后全量测试：`266 passed, 1 warning in 30.43s`。
- `trace-examples` 实际生成 4 组 `CLOSED_SUCCESS` trace；除成功、租约恢复和审核升级外，新增 cardiology / pulmonology 双专科无明确收益负向 fixture，并记录 `NO_CLEAR_SPECIALIST_GAIN` 仲裁结果。JSONL/summary 解析、event ID 反向关联、stage task ID、租约/人工事件和敏感字面量扫描均通过。
- P1-B 页面/API/worker 定向测试：`19 passed, 1 warning in 8.13s`；最终全量测试：`270 passed, 1 warning in 35.10s`。
- 浏览器 QA：1440px/390px 下首页、活动态、`CLOSED_SUCCESS` 和 `ESCALATED` 均无横向溢出；HTMX 本地加载、轮询停止、人工表单、citation verdict、报告和无 JS 303 fallback 通过，控制台 0 error。
- 配置 CLI：`Configuration validation: OK`。
- 实际 leakage gate：300 个 PubMedQA 样本、1927 个 chunks，`OK: no data leakage detected`。
- P2 人工 citation 审计切片（2026-07-17）：`15 passed in 0.94s`；只验证模板、双人标注/Kappa/裁决与报告阻断逻辑，不产生正式医学指标。
- P2 审计切片纳入后的全量回归（2026-07-17）：`274 passed, 1 warning in 33.27s`。
- 实时 DeepSeek adapter 单测（2026-07-17）：`3 passed in 1.00s`；与 provider runtime、worker、演示页的回归为 `17 passed, 1 warning in 4.53s`。
- 官方接口 smoke（2026-07-23）：以非医疗 JSON 请求验证 `https://api.deepseek.com/chat/completions`、`deepseek-v4-flash`、响应 ID 和 usage 字段可用；未输出或保存 API key、原始回复和病例数据。`thinking` 显式关闭，客户端不再发送未确认兼容性的 `seed`。
- 自动缓存 smoke（2026-07-23）：两次调用共享长稳定前缀，第二次返回 `1408` cache-hit tokens 与 `48` cache-miss tokens；该结果只证明 DeepSeek 自动缓存字段和公共前缀策略可用，不是医学样本或正式项目指标。
- 本轮完整回归（2026-07-23）：`296 passed, 1 warning in 27.81s`。测试注入确定性 token-hashing embedding encoder，不再自动访问 Hugging Face；正式 runner 仍使用配置锁定的第三方 embedding 模型。

warning 是 FastAPI/Starlette TestClient 当前 httpx adapter 的弃用提示，不是行为失败。这些结果只证明当前自动化开发门禁、确定性 provider 工程闭环和 provider 调用边界通过，不代表真实模型工作流、正式 NLI、人工标注或第一阶段完成。

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
| Judge | `cross-encoder/nli-MiniLM2-L6-H768` | 已锁定的固定 NLI judge；当前没有正式结果 |
| Retrieval | FAISS、`rank-bm25` | 第三方索引和检索库 |
| Backend | FastAPI、SQLAlchemy、Alembic | 第三方框架；六个 MVP API 已集成 |
| Demo UI | Jinja2、HTMX 2.0.4 | 服务端模板与局部刷新；HTMX 以固定本地 BSD 2-Clause 资产集成 |
| Test/config | pytest、PyYAML | 测试与配置工具 |

第三方模型、框架和数据集只算集成，不算核心创新。

## My Contributions

以下内容有当前源码或测试文件支撑，但完成度以本 README 的 Current Status 为准：

- 面向医疗诊断场景的 14 状态执行器、触发主体与非法跳转校验。
- SQLAlchemy 数据模型、状态与事件同事务的执行器原型、幂等服务、乐观锁重试和任务租约原型。
- P0-B 数据库一致性补强：三组复合唯一约束、`active_task_id + version` 启动 CAS、不可复活的 lease renew、原子 reclaim/result CAS 与 `TASK_LEASE_LOST` 独立事件事务。
- P0-C 单机闭环：六个 FastAPI API、配置化 worker/lease scanner、阶段产物、人工升级回流、结构化报告和崩溃接管恢复。
- Provider 调用可靠性边界：七类阶段 schema、timeout/429/瞬时 5xx 有限重试、非重试错误拒绝、provider request ID 和 retry decision 事件审计。
- P1-A trace exporter：统一 trace schema、事件与 artifact/agent run 关联、敏感键与直接标识符脱敏、raw/summary 反向关联和原子文件写入。
- 四类确定性工程案例：正常成功、租约 reclaim + 旧写入拒绝、审核升级 + 人工批准恢复、双专科无明确收益负向 fixture；不把它们解释为医学效果或多 Agent 收益。
- P1-B 最小演示页：病例创建/最近列表、状态与时间线、2 秒局部轮询、升级人工处置、恢复任务、证据/citation/审核/报告展示和无 JS POST fallback。
- 医学术语归一化、BM25/embedding/证据等级组合排序和 cross-encoder rerank 接口。
- 单 Agent、固定双专科、动态双专科和仲裁的实验组件。
- CitationVerifier、ClinicalLogicReviewer 与 ComplianceGuard 规则组件。
- 本轮新增的配置驱动评测门禁：实验族隔离、正式 judge fail-closed、eligible 指标分母、leakage 前置检查、run provenance 与探索性结果隔离。

不把尚未接入真实 provider 的组件描述为完整多 Agent 协作系统；在独立实验支持收益前，只称为“流水线式 Agent 编排 + 双专科仲裁实验”。

## Evaluation Contract

实验分为两个互不耦合的族：

- RAG：`rag_embedding`、`rag_bm25`、`rag_evidence_weight`、`rag_term_norm`、`rag_citation_review`、`rag_full`。前四个单变量组相对 pure embedding baseline 只改变一个开关；`rag_full` 只报告组合效果。
- Agent：`agent_single`、`agent_fixed_pair`、`agent_dynamic_pair`。三组固定使用 `rag_full`，只改变 Agent topology。

当前 `eval/config.yaml` 默认为 `evaluation.mode: development`：embedding、rerank 和本地 NLI judge 使用已下载的固定 Hugging Face commit SHA，generation 保持 `deepseek-v4-flash` 且标记为 `development-unpinned`，judge 为 `rule_fallback`。该配置可用于开发，但 raw manifest 会写入 `report_eligible: false`。正式模式必须满足：

- Hugging Face embedding、rerank 和 NLI judge 使用不变 commit SHA；generation 使用声明的 provider 模型标识。
- 当 provider 不提供 `system_fingerprint` / snapshot 时，可使用 `provider_response_id` 模式；必须记录每条 Agent 样本的 `response.id`，并在报告中明示不具备不变快照。
- `judge.method: nli`，模型加载或推理失败立即终止；
- 不使用 `--limit` 或 `--dry-run`；
- 固定 100 样本 Agent manifest 存在且通过 schema/唯一性校验；
- leakage gate 通过；
- run 后从实际 claim-citation pair 创建不少于 20% 的双人复核样本；审计、裁决和 Kappa 门禁通过。

核心指标口径：

| 指标 | 口径 |
|---|---|
| Evidence Recall@5 | eligible 样本中 top-5 至少命中一条 gold evidence 的样本数 / `gold_evidence_ids` 非空样本数 |
| Gold Evidence Coverage | `gold_evidence_ids` 非空样本数 / 全部样本数 |
| Citation Precision | `SUPPORTED` claim-citation pair 数 / 所有实际输出的 claim-citation pair 数 |
| Unsupported Claim Rate | 最佳有效 citation 仍为 `UNSUPPORTED` 的 claim 数 / 全部 claim 数 |
| Workflow Success Rate | 持久化终态为 `CLOSED_SUCCESS` 的病例数 / 全部工作流病例数；当前 eval runner 不生成该值 |

## Development Commands

### 在本地跑 CI 的同一组检查

`.github/workflows/ci.yml` 只在 `push` 到 `main` 和 `pull_request` 时触发。本仓库
目前没有配置 remote，全部工作都在特性分支上进行，因此 CI 实际上不会运行——落盘前
必须在本地跑完同样三步。三步与 CI 中的步骤逐条对应：

```powershell
conda activate medidiag

# 1. ruff：阻断
python -m ruff check .

# 2. pytest + 覆盖率：阻断（覆盖率只报告，不设阈值门禁）
python -m pytest -q --cov=medidiag --cov=eval --cov-report=term-missing

# 3. mypy：自 2026-07-26 起阻断（存量已清零）
python -m mypy src
```

CI 另外设置了 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，使任何非预期的
Hugging Face 下载变成显式失败而不是缓慢的网络依赖式通过。想完全复现 CI 环境时
在本地一并设置：

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
```

注意用 conda `medidiag` 环境而不是 base：base 缺少 `sentence_transformers` 与
`faiss`，会在 `tests/test_rag.py` 产生 12 个与代码无关的错误。

### 其余命令

```powershell
pip install -e ".[dev]"

python -m pytest -q
python -m eval.runner --config eval/config.yaml --validate
python -m eval.runner --config eval/config.yaml --show-config

# 开发态检索检查：结果明确不可用于正式报告
python -m eval.runner --config eval/config.yaml --experiment rag_retrieval --dry-run --limit 20

# 独立 leakage gate
python -m eval.leakage_check `
  --config eval/config.yaml `
  --eval-set eval/datasets/eval_set_pubmedqa.jsonl `
  --kb eval/datasets/knowledge_chunks.jsonl

# 工程测量工具：均不调用 provider、不产生评测指标
# prompt 体积（formal run 起飞前门禁之一）
python -m eval.prompt_size_diagnostics --config eval/config.formal.yaml
# 路由代码行为（分支分布与分项得分，不是路由准确率）
python -m eval.routing_diagnostics --config eval/config.formal.yaml
# 费用折算：读上面的实测字符数 × DeepSeek 官方单价，输出上限与下界
python -m eval.cost_estimate
# run 完成后回填实测 token 与实际费用
python -m eval.cost_estimate --run-dir reports/raw/<run_id>

# P0-C 本地工程闭环（当前 worker 使用确定性非诊断 provider）
alembic upgrade head
uvicorn medidiag.api.app:app --host 127.0.0.1 --port 8400
medidiag worker --once
medidiag lease-scan --once

# 人工升级页面的确定性开发场景
medidiag worker --once --review-verdict ESCALATED

# Provider 可靠性与 worker 审计定向测试
python -m pytest tests/test_provider_runtime.py tests/test_worker.py -q

# P1-A：导出已有病例或生成四类确定性工程案例
medidiag trace-export --case-id <case_id> --output-root traces
medidiag trace-examples --output-root traces
```

```powershell
# P2：正式 NLI run 后的人工 citation 复核与自动报告
# 详细 schema、人工填写规则与正式配置步骤见 docs/evaluation_protocol.md
python -m eval.annotation_audit prepare `
  --run-dir reports/raw/<run_id> `
  --output eval/annotations/citation_sample_v1.jsonl --ratio 0.20 --seed 42
python -m eval.annotation_audit audit `
  --run-dir reports/raw/<run_id> `
  --sample eval/annotations/citation_sample_v1.jsonl `
  --annotator-a eval/annotations/citation_labels_a_v1.jsonl `
  --annotator-b eval/annotations/citation_labels_b_v1.jsonl `
  --adjudication eval/annotations/citation_adjudication_v1.jsonl `
  --output reports/raw/<run_id>/citation_annotation_audit.json
python -m eval.annotation_audit report `
  --run-dir reports/raw/<run_id> `
  --audit reports/raw/<run_id>/citation_annotation_audit.json `
  --output reports/final_eval.md
```

评测输出写入 `reports/raw/<run_id>/`。每个 run 包含 config snapshot、manifest 和各实验原始结果。即使满足 formal 配置，runner 也只标记 `formal_candidate: true`，不会直接设置 `report_eligible`；只有 `eval.annotation_audit` 验证实际输出的 20% 双人复核、Kappa 与所有分歧裁决后，`eval/report.py` 才允许生成正式报告。

## MVP API

| Method | Path | 作用 |
|---|---|---|
| POST | `/api/v1/cases` | 使用 `Idempotency-Key` 创建公开/脱敏模拟病例 |
| POST | `/api/v1/cases/{case_id}/workflow` | 幂等启动或恢复工作流 |
| GET | `/api/v1/cases/{case_id}` | 查询 version、active task 和人工动作 |
| GET | `/api/v1/cases/{case_id}/events` | 使用稳定 event ID cursor 分页 |
| GET | `/api/v1/cases/{case_id}/report` | 获取已生成的结构化报告 |
| POST | `/api/v1/cases/{case_id}/human-decisions` | 仅从 `ESCALATED` 执行三类人工决策 |

`DeepSeekWorkflowProvider` 已通过官方 OpenAI-compatible `/chat/completions` 接入实时起草：默认从 `.env` 读取 `DEEPSEEK_BASE_URL=https://api.deepseek.com`、`DEEPSEEK_MODEL=deepseek-v4-flash` 与 `DEEPSEEK_API_KEY`，只在 generation 阶段进行实时调用，并把 provider response ID、重试决策、阶段产物及非敏感 token/cache usage 持久化或汇总到 manifest。客户端把稳定系统策略放在动态病例和证据之前，以复用 DeepSeek 自动上下文缓存的公共前缀；这是 provider 自动缓存优化，不是自建缓存层。其 retrieval 是明确标注的本地演示 fixture，citation 判定是 `demo_structure_binding_not_nli`，因此不构成医学 RAG、固定 NLI 验证或可写入简历的指标。`DeterministicWorkflowProvider` 继续仅用于无网络测试与工程 fixture。

最小演示页位于 `http://127.0.0.1:8400/demo`。`medidiag demo --provider deepseek` 在同一进程启动 FastAPI 与单机 worker；页面创建任务后每 2 秒刷新局部视图，进入 `ESCALATED` 或 `CLOSED_*` 后停止。若 generation 阶段的网关调用在有限重试后仍失败，worker 会在同一事务内将任务标为 `FAILED`、病例转为 `ESCALATED` 并写入 `stage_failed` 事件；页面显示安全错误提示且明确未生成报告或医疗结论，后续只能由人工选择处置。实时草稿会在页面中显示本地 fixture 来源与 `PARTIAL / demo_structure_binding_not_nli` 限制。页面不包含登录、真实患者档案、WebSocket、生产 Dashboard 或医院系统集成。

trace raw 文件位于 `traces/raw/<trace_id>.jsonl`，脱敏摘要位于 `traces/summary/<trace_id>.json`。摘要只保留 evidence 元数据、claim hash、citation verdict、阶段耗时、恢复与人工事件，不保存病例问题或证据正文；`raw_event_ids` 可反向定位 raw event。生成文件属于本地运行产物，不提交 Git。

## Documents

- `SUPPLEMENT_PLAN.md`：当前直接实施与验收依据。
- `overview.md`：本轮实施记录、验证状态和下一步。
- `docs/architecture.md`：当前实现边界与目标数据流。
- `docs/design-decisions.md`：评测模式、实验隔离、judge 与结果资格决策。
- `docs/evaluation_protocol.md`：正式 run、20% 双人 citation 复核、Kappa/裁决审计和自动报告步骤。
- `docs/archive/overview_historical_2026-07-06.md`：历史阶段记录，不代表当前验收结论。
- `reports/archive/`：旧探索性报告，不得用于简历。
- `THIRD_PARTY_NOTICES.md`：vendored 前端资产版本、来源、哈希和许可证。

## License

MIT，全文见 [`LICENSE`](LICENSE)。仅用于工程演示与公开数据评测。
