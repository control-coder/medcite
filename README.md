# MediDiag-Agent EvidenceFlow

> 医疗循证诊断多 Agent 工作流平台（模拟，仅用于工程演示与公开数据评测，**不用于真实医疗诊断**）。

## Scope 声明

**第一阶段（MVP）只承诺：**

- 单机 worker、SQLite 单库、本地 RAG 检索
- 公开数据集（MedQA / PubMedQA）改造病例 + 模拟病例
- 面向医疗诊断场景的状态机执行器（14 状态 + 合法跳转 + 触发主体）
- 任务租约、幂等键、乐观锁冲突重试、worker 崩溃恢复
- 证据前置检索 + 医学 RAG 消融实验（A-F 组单变量）
- 引用校验 + 诊断逻辑审核 + 合规输出管控
- 双专科并行诊断 + 仲裁实验（含单 Agent baseline 对比）
- 可复现评测：`eval/config.yaml` 锁定模型 / 温度 / seed / 数据集版本 / 检索权重 / 运行命令
- 数据泄露校验 + Cohen's Kappa 标注一致性
- 结构化日志 + event log + 分阶段 latency 埋点

**二期预留（第一阶段不承诺）：**

- 多 worker 横向扩展、Celery / 分布式任务队列
- 生产级告警平台、Prometheus + Grafana
- 真实医疗合规认证、真实患者数据接入
- 容器级隔离、seccomp、进程级资源隔离
- 全量公开基准跑分（如 MMLU-clinical 全集）
- 前端 Dashboard、WebSocket 实时推送

**免责声明：** 本系统只用于 Agent 工程实验和公开数据评测，不构成医疗建议，不用于真实医疗诊断。技术层面已加入超范围拒答、绝对化措辞拦截和强制风险提示，而非仅靠 README 免责。

---

## Upstream Reference

本项目参考但不复用以下来源的代码与架构思想：

- **edict**（`../edict/`）：参考其事件驱动 Agent 工程模式——FastAPI + 事件日志 + thoughts/todo 结构化 + 可回放 trace。**未复用其代码**，仅借鉴工程组织方式。edict 的"朝堂议政"多 Agent 角色设计与本项目的医疗诊断场景无关。
- **MedQA**：公开医学考试题数据集，用于评测集公开题改造（占比 ≥ 40%）。
- **PubMedQA**：公开生物医学问答数据集，用于评测集公开题改造与知识库 chunk 构建。
- **MeSH descriptor**：NLM 公开医学主题词表，用于术语归一化的外部标准映射层。

## Third-party Components

以下为第三方组件，**只算集成，不计为个人核心创新**：

| 类别 | 组件 | 用途 |
|---|---|---|
| LLM | DeepSeek API（`deepseek-v4-flash-free`，OpenAI 兼容） | 诊断生成、Agent 推理 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 文本向量化 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 检索结果重排 |
| Judge (NLI) | `microsoft/deberta-v3-base-mnli` | Citation Precision 判定（SUPPORTED / PARTIAL / UNSUPPORTED） |
| 向量库 | FAISS (`faiss-cpu`) | embedding 检索 |
| 稀疏检索 | `rank-bm25` | BM25 检索 |
| Web 框架 | FastAPI + Uvicorn | API 层 |
| ORM | SQLAlchemy 2.0 + Alembic | 数据模型与迁移 |
| 配置 | Pydantic Settings + PyYAML | 环境变量与评测配置 |
| 术语词表 | MeSH descriptor XML | 术语归一化第三层 |

## My Contributions

以下为本人新增或重写的模块，每一项均可被源码、测试或评测报告验证：

1. **面向医疗诊断场景的状态机执行器**（`src/medidiag/workflow/state_machine.py`）
   - 14 状态枚举 + 合法跳转表 + 触发主体标注
   - `ESCALATED` 中间等待态、`CLOSED_*` 终态语义
   - 非法跳转拦截、自动/人工触发权限边界

2. **任务执行与一致性**（`src/medidiag/workflow/` + `src/medidiag/db/`）
   - worker 租约（lease_owner / lease_until / heartbeat / attempt）
   - 幂等键（创建病例 / 启动工作流 / Agent 执行三层）
   - 乐观锁冲突自动重试（3 次退避 50/100/200ms）
   - `RUNNING` 重复请求处理 + 旧 worker 迟到写入防护（`TASK_LEASE_LOST`）
   - 事务边界：外部 IO 不进事务、状态+事件同事务

3. **医学 RAG 与术语归一化**（`src/medidiag/rag/`）
   - 三层术语归一化：轻量词典 + 数据集派生 synonym map + MeSH descriptor 映射
   - 检索排序公式：`final_score = w1*bm25 + w2*embedding + w3*evidence_level + w4*term_overlap`
   - 权重写入 `eval/config.yaml`，参与消融实验

4. **审核与合规**（`src/medidiag/review/` + `src/medidiag/compliance/`）
   - `CitationVerifier`：NLI/cross-encoder 判定，LLM judge 仅辅助解释
   - `ClinicalLogicReviewer`：证据/风险提示/检查建议/不确定性检查
   - `ComplianceGuard`：超范围拒答、绝对化措辞拦截、强制风险提示
   - `ArbitrationAgent`：双专科意见冲突仲裁
   - 审核驳回闭环 + 连续失败人工升级（不进死状态）

5. **评测体系**（`eval/`）
   - `eval/config.yaml` 锁定全部变量
   - `eval/runner.py`：CLI + 配置加载 + 消融分组 + 报告生成
   - `eval/leakage_check.py`：数据泄露校验（输出 `EVAL_DATA_LEAKAGE_DETECTED`）
   - Cohen's Kappa 标注一致性统计
   - 单变量消融（A-F 组）+ 全量组合

6. **Agent 编排**（`src/medidiag/agents/`）
   - 病例归一化 / 证据检索 / 诊断生成 worker
   - 双专科并行 Agent + 仲裁 Agent
   - 单 Agent baseline 对比实验

7. **可观测性**（`src/medidiag/errors.py` + event log）
   - 错误码五级分类（4xx/42x/52x/53x/55x）
   - 分阶段 latency 埋点
   - 错误码告警阈值

---

## 状态机概览

```
CREATED -> NORMALIZED -> EVIDENCE_RETRIEVED -> PLAN_GENERATED
  -> SPECIALIST_REVIEWING -> ARBITRATION_REVIEWING
  -> APPROVED -> REPORT_GENERATED -> CLOSED_SUCCESS

分支：
  - 任意阶段 -> ESCALATED（中间等待态，人工回流）
  - ESCALATED -> REVISION_REQUIRED / APPROVED / CLOSED_ESCALATED
  - REVISION_REQUIRED -> PLAN_GENERATED（重试）/ CLOSED_FAILED
  - CREATED -> CLOSED_CANCELLED（用户取消）
```

完整状态转移表见 `docs/architecture.md`。

## 评测概览

| 指标 | 公式 |
|---|---|
| Evidence Recall@5 | 至少命中 1 条 gold_evidence 的样本数 / 总样本数 |
| Citation Precision | SUPPORTED citation 数 / 系统输出 citation 总数 |
| Judge Agreement | judge 判定与人工抽样复核一致样本数 / 抽样复核样本数 |
| Unsupported Claim Rate | UNSUPPORTED claims / total claims |
| Terminology Normalization Gain | Recall@5(with norm) - Recall@5(without norm) |
| Workflow Success Rate | CLOSED_SUCCESS case 数 / 总 case 数 |

消融组别（A-F）见 `reports/baseline.md`。

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 运行测试
pytest

# 4. 评测 CLI（阶段 0 仅骨架）
python -m eval.runner --help
python -m eval.leakage_check --help
```

## 目录结构

```
medidiag/
  README.md
  docs/
    architecture.md
    design-decisions.md
  src/medidiag/
    workflow/        # 状态机执行器、任务租约
    db/              # SQLAlchemy 模型、迁移
    rag/             # 术语归一化、BM25、embedding、rerank
    agents/          # 诊断生成、双专科、仲裁
    review/          # CitationVerifier、ClinicalLogicReviewer
    compliance/      # ComplianceGuard
    errors.py        # 错误码分级
    config.py        # 配置加载
  tests/
  eval/
    config.yaml      # 锁定模型/温度/seed/数据集/权重/命令
    runner.py        # 评测 CLI
    leakage_check.py # 数据泄露校验
  reports/
    baseline.md
    final_eval.md
    raw/
  traces/
    raw/
    summary/
  examples/
  scripts/
```

## License

MIT。仅用于工程演示与公开数据评测。
