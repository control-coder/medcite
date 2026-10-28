# MedCite

基于公开证据的医疗信息辅助分析原型。每条展示的内容都能追溯到原文片段和来源，证据不足时系统拒答。

[![CI](https://github.com/control-coder/medcite/actions/workflows/ci.yml/badge.svg)](https://github.com/control-coder/medcite/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> [!IMPORTANT]
> 本项目是工程原型，只处理公开资料与模拟输入，不提供诊断、处方或治疗建议，也不接收真实患者数据。

## 简介

用户提交一个健康相关的问题，系统在公开语料中检索证据，由后台任务完成检索、生成、审核和结果投影，只展示能关联到原文片段的内容，并附上来源链接、局限和风险提示。检索不到支持的证据时，返回“证据不足”，不生成看似确定的回答。

- 可追溯引用：每条展示内容都绑定到本次检索返回的原文片段，没有引用的内容不会出现在结果页。
- 拒答与失败处理：检索为空时不调用模型，直接拒答；模型改写了原文、引用了不存在的片段、输出格式错误或被截断时，结果一律被拒绝。
- 任务一致性：任务状态以数据库为准，消息队列只负责通知 worker。重复投递、消息丢失、worker 中途退出、任务取消后才到达的写入，都有处理逻辑和测试。

## 功能

- 前端使用 React + TypeScript，包含咨询提交、任务进度、结果与证据、历史记录四个页面，刷新页面后可恢复同一任务。
- 后端是 FastAPI 应用加独立的 worker 进程。本地演示使用 SQLite，服务模式使用 PostgreSQL + Redis/Celery。
- 任务状态机；幂等键去重；乐观锁；worker 领取任务后在租约内独占并定期续约；各阶段结果落库，中断后可从断点继续；定时扫描数据库，补发已保存但未发出的队列消息。
- 服务端签发匿名会话，每个用户只能访问自己的数据，访问他人或不存在的数据一律返回 404。
- 三种运行模式，互不静默切换：`fake_offline`（固定证据与输出，用于无依赖地演示流程）、`retrieval_mock`（真实检索加固定规则摘录，默认演示，不需要模型或密钥）、`mimo_grounded`（真实检索，由 LLM 从检索结果中选择原文片段，有调用预算限制）。
- 四种检索配置，需显式指定：BM25（`configs/application.yaml`，默认，中文按相邻两字切分）、向量检索（`application_dense.yaml`，固定版本的 bge-small-zh，只读本地缓存）、BM25 加查询改写（`application_rewrite.yaml`）、BM25 加检索 Agent（`application_agent.yaml`）、向量检索加检索 Agent（`application_dense_agent.yaml`，v3 语料，与测试集评测的方案一致）。后四种配置搭配 `mimo_grounded` 使用时，输出校验失败会带反馈重试一次。选型依据见 [架构与取舍](docs/architecture.md)。
- 真实模型调用有预算账本：先扣额度再发请求，失败不退还，重启后不重置，不跟随重定向。
- 评测代码在 `eval/`，真实模型调用的请求与响应存为调用记录入库，CI 中可不联网回放。

### v4 RAG 链路（M4–M15，需显式配置开启）

- 离线：多来源多格式采集（WHO 网页与出版物 PDF、国家文件 PDF/DOCX、疫情月报表格、维基 JSON）→ 解析、清洗、MinHash 近重复检测 → 父子分块（parent-child chunking）→ 向量化 → NumPy 持久缓存或 pgvector 存储 → 按内容哈希增量同步（`medidiag corpus sync`）。
- 在线：查询路由与元数据过滤 → 混合检索（BM25 + 向量，RRF 融合）→ 重排（rerank）→ MMR 去冗余 → 检索 Agent v2（HyDE、问题分解并行检索、CRAG 式检索评估）→ 抽取式回答（代码中称 constrained excerpt）或生成式回答（句子级引用、确定性校验、蕴含判定）→ 任务进度 SSE 推送。
- 配置：`configs/application_v4_recommended.yaml`（抽取式回答）、`configs/application_v4_recommended_generative.yaml`（`answer_mode: generative`）。现有配置与 v2/v3 语料不变，不选新配置时运行路径与原来相同。
- 运行前提：v4 语料全文只在本机 `.cache/corpus/`，复现步骤见 [examples/public_health_v4/README.md](examples/public_health_v4/README.md)；embedding 与重排模型须已在本地缓存，不会自动下载；调用真实模型须带 `--live-budget` 预算账本。详细设计见 [架构与取舍](docs/architecture.md#v4-rag-链路)。

## 截图

截图由浏览器自动测试生成，使用默认的 BM25 检索和固定规则摘录，重新生成的方法见 [开发与运行](docs/development.md)。

结果与证据页：每条摘录都标明“检索摘录，不代表问题已获解答”，右侧列出原始来源，可打开原文。

![结果与证据页](docs/images/result-with-evidence-desktop.png)

手机宽度下的拒答：问题与语料无关时检索不到证据，页面显示“证据不足”。

<img src="docs/images/abstain-mobile.png" alt="手机宽度下检索不到证据时的结果页" width="320">

## 架构

```mermaid
flowchart LR
    UI[React /app/] --> API[FastAPI]
    API --> ID[匿名会话 / owner 隔离]
    ID --> DB[(PostgreSQL / SQLite<br/>任务状态以此为准)]
    S[任务派发与补发扫描] --> DB
    S --> Q[(Redis<br/>仅携带 task_id)]
    Q --> W[Celery / 本地 worker]
    W --> DB
    W --> P{Provider}
    P --> R[BM25 / 向量检索]
    R --> G[摘录原文 / 模型只能选择原文片段]
    G --> V[引用绑定与合规检查]
    V --> DB
    DB --> PRJ[面向用户的结果视图] --> UI
```

设计取舍、一致性保证和已知限制见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

环境要求：Python 3.11、Node.js 20.19+。默认演示不需要 Docker、模型下载或 API Key。

```bash
# 1. 安装依赖（建议使用独立的 conda 环境）
conda env create -f environment.yml
conda activate medidiag
python -m pip install -e ".[dev]"
npm ci --prefix frontend
npm run build --prefix frontend

# 2. 使用独立的演示数据库并完成迁移
mkdir -p .cache/demo
export DATABASE_URL="sqlite:///./.cache/demo/public-demo.db"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m alembic upgrade head

# 3. 启动（API + 本地 worker 线程）
python -m medidiag.cli demo --provider retrieval_mock --app-config configs/application.yaml --port 8400
```

打开 <http://127.0.0.1:8400/app/>，点击“填入公开检索示例”并提交。Windows PowerShell 下用 `$env:DATABASE_URL = "..."` 设置环境变量。PostgreSQL + Redis/Celery、真实模型模式和排障见 [docs/development.md](docs/development.md)。

## 测试

```bash
ruff check .
mypy src
python -m pytest -q
python scripts/verify_offline.py --provider retrieval_mock --browser   # 端到端：独立 API/worker 加浏览器
```

CI 在每次推送到 `main` 和每个 Pull Request 上运行三个作业：`check`（ruff、mypy 严格模式、pytest）、`eval-regression`（不联网回放调用记录并运行回归门槛，误答必须为 0，见 [examples/rag_regression](examples/rag_regression/README.md)）、`e2e`（真实 API、独立 worker 和 Chromium，使用 `retrieval_mock`）。

## 评测结果

评测使用项目内部构造的 168 条查询（可回答 112 条，不可回答 56 条），按开发集和测试集各半划分。结果不代表医学准确率，也不代表真实用户的问题。方法、逐项结果、复现命令和原始报告见 [docs/evaluation.md](docs/evaluation.md)。

术语：Hit@3 指前 3 个检索结果中包含相关片段的可回答问题所占比例；误答率指不可回答的问题中模型给出了回答的比例；过度拒答率指可回答的问题中模型拒答的比例；F1 是回答精确率和召回率的调和平均。

v3 语料（131 篇页面，498 个片段）测试集结果，模型为 `mimo-v2.6-flash`，输出被拦截时带反馈重试一次：

| 条件 | Hit@3 | 正确作答（/56） | 回答 F1 [95% 置信区间] | 误答（/28） |
| --- | --- | --- | --- | --- |
| BM25 | 0.679 | 36 | 0.783 [0.675, 0.867] | 0 |
| BM25 + 查询改写 | 0.875 | 46 | 0.893 [0.820, 0.952] | 0 |
| BM25 + 检索 Agent | 0.821 | 44 | 0.880 [0.805, 0.941] | 0 |
| 向量检索 | 0.929 | 52 | 0.954 [0.902, 0.992] | 0 |
| 向量检索 + 检索 Agent | 0.964 | 54 | 0.973 [0.928, 1.000] | 0 |

- 语料规模：页面从 8 篇增加到 131 篇，BM25 的 Hit@3 从 0.796 降到 0.661，向量检索从 0.991 降到 0.955。
- 误答：五个条件下均为 0/28。样本只有 28 条，95% 置信区间上限约 12%，只能说没有观测到误答。48 次端到端失败中，45 次是检索未命中，3 次是检索噪声，没有观测到模型编造内容。
- 检索 Agent：BM25 条件下，Agent 与固定的查询改写在 112 条可回答问题上命中相关片段的数量都是 92，端到端正确作答 44 对 46。向量检索加 Agent 只多 2 条（p = 0.5）。
- 可靠性：60 个病例、4 个 worker 进程，随机强制终止 8 次并暂停 2 次，SQLite 和 PostgreSQL 各用 3 个随机种子。修复两个调度缺陷后全部为 180/180 成功，语义为至少执行一次、结果恰好提交一次。
- 同一请求重复运行时结果有波动，BM25 条件 3 次运行的正确作答数在 33 到 36 之间，相差 1 到 3 条的比较不可信。
- 拒答门槛：在向量检索加 Agent 加重排前加重排分数门槛，阈值只在开发集上选。模型拒答已经没有观测到误答，门槛只增加过度拒答，应用不加门槛。
- 多轮追问（66 条 AI 撰写的两轮对话，test）：只用追问原文时正确作答 9/46，把历史问题直接拼接为 35/46（误答 3/20），检索前用模型补全问题为 42/46（误答 0/20），以源查询为准的上限参照为 41/46。补全对只用追问原文 33 / 0，p < 0.0001；对拼接 9 / 2，p = 0.065。

### v4 语料 test 最终对照（M15，单次运行）

v4 语料（498 份快照，M6 父子分块后 39,752 个子块）与 v4 评测集 test 202 条（科普 155：可回答 129、无答案 26；非科普 47，含急症 6），模型为 `mimo-v2.6-flash`。v4 的数字与上面的 v3 结果使用不同语料和查询，不能直接比较。相关性标签由 LLM 给出，未经人工复核；评审与被评系统同源。

| 指标 | ① v3 推荐配置（换 v4 语料） | ② v4 推荐，抽取式 | ③ v4 推荐，生成式 |
| --- | --- | --- | --- |
| 正确回答（/129） | 104 | 108 | 118 |
| 过度拒答（/129） | 23 | 12 | 8 |
| 无答案被作答（/26） | 2 | 3 | 4 |
| 急症给出急症提示（/6） | 0 | 6 | 6 |
| 引用召回率 / 引用精确率 / 忠实度 | — | — | 0.877 / 0.854 / 0.914 |
| 估算费用（元） | 0.51 | 0.83 | 1.03（评审另 0.91） |
| 延迟 p50 / p90（s） | 16.7 / 35.9 | 27.2 / 74.4 | 31.6 / 82.2 |

- ②③ 的过度拒答含 6 条被查询路由误拒的科普问题；③ 复用 ② 的路由与检索记录。急症问题在三个条件下都没有被作答。
- 差异来自整条链路的组合变化，不拆分各模型组件的单独贡献；单次运行，不做显著性结论。
- 部分 v3 dev 查询在 v4 划分中落入 test，对 ① 略有利。无答案被作答只有 2–4 条，条件之间的差别不可靠。
- 数据与分项见 [评测与验证](docs/evaluation.md#v4-评测汇总m15)。

## 目录结构

```text
src/medidiag/
  api/            HTTP 接口、匿名会话、用户安全投影、SSE 进度、兼容 HTML 页面
  workflow/       状态机、任务领取与租约、派发补偿、worker、应用 Provider、检索 Agent、查询路由、生成式回答
  rag/            BM25、向量与混合检索、证据包；ingest/ 解析清洗分块，store/ 向量存储
  review/  compliance/   复核轮次上限判定、输出合规边界
  llm/            模型适配、预算账本、调用记录与回放
  db/  observability/    数据模型、会话、结构化日志与追踪
frontend/         React + TypeScript + Vite 应用与 Playwright 测试
migrations/       Alembic 数据库迁移
configs/          应用运行配置
examples/         公开语料、固定查询与回归集
scripts/          数据准备、故障注入测试与端到端验证脚本
eval/             评测代码、调用记录
artifacts/        评测报告
tests/            自动化测试
```

内部 Python 包、conda 环境和数据库沿用早期名称 `medidiag`。

## 文档

| 文档 | 内容 |
| --- | --- |
| [架构与取舍](docs/architecture.md) | 数据权威、任务一致性、用户隔离、检索与模型路线、选型依据 |
| [API 与前端行为](docs/api.md) | 接口、输入约束、结果投影与失败语义 |
| [开发与运行](docs/development.md) | 安装、各运行模式、验证命令、演示路线与排障 |
| [评测与验证](docs/evaluation.md) | 指标口径、数据集、检索/生成/端到端矩阵、规模影响、可靠性、错误分析、未验证范围 |
| [评测代码说明](eval/README.md)、[报告索引](artifacts/reports/README.md) | 评测代码、调用记录与报告说明 |

## 局限

- 引用关联只证明内容可追溯到原文，不证明语义正确或医学适用；中文应用路径没有使用 NLI 做语义审核。
- 默认配置是 BM25，应用配置中的语料是 178 个片段的 v2，没有换成 v3。向量检索需要本机先准备模型缓存，且每个问题都会调用一次模型。
- 评测查询没有第二位标注者，不可回答的问题只有 28 条。真实用户问题上的表现、更大规模的语料和其他嵌入模型都没有测试过。
- v4 链路未验证：真实 `mimo_grounded` 经数据库 worker 的完整端到端、pgvector 下的元数据过滤路径、PostgreSQL 与多 worker 下的 SSE、浏览器内点击引用高亮。生成式回答的蕴含判定由模型完成，不是医学审核。
- 匿名会话不支持账号找回和跨设备迁移；公网部署、多机容量和临床效果都没有验证。

## 许可证

本项目代码采用 [MIT License](LICENSE)。第三方组件与数据（HTMX、WHO 科普文本摘录）的来源和使用条件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
