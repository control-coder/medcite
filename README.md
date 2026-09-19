# MediDiag：医疗信息辅助分析助手

面向公开资料与脱敏模拟输入的应用工程原型。用户提交症状与背景，系统先检索证据，再生成带引用、局限说明与风险提示的辅助分析。

> 仅用于软件工程演示与公开数据研究，不用于真实患者诊断、处方、治疗决策或患者服务。

## 当前与目标

当前可运行：FastAPI、SQLite、状态机与单机 worker、离线模拟 Provider、医学 RAG/引用审核组件，以及 /assistant 用户页和 /demo 工程工作台。

下一步：React + TypeScript + Vite 用户界面、PostgreSQL、Redis/Celery 派发与补偿、用户归属过滤。它们是 [实施计划](项目实施计划.md) 中的后续任务，不是本轮已经完成的功能。

历史研究评测独立保留。人工标注不再阻塞应用交付；旧报告的语言兼容性问题、report_eligible=false 和混合实现版本事实不改写，也不把历史分数当成医学有效性证据。

## 快速开始

在项目根目录执行：

~~~powershell
conda activate medidiag
python -c "import sys; print(sys.executable)"
# 初次安装或依赖声明变化后执行，禁止安装到 base。
python -m pip install -e ".[dev]"

# 迁移现有配置指定的数据库。默认保留根目录 medidiag.db。
python -m alembic upgrade head

# 本地离线演示，不下载模型、不调用付费接口。
medidiag demo --provider fake_offline
~~~

访问 http://127.0.0.1:8400/assistant；工程排障入口为 /demo。程序启动参数以 `medidiag demo --help` 为准。不要在共享网络上公开此工程演示服务。

更多环境、独立 worker、可选研究命令与故障说明见 [运行手册](docs/development/runbook.md)。新机器先用 `conda env create -f environment.yml` 创建环境；本机已有 medidiag 时直接激活，不重复创建。

## 按改动验证

~~~powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m pytest tests/test_api.py tests/test_assistant_ui.py tests/test_cli.py tests/test_migrations.py -q
~~~

修改检索、任务租约或 Provider 时运行对应现有测试；不要求每轮都执行全量研究评测、人工标注或大规模新增测试。

## 目录导航

| 路径 | 职责 |
| --- | --- |
| src/medidiag/ | 应用业务源码，已有 api、rag、workflow、llm、db 等职责模块 |
| tests/ | 自动化测试 |
| migrations/ | 数据库迁移，保留原 revision 标识 |
| eval/ | 独立研究评测程序、公开数据、配置与标注 |
| docs/ | 当前结构、运行说明与开发记录 |
| docs/archive/research/ | 旧章程、研究协议、计划、日志与历史状态 |
| artifacts/reports/ | 历史报告和评测产物 |
| artifacts/traces/ | 脱敏运行追踪 |
| scripts/、examples/ | 辅助脚本与模拟输入 |

frontend/ 在 React 实施轮次创建；根目录现有 medidiag.db 暂保留以兼容本地配置，不自动删除或移动。详细职责见 [目录说明](docs/structure.md)。

## 当前 API

- POST /api/v1/cases：创建公开或模拟病例，使用 Idempotency-Key。
- POST /api/v1/cases/{case_id}/workflow：启动或恢复任务。
- GET /api/v1/cases/{case_id}：查询任务与阶段状态。
- GET /api/v1/cases/{case_id}/events：分页查询事件。
- GET /api/v1/cases/{case_id}/report：读取结构化报告。
- POST /api/v1/cases/{case_id}/human-decisions：现有工程升级处置入口，不代表临床审核。

## 开发入口

- [AGENTS.md](AGENTS.md)：中文、独立环境、分轮验证提交、清理与审查尺度。
- [项目实施计划](项目实施计划.md)：当前交付范围与分轮验收。
- [实际状态](docs/status.md)：当前完成与未完成事项。
- [历史研究状态](docs/archive/research/status.md)：历史结果及限制，不能当成当前排期。
- [历史评测报告](artifacts/reports/final_eval.md)：仅作原始研究记录，不作医学效果宣传。

许可证为 MIT，原文见 LICENSE；第三方通知见 THIRD_PARTY_NOTICES.md。
