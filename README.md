# MediDiag：医疗信息辅助分析助手

公开资料与脱敏模拟输入上的应用工程原型，展示咨询提交、后台任务、证据与引用、失败反馈和历史查询。

> 仅用于软件工程演示与公开数据研究，不提供真实患者诊疗服务，不构成诊断、处方或治疗建议。

## 已实现与验证边界

- React + TypeScript + Vite 四页，入口 `/app/`；保留 `/assistant` 和 `/demo`。
- FastAPI 模块化单体、独立 worker；兼容 SQLite 离线演示与 PostgreSQL。
- Redis/Celery 只传 task_id；数据库扫描补偿、CAS 领取、租约、阶段恢复与迟到写入保护。接受重复投递，不宣称 exactly-once。
- 服务端 30 天匿名会话与 owner 过滤；历史、报告、事件和操作统一隔离。不是实名账号，清 Cookie/换浏览器不能恢复旧记录。
- 引用与实际返回证据关联、无证据弃答、阶段耗时与部分用量观测。未知费用不是零费用。
- 默认闭环使用固定离线 fixture；小型 RAG 对照另行验证真实 BM25/词典软件路径。它们都不等于真实模型、引用语义或临床效果验收。

当前交付见 [实际状态](docs/status.md)，范围见 [实施计划](项目实施计划.md)。历史研究的语言兼容性问题、report_eligible=false 和混合实现版本事实保持不变，人工标注不作为应用门禁。

## 快速开始

在项目根目录执行。已有 `medidiag` 时直接激活，不重复创建；仅新机器首次用 `conda env create -f environment.yml`。

~~~powershell
conda activate medidiag
python -c "import sys; print(sys.executable, sys.version)"
# 首次安装或依赖声明变化时执行；仅此环境，不使用 base/.venv。
python -m pip install -e ".[dev]"
npm ci --prefix frontend
npm run build --prefix frontend

# 显式使用独立演示库，绝不默认迁移用户根目录 medidiag.db。
New-Item -ItemType Directory -Force .cache/demo | Out-Null
$env:DATABASE_URL = "sqlite:///./.cache/demo/demo.db"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m alembic upgrade head
python -m medidiag.cli demo --provider fake_offline --port 8400
~~~

浏览器打开 `http://127.0.0.1:8400/app/`，点击“填入模拟示例”、确认非敏感数据后提交。工程排障入口 `/demo`，兼容用户页 `/assistant`；Ctrl+C 停止。不要公开到共享网络。本轮使用 Python 3.11.15、Node 24.16.0、npm 11.13.0。

## 可复现验收

~~~powershell
# 自动创建独立数据库，启动独立 API/worker，完成 HTTP 闭环后停止子进程。
python scripts/verify_offline.py
# 固定 20 条模拟问题的词法检索对照，不下载模型。
python scripts/verify_rag.py
# 已启动 8400 演示服务时运行，要求本机安装 Edge。
npm run test:e2e --prefix frontend
~~~

`verify_offline.py` 只接受 conda medidiag / Python 3.11，使用 `.cache/implementation/offline-*`，不覆盖旧数据。新数据库、独立进程和浏览器闭环已在现有指定环境验证；未声称全新机器/全新 conda 安装已验收。

PostgreSQL/Redis/Celery 独立启动、专用故障注入验收、故障恢复见 [运行手册](docs/development/runbook.md)。架构与工程取舍见 [架构说明](docs/architecture.md)，贡献/失败复盘与简历材料见 [交付复盘](docs/development/delivery-review.md)。

## API 摘要

客户端先访问 `GET /api/v1/session` 并保存服务端 Cookie，不得自填 owner 或将 X-User-Scope 当授权。

| 接口 | 用途 |
| --- | --- |
| POST /api/v1/consultations | 咨询输入、非敏感确认、Idempotency-Key |
| POST /api/v1/cases | 兼容创建接口 |
| GET /api/v1/cases | 本人历史游标分页 |
| POST /api/v1/cases/{id}/workflow | 启动/恢复，Idempotency-Key |
| GET /api/v1/cases/{id}、/events | 本人任务与事件 |
| GET /api/v1/cases/{id}/analysis | 用户安全投影与运行观测 |
| GET /api/v1/cases/{id}/report | 原结构化报告，供受控工程排障 |
| POST /api/v1/cases/{id}/cancel | 取消自动处理，人工升级和终态除外 |
| POST /api/v1/cases/{id}/human-decisions | 工程升级处置，不代表临床审核 |

## 目录

`src/medidiag/` 后端；`frontend/` React；`tests/` 自动化测试；`migrations/` 迁移；`scripts/` 与 `examples/` 辅助入口和模拟数据；`eval/` 独立研究子系统；`docs/` 当前资料；`docs/archive/research/` 历史文档；`artifacts/reports/application/` 本轮结果，其余历史报告原样保留。详情见 [目录说明](docs/structure.md)。

遵循 [AGENTS.md](AGENTS.md)：中文说明、独立环境、分轮验证提交，不推送、不泄露凭据、不混入用户修改。许可证 MIT，见 LICENSE；第三方通知见 THIRD_PARTY_NOTICES.md。


## 公开中文正文检索（第 8A 轮）

已有 8 篇 WHO 中文科普页面短引、16 条固定工程查询和应用专用 configs/application.yaml。显式使用 `--provider retrieval_mock --app-config configs/application.yaml` 可展示实际检索片段与来源；生成仍是确定性摘录，不是在线模型。启动时请按 [运行手册](docs/development/runbook.md) 指向独立演示库。

`python scripts/verify_public_rag.py` 重放两组对照；`python scripts/verify_offline.py --provider retrieval_mock --browser` 自动验证本机独立进程与 Edge 页面。仍有口语漏检和词面误命中，见 [检索记录](docs/development/rag-delivery.md)。第 8B 真实模型联合验收未执行；第 7 轮全新环境安装复验可选且本次跳过。
