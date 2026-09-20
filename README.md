# MediDiag：公开证据辅助分析工程原型

用于公开资料和模拟输入的工程演示，不提供真实患者诊疗服务，不构成诊断、处方或治疗建议。

## 能力与边界

- React + TypeScript 四页贯通咨询、任务进度、结果与证据、本人历史；保留 `/assistant` 和工程排障入口 `/demo`。
- 模块化 FastAPI 单体与独立 worker；支持 SQLite 演示、PostgreSQL 与 Redis/Celery 派发补偿，数据库是任务状态的权威来源。
- 服务端匿名会话与 owner 隔离，支持幂等提交、阶段恢复和取消后的迟到写入保护；不是实名账号或跨设备找回系统。
- 三种显式应用模式：`fake_offline` 固定演示；`retrieval_mock` 公开中文短引真实检索、模拟摘录；`mimo_grounded` 真实 MiMo 受约束完整摘录。后两者均不声称 NLI 或医学语义审核。
- 已完成有界的 `mimo-v2.5` 页面联合验收，成功与弃答失败均保留；这不是长期付费授权，也不是临床效果或公网生产验收。

当前轮次与证据见 [实际状态](docs/status.md)，允许实施的范围见 [项目实施计划](项目实施计划.md)。

## 最短安全启动

在项目根目录、已安装依赖的 `medidiag` / Python 3.11 环境执行；首次准备见 [运行手册](docs/development/runbook.md#准备环境)。显式选择离线模式，即使本地有密钥也不会调用模型：

```powershell
conda activate medidiag
npm run build --prefix frontend
New-Item -ItemType Directory -Force .cache/demo | Out-Null
$env:DATABASE_URL = "sqlite:///./.cache/demo/public-demo.db"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m alembic upgrade head
python -m medidiag.cli demo --provider retrieval_mock --app-config configs/application.yaml --port 8400
```

浏览器打开 `http://127.0.0.1:8400/app/`，填入公开检索示例并确认非敏感数据；Ctrl+C 停止。仅回环使用，不公开到共享网络，不操作用户根目录 `medidiag.db`。此命令采用本地 worker 线程；独立 API/worker、队列、回归和真实模式的操作统一见运行手册。

## 文档入口

| 要找的内容 | 唯一主入口 |
| --- | --- |
| 范围、轮次验收条件、停止条件 | [实施计划](项目实施计划.md) |
| 当前完成状态、提交与证据索引 | [实际状态](docs/status.md) |
| 安装、启动、演示、验证与排障 | [运行手册](docs/development/runbook.md) |
| API、用户投影与前端行为 | [应用契约](docs/development/application-contract.md) |
| 数据一致性、隔离、模型路线取舍 | [架构](docs/architecture.md) |
| RAG 对照、真实调用、失败与限制 | [检索与模型验收](docs/development/rag-delivery.md) |
| 目录职责、历史与研究入口 | [目录与文档导航](docs/structure.md) |

开发遵循 [AGENTS.md](AGENTS.md)。历史研究的语言兼容性、`report_eligible=false` 与混合实现版本事实不因工程验收改变。许可证见 [LICENSE](LICENSE)，第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
