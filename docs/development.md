# 开发与运行

所有命令在项目根目录执行，使用 Python 3.11。示例以 PowerShell 为主；bash 下把 `$env:NAME = "value"` 换成 `export NAME="value"`。接口与页面行为见 [API 文档](api.md)，设计见 [架构](architecture.md)。

## 环境准备

```powershell
conda env create -f environment.yml   # 已有环境可跳过
conda activate medidiag
python -m pip install -e ".[dev]"
python -m pip check
npm ci --prefix frontend
npm run build --prefix frontend
```

开发环境：Python 3.11、Node.js 20.19+（已在 Node 24 上验证）。FastAPI 会挂载 `frontend/dist`；未构建前端时 `/app/` 不存在，兼容页面 `/assistant`、`/demo` 仍可用。

复制 `.env.example` 为 `.env` 按需填写，`.env` 不提交。**所有 demo/worker 命令都需要显式传 `--provider`**，即使本地配置了密钥，也不会因为省略参数而调用在线模型。

## 本地演示

默认使用真实 BM25 检索 + 确定性摘录，不需要 Docker、模型下载或 API Key：

```powershell
conda activate medidiag
New-Item -ItemType Directory -Force .cache/demo | Out-Null
$env:DATABASE_URL = "sqlite:///./.cache/demo/public-demo.db"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m alembic upgrade head
python -m medidiag.cli demo --provider retrieval_mock --app-config configs/application.yaml --port 8400
```

打开 <http://127.0.0.1:8400/app/>，点击“填入公开检索示例”。该命令在 API 进程内附带一个本地 worker 线程，仅监听回环地址。需要固定 fixture 时改用 `--provider fake_offline`。

### 独立 API 与 worker

需要分别控制任务领取与取消时，在两个终端中使用相同的数据库与离线环境变量：

```powershell
# 终端 1
python -m uvicorn medidiag.api.app:create_app --factory --host 127.0.0.1 --port 8400
# 终端 2
python -m medidiag.cli worker --loop --provider retrieval_mock --app-config configs/application.yaml
```

### 前端热更新

后端运行在 8400 端口时，执行 `npm run dev --prefix frontend` 并打开终端提示的 `/app/` 地址，Vite 会把 `/api` 代理到后端。

## PostgreSQL 与 Redis/Celery

`compose.yaml` 提供只监听回环地址的开发服务（端口 15432 / 16379，trust 认证仅限本机开发）：

```powershell
docker compose up -d --wait
$env:DATABASE_URL = "postgresql+psycopg://medidiag@127.0.0.1:15432/medidiag"
$env:MEDIDIAG_BROKER_URL = "redis://127.0.0.1:16379/0"
python -m alembic upgrade head
```

再开三个终端，设置相同的数据库与 broker 变量。消费者需设置 `MEDIDIAG_APP_PROVIDER=fake_offline`，或设置 `retrieval_mock` 并同时设置 `MEDIDIAG_APP_CONFIG=configs/application.yaml`。Celery 消费者暂不支持真实模型模式。

```powershell
# 终端 1：API
python -m uvicorn medidiag.api.app:create_app --factory --host 127.0.0.1 --port 8400
# 终端 2：Celery 消费者（Windows 使用 solo pool）
python -m celery -A medidiag.workflow.queue:celery_app worker --pool=solo --concurrency=1 --loglevel=WARNING
# 终端 3：派发与补偿扫描
python -m medidiag.workflow.dispatcher
```

停止时先关闭三个进程，再执行 `docker compose stop`。不要使用 `down -v`，它会删除数据卷。

### 队列故障验证

在专用数据库上验证重复投递、发布异常后的补偿，以及 worker 异常退出后的租约恢复：

```powershell
docker compose exec -T postgres createdb -U medidiag medidiag_acceptance
$env:DATABASE_URL = "postgresql+psycopg://medidiag@127.0.0.1:15432/medidiag_acceptance"
python -m alembic upgrade head
python scripts/verify_queue.py
```

脚本只接受专用库名，会自行启动和停止 Celery。运行前先停止其他消费者。

## 测试与验证

```powershell
ruff check .
mypy src
python -m pytest -q
```

端到端验证脚本会新建数据库、启动独立的 API 与 worker，结束后自动关闭，记录写入 `.cache/implementation/`：

```powershell
# 前提：前端已构建。HTTP 流程、幂等、引用、历史与双身份隔离
python scripts/verify_offline.py
# 追加 Edge 浏览器用例：正常流程，以及取消、重复点击、会话丢失
python scripts/verify_offline.py --provider retrieval_mock --browser
python scripts/verify_offline.py --provider fake_offline --browser
# 检索回归（模拟主题集 / 公开中文语料），不调用模型
python scripts/verify_rag.py
python scripts/verify_public_rag.py
```

服务已在 8400 端口运行时，可单独执行 `npm run test:e2e --prefix frontend`；其他端口设置 `MEDIDIAG_WEB_URL`。真实模型的浏览器用例需显式授权变量，日常测试不会触发付费请求。

**重新生成 README 里的截图**：运行上面带 `--provider retrieval_mock --browser` 的命令，截图会写到 `artifacts/visual/`（该目录不提交）。README 用到的三张是 `public-rag-form-desktop.png`、`public-rag-desktop.png`、`public-rag-empty-mobile.png`，复制到 `docs/images/` 并改成对应的文件名后提交。界面改动后需要重新生成。

## 真实模型模式

`mimo_grounded` 会在用户提交后调用 `mimo-v2.6-flash`（模型名由 `src/medidiag/llm/models.py` 决定，`.env` 里的 `MIMO_MODEL` 只是默认值），需要在 `.env` 中配置 `MIMO_API_KEY`、`MIMO_BASE_URL`，并指定持久化预算账本：

```powershell
python -m medidiag.cli demo --provider mimo_grounded --live-budget .cache/implementation/<ledger>/calls.db --port 8400
# 自动化端到端验证（需要至少 3 次剩余额度）
python -m scripts.verify_live_application --allow-live --ledger .cache/implementation/<ledger>/calls.db
```

账本最多允许 8 次请求，先占额度后发请求，失败不退额、重启不重置；不要通过新建或删除账本来绕过上限。每次网络操作超时 45 秒，网络与阶段自动重试均为 0。账本不保存凭据、请求正文或上游错误原文。

### 向量检索方案

默认配置只用 BM25。要换成评测中表现更好的向量检索（固定版本的 `BAAI/bge-small-zh-v1.5`，v2 语料），需要先在本机准备好模型缓存（应用不会自动下载），然后指定另一份配置：

```powershell
python -m medidiag.cli demo --provider mimo_grounded --app-config configs/application_dense.yaml --live-budget .cache/dense-ledger.db --port 8400
# 6 个问题的真实调用冒烟（最多 8 次请求），结果写入 artifacts/reports/application/
python -I scripts/verify_dense_application.py --allow-live --ledger .cache/dense-ledger.db
```

该配置还让模型输出不合格（摘录不是原文、格式错误）时，带上被拒原因再请求一次，最多多 1 次调用；其他配置默认不重试。向量检索方案也可以配合 `--provider retrieval_mock` 使用，此时不调用模型。首次启动需要几秒加载模型并给 178 条短引建索引。

### BM25 加问题改写

没有向量模型缓存时，可以用 `configs/application_rewrite.yaml`：检索前先让模型把问题改写成规范表述，再和原问题一起做 BM25 检索，改写失败时回退到原问题（原因写在检索结果的 `query_rewrite`）。必须搭配 `mimo_grounded`，每个问题多一次改写调用，单轮冒烟用 `--cases` 挑 3 个问题以留在 8 次请求的上限内：

```powershell
python -I scripts/verify_dense_application.py --allow-live --ledger .cache/rewrite-ledger.db --app-config configs/application_rewrite.yaml --cases 2 3 6
```

### BM25 加检索智能体

`configs/application_agent.yaml`：先按原问题检索一次，再由模型看结果，决定是否调用 `search_kb` 换种说法继续检索（最多再检索 2 次，模型最多调用 3 次，开启思考），合并后的前 3 段作为证据交给受约束摘录。出错时沿用已拿到的证据，步骤记录在检索结果的 `search_agent`。不能和问题改写同时开启。每个问题的模型调用最多 3 次检索判断加 1 到 2 次生成，所以 8 次请求的上限下一次只能试 1 到 2 个问题，需要换新的账本文件：

```powershell
python -I scripts/verify_dense_application.py --allow-live --ledger .cache/agent-ledger.db --app-config configs/application_agent.yaml --cases 2
```

### 评测的调用记录

评测（`eval/llm_pipeline_eval.py`）把每次模型调用的请求和回复存成调用记录（`eval/cassettes/`），之后可以零网络、零费用回放。`--model` 指定请求用的模型名：新的调用并记录用默认的 `mimo-v2.6-flash`，回放 v2.5 的旧记录要加 `--model mimo-v2.5`。改动提示词、检索分词、证据排序或智能体的提示词与工具定义，都会使请求哈希变化而回放失败，需要重新调用并记录（付费，先确认账本额度）。回放测试为 `tests/test_llm_replay_committed.py`（v2.5）和 `tests/test_llm_replay_flash.py`（v2.6-flash）。

## 演示路线（约 4 分钟）

| 时间 | 操作 |
| --- | --- |
| 0:00–0:40 | 填入公开检索示例，确认非敏感数据并提交 |
| 0:40–1:20 | 刷新任务页：URL 保留任务 ID，从后端恢复，不会新建任务 |
| 1:20–2:20 | 查看摘录、证据与 WHO 来源链接；说明引用关联不等于语义审核 |
| 2:20–2:50 | 查看历史记录并返回同一任务 |
| 2:50–3:40 | 提交“模拟提问：量子纠缠计算芯片”，展示空证据拒答 |
| 3:40–4:20 | 演示取消任务与会话丢失后的提示 |

体验取消时，按“独立 API 与 worker”启动，停止 worker 后提交并取消，再重启 worker。

## 排障

| 现象 | 处理 |
| --- | --- |
| `/app/` 404 | 构建前端后重启 API |
| 迁移字段缺失 | 确认 `DATABASE_URL` 指向的数据库，并对其执行迁移 |
| 401 / 404 | 刷新页面会建立新匿名会话；旧会话的记录无法找回，越权与不存在统一返回 404 |
| 400 Host / 403 Origin | 使用同源回环地址；自定义部署需配置 `MEDIDIAG_ALLOWED_HOSTS`（JSON 数组）、HTTPS 与可信代理 |
| 任务一直排队 | 确认 API、消费者与扫描器使用相同的数据库和 broker，查看各进程日志 |
| 证据不足 | 属于正常拒答，系统不会把无支持的文本包装成结论 |
| 升级 / 失败 | 查看安全错误码与 `/demo` 中的事件，重新提交咨询 |
| 预算不足 | 停止真实请求，不要更换账本绕过上限 |

## 研究评测

研究子系统的数据、配置与 formal 流程见 [eval/README.md](../eval/README.md) 与 [研究评测协议](research/evaluation-protocol.md)。它独立于应用运行，不是应用测试的必要步骤。
