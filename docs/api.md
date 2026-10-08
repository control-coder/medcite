# API 与前端行为

本文描述 HTTP 接口、用户结果投影与前端约束。字段定义以 [schemas.py](../src/medidiag/api/schemas.py)、[app.py](../src/medidiag/api/app.py) 和 [analysis.py](../src/medidiag/api/analysis.py) 为准。

## 身份与访问

客户端先 `GET /api/v1/session` 并保存服务端 `medidiag_session` Cookie，再读取或修改自己的数据。会话有效期 30 天，HttpOnly、SameSite=Strict，HTTPS 时附加 Secure；数据库仅保存随机凭据的 SHA-256 摘要、owner 与到期时间。

- 无效/过期凭据返回 401 并清 Cookie；刷新可创建新身份。清 Cookie、过期或换浏览器后不能恢复旧记录。
- 新旧 API、`/assistant`、`/demo` 的列表、详情、报告、事件与操作均 owner 过滤；越权和不存在统一 404，不透露记录是否存在。
- `X-User-Scope` 不作为身份，客户端不传 owner；幂等键按服务器 owner 隔离。NULL owner 的历史病例不自动认领。
- 修改请求拒绝异站 Origin / Sec-Fetch-Site，默认仅允许本机 Host。部署配置与排障见 [开发与运行](development.md#排障)。

## 接口

| 接口 | 用途与约束 |
| --- | --- |
| `GET /api/v1/session` | 建立/确认匿名会话 |
| `POST /api/v1/consultations` | 当前咨询入口，要求非敏感确认和 `Idempotency-Key` |
| `POST /api/v1/cases` | 兼容旧创建入口，仍受相同 owner/幂等约束 |
| `GET /api/v1/cases` | 当前会话的历史，游标分页 |
| `POST /api/v1/cases/{id}/workflow` | 显式启动/可恢复处理，`Idempotency-Key` |
| `GET /api/v1/cases/{id}` | 病例状态与当前任务 |
| `GET /api/v1/cases/{id}/events` | 当前会话的事件，分页 |
| `GET /api/v1/cases/{id}/analysis` | `consultation-v1` 用户安全投影与观测 |
| `GET /api/v1/cases/{id}/report` | 原结构化报告，受控工程排障，不作为前端自由展示内容 |
| `POST /api/v1/cases/{id}/cancel` | 自动处理阶段取消，禁止绕过人工升级与终态 |
| `POST /api/v1/cases/{id}/human-decisions` | 工程升级处置，不代表临床审核 |

创建咨询与启动工作流是两个请求。创建已成功、启动请求失败时可沿同一启动幂等键重试；重新提交终态病例则创建新咨询，不重开不可逆终态。

### 咨询输入

- `symptoms`：10–8000 字符；`duration`：1–200 字符；`background`：可空，最多 2000 字符。文本首尾去空白。
- `input_kind`：`public_dataset` 或 `deidentified_simulation`；`source_ref` 可空、最多 128 字符。
- `non_sensitive_confirmed` 必须为 `true`。这些字段不意味着允许输入真实患者隐私。
- 旧 `cases` 入口沿用 `question`、`input_kind`、`source_ref`，不为兼容而绕过归属控制。

## 用户投影与失败语义

`analysis` 返回处理状态、`execution_mode`、`outcome`、claims/evidence、局限、风险、后续动作及运行观测。只将 claim 关联到本次实际返回的 chunk；摘要从保留的 claims 生成，原报告不覆盖。

| outcome | 用户语义 |
| --- | --- |
| `processing` | 尚在处理，刷新/轮询同一任务 |
| `ready` | 工程成功且有可关联内容；不等于 NLI、临床或医学正确性审核 |
| `insufficient_evidence` | 检索为空、模型拒答或无可关联 claim；不输出确定分析 |
| `failed` | 未交付兼容报告或处理升级/失败，显示安全错误码，不泄露上游异常正文 |
| `cancelled` | 已取消，没有可用报告 |

空检索保留拒答阶段产物但跳过生成调用。`ESCALATED` 在投影中属未交付，并非成功终态；底层 HUMAN 状态机不变。`retry_action=resubmit` 只表示新建咨询。

`execution_mode` 可为 `fake_offline`、`retrieval_mock`、`mimo_grounded`、`model_pipeline`、`unknown`。前端明示固定演示、真实检索/模拟生成、MiMo 受约束摘录等区别；`unknown` 不猜测为在线成功，任何标签都不能证明某个任务实际联网（空证据尤其不会生成）。路线设计见 [架构](architecture.md#检索与模型路线)。

### 运行观测

- `recorded_stage_latency_ms` 是最新任务已保存阶段耗时之和，不含排队和未提交阶段；不能当页面端到端延迟。
- `provider_attempts` 是阶段 Provider 事件计数，不等于付费 LLM 请求数。
- 输入/输出 Token 缺失时为 `null`、`usage_status=not_recorded`；有记录时仍标 `partial`，不是覆盖失败重试的完整账单。
- 未核实计价与完整账单时 `cost_usd=null`，未知不表示免费。

## 前端约束

四页为咨询提交、任务进度、结果与证据、历史列表，入口 `/app/`。Hash 路由保存任务 ID，刷新向后端恢复；不在浏览器持久保存症状/背景，历史从服务端查询，不依赖旧 localStorage 病例列表。

提交中按钮禁用以防重复点击，服务端仍必须幂等。终态停止误导性的“持续执行”提示；会话丢失显示统一归属说明，不无限加载。引用可定位证据和安全的 HTTP/HTTPS 来源链接；失败时不静默切换 Provider 或替换旧结果。

构建、Vite 开发代理与浏览器测试见 [开发与运行](development.md)。`/assistant` 与 `/demo` 保留兼容，不绕过本页隔离或取消规则。
