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
| `GET /api/v1/cases/{id}/events/stream` | SSE 推送事件，见下文「进度推送（SSE）」 |
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
- 病例响应带 `question`：提交时的原始问题，咨询入口为“症状：…\n持续时间：…\n背景：…”拼接文本；与其他字段一样只返回给归属者，前端从中取出问题用于标题与历史列表。
- 多轮追问：`cases` 入口可选 `parent_case_id`，指向当前会话自己的、已结束的上一轮病例。不存在或不属于当前会话时返回 404，上一轮未结束时返回 400 `CASE_INVALID_INPUT`。带 `parent_case_id` 时 `question` 下限放宽到 2 个字符，不带时仍为 10 个字符。病例响应多一个 `parent_case_id` 字段，单轮为 `null`。咨询入口和前端暂不提供追问。
- 追问的处理：worker 沿 `parent_case_id` 取最近 3 轮历史（每轮是当时的问题和成功关闭时交付的摘录，或拒答；未成功结束的记为没有回答）。配置开启 `generation.context_rewrite` 时，检索前先由模型把追问补全成可以单独理解的问题，检索与生成都用补全后的问题，失败时回退到原问题，结果记在检索产物的 `context_rewrite`。没有开启或 provider 不支持时，追问按单轮问题处理。历史不交给生成步骤，摘录与拒答规则不变。

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

`execution_mode` 可为 `fake_offline`、`retrieval_mock`、`mimo_grounded`、`model_pipeline`、`unknown`。前端明示固定演示、真实检索/模拟生成、MiMo 抽取式回答等区别；`unknown` 不猜测为在线成功，任何标签都不能证明某个任务实际联网（空证据尤其不会生成）。路线设计见 [架构](architecture.md#检索与模型路线)。

### 生成式回答与引用字段

配置 `generation.answer_mode: generative`（须显式开启，默认 `extractive`）时，`analysis` 增加以下可选字段，未开启时取默认值，原有客户端不受影响：

- `answer_mode`：`extractive`（默认，抽取式回答，代码中称 constrained excerpt）或 `generative`（生成式回答）。仅当报告标明生成式时为 `generative`；生成式回答在蕴含判定全部删句后降级为抽取式，此时为 `extractive`。
- `claims[].citations`：句子级引用列表 `[{chunk_id, quote, entailment}]`，默认空数组。`quote` 是该 `chunk_id` 证据中的逐字片段；`entailment` 为 `entailed`（蕴含）或 `partial`（部分蕴含，前端加“部分支持”标记）。不蕴含的句子不会出现。`citations` 里 `chunk_id` 不在本任务证据中的条目会被丢弃。
- `claims[].text` 是回答句，`evidence_ids` 是其引用的 `chunk_id`，与抽取式回答同一结构；每个 claim 的全部 `evidence_ids` 都必须能关联到本任务证据，否则该 claim 被隐藏。

`entailed` 与 `partial` 是模型的蕴含判定，只说明引用是否支持这句话，不是医学正确性审核。

### 路由拒答

查询路由（`generation.query_router`，如 `application_v4_router.yaml`）在检索前判定急症、个体诊疗或超出范围时，不检索、不调用生成模型，任务正常结束并生成拒答报告。此时 `analysis` 的 `outcome` 为 `insufficient_evidence`，`claims` 与 `evidence` 为空。路由信息写在原结构化报告（`GET /api/v1/cases/{id}/report`）里：

- `title`：`无法作答说明`；`summary` 为路由给出的固定提示。
- `limitations`：写明判定的意图和来源（规则或模型）。
- `next_steps`：急症为“请立即拨打 120 或前往就近急诊。”，其他为“真实健康问题请咨询专业人员。”
- `provenance`：`{guard: "query-router-v1", intent, source, rule}`；`intent` 取 `emergency`、`individual_care` 或 `out_of_scope`，`source` 为 `rule` 或模型来源，`rule` 为命中的规则名（模型判定时可能为空）。

路由的 LLM 调用失败时按“科普可答”继续，由证据门槛决定是否拒答，不会因此失败。`analysis` 不单独暴露 `intent` 字段，`message` 为路由给出的固定提示（与报告 `summary` 相同），前端据此显示急症或无法作答说明；原报告用于排障。

### 进度推送（SSE）

`GET /api/v1/cases/{id}/events/stream` 转发该病例的事件日志，`Content-Type: text/event-stream`，带 `Cache-Control: no-cache` 与 `X-Accel-Buffering: no`。

- 事件：`case_event` 的 `data` 是与 `GET /events` 相同的 `EventResponse` JSON（`event_id`、`event_type`、`from_status`、`to_status`、`detail` 等），帧 `id` 为 `event_id`；结束时发 `end`，`data` 为 `{"reason": "finished"}`（病例已终态或等待人工升级，且事件已发完）或 `{"reason": "timeout"}`。
- 续传：`cursor` 查询参数或 `Last-Event-ID` 请求头给出已收到的最大 `event_id`，两者取较大值，只推送之后的事件。浏览器 `EventSource` 重连时自动带 `Last-Event-ID`。
- 心跳：连续 15 秒没有事件时发注释帧 `: keep-alive`。
- 时长：单个连接最多 300 秒，超时发 `end`（`timeout`）后关闭；客户端可带 `cursor` 重新订阅。
- 鉴权：与轮询端点相同。流开始前校验匿名会话和病例归属，越权与不存在统一返回 404；流内每轮重新按同一 owner 读取，病例读取失败时直接关闭。
- 实现为服务端每 0.5 秒查询事件表，没有新增消息通道；不推送逐 token 输出。前端连接出错时回退为轮询 `GET /events`。

未验证：PostgreSQL 与多 worker 部署下的 SSE 行为；浏览器只在 `fake_offline` 下确认过推送与回退轮询。

### 运行观测

- `recorded_stage_latency_ms` 是最新任务已保存阶段耗时之和，不含排队和未提交阶段；不能当页面端到端延迟。
- `provider_attempts` 是阶段 Provider 事件计数，不等于付费 LLM 请求数。
- 输入/输出 Token 缺失时为 `null`、`usage_status=not_recorded`；有记录时仍标 `partial`，不是覆盖失败重试的完整账单。
- 未核实计价与完整账单时 `cost_usd=null`，未知不表示免费。

## 前端约束

四个视图为提问、任务进度、回答与来源、历史记录，入口 `/app/`。任务进度与回答共用 `#/cases/{id}`（旧链接 `#/cases/{id}/result` 仍可用），任务结束后原地显示回答。Hash 路由保存任务 ID，刷新向后端恢复；不在浏览器持久保存问题/背景，历史从服务端查询，不依赖旧 localStorage 病例列表。提问页的持续时间留空时提交“不适用”。

提交中按钮禁用以防重复点击，服务端仍必须幂等。终态停止误导性的“持续执行”提示；会话丢失显示统一归属说明，不无限加载。引用可定位证据和安全的 HTTP/HTTPS 来源链接；失败时不静默切换 Provider 或替换旧结果。

构建、Vite 开发代理与浏览器测试见 [开发与运行](development.md)。`/assistant` 与 `/demo` 保留兼容，不绕过本页隔离或取消规则。
