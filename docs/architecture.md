# 应用架构与取舍

模块化单体 + 独立 worker。本文解释设计与限制；实际验收见 [状态](status.md)，接口字段见 [应用契约](development/application-contract.md)，启动命令见 [运行手册](development/runbook.md)。

```mermaid
flowchart LR
    U[React 四页 /app/] --> API[FastAPI 单体]
    H[/assistant 与 /demo] --> API
    API --> ID[服务端匿名会话与 owner 过滤]
    ID --> DB[(PostgreSQL 权威状态)]
    D[数据库派发与补偿扫描] --> DB
    D --> R[(Redis 仅携带 task_id)]
    R --> C[Celery 独立消费者]
    C --> W[共用状态机 CAS 租约与阶段执行器]
    L[CLI 独立 worker 或 demo 线程] --> W
    W --> DB
    W --> P[显式选择 Provider]
    P --> F[fake_offline 固定 fixture]
    P --> B[应用配置 纯 BM25 中文双字检索]
    B --> M[retrieval_mock 确定性摘录]
    B --> G[mimo_grounded 有界真实 MiMo 摘录]
    DB --> O[用户安全投影与证据关联]
    O --> U
    E[独立研究 eval] --> A[既有多专科与 NLI 路线]
```

图中共享执行器不表示所有入口都支持全部模式：Celery 当前只配置离线应用模式；真实 MiMo 通过显式账本的 CLI 入口验收，未做真实模型与 PostgreSQL/Redis 的联合复验。最短 `demo` 将数据库换成 SQLite 并附带本地 worker 线程；自动验收脚本用独立 API/worker 进程，业务执行逻辑共用。

## 数据权威与任务一致性

- `Case` 保存 owner、状态、版本与 active_task；`WorkflowTask` 保存 attempt、租约、领取者与错误码；`StageArtifact`、`CaseReport`、`CaseEventLog` 保存阶段输出、版本化报告与事件。
- 创建/启动幂等键按服务端 owner 隔离。消息只是唤醒信号，即使 lease_owner 相同也须重新 CAS 领取，不能凭重复消息重入有效租约。
- 提交数据库与发布消息之间并非原子事务。扫描器默认每 5 秒、每批最多 100 条检查 PENDING 和租约过期 RUNNING，派发失败留待补偿；接受重复投递，不宣称 exactly-once。
- Provider I/O 前结束读取事务，完成后重新校验 owner/attempt/lease fencing。中断可从已提交阶段恢复，达到恢复上限后升级，不无限重试；不同 Provider 的调用重试预算另行约束。
- 取消在同一事务内终止病例并使任务失效，迟到写入被拒绝；取消不能撤回已发远程请求或退还费用。`ESCALATED` 仍受 HUMAN 边界保护，不借取消绕过。

**取舍：** 当前单人维护原型用本地事务即可表达核心一致性，独立 worker 已隔离耗时处理。没有实测容量/组织边界需要微服务；扫描补偿已覆盖现有故障场景，不同时维护第二套 outbox、Kafka 或复杂缓存。

## 归属与用户安全投影

`WebSession` 保存随机凭据摘要、服务端 owner 与到期时间，不接受客户端自报归属。匿名身份首先保证隔离，不提供实名、找回或跨设备迁移；旧 NULL owner 数据不会交给首个访问者。具体 Cookie、HTTP 错误和同源要求由应用契约统一定义。

前端只保留路由所需任务 ID，内容从后端恢复。结果投影只显示本任务证据可关联的 claims，摘要从保留的 claims 生成，避免隐藏无引用正文后仍泄露其自由摘要。原报告保留用于受控工程排障，不以修改历史报告“修复”展示。

## 检索与模型路线

| 路线 | 检索与生成 | 审核与限制 |
| --- | --- | --- |
| `fake_offline` | 固定证据和确定性 fixture | 只证明工程流程，不证明真实检索/模型 |
| `retrieval_mock` | 应用纯 BM25、中文双字切分；确定性短引摘录 | 结构关联与 ComplianceGuard；PARTIAL、confidence=null，非 NLI |
| `mimo_grounded` | 复用同一检索；`mimo-v2.5` 单路 JSON 选择完整原文 | 二次完整原文绑定与 ComplianceGuard，非医学语义审核 |
| 研究/既有模型链路 | 独立 `eval` 配置、既有适配器与多专科/NLI | 保留原研究门禁；`model_pipeline` 标签不等于真实联网 |

应用安全工厂校验显式配置，拒绝隐式向量/重排、研究配置替代或关闭泄露检查，不因 `.env` 中有密钥切到在线。纯 BM25 无需模型下载。空证据由共用分支在生成前弃答，词面命中却缺答案仍可能发生，不能承诺识别所有证据不足。

真实模式仅接受最多三条绑定完整原文的 claim，每条绑定一个本次 chunk；改写、拼接、重复、未知引用、状态矛盾、格式错误、非预期返回模型、缺响应标识或截断均拒绝。持久 SQLite 账本先占额度后发请求，失败不返额、重启不重置，不跟随重定向；上限与本次用量统一见 [8B 记录](development/rag-delivery.md#第-8b-轮真实-mimo-小预算验收)。不保存思维链。

**取舍：** 合法引用 ID 只证明可追踪，完整原文绑定也不保证相关性或现行医学适用性。中文应用不借英文研究 NLI 给结论背书；少量工程抽查不替代医学研究标注。用小语料保留漏检/误命中与保守弃答，比增加未验证的多 Agent 更符合当前范围。

## 观测与未实施范围

页面观测只覆盖已保存阶段，不是完整端到端计费账单；字段语义见应用契约。RAG 单次耗时与有界模型验收不外推为吞吐、稳定 P95 或诊断准确率。

未实施或未验收：账号恢复/历史自动认领、公网生产 TLS 与运维平台、多机容量、临床效果。工程 HUMAN 操作也不是临床专家审核。研究的语言兼容性、无效报告与混合实现版本事实保持独立。
