# 应用架构与取舍

更新：2026-09-20。模块化单体 + 独立 worker，不引入微服务集群。

```mermaid
flowchart LR
    U[React 四页 /app/] --> API[FastAPI 单体]
    H[/assistant 与 /demo] --> API
    API --> ID[服务端匿名会话与 owner 过滤]
    ID --> DB[(PostgreSQL 权威状态)]
    D[数据库派发与补偿扫描] --> DB
    D --> R[(Redis 标识消息)]
    R --> C[Celery 独立 worker]
    C --> W[共用状态机 CAS 租约与阶段执行器]
    W --> DB
    W --> P[Provider 接口]
    P --> F[默认确定性离线 fixture]
    P -. 单独授权与验证 .-> L[既有真实模型/RAG 适配器]
    DB --> O[报告安全投影与证据关联]
    O --> U
    E[独立研究 eval 与固定 RAG 小回归] -. 不作为交付门禁 .-> L
```

离线最短模式将 PostgreSQL 换成新建 SQLite，将队列换成同进程演示 worker；`verify_offline.py` 则使用 SQLite 与独立 API/worker 两个进程。这些入口复用执行器，而非复制业务逻辑。

## 数据与一致性

- Case 持有 owner、状态、版本与 active_task；WorkflowTask 持有 attempt、租约、领取者与错误码。
- StageArtifact、CaseReport、CaseEventLog 保存阶段输出、版本化报告与可追踪事件。外部 Provider 调用前结束读取事务，写入时重新 fencing/CAS。
- WebSession 只保存随机凭据摘要、服务端 owner 与到期时间，不接受用户自报归属。历史 NULL owner 不迁给首个访问者。
- 数据库提交与队列发布有窗口，采用定期扫描补偿；允许重复消息，重新领取并校验有效租约，不借已有租约重复运行。
- 用户取消使任务失效与病例终止同事务提交；迟到结果被拒绝。状态机仍禁止 API 绕过 ESCALATED 的 HUMAN 边界。

## 边界与取舍

1. 简单匿名持有凭据身份，先保障归属隔离，不堆账号注册/恢复系统。不是实名医疗服务，不承诺公网生产安全。
2. 不同时实现 outbox、Kafka 或复杂缓存；默认每 5 秒扫描、每批 100，足够本轮工程验收，不宣传吞吐量。
3. 先展示引用关联与明确弃答。关联只证明 ID 可追踪，不证明证据支持每个医学命题；NLI 分数不等于医学正确性。
4. 默认演示固定 fixture，不隐式加载大模型或付费 API。保留既有研究与真实 Provider 接口，但本轮只使用离线/模拟传输验收。
5. 运行观测区分已记录耗时、部分 Token 与未知费用；没有完整账单不估算零费用。
6. React 通过 hash 路由保存任务标识，刷新从后端恢复；浏览器不持久保存症状。本人历史是服务端查询，不依赖 localStorage 列表。

未实施：账号找回/跨设备迁移、自动认领历史病例、生产 TLS/监控/审计平台、多机容量验收、临床验证。均不在本次交付中隐含宣称已具备。
