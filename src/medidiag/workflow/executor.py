"""任务执行器占位。

阶段 3 实现：
- worker 租约（lease_owner / lease_until / heartbeat_at / attempt）
- 幂等键（创建病例 / 启动工作流 / Agent 执行）
- 乐观锁冲突自动重试（3 次退避 50/100/200ms）
- RUNNING 重复请求处理
- 租约超时扫描器
- 旧 worker 迟到写入防护（TASK_LEASE_LOST）
"""

from __future__ import annotations


class TaskExecutor:
    """任务执行器骨架。

    TODO[阶段3]: 实现完整的租约、幂等、乐观锁、租约扫描、脑裂防护。
    """

    def __init__(self) -> None:
        raise NotImplementedError("TaskExecutor 将在阶段 3 实现")
