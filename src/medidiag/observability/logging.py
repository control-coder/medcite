"""结构化日志。

设计约束（与 `trace_exporter` 相同的脱敏契约）：

- **只记录标识符、计数、状态和错误码**。不得记录病例问题正文、证据文本、
  prompt、完整 provider 响应或 API key。数据库事件日志与 trace 导出负责
  「发生了什么内容」，日志负责「进程当时在做什么」。
- 脱敏处理器是兜底，不是许可：它按 `medidiag.observability.redaction` 的
  规则屏蔽已知敏感键名与模式，并把超长字符串截断为长度标记——一段被误传进来的
  prompt 或 provider 响应会以 ``[TRUNCATED:1842]`` 的形式暴露出来，而不是
  完整落盘。
- 日志不是审计证据。任何评测或合规结论仍只能来自数据库事件与 trace 产物。

用法::

    from medidiag.observability.logging import get_logger

    logger = get_logger(__name__)
    logger.info("worker.stage_completed", case_id=..., stage=..., latency_ms=...)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any, TextIO

import structlog

from medidiag.config import get_settings
from medidiag.observability.redaction import sanitize

#: 超过该长度的字符串值不会原样输出。正常的日志字段是 ID、状态和错误码，
#: 没有一个需要接近这个长度；触发截断本身就是"有人传了正文"的信号。
MAX_VALUE_LENGTH = 256

_configured = False


def _redact_processor(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """按共享脱敏契约处理整个事件字典，并截断超长值。"""
    redacted = sanitize(dict(event_dict))
    return {key: _truncate(value) for key, value in redacted.items()}


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_VALUE_LENGTH:
        return f"[TRUNCATED:{len(value)}]"
    return value


def configure_logging(
    *, stream: TextIO | None = None, force: bool = False
) -> None:
    """配置 structlog。重复调用是无操作，除非 ``force=True``。

    ``LOG_LEVEL`` 控制级别，``STRUCTLOG_DEV`` 非零时使用彩色控制台渲染，
    否则输出单行 JSON（便于日后接入日志采集）。
    """
    global _configured
    if _configured and not force:
        return
    settings = get_settings()
    level = logging.getLevelNamesMapping().get(
        str(settings.log_level).upper(), logging.INFO
    )
    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=False)
        if settings.structlog_dev
        else structlog.processors.JSONRenderer(ensure_ascii=False, sort_keys=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # 脱敏必须紧邻渲染器之前，覆盖上面所有处理器加入的字段。
            _redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str) -> Any:
    """返回绑定组件名的 logger，首次调用时惰性完成配置。

    惰性配置保证脱敏处理器总是生效：任何组件都不可能在未配置的情况下用
    structlog 默认链路输出未脱敏内容。
    """
    configure_logging()
    return structlog.get_logger(name)
