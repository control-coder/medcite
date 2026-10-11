"""配置加载。

环境变量（pydantic-settings）：API key、数据库 URL、运行时参数。
应用的检索与生成方案在 ``configs/application*.yaml``，由 ``workflow/application.py`` 校验后加载。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时环境变量配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 数据库
    database_url: str = "sqlite:///./medidiag.db"

    medidiag_allowed_hosts: list[str] = ["127.0.0.1", "localhost", "[::1]", "testserver"]

    # 队列只保存任务标识，独立扫描器负责提交与派发窗口的补偿。
    medidiag_broker_url: str = "redis://127.0.0.1:16379/0"
    medidiag_dispatch_seconds: int = 5

    # Hugging Face 模型缓存
    hf_home: str = ".cache/huggingface"

    # 应用队列默认 fixture；真实检索也不允许隐式调用模型。
    medidiag_app_provider: Literal["fake_offline", "retrieval_mock"] = "fake_offline"
    medidiag_app_config: str = "configs/application.yaml"

    # 工作流运行时
    medidiag_lease_seconds: int = 60
    medidiag_heartbeat_seconds: int = 20
    medidiag_lease_scan_seconds: int = 30
    # 复核轮次上限：被驳回后连续失败达到该次数就升级人工。
    medidiag_max_review_rounds: int = 3

    # 日志
    log_level: str = "INFO"
    structlog_dev: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例 Settings。"""
    return Settings()
