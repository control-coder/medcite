"""配置加载。

两层配置：
1. 环境变量（pydantic-settings）：API key、数据库 URL、运行时参数。
2. 评测配置（eval/config.yaml）：锁定模型、温度、seed、数据集版本、检索权重、运行命令。

评测配置必须可复现，换任何字段需重跑全部评测。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时环境变量配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    deepseek_api_key: str = ""
    # DeepSeek 官方 OpenAI-compatible API。
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: int = 60

    # Database
    database_url: str = "sqlite:///./medidiag.db"

    # HuggingFace
    hf_home: str = ".cache/huggingface"

    # Workflow runtime
    medidiag_lease_seconds: int = 60
    medidiag_heartbeat_seconds: int = 20
    medidiag_lease_scan_seconds: int = 30
    # 复核轮次上限。与 eval/config.yaml 的 workflow.review.max_review_rounds
    # 有意重复：运行时 worker 只读 Settings，评测器只读该 YAML，两条链路互不
    # 依赖。改动其一时需同步另一处。
    medidiag_max_review_rounds: int = 3

    # Logging
    log_level: str = "INFO"
    structlog_dev: int = 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例 Settings。"""
    return Settings()


@lru_cache(maxsize=8)
def load_eval_config(path: str | Path) -> dict[str, Any]:
    """加载评测配置 YAML。

    评测配置是可复现性的核心，任何字段变更都意味着需要重跑 baseline + 消融。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"eval config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"eval config must be a mapping, got {type(cfg)}")
    return cfg
