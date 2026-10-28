"""按阶段汇总 p50 / p90 耗时：读取数据库里 worker 落盘的阶段产物（stage_artifacts.latency_ms）。

用法：python scripts/stage_latency.py [--database-url URL] [--json 输出路径]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select

from medidiag.config import get_settings
from medidiag.db.models import StageArtifact
from medidiag.db.session import create_db_engine, get_session_factory


def percentile(values: list[int], q: float) -> float:
    """最近秩分位数（不插值），与小样本下的直观读法一致。"""
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


def summarize(rows: list[tuple[str, int]]) -> dict[str, dict[str, float]]:
    by_stage: dict[str, list[int]] = defaultdict(list)
    for stage, latency in rows:
        by_stage[stage].append(latency)
    return {
        stage: {"count": len(values), "p50_ms": percentile(values, 0.5), "p90_ms": percentile(values, 0.9),
                "max_ms": float(max(values))}
        for stage, values in sorted(by_stage.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None, help="默认读取 DATABASE_URL 配置")
    parser.add_argument("--json", type=Path, default=None, help="同时把汇总写成 JSON")
    args = parser.parse_args()
    engine = create_db_engine(args.database_url or get_settings().database_url)
    with get_session_factory(engine)() as session:
        rows = [(stage, int(latency)) for stage, latency in
                session.execute(select(StageArtifact.stage, StageArtifact.latency_ms))]
    summary = summarize(rows)
    if not summary:
        print("没有阶段产物，先跑过至少一个任务。")
    else:
        print(f"{'阶段':<16}{'次数':>6}{'p50(ms)':>10}{'p90(ms)':>10}{'max(ms)':>10}")
        for stage, item in summary.items():
            print(f"{stage:<16}{item['count']:>6}{item['p50_ms']:>10.0f}{item['p90_ms']:>10.0f}{item['max_ms']:>10.0f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
