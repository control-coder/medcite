"""真实验收的持久调用限额；先占用额度再发包，进程重启不重置预算。"""
from __future__ import annotations

import json as jsonlib
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from medidiag.errors import MediDiagError


class BudgetedTransport:
    """仅支持本轮指定官方接口；不记录请求正文、请求头或异常原文。"""

    def __init__(self, ledger: str | Path, *, max_calls: int = 8, post=None) -> None:
        self.path = Path(ledger).resolve()
        if type(max_calls) is not int or not 1 <= max_calls <= 8:
            raise ValueError("单次验收最多允许 8 次真实请求")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.post = post or httpx.post
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?)", (max_calls,))
            if db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0] != max_calls:
                raise ValueError("不能通过重启修改已建立的调用预算")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, state TEXT NOT NULL, elapsed_ms INTEGER,
                http_status INTEGER, model TEXT, response_id TEXT, usage TEXT, error_type TEXT)""")

    def __call__(self, url: str, *, headers: dict, json: dict, timeout: float) -> httpx.Response:
        target = urlsplit(url)
        if (target.scheme != "https" or target.hostname != "api.xiaomimimo.com"
                or target.path != "/v1/chat/completions" or target.query or target.fragment
                or target.username or target.password or target.port not in (None, 443)):
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="验收只允许指定官方 HTTPS 接口")
        if (json.get("model") != "mimo-v2.5" or type(json.get("max_tokens")) is not int
                or not 1 <= json.get("max_tokens", 0) <= 2048
                or not 0 < timeout <= 60 or len(str(json.get("messages", []))) > 12000):
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="请求超出本轮模型、Token、输入或超时边界")
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] >= self.max_calls:
                raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="真实验收调用额度已用尽")
            call_id = db.execute("INSERT INTO calls(state) VALUES ('reserved')").lastrowid
        started = time.perf_counter()
        try:
            response = self.post(url, headers=headers, json=json, timeout=timeout, follow_redirects=False)
        except Exception as exc:
            self._finish(call_id, started, "transport_error", error_type=type(exc).__name__)
            raise
        try:
            body = response.json()
            if not isinstance(body, dict):
                body = {}
        except ValueError:
            body = {}
        raw_usage = body.get("usage")
        usage = {k: v for k, v in (raw_usage if isinstance(raw_usage, dict) else {}).items()
                 if k in {"prompt_tokens", "completion_tokens", "total_tokens"} and type(v) is int and v >= 0}
        # 只存必要计量；不保存思维链、错误响应文本、提示词或任意服务器字段。
        self._finish(call_id, started, "response", http_status=response.status_code,
                     model=str(body.get("model", ""))[:128], response_id=str(body.get("id", ""))[:128],
                     usage=jsonlib.dumps(usage))
        if 300 <= response.status_code < 400:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="真实验收禁止重定向")
        return response

    def _finish(self, call_id: int, started: float, state: str, **fields: Any) -> None:
        fields.update(state=state, elapsed_ms=round((time.perf_counter() - started) * 1000))
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE calls SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                       [*fields.values(), call_id])

    def records(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute("SELECT * FROM calls ORDER BY id")]
        for row in rows:
            row["usage"] = jsonlib.loads(row["usage"]) if row["usage"] else None
        return rows
