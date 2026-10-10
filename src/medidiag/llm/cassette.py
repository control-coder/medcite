"""LLM 响应录制文件：录制真实调用，之后可零网络、零费用回放。

两个类都是 ``post`` 兼容的可调用对象，可直接注入 ``OpenAICompatibleProvider(post=...)``。

- ``RecordingTransport``：先查录制文件，命中则不联网；未命中才占用持久预算后请求并写入。
- ``ReplayTransport``：只读录制文件，未命中直接失败，绝不联网。

录制文件以请求体的哈希为键（加 trial 序号，用于对同一请求做重复采样），只保存响应的必要字段
（id、model、首个 choice 的 content 与 finish_reason、usage），不保存请求头、密钥、思维链或错误响应。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from medidiag.errors import MediDiagError

ALLOWED_HOST = "api.xiaomimimo.com"


def request_key(body: dict[str, Any], trial: int = 0) -> str:
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{canonical}#{trial}".encode()).hexdigest()


def sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    """只留回放需要的字段；丢弃 reasoning_content 等其他服务器字段。"""
    choices = payload.get("choices")
    choice: dict[str, Any] = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    raw_message = choice.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    raw_usage = payload.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return {
        "id": str(payload.get("id", ""))[:128],
        "model": str(payload.get("model", ""))[:128],
        "choices": [{"message": {"role": "assistant", "content": message.get("content", "")},
                     "finish_reason": choice.get("finish_reason")}],
        "usage": {k: v for k, v in usage.items()
                  if k in {"prompt_tokens", "completion_tokens", "total_tokens"} and type(v) is int and v >= 0},
    }


class Cassette:
    """追加写的 JSONL 录制文件；读写都受锁保护，可被多线程共享。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, body: dict[str, Any], elapsed_ms: int) -> None:
        entry = {"key": key, "status": 200, "elapsed_ms": elapsed_ms, "body": sanitize(body)}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            self._entries[key] = entry


def _response(url: str, entry: dict[str, Any]) -> httpx.Response:
    return httpx.Response(entry["status"], json=entry["body"], request=httpx.Request("POST", url))


class ReplayTransport:
    """只读回放；未命中即失败。"""

    def __init__(self, cassette: str | Path | Cassette, *, trial: int = 0) -> None:
        self.cassette = cassette if isinstance(cassette, Cassette) else Cassette(cassette)
        self.trial = trial

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float,
                 **_: Any) -> httpx.Response:
        entry = self.cassette.get(request_key(json, self.trial))
        if entry is None:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="录制文件未命中：请求与录制时不一致")
        return _response(url, entry)


class RecordingTransport:
    """录制 + 去重：已录过的请求不再付费；未录过的请求受持久预算上限约束。"""

    def __init__(self, cassette: str | Path | Cassette, ledger: str | Path, *, max_calls: int, trial: int = 0,
                 post: Callable[..., httpx.Response] | None = None) -> None:
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls 必须是正整数")
        self.cassette = cassette if isinstance(cassette, Cassette) else Cassette(cassette)
        self.ledger = Path(ledger).resolve()
        self.max_calls = max_calls
        self.trial = trial
        self.post = post or httpx.post
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.ledger) as db:
            db.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), cap INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO budget VALUES (1, ?)", (max_calls,))
            if db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0] != max_calls:
                raise ValueError("不能通过重启修改已建立的调用预算")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, state TEXT NOT NULL, elapsed_ms INTEGER, http_status INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER, error_type TEXT)""")

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float,
                 **_: Any) -> httpx.Response:
        key = request_key(json, self.trial)
        with self._lock:
            key_lock = self._inflight.setdefault(key, threading.Lock())
        with key_lock:  # 并发的相同请求只付一次费，后到者读取先到者刚写入的录像
            cached = self.cassette.get(key)
            if cached is not None:
                return _response(url, cached)
            return self._request(key, url, headers, json, timeout)

    def _request(self, key: str, url: str, headers: dict[str, str], json: dict[str, Any],
                 timeout: float) -> httpx.Response:
        target = urlsplit(url)
        if (target.scheme != "https" or target.hostname != ALLOWED_HOST or target.path != "/v1/chat/completions"
                or target.query or target.fragment or target.username or target.password
                or target.port not in (None, 443)):
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="只允许指定官方 HTTPS 接口")
        if (json.get("model") != "mimo-v2.5" or type(json.get("max_tokens")) is not int
                or not 1 <= json["max_tokens"] <= 2048 or not 0 < timeout <= 60
                or len(str(json.get("messages", []))) > 12000):
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="请求超出模型、Token、输入或超时边界")
        with self._lock, sqlite3.connect(self.ledger, timeout=10) as db:
            if db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] >= self.max_calls:
                raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="真实调用额度已用尽")
            call_id = db.execute("INSERT INTO calls(state) VALUES ('reserved')").lastrowid
        started = time.perf_counter()
        try:
            response = self.post(url, headers=headers, json=json, timeout=timeout, follow_redirects=False)
        except Exception as exc:
            self._finish(call_id, started, "transport_error", error_type=type(exc).__name__)
            raise
        elapsed = round((time.perf_counter() - started) * 1000)
        try:
            body = response.json()
        except ValueError:
            body = None
        body = body if isinstance(body, dict) else {}
        usage = sanitize(body)["usage"]
        self._finish(call_id, started, "response", http_status=response.status_code,
                     prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
        if 300 <= response.status_code < 400:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="禁止重定向")
        if response.status_code == 200 and body:
            self.cassette.put(key, body, elapsed)
        return response

    def _finish(self, call_id: Any, started: float, state: str, **fields: Any) -> None:
        fields.update(state=state, elapsed_ms=round((time.perf_counter() - started) * 1000))
        with self._lock, sqlite3.connect(self.ledger) as db:
            db.execute("UPDATE calls SET " + ",".join(f"{k}=?" for k in fields) + " WHERE id=?",
                       [*fields.values(), call_id])

    def spent(self) -> dict[str, Any]:
        with sqlite3.connect(self.ledger) as db:
            calls, ok, prompt, completion = db.execute(
                "SELECT COUNT(*), SUM(http_status=200), SUM(prompt_tokens), SUM(completion_tokens) FROM calls").fetchone()
        return {"calls": calls, "http_200": ok or 0, "prompt_tokens": prompt or 0, "completion_tokens": completion or 0}
