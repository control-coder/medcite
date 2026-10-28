"""进程内检索结果缓存：LRU + TTL，线程安全。

只缓存检索结果（证据快照与过滤轨迹），不缓存模型回答。键里带语料哈希、语料版本与检索配置哈希，
所以换语料或换检索参数不会复用旧结果；进程重启即清空。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

DEFAULT_MAX_ENTRIES = 256
DEFAULT_TTL_S = 3600.0


def cache_key(query: str, *, top_k: int | None, filters: Mapping[str, Any] | None,
              corpus_hash: str, corpus_version: str, config_hash: str) -> str:
    """规范化问题（合并空白）、过滤条件、top_k 与语料/配置版本，取 sha256。"""
    payload = {
        "q": " ".join(query.split()), "top_k": top_k,
        "filters": dict(filters) if filters else None,
        "corpus_hash": corpus_hash, "corpus_version": corpus_version, "config_hash": config_hash,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RetrievalCache:
    """定长 LRU，条目超过 TTL 视为未命中并删除。值在存取时深拷贝，调用方改动不会污染缓存。"""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, ttl_s: float = DEFAULT_TTL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_entries < 1 or ttl_s <= 0:
            raise ValueError("缓存容量与 TTL 必须为正数")
        self.max_entries, self.ttl_s, self._clock = max_entries, ttl_s, clock
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            entry = self._items.get(key)
            if entry is not None and self._clock() - entry[0] <= self.ttl_s:
                self._items.move_to_end(key)
                self.hits += 1
                return True, copy.deepcopy(entry[1])
            if entry is not None:
                del self._items[key]
            self.misses += 1
            return False, None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = (self._clock(), copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._items)}
