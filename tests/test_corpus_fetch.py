"""corpus_fetch 的离线单元测试：限速、失败记录与跳过已有文件，不访问网络。"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import corpus_fetch  # noqa: E402


class _FakeHeaders:
    def get_content_type(self):
        return "text/html"


class _Resp:
    def __init__(self, data: bytes, status: int = 200, url: str = "https://example.org/x"):
        self._data, self.status, self._url = data, status, url
        self.headers = _FakeHeaders()

    def read(self):
        return self._data

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Opener:
    """模拟 urllib opener；按队列返回成功或抛错，并记录请求。"""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.requests: list[str] = []

    def open(self, request, timeout=None):
        url = request.full_url
        self.requests.append(url)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome, url=url)


def _no_sleep(_seconds: float) -> None:
    return None


def test_throttle_按主机限速():
    sleeps: list[float] = []
    corpus_fetch._LAST_REQUEST.clear()
    url = "https://a.example.org/1"
    corpus_fetch._throttle(url, sleeps.append)
    corpus_fetch._throttle(url, sleeps.append)
    corpus_fetch._throttle("https://b.example.org/1", sleeps.append)  # 不同主机不限速
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= corpus_fetch.HOST_DELAY


def test_fetch_bytes_4xx不重试():
    opener = _Opener([urllib.error.HTTPError("https://example.org/x", 403, "Forbidden", {}, None)])
    result = corpus_fetch.fetch_bytes(opener, "https://example.org/x", _no_sleep)
    assert result["http_status"] == 403
    assert "403" in result["error"]
    assert len(opener.requests) == 1  # 客户端错误不重试


def test_fetch_bytes_网络错误重试后仍失败则记录():
    err = urllib.error.URLError("timeout")
    opener = _Opener([err, err, err])
    result = corpus_fetch.fetch_bytes(opener, "https://example.org/x", _no_sleep)
    assert result["error"] is not None
    assert result["sha256"] is None
    assert len(opener.requests) == corpus_fetch.MAX_RETRIES + 1


def test_fetch_documents_跳过已存在且哈希一致的文件(tmp_path):
    raw = tmp_path / "raw"
    doc = {"doc_id": "t1", "source_type": "who_factsheet", "url": "https://example.org/a", "format": "html",
           "license_conclusion": "纳入"}
    data = b"<html>ok</html>"
    import hashlib

    (raw / "who_factsheet").mkdir(parents=True)
    (raw / "who_factsheet" / "t1.html").write_bytes(data)
    (raw / "manifest.json").write_text(json.dumps({"t1": {"sha256": hashlib.sha256(data).hexdigest()}}), encoding="utf-8")

    opener = _Opener([data])
    report = corpus_fetch.fetch_documents([doc], raw, {"direct": opener, "wiki": opener}, _no_sleep)
    assert report[0]["skipped"] == "已存在且哈希一致"
    assert report[0]["error"] is None

    # 内容变化时应重写
    new_data = b"<html>new</html>"
    opener2 = _Opener([new_data])
    report2 = corpus_fetch.fetch_documents([doc], raw, {"direct": opener2, "wiki": opener2}, _no_sleep)
    assert report2[0].get("skipped") is None
    assert (raw / "who_factsheet" / "t1.html").read_bytes() == new_data


def test_fetch_documents_失败条目单列原因(tmp_path):
    raw = tmp_path / "raw"
    doc = {"doc_id": "bad1", "source_type": "nhc_document", "url": "https://example.org/404", "format": "html",
           "license_conclusion": "纳入"}
    opener = _Opener([urllib.error.HTTPError(doc["url"], 404, "Not Found", {}, None)])
    report = corpus_fetch.fetch_documents([doc], raw, {"direct": opener, "wiki": opener}, _no_sleep)
    assert report[0]["http_status"] == 404
    assert report[0]["error"] is not None
    assert not (raw / "nhc_document" / "bad1.html").exists()


def test_build_report_汇总成功与失败():
    report = [
        {"doc_id": "a", "url": "u1", "error": None, "http_status": 200},
        {"doc_id": "b", "url": "u2", "error": "HTTP 403", "http_status": 403},
    ]
    full = corpus_fetch.build_report(report, "2026-10-27T00:00:00Z", ["who_factsheet"])
    assert full["total"] == 2 and full["succeeded"] == 1 and full["failed_count"] == 1
    assert full["failed"] == [{"doc_id": "b", "url": "u2", "http_status": 403, "error": "HTTP 403"}]
