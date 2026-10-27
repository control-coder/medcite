"""v4 公开语料采集：按来源抓取快照并写入 .cache/corpus/raw/，只存原始字节，不做转换。

数据源来自 ``examples/public_health_v4/sources.json``；抓取结果（HTTP 状态、最终 URL、
Content-Type、字节数、响应 sha256、抓取时间）写入本机清单 ``.cache/corpus/raw/manifest.json``
以及可提交的 ``examples/public_health_v4/fetch_report.json``。

约定：
- 同一主机请求间隔至少 ``HOST_DELAY`` 秒；单个请求失败最多重试 ``MAX_RETRIES`` 次。
- User-Agent 使用描述性字符串；不绕过任何访问控制，遇到 403/412 等拦截直接记为失败。
- 维基百科需要经过本机系统代理访问；代理地址从 Windows 注册表读取，
  也可用 ``--proxy`` 或环境变量 ``MEDCITE_PROXY`` 显式指定。
- 支持 ``--source-type`` 只抓某一类来源；已存在且哈希一致的文件跳过。

用法：
    python scripts/corpus_fetch.py --sources examples/public_health_v4/sources.json
    python scripts/corpus_fetch.py --sources examples/public_health_v4/sources.json --source-type wikipedia
    python scripts/corpus_fetch.py --sources examples/public_health_v4/sources.json --report .cache/fetch_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

USER_AGENT = "MedCite/0.1 (public health RAG engineering prototype; non-commercial; contact: local dev)"
HOST_DELAY = 1.5  # 同一主机两次请求的最小间隔（秒）
MAX_RETRIES = 2  # 单个请求失败后的最大重试次数
TIMEOUT = 60
WIKI_API = "https://zh.wikipedia.org/w/api.php"
WIKI_PAGE_BASE = "https://zh.wikipedia.org/wiki/"
# 维基媒体基金会对未声明联系方式的批量访问限速较严，单独放宽间隔与重试
WIKI_HOST_DELAY = 3.0
WIKI_MAX_RETRIES = 4

_LAST_REQUEST: dict[str, float] = {}


def _throttle(url: str, sleep: Callable[[float], None] = time.sleep, host_delay: float = HOST_DELAY) -> None:
    """保证同一主机两次请求间隔不少于 host_delay。"""
    host = urllib.parse.urlparse(url).netloc
    now = time.monotonic()
    last = _LAST_REQUEST.get(host)
    if last is not None:
        wait = host_delay - (now - last)
        if wait > 0:
            sleep(wait)
    _LAST_REQUEST[host] = time.monotonic()


def _build_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    """构造 opener；HTTPS 走默认 TLS 上下文。代理为 None 时直连。"""
    context = ssl.create_default_context()
    handlers: list[Any] = [urllib.request.HTTPSHandler(context=context)]
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def system_proxy() -> str | None:
    """从 Windows 注册表读取 IE 代理；非 Windows 或无代理返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not enabled:
                return None
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
        if server and "=" not in server:
            return f"http://{server}"
        return server or None
    except OSError:
        return None


def fetch_bytes(opener: Any, url: str, sleep: Callable[[float], None] = time.sleep,
                host_delay: float = HOST_DELAY, max_retries: int = MAX_RETRIES) -> dict[str, Any]:
    """抓取一个 URL，返回原始字节与响应元数据；失败时返回 error，不抛异常。"""
    record: dict[str, Any] = {"final_url": None, "http_status": None, "content_type": None,
                              "bytes": None, "sha256": None, "error": None, "data": None}
    # URL 必须 ASCII 化：quote path 与 query 中的非 ASCII 字符（如 é、中文）
    parts = urllib.parse.urlsplit(url)
    safe_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc,
                                        urllib.parse.quote(parts.path, safe="/:@!$&'()*+,;=-._~()%"),
                                        urllib.parse.quote(parts.query, safe="=&?/:@!$'()*+,;=-._~()%"),
                                        parts.fragment))
    for attempt in range(max_retries + 1):
        _throttle(safe_url, sleep, host_delay)
        request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
        try:
            with opener.open(request, timeout=TIMEOUT) as response:  # noqa: S310 公开网页只读
                data = response.read()
                record.update(final_url=response.geturl(), http_status=response.status,
                              content_type=response.headers.get_content_type(), bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest(), data=data, error=None)
                return record
        except urllib.error.HTTPError as exc:
            record.update(final_url=url, http_status=exc.code, error=f"HTTP {exc.code} {exc.reason}")
            if 400 <= exc.code < 500 and exc.code != 429:  # 客户端错误（含反爬 403/412）不重试；429 等待后重试
                break
        except http.client.IncompleteRead as exc:
            # 代理或连接中断导致的 chunked 截断；记录已读字节数，按失败重试
            record["error"] = f"IncompleteRead: {len(exc.partial or b'')} bytes read"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            sleep(2 * (attempt + 1) * host_delay)
    return record


def fetch_documents(documents: list[dict[str, Any]], raw_dir: Path,
                    openers: dict[str, Any], sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:
    """按清单逐条抓取，快照写入 raw_dir/<source_type>/<doc_id>.<format>，返回报告条目。"""
    manifest_path = raw_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    report: list[dict[str, Any]] = []
    for doc in documents:
        doc_id, source_type = doc["doc_id"], doc["source_type"]
        entry: dict[str, Any] = {"doc_id": doc_id, "url": doc["url"], "final_url": None, "http_status": None,
                                 "content_type": None, "bytes": None, "sha256": None,
                                 "accessed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "error": None}
        if source_type == "wikipedia":
            result = _fetch_wikipedia(doc, openers["wiki"], sleep)
        else:
            result = fetch_bytes(openers["direct"], doc["url"], sleep)
        entry.update({k: result[k] for k in ("final_url", "http_status", "content_type", "bytes", "sha256", "error")})
        if result["error"] is None and result["data"] is not None:
            ext = doc.get("format") or "bin"
            out_path = raw_dir / source_type / f"{doc_id}.{ext}"
            old = manifest.get(doc_id)
            if out_path.exists() and old and old.get("sha256") == result["sha256"]:
                entry["skipped"] = "已存在且哈希一致"
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(result["data"])
            manifest[doc_id] = {"url": entry["final_url"], "sha256": result["sha256"],
                                "bytes": result["bytes"], "accessed_at": entry["accessed_at"],
                                "source_type": source_type, "format": ext}
            if result.get("revid") is not None:
                manifest[doc_id]["revid"] = result["revid"]
                entry["revid"] = result["revid"]
        report.append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _fetch_wikipedia(doc: dict[str, Any], opener: Any, sleep: Callable[[float], None]) -> dict[str, Any]:
    """调用 MediaWiki API action=parse（variant=zh-cn，保存原始 JSON 响应）。"""
    title = urllib.parse.unquote(doc["url"].removeprefix(WIKI_PAGE_BASE))
    params = {"action": "parse", "page": title, "prop": "text|revid|displaytitle",
              "variant": "zh-cn", "format": "json", "formatversion": "2", "redirects": "1"}
    result = fetch_bytes(opener, WIKI_API + "?" + urllib.parse.urlencode(params), sleep,
                         host_delay=WIKI_HOST_DELAY, max_retries=WIKI_MAX_RETRIES)
    if result["error"] is None and result["data"] is not None:
        try:
            payload = json.loads(result["data"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            result["error"] = f"invalid json: {exc}"
            return result
        if "error" in payload:
            result["error"] = f"mediawiki: {payload['error'].get('code', 'unknown')}"
        else:
            result["revid"] = payload.get("parse", {}).get("revid")
    return result


def build_report(report: list[dict[str, Any]], run_at: str, source_types: list[str]) -> dict[str, Any]:
    failed = [r for r in report if r["error"]]
    return {"version": "public-health-v4-fetch", "run_at": run_at,
            "user_agent": USER_AGENT, "host_delay_seconds": HOST_DELAY, "max_retries": MAX_RETRIES,
            "source_types": source_types, "total": len(report), "succeeded": len(report) - len(failed),
            "failed_count": len(failed),
            "failed": [{"doc_id": r["doc_id"], "url": r["url"], "http_status": r["http_status"],
                        "error": r["error"]} for r in failed],
            "records": [{k: v for k, v in r.items() if v is not None} for r in report]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, default=Path("examples/public_health_v4/sources.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path(".cache/corpus/raw"))
    parser.add_argument("--report", type=Path, default=Path("examples/public_health_v4/fetch_report.json"))
    parser.add_argument("--source-type", choices=["who_factsheet", "who_publication", "nhc_document", "epidemic_report", "wikipedia"])
    parser.add_argument("--proxy", default=None, help="维基百科使用的代理；缺省读系统代理，可用 MEDCITE_PROXY 覆盖")
    args = parser.parse_args()

    import os

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    documents = sources["documents"]
    if args.source_type:
        documents = [d for d in documents if d["source_type"] == args.source_type]
    included = [d for d in documents if d.get("license_conclusion") == "纳入"]
    excluded = [d for d in documents if d.get("license_conclusion") != "纳入"]

    proxy = args.proxy or os.environ.get("MEDCITE_PROXY") or system_proxy()
    openers = {"direct": _build_opener(None), "wiki": _build_opener(proxy)}

    run_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = fetch_documents(included, args.raw_dir, openers)
    source_types = sorted({d["source_type"] for d in documents})
    full_report = build_report(report, run_at, source_types)
    if excluded:
        full_report["not_fetched_license_excluded"] = [
            {"doc_id": d["doc_id"], "reason": d.get("exclusion_reason")} for d in excluded]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(full_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: full_report[k] for k in ("total", "succeeded", "failed_count")}, ensure_ascii=False))
    if full_report["failed"]:
        print(json.dumps(full_report["failed"], ensure_ascii=False, indent=2))
    return 0 if not full_report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
