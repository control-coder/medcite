"""公开中文语料的抓取与逐字核对工具；只读公开网页，不提交网页全文。

fetch   把 URL 抓取为纯文本，写入调用方指定的（仓库外）缓存目录。
verify  对照缓存正文，逐字核对 chunks.jsonl 中每条摘录，输出未通过项。

核对时忽略所有空白字符的差异，其余字符必须完全一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (medidiag corpus verification; non-commercial research)"
_SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
_BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "div", "br", "tr", "section", "article"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._depth += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self._depth:
            self._depth -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (re.sub(r"[ \t　\xa0]+", " ", line).strip() for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def squash(text: str) -> str:
    """去掉全部空白，用于逐字比较。"""
    return re.sub(r"\s+", "", text)


def fetch(urls: dict[str, str], out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for doc_id, url in urls.items():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=40) as response:  # noqa: S310 公开网页只读
                status = response.status
                html = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # 记录失败，不中断其余页面
            manifest[doc_id] = {"url": url, "status": None, "error": str(exc)}
            continue
        text = html_to_text(html)
        (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        manifest[doc_id] = {"url": url, "status": status, "chars": len(text),
                            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return manifest


def verify(chunks_path: Path, pages_dir: Path) -> list[str]:
    """返回未通过核对的 chunk_id 与原因；source_id 必须对应缓存文件名。"""
    problems: list[str] = []
    cache: dict[str, str] = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        source_id = chunk["source_id"]
        if source_id not in cache:
            page = pages_dir / f"{source_id}.txt"
            cache[source_id] = squash(page.read_text(encoding="utf-8")) if page.exists() else ""
        if not cache[source_id]:
            problems.append(f"{chunk['chunk_id']}: 缺少缓存页面 {source_id}.txt")
        elif squash(chunk["text"]) not in cache[source_id]:
            problems.append(f"{chunk['chunk_id']}: 摘录未逐字出现在页面正文")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = sub.add_parser("fetch")
    fetch_cmd.add_argument("--out", type=Path, required=True)
    fetch_cmd.add_argument("pairs", nargs="+", help="doc_id=URL")
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--chunks", type=Path, required=True)
    verify_cmd.add_argument("--pages", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        urls = dict(pair.split("=", 1) for pair in args.pairs)
        print(json.dumps(fetch(urls, args.out), ensure_ascii=False, indent=2))
        return 0
    problems = verify(args.chunks, args.pages)
    print("\n".join(problems) if problems else "全部摘录逐字核对通过")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
