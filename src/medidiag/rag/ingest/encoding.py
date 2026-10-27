"""编码检测：统一把原始字节转为 UTF-8 文本。

政府网站可能是 GBK 或 GB18030。优先级：HTTP/HTML meta 声明 > charset-normalizer
检测 > UTF-8 兜底。检测结果写入文档元数据，供报告与排查使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_META_CHARSET = re.compile(
    r"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)
_META_CONTENT_TYPE = re.compile(
    r"""<meta[^>]+content\s*=\s*["'][^"']*charset=\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)

# 对中文场景做归一：GBK/GB2312 统一按 GB18030 解码（GB18030 是 GBK 的超集）
_NORMALIZE = {
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "gb18030": "gb18030",
    "utf-8": "utf-8",
    "utf8": "utf-8",
}


@dataclass
class EncodingResult:
    """解码结果与检测信息。"""

    text: str
    encoding: str  # 实际用于解码的编码（归一后）
    declared: str | None  # HTTP 头或 HTML meta 声明的编码
    detected: str | None  # charset-normalizer 检测的编码
    source: str  # 编码来源：declared / detected / fallback


def _normalize(name: str | None) -> str | None:
    if not name:
        return None
    return _NORMALIZE.get(name.strip().lower(), name.strip().lower())


def _declared_from_html(raw: bytes) -> str | None:
    """从 HTML 前 4KB 的 meta 标签里读编码声明（不触发完整解码）。"""
    head = raw[:4096].decode("latin-1", errors="ignore")
    for pattern in (_META_CHARSET, _META_CONTENT_TYPE):
        match = pattern.search(head)
        if match:
            return _normalize(match.group(1))
    return None


def decode_bytes(raw: bytes, declared: str | None = None) -> EncodingResult:
    """把原始字节解码为文本。

    ``declared`` 是调用方已知的声明编码（例如 HTTP Content-Type）；HTML 内容
    还会再检查 meta 声明。两者都没有时用 charset-normalizer 检测。
    """
    declared_norm = _normalize(declared) or _declared_from_html(raw)

    if declared_norm:
        try:
            return EncodingResult(
                text=raw.decode(declared_norm, errors="replace"),
                encoding=declared_norm,
                declared=declared_norm,
                detected=None,
                source="declared",
            )
        except LookupError:
            pass  # 声明了系统不认识的编码名，继续走检测

    from charset_normalizer import from_bytes

    # 中文项目语境：检测候选限定在常见中文编码与 UTF-8。
    # charset-normalizer 在短中文文本上容易误判成韩文/日文编码，
    # 按候选顺序取第一个能解码且不含替换符的；GB18030 兼容 GBK。
    candidates: list[str] = []
    for match in from_bytes(raw):
        name = _normalize(match.encoding)
        if name and name in {"utf-8", "gb18030", "big5"} and name not in candidates:
            candidates.append(name)
    for name in ("utf-8", "gb18030"):
        if name not in candidates:
            candidates.append(name)

    for detected in candidates:
        try:
            text = raw.decode(detected, errors="strict")
            return EncodingResult(
                text=text,
                encoding=detected,
                declared=declared_norm,
                detected=detected,
                source="detected",
            )
        except (LookupError, UnicodeDecodeError):
            continue

    return EncodingResult(
        text=raw.decode("utf-8", errors="replace"),
        encoding="utf-8",
        declared=declared_norm,
        detected=None,
        source="fallback",
    )
