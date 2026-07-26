"""合规管控模块。

技术层面合规管控（不是真正医疗合规认证）:
- 超范围医疗问题拒答
- 绝对化诊断措辞拦截
- 强制风险提示插入
- 不当输出过滤
- 合规命中必须写入 trace / event log
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ComplianceResult:
    """合规检查结果。"""

    blocked: bool = False
    """是否拦截（绝对化措辞/超范围/不当输出）。"""

    block_reasons: list[str] = field(default_factory=list)
    """拦截原因列表。"""

    sanitized_text: str = ""
    """过滤后的文本（绝对化措辞替换为 ***）。"""

    disclaimer_added: bool = False
    """是否添加了强制免责声明。"""

    out_of_scope: bool = False
    """是否超范围问题。"""

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "block_reasons": self.block_reasons,
            "disclaimer_added": self.disclaimer_added,
            "out_of_scope": self.out_of_scope,
        }


# 绝对化措辞（从 eval/config.yaml compliance.block_absolute_terms 同步）
BLOCKED_ABSOLUTE_TERMS: list[str] = [
    "确诊",
    "保证治愈",
    "无需就医",
    "根治",
    "百分之百",
    "guaranteed cure",
    "definitely cure",
    "100% cure",
    "certain cure",
]

# 强制免责声明
MANDATORY_DISCLAIMER: str = "仅供学习和工程演示，不构成医疗建议。"

# 超范围问题关键词（非医学问题）
OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "股票",
    "投资",
    "法律建议",
    "心理咨询",
    "美容手术",
    "stock",
    "investment",
    "legal advice",
]


# 中文否定前缀。CJK 没有词边界，`\b` 在汉字之间不成立，因此不能用词边界排除
# 「尚未确诊」这类反例，只能显式列出否定语境。命中词紧邻其后时不算命中。
CJK_NEGATION_PREFIXES: tuple[str, ...] = (
    "尚未",
    "暂未",
    "还未",
    "并未",
    "从未",
    "没有",
    "无法",
    "不能",
    "难以",
    "无需",
    "排除",
    "未",
    "不",
    "非",
)


def find_term_spans(text: str, term: str) -> list[tuple[int, int]]:
    """返回 ``term`` 在 ``text`` 中的真实命中区间。

    ASCII 词条按词边界匹配：``stock`` 不得命中神经科真实体征
    ``stocking-glove distribution``。CJK 词条没有可用的词边界，改为显式排除
    ``CJK_NEGATION_PREFIXES`` 中的否定语境：``确诊`` 不得命中「尚未确诊」。

    两种策略都只降低误报，不试图理解语义；仍可能漏判更复杂的否定表达。
    """
    if not term:
        return []
    if term.isascii():
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        return [match.span() for match in pattern.finditer(text)]
    spans: list[tuple[int, int]] = []
    for match in re.finditer(re.escape(term), text):
        prefix = text[: match.start()]
        if any(prefix.endswith(negation) for negation in CJK_NEGATION_PREFIXES):
            continue
        spans.append(match.span())
    return spans


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠区间并按倒序返回，使从后往前替换不会互相破坏下标。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return list(reversed(merged))


class ComplianceGuard:
    """合规管控器。

    用法:
        guard = ComplianceGuard()
        result = guard.check(diagnosis_text)
        if result.blocked:
            # 拦截，记录 trace
        else:
            # 使用 result.sanitized_text
    """

    def __init__(
        self,
        blocked_terms: list[str] | None = None,
        disclaimer: str | None = None,
        out_of_scope_keywords: list[str] | None = None,
    ) -> None:
        self.blocked_terms = blocked_terms or BLOCKED_ABSOLUTE_TERMS
        self.disclaimer = disclaimer or MANDATORY_DISCLAIMER
        self.out_of_scope_keywords = (
            out_of_scope_keywords or OUT_OF_SCOPE_KEYWORDS
        )

    def check(self, text: str) -> ComplianceResult:
        """检查文本合规性。

        Args:
            text: 待检查的文本（诊断建议等）。

        Returns:
            ComplianceResult。
        """
        reasons: list[str] = []
        sanitized = text
        out_of_scope = False

        # 1. 超范围问题检查
        for kw in self.out_of_scope_keywords:
            if find_term_spans(text, kw):
                out_of_scope = True
                reasons.append(f"out_of_scope: {kw}")
                break

        # 2. 绝对化措辞拦截。只替换真实命中的区间，被否定语境排除的出现保持原样。
        blocked_spans: list[tuple[int, int]] = []
        for term in self.blocked_terms:
            spans = find_term_spans(text, term)
            if spans:
                reasons.append(f"absolute_term_blocked: {term}")
                blocked_spans.extend(spans)
        for start, end in _merge_spans(blocked_spans):
            sanitized = sanitized[:start] + "***" + sanitized[end:]

        # 3. 强制免责声明
        disclaimer_added = False
        if self.disclaimer not in sanitized:
            sanitized = sanitized.rstrip() + "\n" + self.disclaimer
            disclaimer_added = True

        # 4. 判定是否拦截
        blocked = len(reasons) > 0

        return ComplianceResult(
            blocked=blocked,
            block_reasons=reasons,
            sanitized_text=sanitized,
            disclaimer_added=disclaimer_added,
            out_of_scope=out_of_scope,
        )

    def check_output(self, output_dict: dict) -> ComplianceResult:
        """检查 Agent 输出字典的合规性。

        对输出中的所有文本字段进行检查。

        Args:
            output_dict: Agent 输出的字典形式。

        Returns:
            ComplianceResult。
        """
        # 合并所有文本字段
        text_parts: list[str] = []

        for d in output_dict.get("differential_diagnosis", []):
            text_parts.append(d.get("diagnosis", ""))

        for c in output_dict.get("claims", []):
            text_parts.append(c.get("text", ""))

        text_parts.extend(output_dict.get("risk_flags", []))
        text_parts.extend(output_dict.get("recommended_tests", []))

        if output_dict.get("uncertainty"):
            text_parts.append(output_dict["uncertainty"])

        combined_text = " ".join(text_parts)
        return self.check(combined_text)

    def get_out_of_scope_reject_template(self) -> str:
        """获取超范围问题拒答模板。"""
        return (
            "该问题超出本系统的模拟范围，无法提供诊断建议。"
            "请咨询专业医生。\n"
            + self.disclaimer
        )
