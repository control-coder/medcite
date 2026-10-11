"""合规守卫与复核轮次上限的测试。"""

from __future__ import annotations

import pytest

from medidiag.compliance.guard import ComplianceGuard
from medidiag.review.logic import ClinicalLogicReviewer


def test_should_escalate_after_consecutive_failures() -> None:
    """连续失败达到上限（默认 3 次）才升级人工。"""
    assert ClinicalLogicReviewer.should_escalate(2) is False
    assert ClinicalLogicReviewer.should_escalate(3) is True
    assert ClinicalLogicReviewer.should_escalate(5) is True
    assert ClinicalLogicReviewer.should_escalate(1, max_rounds=1) is True


class TestComplianceGuard:
    @pytest.fixture
    def guard(self) -> ComplianceGuard:
        return ComplianceGuard()

    def test_absolute_term_blocked(self, guard: ComplianceGuard) -> None:
        """绝对化措辞拦截。"""
        result = guard.check("确诊为心肌梗死，保证治愈")
        assert result.blocked is True
        assert any("absolute_term" in r for r in result.block_reasons)
        assert "***" in result.sanitized_text

    def test_disclaimer_added(self, guard: ComplianceGuard) -> None:
        """强制免责声明。"""
        result = guard.check("patient has fever")
        assert result.disclaimer_added is True
        assert "仅供学习" in result.sanitized_text

    def test_disclaimer_not_duplicated(self, guard: ComplianceGuard) -> None:
        """已有免责声明不重复添加。"""
        text = "patient has fever\n仅供学习和工程演示，不构成医疗建议。"
        result = guard.check(text)
        assert result.disclaimer_added is False

    def test_out_of_scope(self, guard: ComplianceGuard) -> None:
        """超范围问题。"""
        result = guard.check("请推荐股票投资策略")
        assert result.out_of_scope is True
        assert result.blocked is True

    def test_normal_text_not_blocked(self, guard: ComplianceGuard) -> None:
        """正常文本不拦截。"""
        result = guard.check("patient has chest pain and fever")
        assert result.blocked is False
        assert result.disclaimer_added is True

    def test_reject_template(self, guard: ComplianceGuard) -> None:
        """拒答模板。"""
        template = guard.get_out_of_scope_reject_template()
        assert "超出" in template or "模拟范围" in template
        assert "仅供学习" in template

    def test_ascii_terms_use_word_boundaries(self, guard: ComplianceGuard) -> None:
        """`stock` 不得命中神经科真实体征 stocking-glove distribution。

        命中即 blocked=True 并把病例推向 ESCALATED，子串匹配的代价是把正常
        神经科描述判成超范围问题。
        """
        result = guard.check(
            "Examination shows a stocking-glove distribution of sensory loss."
        )
        assert result.out_of_scope is False
        assert result.blocked is False

    def test_ascii_terms_still_match_as_whole_words(
        self, guard: ComplianceGuard
    ) -> None:
        result = guard.check("Please recommend a stock portfolio.")
        assert result.out_of_scope is True
        assert result.blocked is True

    def test_cjk_negation_context_is_not_a_hit(self, guard: ComplianceGuard) -> None:
        """`确诊` 不得命中「尚未确诊」。CJK 之间不存在可用的词边界。"""
        result = guard.check("患者尚未确诊，建议进一步检查。")
        assert result.blocked is False
        assert "***" not in result.sanitized_text

    def test_cjk_term_without_negation_is_still_blocked(
        self, guard: ComplianceGuard
    ) -> None:
        result = guard.check("患者已确诊为心肌梗死。")
        assert result.blocked is True
        assert "***" in result.sanitized_text

    def test_only_real_hits_are_masked(self, guard: ComplianceGuard) -> None:
        """同一词条既有否定出现又有真实出现时，只屏蔽真实出现。"""
        result = guard.check("入院时尚未确诊；复查后确诊为肺炎。")
        assert result.blocked is True
        assert result.sanitized_text.count("***") == 1
        assert "尚未确诊" in result.sanitized_text

    def test_check_output_dict(self, guard: ComplianceGuard) -> None:
        """检查 Agent 输出字典。"""
        output_dict = {
            "differential_diagnosis": [{"diagnosis": "确诊心肌梗死"}],
            "claims": [{"text": "保证治愈"}],
            "risk_flags": [],
            "recommended_tests": ["ECG"],
            "uncertainty": "some",
        }
        result = guard.check_output(output_dict)
        assert result.blocked is True
        assert any("absolute_term" in r for r in result.block_reasons)

    def test_result_to_dict(self, guard: ComplianceGuard) -> None:
        """结果可序列化。"""
        result = guard.check("test")
        d = result.to_dict()
        assert "blocked" in d
        assert "disclaimer_added" in d

