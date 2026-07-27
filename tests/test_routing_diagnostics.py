"""`eval/routing_diagnostics.py` 的纯函数测试（不加载模型、不访问网络）。

`_classify` 复制了 `SpecialistRouter.route` 的规则顺序，用于把路由结果归类到
四个分支。复制就会漂移：路由改了规则顺序而诊断工具没跟上时，测量出来的分布会
静默地对不上实际行为。这里逐条比对两者，使漂移变成失败而不是错误的报表。
"""

from __future__ import annotations

from eval.routing_diagnostics import _classify, _percentiles
from medidiag.agents.router import SpecialistRouter
from medidiag.schemas import KnowledgeChunk


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        text=text,
        source="unit_test_fixture",
        source_id="local-fixture",
        evidence_level="level_2_review",
    )


class TestPercentiles:
    def test_empty_input_is_all_zero(self) -> None:
        assert _percentiles([]) == {
            "min": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    def test_single_value_collapses_to_that_value(self) -> None:
        assert _percentiles([2.5]) == {
            "min": 2.5,
            "p10": 2.5,
            "p50": 2.5,
            "p90": 2.5,
            "max": 2.5,
        }

    def test_ordering_is_monotonic_and_input_order_independent(self) -> None:
        forward = _percentiles([float(n) for n in range(11)])
        shuffled = _percentiles([3.0, 10.0, 0.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0, 5.0])
        assert forward == shuffled
        assert forward["min"] == 0.0
        assert forward["p50"] == 5.0
        assert forward["max"] == 10.0


class TestClassifyMatchesTheRouter:
    """`_classify` 的分支必须与 `route()` 实际走的分支一致。"""

    QUERIES = [
        # 无医学词汇 -> 规则 1
        ("hello world foo bar", []),
        # 单一专科强信号 -> top1 明显领先
        (
            "chest pain with ST elevation on the electrocardiogram and troponin rise",
            [_chunk("c1", "myocardial infarction coronary angina troponin")],
        ),
        # 两个专科同时强 -> 比值接近 1
        (
            "chest pain and cough with dyspnea, pneumonia versus myocardial infarction, "
            "wheezing and troponin and angina",
            [
                _chunk("c1", "pneumonia lung pulmonary respiratory wheezing"),
                _chunk("c2", "cardiac coronary angina myocardial troponin"),
            ],
        ),
        # 中等强度单专科
        ("fever and chills with a positive blood culture", []),
        ("headache with seizure and hemiparesis after a stroke", []),
        ("abdominal pain, vomiting, diarrhea and jaundice with ascites", []),
    ]

    def test_every_query_classifies_to_the_branch_route_took(self) -> None:
        router = SpecialistRouter()
        for question, evidence in self.QUERIES:
            result = router.route(question, evidence)
            ranked = sorted(result.scores.values(), key=lambda s: s.total, reverse=True)
            top1 = ranked[0]
            top2_total = ranked[1].total if len(ranked) > 1 else 0.0
            branch = _classify(top1.total, top2_total, result.confidence)

            if branch in {
                "rule_1_primary_below_threshold",
                "rule_4_low_confidence",
            }:
                assert result.is_fallback is True, (question, branch)
            else:
                assert result.is_fallback is False, (question, branch)

            if branch == "rule_3_dynamic_top2":
                # 动态 Top2 必须真的返回前两名，而不是掺入 evidence_skeptic。
                assert result.specialty_pair == (top1.specialty, ranked[1].specialty)
            if branch == "rule_2_top1_plus_skeptic":
                assert result.specialty_pair[0] == top1.specialty

    def test_the_four_branches_are_mutually_exclusive_and_ordered(self) -> None:
        # 规则 1 先于规则 4：总分不足时不看比值。
        assert _classify(0.5, 0.5, 1.0) == "rule_1_primary_below_threshold"
        # 规则 4 先于规则 2：这正是被记录为语义颠倒的顺序。
        assert _classify(3.0, 0.1, 0.0333) == "rule_4_low_confidence"
        # 第二名不足 -> 规则 2。
        assert _classify(3.0, 1.0, 0.5) == "rule_2_top1_plus_skeptic"
        # 两名均达标且比值高 -> 规则 3。
        assert _classify(3.0, 2.5, 0.8333) == "rule_3_dynamic_top2"
