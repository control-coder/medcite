"""阶段 5 Agent 路由与仲裁测试。

覆盖:
1. 路由器打分（关键词/术语/证据/规划提示）
2. Top2 筛选 + 兜底逻辑
3. 仲裁（一致/冲突/弃权/证据对比）
4. Agent 基类（无 LLM 时弃权、输出序列化）
5. 消融三组配置
"""

from __future__ import annotations

import pytest

from medidiag.agents.arbitration import ArbitrationAgent
from medidiag.agents.base import AgentOutput, Claim, DiagnosisItem
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialist import SpecialistAgent
from medidiag.agents.specialty_data import (
    BASELINE_PAIR,
    FALLBACK_PAIR,
    MATCH_SATURATION_COUNT,
    SPECIALTIES,
    SPECIALTY_KEYWORDS,
    THRESHOLDS,
)
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.schemas import KnowledgeChunk

# ===== 路由器测试 =====


class TestRouter:
    @pytest.fixture
    def router(self) -> SpecialistRouter:
        return SpecialistRouter()

    def test_cardiology_keywords_high_score(self, router: SpecialistRouter) -> None:
        """心脏病关键词 → cardiology 得分高。"""
        chunks = [
            KnowledgeChunk(
                chunk_id="c1", source="test", source_id="b1",
                text="Myocardial infarction causes chest pain and ECG ST elevation.",
                evidence_level="level_2_review",
            )
        ]
        router.route(
            "patient has chest pain and ECG shows ST elevation", chunks
        )
        scores = router._compute_all_scores(
            "patient has chest pain and ECG shows ST elevation", chunks
        )
        assert scores["cardiology"].keyword_score > 0
        assert scores["cardiology"].total > scores["respiratory"].total

    def test_fallback_when_no_match(self, router: SpecialistRouter) -> None:
        """无匹配关键词 → 兜底组合。"""
        result = router.route("hello world foo bar", [])
        assert result.is_fallback is True
        assert result.specialty_pair == FALLBACK_PAIR

    def test_fallback_low_score(self, router: SpecialistRouter) -> None:
        """最高分 < 2.0 → 兜底。"""
        result = router.route("fever", [])
        if result.scores:
            top_score = max(s.total for s in result.scores.values())
            if top_score < THRESHOLDS["MIN_PRIMARY_SCORE"]:
                assert result.is_fallback is True

    def test_confidence_range(self, router: SpecialistRouter) -> None:
        """置信度在 0~1 之间。"""
        chunks = [
            KnowledgeChunk(
                chunk_id="c1", source="test", source_id="b1",
                text="chest pain and cough with fever",
                evidence_level="level_3_primary_study",
            )
        ]
        result = router.route("patient has chest pain and cough", chunks)
        assert 0 <= result.confidence <= 1.0

    def test_all_specialties_scored(self, router: SpecialistRouter) -> None:
        """所有 10 个专科都有得分。"""
        scores = router._compute_all_scores("fever and headache", [])
        assert len(scores) == 10
        for sp in SPECIALTIES:
            assert sp in scores

    def test_routing_result_to_dict(self, router: SpecialistRouter) -> None:
        """路由结果可序列化。"""
        result = router.route("test query", [])
        d = result.to_dict()
        assert "specialty_pair" in d
        assert "confidence" in d
        assert "is_fallback" in d
        assert "reason" in d

    def test_specialty_score_to_dict(self) -> None:
        """分项得分可序列化。"""
        from medidiag.agents.router import SpecialtyScore

        s = SpecialtyScore(specialty="cardiology", keyword_score=0.5, total=1.5)
        d = s.to_dict()
        assert d["specialty"] == "cardiology"
        assert d["keyword_score"] == 0.5

    def test_evidence_score_uses_level(self, router: SpecialistRouter) -> None:
        """证据分数使用证据等级加权。"""
        chunks_high = [
            KnowledgeChunk(
                chunk_id="c1", source="test", source_id="b1",
                text="chest pain cardiac myocardial",
                evidence_level="level_1_guideline",
            )
        ]
        chunks_low = [
            KnowledgeChunk(
                chunk_id="c1", source="test", source_id="b1",
                text="chest pain cardiac myocardial",
                evidence_level="level_5_other",
            )
        ]
        score_high = router._compute_evidence_score(
            "cardiology", [c.text for c in chunks_high], chunks_high
        )
        score_low = router._compute_evidence_score(
            "cardiology", [c.text for c in chunks_low], chunks_low
        )
        assert score_high >= score_low

    def test_plan_hint_component_is_gone(self, router: SpecialistRouter) -> None:
        """plan_hint 分项已删除（DD-023）：它恒为 0，不得再出现在打分公式或产物里。"""
        from medidiag.agents.specialty_data import ROUTING_WEIGHTS

        assert "plan_hint" not in ROUTING_WEIGHTS
        assert not hasattr(router, "_compute_plan_hint_score")
        scores = router._compute_all_scores("chest pain", [])
        assert "plan_hint_score" not in scores["cardiology"].to_dict()


class TestScoreDenominators:
    """三个分项的分母必须与词表长度无关（DD-023）。"""

    @pytest.fixture
    def router(self) -> SpecialistRouter:
        return SpecialistRouter(normalizer=TerminologyNormalizer())

    def test_adding_vocabulary_cannot_lower_a_matching_specialty(
        self, monkeypatch: pytest.MonkeyPatch, router: SpecialistRouter
    ) -> None:
        """给词表加词不得降低一个已经命中的专科在同一查询上的分数。

        这是修复前的实际行为：`_compute_term_score` 与 `_compute_evidence_score`
        都除以 `len(SPECIALTY_KEYWORDS[specialty])`，因此扩表与打分互相打架
        （见 `612004e` 的 commit message）。
        """
        question = "chest pain with ST elevation and troponin rise and angina"
        chunks = [
            KnowledgeChunk(
                chunk_id="c1", source="test", source_id="b1",
                text=(
                    "myocardial infarction with coronary artery disease, "
                    "angina, troponin elevation and cardiac arrhythmia"
                ),
                evidence_level="level_2_review",
            )
        ]
        before = router._compute_all_scores(question, chunks)["cardiology"]

        # 追加 40 个不会在本查询或证据中命中的心内科词条。
        padded = dict(SPECIALTY_KEYWORDS)
        padded["cardiology"] = [
            *SPECIALTY_KEYWORDS["cardiology"],
            *[f"zzz placeholder term {n}" for n in range(40)],
        ]
        monkeypatch.setattr(
            "medidiag.agents.router.SPECIALTY_KEYWORDS", padded
        )
        after = router._compute_all_scores(question, chunks)["cardiology"]

        assert after.keyword_score == before.keyword_score
        assert after.normalized_term_score == before.normalized_term_score
        assert after.evidence_score == before.evidence_score
        assert after.total == before.total

    def test_all_three_components_use_the_same_saturation_count(
        self, router: SpecialistRouter
    ) -> None:
        """关键词、术语、证据三项共用 `MATCH_SATURATION_COUNT`。"""
        # 四个心内科词条 -> 关键词分饱和到 1.0。
        four = "chest pain, palpitation, syncope and murmur"
        assert router._compute_keyword_score(
            "cardiology", four.lower(), four.lower()
        ) == pytest.approx(1.0)
        # 三个词条 -> 3/4。
        three = "chest pain, palpitation and syncope"
        assert router._compute_keyword_score(
            "cardiology", three.lower(), three.lower()
        ) == pytest.approx(3.0 / MATCH_SATURATION_COUNT)

    def test_evidence_score_denominator_is_chunk_count_only(
        self, router: SpecialistRouter
    ) -> None:
        """证据分只按 chunk 数取平均：单条满命中的 level_1 chunk 应得满分。"""
        text = "chest pain palpitation syncope murmur troponin"
        chunk = KnowledgeChunk(
            chunk_id="c1", source="test", source_id="b1",
            text=text, evidence_level="level_1_guideline",
        )
        router.evidence_level_scores = {"level_1_guideline": 1.0}
        assert router._compute_evidence_score(
            "cardiology", [text], [chunk]
        ) == pytest.approx(1.0)

    def test_synonym_variants_of_one_term_count_once(
        self, router: SpecialistRouter
    ) -> None:
        """`ECG` 与 `electrocardiogram` 同时出现时只算一次命中。

        关键词分同时打分原文与归一化文本，若不折叠同义词，一个临床事实会被
        数成两次甚至三次命中。
        """
        query = "the ECG and the electrocardiogram both show ST elevation"
        both = router._compute_keyword_score(
            "cardiology", query.lower(), query.lower()
        )
        single = "the electrocardiogram shows ST elevation"
        one = router._compute_keyword_score(
            "cardiology", single.lower(), single.lower()
        )
        assert both == one


class TestKeywordScoringSeesBothTexts:
    """`612004e` 声称修好的「归一化改写导致词条永不命中」问题的回归保护。"""

    @pytest.fixture
    def router(self) -> SpecialistRouter:
        return SpecialistRouter(normalizer=TerminologyNormalizer())

    def test_cachexia_scores_for_oncology(self, router: SpecialistRouter) -> None:
        """`cachexia` 被改写为 `weight loss`（内分泌科词条），但仍须给肿瘤科计分。

        修复前：关键词分只看归一化后的文本，`cachexia` 不可能命中，且改写结果
        `weight loss` 反而给内分泌科加分——一个肿瘤科词条为别的专科加分。
        """
        question = "a 62-year-old man with cachexia and lymphadenopathy"
        scores = router._compute_all_scores(question, [])
        assert scores["oncology"].keyword_score > 0
        assert scores["oncology"].total > scores["endocrinology"].total

    def test_normalizer_rewritten_keywords_still_match(
        self, router: SpecialistRouter
    ) -> None:
        """被归一化器改写掉的词条仍应命中自己所属的专科。"""
        for keyword, specialty in [
            ("ECG", "cardiology"),
            ("EKG", "cardiology"),
            ("dyspnea", "respiratory"),
            ("COPD", "respiratory"),
            ("vertigo", "neurology"),
            ("febrile", "infectious_disease"),
        ]:
            question = f"the patient reports {keyword} today"
            score = router._compute_all_scores(question, [])[specialty]
            assert score.keyword_score > 0, (keyword, specialty)

    def test_no_keyword_is_unreachable_in_context(
        self, router: SpecialistRouter
    ) -> None:
        """词表中不得存在任何在真实查询里永不可能命中的词条。

        `tests/test_agents.py::TestSpecialtyKeywords` 只检查词条不会被改写到
        本表之外；这里检查更强的性质：把词条放进一个句子后，它所属的专科必须
        真的得分。
        """
        unreachable: list[tuple[str, str]] = []
        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            if not keywords:
                continue
            for keyword in keywords:
                question = f"the patient reports {keyword} today"
                if router._compute_all_scores(question, [])[
                    specialty
                ].keyword_score <= 0:
                    unreachable.append((specialty, keyword))
        assert unreachable == [], f"keywords that can never score: {unreachable}"


class TestSpecialtyKeywords:
    """守护 `specialty_data.py` 文件头记录的三条词表编写约定。"""

    def test_a_rewritten_keyword_never_favours_another_specialty(self) -> None:
        """归一化改写不得让别的专科在只含该词条的查询上超过词条的归属专科。

        2026-07-27 前的约定更强也更脆：词条**必须**能在归一化后的查询里存活，
        否则永不命中（`ECG`/`EKG` 曾因此白丢 5 次 manifest 命中）。现在关键词分
        同时打分原文与归一化文本（DD-023），存活不再是必要条件，但改写落到别的
        专科词表上仍然有害——`cachexia` 会被改写成内分泌科的 `weight loss`。
        本测试守护那个仍然重要的部分。
        """
        router = SpecialistRouter(normalizer=TerminologyNormalizer())
        offenders: list[tuple[str, str, str, float, float]] = []
        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            if not keywords:
                continue
            for keyword in keywords:
                scores = router._compute_all_scores(
                    f"the patient reports {keyword} today", []
                )
                own = scores[specialty].keyword_score
                for other, score in scores.items():
                    if other != specialty and score.keyword_score > own:
                        offenders.append(
                            (specialty, keyword, other, own, score.keyword_score)
                        )
        assert offenders == [], (
            "normalization makes another specialty outscore the owning one: "
            f"{offenders}"
        )

    def test_keywords_are_not_cross_listed(self) -> None:
        """同一词条不得出现在两个专科：它同时抬高两侧，不产生区分度。"""
        owners: dict[str, list[str]] = {}
        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            for keyword in keywords:
                owners.setdefault(keyword.lower(), []).append(specialty)
        shared = {k: v for k, v in owners.items() if len(v) > 1}
        assert shared == {}, f"cross-listed keywords: {shared}"

    def test_no_duplicate_keywords_within_a_specialty(self) -> None:
        """表内重复只会稀释 term/evidence 分项的分母，不增加命中。"""
        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            lowered = [keyword.lower() for keyword in keywords]
            assert len(lowered) == len(set(lowered)), (
                f"{specialty} has duplicate keywords"
            )

    def test_fallback_roles_stay_unscored(self) -> None:
        """兜底角色刻意没有关键词；其余专科必须有。"""
        for specialty in SPECIALTIES:
            keywords = SPECIALTY_KEYWORDS[specialty]
            if specialty in FALLBACK_PAIR:
                assert keywords == []
            else:
                assert len(keywords) >= 15


# ===== 仲裁测试 =====


class TestArbitration:
    def _make_output(
        self,
        specialty: str,
        diagnoses: list[str],
        claims: list[str],
        abstain: bool = False,
    ) -> AgentOutput:
        return AgentOutput(
            specialty=specialty,
            differential_diagnosis=[
                DiagnosisItem(diagnosis=d, probability=0.8) for d in diagnoses
            ],
            claims=[
                Claim(text=c, citation_chunk_ids=["c1"]) for c in claims
            ],
            risk_flags=["risk1"],
            recommended_tests=["test1"],
            uncertainty="some uncertainty",
            abstain=abstain,
        )

    def test_consensus_approved(self) -> None:
        """一致诊断 → APPROVED。"""
        o1 = self._make_output("cardiology", ["myocardial infarction"], ["claim1"])
        o2 = self._make_output("respiratory", ["myocardial infarction"], ["claim2"])
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        assert result.verdict == "APPROVED"
        assert "myocardial infarction" in result.consensus

    def test_conflict_revision(self) -> None:
        """冲突 → REVISION_REQUIRED。"""
        o1 = self._make_output("cardiology", ["MI"], ["claim1"])
        o2 = self._make_output("respiratory", ["pneumonia"], ["claim2"])
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        assert result.verdict == "REVISION_REQUIRED"
        assert len(result.conflicts) > 0

    def test_both_abstain_escalated(self) -> None:
        """双方弃权 → ESCALATED。"""
        o1 = self._make_output("cardiology", [], [], abstain=True)
        o2 = self._make_output("respiratory", [], [], abstain=True)
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        assert result.verdict == "ESCALATED"

    def test_one_abstain_revision(self) -> None:
        """一方弃权 → REVISION_REQUIRED。"""
        o1 = self._make_output("cardiology", [], [], abstain=True)
        o2 = self._make_output("respiratory", ["pneumonia"], ["claim2"])
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        assert result.verdict == "REVISION_REQUIRED"

    def test_evidence_comparison(self) -> None:
        """证据支撑对比。"""
        o1 = self._make_output("cardiology", ["MI"], ["claim1", "claim2"])
        o2 = self._make_output("respiratory", ["MI"], ["claim3"])
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        assert result.evidence_comparison["specialist_1"]["total_claims"] == 2
        assert result.evidence_comparison["specialist_2"]["total_claims"] == 1

    def test_no_citation_escalated(self) -> None:
        """冲突且双方无引用 → ESCALATED。"""
        o1 = AgentOutput(
            specialty="cardiology",
            differential_diagnosis=[DiagnosisItem(diagnosis="MI")],
            claims=[Claim(text="claim1", citation_chunk_ids=[])],
            risk_flags=["r"], recommended_tests=["t"], uncertainty="u",
        )
        o2 = AgentOutput(
            specialty="respiratory",
            differential_diagnosis=[DiagnosisItem(diagnosis="pneumonia")],
            claims=[Claim(text="claim2", citation_chunk_ids=[])],
            risk_flags=["r"], recommended_tests=["t"], uncertainty="u",
        )
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        assert result.verdict == "ESCALATED"

    def test_arbitration_result_to_dict(self) -> None:
        """仲裁结果可序列化。"""
        o1 = self._make_output("cardiology", ["MI"], ["c1"])
        o2 = self._make_output("respiratory", ["MI"], ["c2"])
        arb = ArbitrationAgent()
        result = arb.arbitrate(o1, o2)
        d = result.to_dict()
        assert "verdict" in d
        assert "consensus" in d
        assert "conflicts" in d


# ===== Agent 基类测试 =====


class TestAgents:
    def test_specialist_no_llm_abstains(self) -> None:
        """无 LLM 时专科 Agent 弃权。"""
        agent = SpecialistAgent("cardiology")
        output = agent.generate("chest pain", [])
        assert output.abstain is True
        assert output.abstain_reason == "no_llm_client"

    def test_diagnosis_no_llm_abstains(self) -> None:
        """无 LLM 时诊断 Agent 弃权。"""
        agent = DiagnosisAgent()
        output = agent.generate("fever", [])
        assert output.abstain is True
        assert output.specialty == "general_diagnosis"

    def test_generation_preserves_provider_response_id(self) -> None:
        """生成输出解析失败时仍保留 provider 响应 ID。"""
        class FakeCompletion:
            content = "not valid json {{{"
            request_id = "chatcmpl-test-id"

        class FakeLLMClient:
            is_configured = True

            def complete(self, prompt: str) -> FakeCompletion:
                return FakeCompletion()

        agent = SpecialistAgent("cardiology", llm_client=FakeLLMClient())
        output = agent.generate("chest pain", [])

        assert output.abstain is True
        assert output.provider_request_id == "chatcmpl-test-id"

    def test_generation_preserves_request_id_when_schema_parse_raises(self) -> None:
        """当 JSON schema 解析失败时仍保留 provider 请求 ID。"""
        class FakeCompletion:
            content = "[]"
            request_id = "chatcmpl-schema-error-id"

        class FakeLLMClient:
            is_configured = True

            def complete(self, prompt: str) -> FakeCompletion:
                return FakeCompletion()

        output = SpecialistAgent("cardiology", llm_client=FakeLLMClient()).generate(
            "chest pain", []
        )

        assert output.abstain is True
        assert output.abstain_reason.startswith("llm_output_parse_error:")
        assert output.provider_request_id == "chatcmpl-schema-error-id"

    def test_output_to_dict(self) -> None:
        """AgentOutput 序列化。"""
        output = AgentOutput(
            specialty="cardiology",
            differential_diagnosis=[
                DiagnosisItem(diagnosis="MI", probability=0.9)
            ],
            claims=[
                Claim(text="claim1", citation_chunk_ids=["c1"], confidence=0.8)
            ],
            risk_flags=["risk1"],
            uncertainty="uncertain",
        )
        d = output.to_dict()
        assert d["specialty"] == "cardiology"
        assert d["differential_diagnosis"][0]["diagnosis"] == "MI"
        assert d["claims"][0]["citation_chunk_ids"] == ["c1"]

    def test_output_from_dict(self) -> None:
        """AgentOutput 反序列化。"""
        data = {
            "specialty": "cardiology",
            "differential_diagnosis": [
                {"diagnosis": "MI", "probability": 0.9, "supporting_claim_indices": [0]}
            ],
            "claims": [
                {"text": "claim1", "citation_chunk_ids": ["c1"], "confidence": 0.8}
            ],
            "risk_flags": ["risk1"],
            "uncertainty": "uncertain",
            "abstain": False,
        }
        output = AgentOutput.from_dict(data)
        assert output.specialty == "cardiology"
        assert output.differential_diagnosis[0].diagnosis == "MI"
        assert output.claims[0].citation_chunk_ids == ["c1"]

    def test_parse_output_invalid_json(self) -> None:
        """JSON 解析失败 → 弃权。"""
        agent = SpecialistAgent("cardiology")
        output = agent.parse_output("not valid json {{{")
        assert output.abstain is True
        assert "json_parse_error" in output.abstain_reason


# ===== 消融三组配置 =====


class TestAblationConfigs:
    def test_baseline_pair(self) -> None:
        """固定双专科实验基线: 心内科+呼吸科。"""
        assert BASELINE_PAIR == ("cardiology", "respiratory")

    def test_fallback_pair(self) -> None:
        """兜底组合: general_internal + evidence_skeptic。"""
        assert FALLBACK_PAIR == ("general_internal", "evidence_skeptic")

    def test_specialties_count(self) -> None:
        """专科池 10 个。"""
        assert len(SPECIALTIES) == 10

    def test_thresholds(self) -> None:
        """阈值常量。"""
        assert THRESHOLDS["MIN_PRIMARY_SCORE"] == 2.0
        assert THRESHOLDS["MIN_SECONDARY_SCORE"] == 1.2
        assert THRESHOLDS["SCORE_GAP"] == 3.0
        assert THRESHOLDS["LOW_CONFIDENCE"] == 0.45

    def test_routing_weights(self) -> None:
        """打分权重：三项，`plan_hint` 已删除（DD-023）。"""
        from medidiag.agents.specialty_data import ROUTING_WEIGHTS

        assert ROUTING_WEIGHTS == {
            "keyword": 3.0,
            "normalized_term": 2.0,
            "evidence": 2.0,
        }
        assert sum(ROUTING_WEIGHTS.values()) == 7.0
        assert MATCH_SATURATION_COUNT == 4.0
