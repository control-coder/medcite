"""阶段 5 Agent 路由与仲裁测试。

覆盖:
1. 路由器打分（关键词/术语/证据/规划提示）
2. Top2 筛选 + 兜底逻辑
3. 仲裁（一致/冲突/弃权/证据对比）
4. Agent 基类（无 LLM 时弃权、输出序列化）
5. 消融三组配置
"""

from __future__ import annotations

import re

import pytest

from medidiag.agents.arbitration import ArbitrationAgent
from medidiag.agents.base import AgentOutput, Claim, DiagnosisItem
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialist import SpecialistAgent
from medidiag.agents.specialty_data import (
    BASELINE_PAIR,
    FALLBACK_PAIR,
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
            "patient has chest pain and ECG shows ST elevation", chunks, ""
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
        scores = router._compute_all_scores("fever and headache", [], "")
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

    def test_plan_hint_score(self, router: SpecialistRouter) -> None:
        """诊断规划提示分。"""
        assert router._compute_plan_hint_score("cardiology", "cardiovascular system") == 1.0
        assert router._compute_plan_hint_score("cardiology", "digestive system") == 0.0
        assert router._compute_plan_hint_score("cardiology", "") == 0.0


class TestSpecialtyKeywords:
    """守护 `specialty_data.py` 文件头记录的三条词表编写约定。"""

    def test_keywords_survive_normalization(self) -> None:
        """词条不得被归一化器改写成本表之外的形式，否则永不命中。

        `_compute_keyword_score` 打分的是归一化**之后**的查询文本。扩表前
        `ECG` / `EKG` 会被改写为 `electrocardiogram`，而该形式不在心内科词表内，
        于是这两个词条在 100 样本 manifest 上白丢 5 次命中。
        """
        normalizer = TerminologyNormalizer()
        dead: list[tuple[str, str, str]] = []
        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            present = {keyword.lower() for keyword in keywords}
            for keyword in keywords:
                preferred = normalizer._synonym_map.get(keyword.lower())
                if not preferred or preferred.lower() == keyword.lower():
                    continue
                # 改写结果里仍能以词边界找回原词条时不算丢失
                # （如 `diabetes` -> `diabetes mellitus`）。
                if re.search(
                    r"\b" + re.escape(keyword.lower()) + r"\b", preferred.lower()
                ):
                    continue
                if preferred.lower() in present:
                    continue
                dead.append((specialty, keyword, preferred))
        assert dead == [], (
            "these keywords are rewritten out of their own specialty list and can "
            f"never match: {dead}"
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
        """打分权重。"""
        from medidiag.agents.specialty_data import ROUTING_WEIGHTS

        assert ROUTING_WEIGHTS["keyword"] == 3.0
        assert ROUTING_WEIGHTS["normalized_term"] == 2.0
        assert ROUTING_WEIGHTS["evidence"] == 2.0
        assert ROUTING_WEIGHTS["plan_hint"] == 1.0
