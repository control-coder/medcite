"""eval/runner.py 执行路径测试。

覆盖 ``run_evaluation`` / ``_run_experiment`` / ``_run_sample`` /
``_generate_outputs``。此前这条链路完全没有测试，缓存跨实验共享和
provider 故障 fail-open 两个缺陷都因此长期存在。

这些测试用假 Retriever / 假 judge / 假 LLM 客户端，不加载任何第三方模型，
也不访问网络。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from eval import runner as runner_module
from eval.configuration import load_config
from eval.runner import (
    AgentOutputCache,
    SampleResult,
    _generate_outputs,
    _run_experiment,
    _run_sample,
    run_evaluation,
)
from medidiag.agents.base import AgentProviderError
from medidiag.agents.llm_client import LLMCompletion
from medidiag.errors import MediDiagError
from medidiag.llm import ProviderCapabilities
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import SearchResult
from medidiag.review.citation import CitationVerifier
from medidiag.schemas import KnowledgeChunk

_AGENT_JSON = json.dumps(
    {
        "differential_diagnosis": [
            {"diagnosis": "condition a", "probability": 0.5,
             "supporting_claim_indices": [0]}
        ],
        "claims": [
            {"text": "evidence supports condition a",
             "citation_chunk_ids": ["kb_test_00001"], "confidence": 0.8}
        ],
        "risk_flags": ["needs clinician review"],
        "missing_info": ["vitals"],
        "recommended_tests": ["basic panel"],
        "uncertainty": "limited evidence",
        "abstain": False,
        "abstain_reason": "",
    },
    ensure_ascii=False,
)


class _FakeLLM:
    """最小 LLMClient 替身：记录调用次数，可选地抛出 provider 故障。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def is_configured(self) -> bool:
        return True

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> LLMCompletion:
        self.prompts.append(prompt)
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMCompletion(
            content=_AGENT_JSON,
            request_id=f"resp_{self.calls:04d}",
            model="fake-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def usage_summary(self) -> dict[str, int]:
        return {"total_tokens": 15 * self.calls}


def _chunks() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id="kb_test_00001",
            source="manual",
            source_id="ref-1",
            text="evidence supports condition a in adult patients",
            evidence_level="level_2_review",
        ),
        KnowledgeChunk(
            chunk_id="kb_test_00002",
            source="manual",
            source_id="ref-2",
            text="alternative finding unrelated to condition a",
            evidence_level="level_5_other",
        ),
    ]


class _FakeRetriever:
    """确定性 Retriever 替身，不构建索引也不加载 embedding 模型。"""

    actual_device = "cpu"

    def __init__(self, chunks: list[KnowledgeChunk] | None = None, **_: Any) -> None:
        self.chunks = chunks if chunks is not None else _chunks()
        self.searches: list[str] = []

    def build_index(self, **_: Any) -> None:
        return None

    def search(self, query: str, top_k: int = 5, experiment_config: Any = None):
        self.searches.append(query)
        return [
            SearchResult(chunk_id=chunk.chunk_id, final_score=1.0 - index * 0.1,
                         chunk=chunk)
            for index, chunk in enumerate(self.chunks[:top_k])
        ]

    def rerank(self, query: str, results, top_k: int = 5):
        return list(results)[:top_k]

    def cache_stats(self) -> dict[str, dict[str, int]]:
        return {"embedding": {"hits": 0, "misses": 0}, "bm25": {"hits": 0, "misses": 0}}


@pytest.fixture
def config() -> dict[str, Any]:
    return load_config("eval/config.yaml")


@pytest.fixture
def verifier() -> CitationVerifier:
    return CitationVerifier(
        model_name="", model_revision="", method="rule_fallback", device="cpu"
    )


def _agent_sample() -> dict[str, Any]:
    return {
        "sample_id": "medqa_0001",
        "question": "adult patient with condition a findings",
        "options": {"A": "condition a", "B": "condition b"},
        "gold_evidence_ids": ["kb_test_00001"],
    }


def _run_agent_sample(
    config: dict[str, Any],
    *,
    name: str = "agent_single",
    llm: Any,
    cache: AgentOutputCache,
    verifier: CitationVerifier | None,
) -> SampleResult:
    experiment = config["experiments"]["agent"][name]
    rag_config = config["experiments"]["rag"][experiment["retrieval_profile"]]["config"]
    return _run_sample(
        name,
        "agent",
        experiment["topology"],
        experiment,
        rag_config,
        _agent_sample(),
        config,
        _FakeRetriever(),
        TerminologyNormalizer(),
        verifier,
        llm,
        cache,
        False,
    )


# ===== A3: 缓存必须按实验隔离 =====


def test_cache_hash_distinguishes_experiments() -> None:
    cache = AgentOutputCache()
    first = cache.compute_hash("agent_single", "q", ["e1"], "cardiology")
    second = cache.compute_hash("agent_fixed_pair", "q", ["e1"], "cardiology")
    assert first != second


def test_cache_hash_distinguishes_claim_language() -> None:
    """语言策略会改变 formal prompt，缓存键必须隔离。"""
    cache = AgentOutputCache()
    default_hash = cache.compute_hash("agent_single", "q", ["e1"], "cardiology")
    english_hash = cache.compute_hash(
        "agent_single", "q", ["e1"], "cardiology", claim_language="en"
    )
    assert default_hash != english_hash


def test_generate_outputs_adds_english_claim_contract_to_formal_prompt() -> None:
    """formal NLI 路径必须把英文 claim 契约传入 Agent prompt。"""
    cache = AgentOutputCache()
    llm = _FakeLLM()
    sample_result = SampleResult(
        sample_id="s1", experiment="agent_single", family="agent", evidence_eligible=True
    )

    _generate_outputs(
        "single",
        {},
        "same question",
        _chunks(),
        ["kb_test_00001", "kb_test_00002"],
        None,
        TerminologyNormalizer(),
        {},
        llm,
        cache,
        sample_result,
        True,
        claim_language="en",
    )

    assert len(llm.prompts) == 1
    assert "claims[].text 必须使用英文完整陈述" in llm.prompts[0]


def test_generate_outputs_does_not_reuse_another_experiment_cache_entry() -> None:
    """三组 agent 实验共用 rag_full，缓存必须按实验名隔离，否则延迟失真。"""
    cache = AgentOutputCache()
    llm = _FakeLLM()
    evidence = _chunks()
    evidence_ids = [chunk.chunk_id for chunk in evidence]
    outputs = []
    for name in ("agent_single", "agent_fixed_pair"):
        sample_result = SampleResult(sample_id="s1", experiment=name, family="agent",
                                     evidence_eligible=True)
        outputs.append(
            _generate_outputs(
                "single", {}, "same question", evidence, evidence_ids, None,
                TerminologyNormalizer(), {}, llm, cache, sample_result, False,
            )
        )
        assert sample_result.cache_hit is False

    assert llm.calls == 2
    assert cache.hits == 0
    assert cache.misses == 2
    assert all(len(item) == 1 for item in outputs)


def test_generate_outputs_reuses_cache_within_one_experiment() -> None:
    cache = AgentOutputCache()
    llm = _FakeLLM()
    evidence = _chunks()
    evidence_ids = [chunk.chunk_id for chunk in evidence]
    for _ in range(2):
        sample_result = SampleResult(sample_id="s1", experiment="agent_single",
                                     family="agent", evidence_eligible=True)
        _generate_outputs(
            "single", {}, "same question", evidence, evidence_ids, None,
            TerminologyNormalizer(), {}, llm, cache, sample_result, False,
        )
    assert llm.calls == 1
    assert (cache.hits, cache.misses) == (1, 1)


def test_run_experiment_reports_per_experiment_cache_stats(
    config: dict[str, Any], verifier: CitationVerifier
) -> None:
    cache = AgentOutputCache()
    llm = _FakeLLM()
    records = [_agent_sample(), _agent_sample()]
    result = _run_experiment(
        "agent_single", "agent", config["experiments"]["agent"]["agent_single"],
        records, config, _FakeRetriever(), TerminologyNormalizer(), verifier, llm,
        cache, False,
    )
    local = result.cache_stats["local_agent_output"]
    assert (local["hits"], local["misses"]) == (1, 1)
    assert len(result.sample_results) == 2
    assert result.sample_results[0].generation_executed is True


# ===== A2: provider 故障不得变成弃权 =====


def test_provider_failure_abstains_in_development_mode(
    config: dict[str, Any], verifier: CitationVerifier
) -> None:
    """development 保留宽松路径，便于本地无网络时继续走通链路。"""
    result = _run_agent_sample(
        config, llm=_FakeLLM(error=RuntimeError("connection refused")),
        cache=AgentOutputCache(), verifier=verifier,
    )
    assert result.agent_abstained is True
    assert result.pipeline_approved is False


def test_provider_failure_aborts_formal_run(
    config: dict[str, Any], verifier: CitationVerifier
) -> None:
    formal = deepcopy(config)
    formal["evaluation"]["mode"] = "formal"
    with pytest.raises(AgentProviderError) as excinfo:
        _run_agent_sample(
            formal, llm=_FakeLLM(error=RuntimeError("connection refused")),
            cache=AgentOutputCache(), verifier=verifier,
        )
    assert "connection refused" in str(excinfo.value)


def test_model_abstention_is_not_a_provider_failure(
    config: dict[str, Any], verifier: CitationVerifier
) -> None:
    """模型主动弃权在 formal 模式下仍是正常样本结果，不终止运行。"""
    formal = deepcopy(config)
    formal["evaluation"]["mode"] = "formal"
    formal["generation"]["provenance_mode"] = "provider_snapshot"
    abstaining = _FakeLLM()
    abstaining.complete = lambda prompt, system_prompt=None: LLMCompletion(  # type: ignore[method-assign]
        content=json.dumps({"claims": [], "abstain": True,
                            "abstain_reason": "insufficient evidence"}),
        request_id="resp_abstain",
        model="fake-model",
    )
    result = _run_agent_sample(
        formal, llm=abstaining, cache=AgentOutputCache(), verifier=verifier
    )
    assert result.agent_abstained is True
    assert result.pipeline_approved is False


# ===== _run_sample 的检索与 dry-run 分支 =====


def test_dry_run_sample_skips_generation(
    config: dict[str, Any], verifier: CitationVerifier
) -> None:
    experiment = config["experiments"]["agent"]["agent_single"]
    rag_config = config["experiments"]["rag"]["rag_full"]["config"]
    result = _run_sample(
        "agent_single", "agent", "single", experiment, rag_config, _agent_sample(),
        config, _FakeRetriever(), TerminologyNormalizer(), verifier, _FakeLLM(),
        AgentOutputCache(), True,
    )
    assert result.generation_executed is False
    assert result.recall_hit is True
    assert "generation" not in result.stage_latency_ms


def test_retrieval_only_rag_sample_records_recall_without_generation(
    config: dict[str, Any]
) -> None:
    rag_config = config["experiments"]["rag"]["rag_embedding"]["config"]
    sample = {"sample_id": "pubmedqa_1", "question": "condition a",
              "gold_evidence_ids": ["kb_test_00002"]}
    result = _run_sample(
        "rag_embedding", "rag", "single",
        config["experiments"]["rag"]["rag_embedding"], rag_config, sample, config,
        _FakeRetriever(), TerminologyNormalizer(), None, None, AgentOutputCache(),
        False,
    )
    assert result.generation_executed is False
    assert result.recall_hit is True
    assert result.citation_results == []


def test_sample_without_gold_evidence_reports_undefined_recall(
    config: dict[str, Any]
) -> None:
    rag_config = config["experiments"]["rag"]["rag_embedding"]["config"]
    sample = {"sample_id": "pubmedqa_2", "question": "condition a",
              "gold_evidence_ids": []}
    result = _run_sample(
        "rag_embedding", "rag", "single",
        config["experiments"]["rag"]["rag_embedding"], rag_config, sample, config,
        _FakeRetriever(), TerminologyNormalizer(), None, None, AgentOutputCache(),
        False,
    )
    assert result.evidence_eligible is False
    assert result.recall_hit is None


# ===== run_evaluation 编排 =====


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


@pytest.fixture
def isolated_config(tmp_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """把数据集换成 tmp 下的最小样本集，避免加载 510M 正式数据集。"""
    kb = tmp_path / "kb.jsonl"
    rag_set = tmp_path / "rag.jsonl"
    agent_set = tmp_path / "agent.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(kb, [
        {"chunk_id": chunk.chunk_id, "source": chunk.source,
         "source_id": chunk.source_id, "text": chunk.text,
         "evidence_level": chunk.evidence_level, "metadata": {}}
        for chunk in _chunks()
    ])
    _write_jsonl(rag_set, [{"sample_id": "pubmedqa_1", "question": "condition a?",
                            "gold_evidence_ids": ["kb_test_00001"]}])
    _write_jsonl(agent_set, [_agent_sample()])
    _write_jsonl(manifest, [{"sample_id": "medqa_0001"}])
    isolated = deepcopy(config)
    isolated["dataset"].update({
        "knowledge_base_path": str(kb),
        "rag_eval_set_path": str(rag_set),
        "agent_eval_set_path": str(agent_set),
        "agent_sample_manifest_path": str(manifest),
    })
    return isolated


@pytest.fixture
def stub_runtime(monkeypatch: pytest.MonkeyPatch) -> _FakeLLM:
    """替换 runner 的检索器、judge 与 LLM 客户端，只保留编排逻辑。"""
    llm = _FakeLLM()
    monkeypatch.setattr(runner_module, "Retriever", _FakeRetriever)
    monkeypatch.setattr(
        runner_module, "LLMClient", lambda **_kwargs: llm
    )
    monkeypatch.setattr(
        runner_module,
        "CitationVerifier",
        lambda **_kwargs: CitationVerifier(
            model_name="", model_revision="", method="rule_fallback", device="cpu"
        ),
    )
    return llm


def test_run_evaluation_writes_experiment_results_and_manifest(
    tmp_path: Path, isolated_config: dict[str, Any], stub_runtime: _FakeLLM
) -> None:
    run_id, results = run_evaluation(
        isolated_config, ["agent_single"], tmp_path / "raw"
    )
    run_dir = tmp_path / "raw" / run_id
    assert (run_dir / "agent_single.json").exists()
    assert (run_dir / "config.snapshot.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["formal_candidate"] is False
    sample = results["agent_single"].sample_results[0]
    assert sample.generation_executed is True
    assert set(sample.stage_artifacts) == {
        "normalize", "retrieval", "generation", "review"
    }
    assert all(
        artifact["schema_version"] == "assistant-stage-artifact-v1"
        for artifact in sample.stage_artifacts.values()
    )


def test_run_evaluation_dry_run_skips_generation_entirely(
    tmp_path: Path, isolated_config: dict[str, Any], stub_runtime: _FakeLLM
) -> None:
    _, results = run_evaluation(
        isolated_config, ["agent_single"], tmp_path / "raw", dry_run=True
    )
    assert stub_runtime.calls == 0
    assert results["agent_single"].sample_results[0].generation_executed is False


def test_formal_run_aborts_without_manifest_on_provider_failure(
    tmp_path: Path,
    isolated_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A2 的核心断言：provider 中途不可用时不得留下可写入的 formal manifest。"""
    formal = deepcopy(isolated_config)
    formal["evaluation"]["mode"] = "formal"
    formal["judge"]["method"] = "nli"
    formal["generation"]["provenance_mode"] = "provider_snapshot"
    monkeypatch.setattr(runner_module, "Retriever", _FakeRetriever)
    class _FailingProvider:
        provider_id = "test"
        profile_id = "test_formal"

        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(structured_output=True, response_id=True)

        def generate(self, *_args: Any, **_kwargs: Any):
            raise MediDiagError("PROVIDER_AUTH_FAILED")

    monkeypatch.setattr(
        runner_module, "build_llm_provider", lambda *_args, **_kwargs: _FailingProvider()
    )
    monkeypatch.setattr(
        runner_module, "CitationVerifier",
        lambda **_kwargs: CitationVerifier(
            model_name="", model_revision="", method="rule_fallback", device="cpu"
        ),
    )
    output_dir = tmp_path / "raw"
    with pytest.raises(MediDiagError):
        run_evaluation(formal, ["agent_single"], output_dir)
    assert list(output_dir.glob("*/manifest.json")) == []


def test_run_evaluation_forbids_limit_in_formal_mode(
    tmp_path: Path, isolated_config: dict[str, Any], stub_runtime: _FakeLLM
) -> None:
    import click

    formal = deepcopy(isolated_config)
    formal["evaluation"]["mode"] = "formal"
    with pytest.raises(click.UsageError):
        run_evaluation(formal, ["agent_single"], tmp_path / "raw", limit=1)


def test_run_evaluation_isolates_cache_between_identical_agent_arms(
    tmp_path: Path, isolated_config: dict[str, Any], stub_runtime: _FakeLLM
) -> None:
    """两组拓扑与专科完全相同的 arm 仍必须各自真实生成。

    这是 A3 的端到端断言：缓存键不含实验名时，后跑的 arm 会全量命中缓存，
    其 ``stage_latency_ms["generation"]`` 测量的是字典查找而不是生成。
    """
    isolated_config["experiments"]["agent"]["agent_fixed_pair_repeat"] = deepcopy(
        isolated_config["experiments"]["agent"]["agent_fixed_pair"]
    )
    arms = ["agent_fixed_pair", "agent_fixed_pair_repeat"]
    _, results = run_evaluation(isolated_config, arms, tmp_path / "raw")

    for name in arms:
        local = results[name].cache_stats["local_agent_output"]
        assert local["hits"] == 0, f"{name} answered from another arm's cache"
        assert local["misses"] == 2
        assert results[name].sample_results[0].cache_hit is False
    # 两个 arm × 双专科 = 4 次真实 provider 调用。
    assert stub_runtime.calls == 4
