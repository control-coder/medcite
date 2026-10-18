"""重排对照的离线核对：复用报告里的检索 Agent 轨迹，与带重排重新运行 Agent 得到同样的最终 3 段。"""
import numpy as np

from eval import rerank_matrix as rm
from eval.retrieval_benchmark import Dataset
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.rag.rerank import Reranker
from medidiag.workflow.retrieval_agent import POOL_SIZE, run_search_agent
from tests.test_retrieval_agent import ScriptedLLM, _reply

TEXTS = ["中暑时应转移到阴凉处并补充水分", "高温天气减少户外活动", "流感通过飞沫传播", "勤洗手可以减少病原体传播",
         "接种疫苗是预防麻疹的最好方法", "室内应定期开窗通风", "高温下老人和儿童更容易中暑", "咳嗽时用纸巾遮住口鼻"]
CHUNKS = [{"chunk_id": f"c{i}", "source_id": f"d{i}", "text": t} for i, t in enumerate(TEXTS)]
QUERIES = [
    {"sample_id": "q1", "split": "test", "bucket": "paraphrase", "question": "天太热人晕了怎么办", "gold_evidence_ids": ["c0"]},
    {"sample_id": "q2", "split": "test", "bucket": "paraphrase", "question": "怎么防止传染给别人", "gold_evidence_ids": ["c3"]},
]
# 每条查询里模型换说法检索一次，然后结束
REFORMULATIONS = {"q1": "中暑 高温 处理", "q2": "传播 预防 洗手"}


def encode(texts: list[str]) -> np.ndarray:
    return np.array([[float(len(t) % 5 + 1), float(sum(ch in t for ch in "热暑温") + 1), 1.0] for t in texts])


def score_by_overlap(pairs: list[tuple[str, str]]) -> list[float]:
    """假重排：问题与片段共有的字越多分数越高。"""
    return [float(len(set(q) & set(t))) for q, t in pairs]


def test_reusing_agent_traces_equals_rerunning_the_agent_with_rerank() -> None:
    dataset = Dataset(chunks=CHUNKS, queries=QUERIES)
    probe = rm.Systems(dataset, {"rewrite": {"rewrites": {}}, "agent": {
        "bm25_agent": {"traces": {}}, "dense_agent": {"traces": {}}}}, encode)
    traces: dict[str, dict] = {"bm25": {}, "dense": {}}
    for base in traces:
        for query in QUERIES:
            outcome = run_search_agent(ScriptedLLM(_reply(REFORMULATIONS[query["sample_id"]]), _reply()),
                                       lambda text, base=base: probe.search(base, text, POOL_SIZE), query["question"],
                                       probe.search(base, query["question"], POOL_SIZE), ACTIVE_MIMO_MODEL)
            traces[base][query["sample_id"]] = {"steps": outcome.steps, "chunks": [h.chunk_id for h in outcome.chunks]}
    report = {"rewrite": {"rewrites": {q["sample_id"]: "" for q in QUERIES}},
              "agent": {f"{base}_agent": {"traces": traces[base]} for base in traces}}
    systems = rm.Systems(dataset, report, encode)
    for base in traces:
        for depth in rm.AGENT_DEPTHS:
            reranker = Reranker(score_by_overlap, model_name="fake", model_revision="0" * 40, candidates=depth)
            for query in QUERIES:
                sid = query["sample_id"]
                assert systems.final(f"{base}_agent", query) == traces[base][sid]["chunks"]

                def merge(lists, question=query["question"], reranker=reranker):
                    return [h for h, _ in reranker.rerank(question, [x for hits in lists for x in hits], POOL_SIZE)]

                rerun = run_search_agent(ScriptedLLM(_reply(REFORMULATIONS[sid]), _reply()),
                                         lambda text, base=base, depth=depth: probe.search(base, text, depth),
                                         query["question"], probe.search(base, query["question"], depth),
                                         ACTIVE_MIMO_MODEL, observe=POOL_SIZE, merge=merge)
                assert systems.final(f"{base}_agent_rerank", query, reranker) == [h.chunk_id for h in rerun.chunks]
                # 模型看到的检索结果与不重排时相同，所以轨迹相同
                assert [s for s in rerun.steps if s["action"] != "merge"] == traces[base][sid]["steps"]
