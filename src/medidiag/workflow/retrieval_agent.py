"""检索智能体：先按原问题检索一次，再由模型看结果，决定是否换一种说法继续检索。

和固定流程的区别：固定流程只检索一次（或把改写句拼在原问题后面再检索一次），而这里每一步做什么由模型决定：
结果够用就结束，不相关就调用 ``search_kb`` 换成规范术语再检索，最多再检索两次。
它只负责准备证据，最终回答仍走 ``MimoGroundedWorkflowProvider`` 的受约束摘录，
所以“摘录必须与证据原文逐字一致”的检查对智能体同样生效，智能体不能绕过它。

任何模型错误都不会让问题失败：停止继续检索，沿用已经拿到的证据，并把原因写进步骤记录。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from medidiag.errors import MediDiagError
from medidiag.llm.contracts import LLMProvider, LLMRequest
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider

MAX_EXTRA_SEARCHES = 2  # 原问题检索之外，最多再检索几次
POOL_SIZE = 3  # 交给生成步骤的证据段数，与固定流程的 top3 保持一致
RRF_K = 60  # 多次检索结果合并时的平滑常数，沿用倒数排名融合的常用取值
MAX_QUERY_CHARS = 120

SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": "在公共卫生科普知识库中检索，返回最相关的几段原文。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索语句，使用规范的书面表述"}},
            "required": ["query"],
        },
    },
}

AGENT_PROMPT = (
    "你是公共卫生科普检索助手，负责为后续的摘录步骤准备证据。你只检索，不回答问题。"
    "用户消息里有问题，以及按原问题检索得到的第一批结果。"
    "如果这些结果已经能直接回答问题，或者换一种说法也不太可能检索到更好的结果，直接回复“结束”。"
    "如果结果和问题不相关，就调用 search_kb，改用世界卫生组织科普页面常用的规范术语再检索一次。"
    "最多再检索两次，不要重复相同的检索语句，不要添加问题中没有的事实、数字或建议。"
)


class Hit(Protocol):
    chunk_id: str
    text: str


@dataclass
class SearchAgentResult:
    chunks: list[Any]  # 合并后的证据，最多 POOL_SIZE 段，元素就是 search 返回的对象
    steps: list[dict[str, Any]]
    model_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    status: str = "ok"  # ok：模型自己结束；fallback：模型出错或输出无法使用，沿用已有证据
    code: str | None = None


def merge_hits(result_lists: Sequence[Sequence[Any]], pool_size: int = POOL_SIZE) -> list[Any]:
    """倒数排名融合：多次检索都靠前的片段排前面；得分相同时先出现的在前。"""
    score: dict[str, float] = {}
    first: dict[str, Any] = {}
    order: dict[str, int] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, 1):
            score[hit.chunk_id] = score.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            first.setdefault(hit.chunk_id, hit)
            order.setdefault(hit.chunk_id, len(order))
    ranked = sorted(score, key=lambda cid: (-score[cid], order[cid]))
    return [first[cid] for cid in ranked[:pool_size]]


def _observation(hits: Sequence[Any]) -> list[dict[str, str]]:
    return [{"chunk_id": hit.chunk_id, "text": hit.text} for hit in hits]


def run_search_agent(llm: LLMProvider, search: Callable[[str], Sequence[Any]], question: str,
                     first_hits: Sequence[Any], model: str, *, max_extra_searches: int = MAX_EXTRA_SEARCHES,
                     pool_size: int = POOL_SIZE, timeout_s: float = 45) -> SearchAgentResult:
    """``first_hits`` 是按原问题检索的结果；``search`` 负责后续检索，返回带 chunk_id 与 text 的对象。"""
    result_lists: list[Sequence[Any]] = [first_hits]
    queries = {question.strip()}
    steps: list[dict[str, Any]] = [
        {"action": "search", "query": question, "chunk_ids": [hit.chunk_id for hit in first_hits]}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": json.dumps(
            {"question": question, "first_search": _observation(first_hits)}, ensure_ascii=False)}]
    outcome = SearchAgentResult(chunks=[], steps=steps)

    def finish() -> SearchAgentResult:
        outcome.chunks = merge_hits(result_lists, pool_size)
        return outcome

    for _ in range(max_extra_searches + 1):  # 最后一次调用只用来让模型有机会说“结束”，不再执行检索
        try:
            reply = llm.generate(
                LLMRequest(messages=messages, model=model, tools=[SEARCH_TOOL], tool_choice="auto",
                           max_tokens=256, reasoning_mode="disabled", prompt_version="retrieval-agent-v1"),
                timeout_s=timeout_s, idempotency_key="mimo-agent-" + uuid.uuid4().hex)
        except MediDiagError as exc:
            outcome.status, outcome.code = "fallback", exc.code
            steps.append({"action": "error", "code": exc.code})
            return finish()
        outcome.model_calls += 1
        for key, value in reply.usage.items():
            outcome.usage[key] = outcome.usage.get(key, 0) + value
        if reply.model != model:
            outcome.status, outcome.code = "fallback", "AGENT_MODEL_MISMATCH"
            steps.append({"action": "error", "code": "AGENT_MODEL_MISMATCH"})
            return finish()
        if not reply.tool_calls:
            steps.append({"action": "stop", "reason": "model_done"})
            return finish()
        call = reply.tool_calls[0]  # 同一轮给出多个调用时只执行第一个，后面的丢弃
        query = _tool_query(call)
        if query is None:
            outcome.status, outcome.code = "fallback", "AGENT_BAD_TOOL_CALL"
            steps.append({"action": "error", "code": "AGENT_BAD_TOOL_CALL"})
            return finish()
        if query in queries:
            steps.append({"action": "stop", "reason": "duplicate_query", "query": query})
            return finish()
        if len(queries) > max_extra_searches:  # 已经用完检索次数
            steps.append({"action": "stop", "reason": "search_limit"})
            return finish()
        queries.add(query)
        hits = search(query)
        result_lists.append(hits)
        steps.append({"action": "search", "query": query, "chunk_ids": [hit.chunk_id for hit in hits]})
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": json.dumps(_observation(hits), ensure_ascii=False)})
    steps.append({"action": "stop", "reason": "search_limit"})
    return finish()


def _tool_query(call: dict[str, Any]) -> str | None:
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "search_kb" or not isinstance(call.get("id"), str):
        return None
    try:
        arguments = json.loads(function.get("arguments") or "")
    except (TypeError, ValueError):
        return None
    query = arguments.get("query") if isinstance(arguments, dict) else None
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        return None
    return query.strip()


@dataclass
class RetrievalAgentWorkflowProvider(MimoGroundedWorkflowProvider):
    """检索阶段由智能体完成，生成阶段沿用受约束摘录；不与问题改写同时使用。"""

    version: str = "retrieval-agent-v1"

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        if self.llm is None:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="未显式装配真实模型")
        rag = self.rag_stage
        outcome = run_search_agent(self.llm, rag.search_evidence, normalized_query,
                                   rag.search_evidence(normalized_query), self.model)
        if outcome.chunks:
            result = rag.freeze_evidence(normalized_query, tuple(outcome.chunks))
        else:  # 与固定流程一致：没有证据时不进入生成，也不调用模型
            result = {"query": normalized_query, "chunks": [], "top_k": 0,
                      "config_hash": rag.retrieval_config_hash, "corpus_hash": rag.corpus_hash}
        return {**result, "execution_mode": "mimo_grounded",
                "search_agent": {"status": outcome.status, "code": outcome.code, "model_calls": outcome.model_calls,
                                 "steps": outcome.steps, "usage": outcome.usage}}
