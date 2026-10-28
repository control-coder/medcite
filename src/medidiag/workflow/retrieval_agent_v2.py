"""检索 Agent v2（M11）：多工具、带检索评估（CRAG）、同一轮并行、有预算上限。

和 v1（``retrieval_agent``，只会换说法再检索）的区别：

- 工具：``search_kb``（可带过滤条件）、``hyde_search``（先让模型写一段假设回答，再用它做向量检索）、
  ``expand_context``（取子块所在的父块原文，只供模型判断，不作为引用）、``finish``（选定证据并给出结论）。
- 检索评估（CRAG，Corrective RAG）：模型每次调用工具时，在参数 ``assessments`` 里给上一批新看到的片段打
  Correct / Ambiguous / Incorrect，不额外增加模型调用。代码规则：结束时只保留被选中、已观察过且不是 Incorrect 的片段；
  全部 Incorrect 或 verdict=insufficient 时证据为空，由既有的弃答分支处理。
- 同一轮里的多个只读调用（检索、HyDE、扩展上下文）并行执行，按 tool_call_id 回填 tool 消息。
- 预算：每题至多 4 次工具执行、5 次模型调用（HyDE 的调用也算）、累计 12000 token；重复检索不再执行。超限即结束，沿用已有证据。
- 任何错误都不让问题失败：保留已有证据，并并入固定链路的检索结果（首轮按原问题检索得到的证据）补齐。

它只负责准备证据；最终回答仍走 ``MimoGroundedWorkflowProvider`` 的抽取式回答（代码旧称 constrained excerpt）。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from medidiag.errors import MediDiagError
from medidiag.llm.contracts import LLMProvider, LLMRequest
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
from medidiag.workflow.query_router import APPLIED_FILTER_KEYS
from medidiag.workflow.retrieval_agent import POOL_SIZE, merge_hits

MAX_TOOL_CALLS = 4  # 每题最多执行几次工具（finish 与被拒绝的调用不计）
MAX_MODEL_CALLS = 5  # 每题最多几次模型调用，HyDE 生成假设回答的调用也算
MAX_TOTAL_TOKENS = 12000  # 累计 token 上限，达到后不再调用模型
MAX_CONTEXT_CHARS = 11000  # 调用记录传输层限制消息总长 12000 字符，留出余量
MAX_QUERY_CHARS = 120
MAX_HYDE_CHARS = 200
OBSERVATION_CHARS = 400  # 单个片段给模型看的最大字数（个别子块很长）
PARENT_CHARS = 800
MAX_TOKENS = 1536  # 带思考的决策调用；评估标签与选择都在这次输出里
REASONING_MODE: Literal["disabled", "enabled"] = "enabled"
LABELS = ("Correct", "Ambiguous", "Incorrect")
PROMPT_VERSION = "retrieval-agent-v2"
HYDE_PROMPT_VERSION = "hyde-v1"

_ASSESSMENTS: dict[str, Any] = {
    "type": "object",
    "description": "给上一批新看到的片段逐个打标签：键是 chunk_id，值是 Correct、Ambiguous 或 Incorrect",
    "additionalProperties": {"type": "string", "enum": list(LABELS)},
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {**properties, "assessments": _ASSESSMENTS},
                       "required": [*required, "assessments"]}}}


SEARCH_TOOL = _tool(
    "search_kb", "在公共卫生科普知识库中检索，返回最相关的几段原文。可同一轮多次调用，分别检索不同的子问题。",
    {"query": {"type": "string", "description": "检索语句，使用规范的书面表述"},
     "filters": {"type": "object", "description": "可选过滤条件，只在问题明确限定时使用",
                 "properties": {"source_type": {"type": "string"}, "period": {"type": "string", "description": "YYYY-MM 或 latest"}}}},
    ["query"])
HYDE_TOOL = _tool(
    "hyde_search", "先写一段假设性的回答，再用它做向量检索；适合问题很口语、检索语句难以写成书面表述时。",
    {"question": {"type": "string", "description": "要回答的问题"}}, ["question"])
EXPAND_TOOL = _tool(
    "expand_context", "取某个片段所在的父块原文，用来判断该片段是否真的回答了问题；内容不能作为引用。",
    {"chunk_id": {"type": "string"}}, ["chunk_id"])
FINISH_TOOL = _tool(
    "finish", "结束检索。selected_chunk_ids 只能填检索结果中出现过的 chunk_id。",
    {"selected_chunk_ids": {"type": "array", "items": {"type": "string"}},
     "verdict": {"type": "string", "enum": ["sufficient", "insufficient"]}},
    ["selected_chunk_ids", "verdict"])

AGENT_PROMPT = (
    "你是公共卫生科普检索助手，负责为后续的摘录步骤准备证据。你只检索，不回答问题。\n"
    "用户消息里有问题，以及按原问题检索得到的第一批结果。检索按字面和语义混合匹配，片段可能只是话题相近。\n"
    "每次调用任何工具时，都要在 assessments 里为你上一批新看到的片段逐个打标签：\n"
    "- Correct：片段明确回答了问题所问的内容；\n"
    "- Ambiguous：话题相关，但看不出是否直接回答，或信息不完整；\n"
    "- Incorrect：只是话题相近或词语相同，没有回答问题。\n"
    "再按标签决定下一步：\n"
    "- 有 Correct 且已足够：调用 finish，selected_chunk_ids 填直接回答问题的片段，verdict 填 sufficient；\n"
    "- 有 Ambiguous：调用 expand_context 看父块上下文，或换说法、补充检索；\n"
    "- 全部 Incorrect：用 search_kb 改用规范书面术语再检索，或用 hyde_search；确实找不到时 finish，"
    "selected_chunk_ids 填空数组，verdict 填 insufficient。\n"
    "问题包含多个部分时，拆成子问题，在同一轮里同时调用多个 search_kb。\n"
    "最多执行 4 次工具，不要重复相同的检索。expand_context 的内容只供你判断，不能当作证据。"
    "不要添加问题中没有的事实、数字或建议。"
)
HYDE_PROMPT = ("你是公共卫生科普写作助手。针对用户问题，写一段不超过 200 字的假设性回答，用世界卫生组织科普页面的书面语气。"
               "它只用于检索，不会展示给用户；不确定时可以泛泛而谈，不要写具体数字。只输出这段文字。")


class _AgentError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class AgentV2Result:
    chunks: list[Any]  # 最终证据，最多 pool_size 段；元素就是检索返回的对象
    steps: list[dict[str, Any]]
    model_calls: int = 0
    tool_calls: int = 0
    parallel_rounds: int = 0  # 同一轮执行了多个只读调用的轮数
    usage: dict[str, int] = field(default_factory=dict)
    # ok：模型自己结束或预算用尽，沿用已有证据；insufficient：评估后证据不足；fallback：出错，并入固定链路补齐
    status: str = "ok"
    code: str | None = None
    stop_reason: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentTools:
    """Agent 依赖的检索能力；hyde 为 None 表示检索器没有向量路，不提供 HyDE 工具。"""

    search: Callable[[str, Mapping[str, str]], Sequence[Any]]
    expand: Callable[[Any], str]  # 片段 -> 父块原文，找不到父块时返回片段本身
    merge: Callable[[Sequence[Sequence[Any]]], list[Any]]
    hyde: Callable[[str, str], Sequence[Any]] | None = None  # (假设回答, 原问题) -> 片段


@dataclass
class _Call:
    raw: dict[str, Any]
    name: str
    args: dict[str, Any]
    assessments: dict[str, str]

    @property
    def id(self) -> str:
        return str(self.raw["id"])


def _text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        return None
    return value.strip()


def parse_call(raw: Any, names: Sequence[str]) -> _Call:
    """解析并校验一次工具调用；任何不合格都按 AGENT_BAD_TOOL_CALL 处理。"""
    function = raw.get("function") if isinstance(raw, dict) else None
    if not isinstance(function, dict) or function.get("name") not in names or not isinstance(raw.get("id"), str):
        raise _AgentError("AGENT_BAD_TOOL_CALL")
    try:
        args = json.loads(function.get("arguments") or "{}")
    except (TypeError, ValueError):
        raise _AgentError("AGENT_BAD_TOOL_CALL") from None
    if not isinstance(args, dict):
        raise _AgentError("AGENT_BAD_TOOL_CALL")
    name = str(function["name"])
    # 评估标签格式不对的项直接忽略，不影响工具调用本身
    given = args.get("assessments")
    assessments = {str(k): v for k, v in given.items() if v in LABELS} if isinstance(given, dict) else {}
    if name == "search_kb":
        query = _text(args.get("query"), MAX_QUERY_CHARS)
        filters = args.get("filters", {})
        if query is None or not isinstance(filters, dict | type(None)):
            raise _AgentError("AGENT_BAD_TOOL_CALL")
        args = {"query": query, "filters": filters or {}}
    elif name == "hyde_search":
        question = _text(args.get("question"), MAX_QUERY_CHARS)
        if question is None:
            raise _AgentError("AGENT_BAD_TOOL_CALL")
        args = {"question": question}
    elif name == "expand_context":
        chunk_id = _text(args.get("chunk_id"), 200)
        if chunk_id is None:
            raise _AgentError("AGENT_BAD_TOOL_CALL")
        args = {"chunk_id": chunk_id}
    else:
        selected, verdict = args.get("selected_chunk_ids"), args.get("verdict")
        if verdict not in ("sufficient", "insufficient") or not (
                selected is None or (isinstance(selected, list) and all(isinstance(i, str) for i in selected))):
            raise _AgentError("AGENT_BAD_TOOL_CALL")
        args = {"selected_chunk_ids": selected, "verdict": verdict}
    return _Call(raw, name, args, assessments)


def _observation(hits: Sequence[Any], shown: set[str]) -> list[dict[str, Any]]:
    """给模型看的结果；之前已展示过原文的片段只给 chunk_id，省 token。"""
    out: list[dict[str, Any]] = []
    for hit in hits:
        if hit.chunk_id in shown:
            out.append({"chunk_id": hit.chunk_id, "seen": True})
        else:
            shown.add(hit.chunk_id)
            out.append({"chunk_id": hit.chunk_id, "text": hit.text[:OBSERVATION_CHARS]})
    return out


def _total_tokens(usage: Mapping[str, Any]) -> int:
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    return sum(v for k, v in usage.items() if k in ("input_tokens", "output_tokens") and isinstance(v, int))


class _Run:
    """一道题的执行状态；所有可变状态集中在这里，HyDE 的模型调用在线程里更新用量，所以加锁。"""

    def __init__(self, llm: LLMProvider, tools: AgentTools, question: str, first_hits: Sequence[Any],
                 model: str, default_filters: Mapping[str, str], pool_size: int, timeout_s: float) -> None:
        self.llm, self.tools, self.question, self.model = llm, tools, question, model
        self.default_filters = dict(default_filters)
        self.pool_size, self.timeout_s = pool_size, timeout_s
        self.first_hits = list(first_hits)
        self.result_lists: list[Sequence[Any]] = [self.first_hits]
        self.observed: dict[str, Any] = {h.chunk_id: h for h in self.first_hits}
        self.shown: set[str] = set()
        self.labels: dict[str, str] = {}
        self.seen_keys: set[tuple[Any, ...]] = {("search", question.strip(), tuple(sorted(self.default_filters.items())))}
        self.lock = threading.Lock()
        self.out = AgentV2Result(chunks=[], steps=[
            {"action": "search", "tool": "first_search", "query": question, "filters": self.default_filters,
             "chunk_ids": [h.chunk_id for h in self.first_hits]}])
        self._last_ms = 0
        self.names = ["search_kb", "expand_context", "finish"] + (["hyde_search"] if tools.hyde else [])

    # ---- 模型调用 ----
    def _add_usage(self, usage: Mapping[str, Any]) -> None:
        with self.lock:
            for key, value in usage.items():
                if isinstance(value, int):
                    self.out.usage[key] = self.out.usage.get(key, 0) + value

    def _tokens(self) -> int:
        return _total_tokens(self.out.usage)

    def _decide(self, messages: list[dict[str, Any]], finish_only: bool) -> Any:
        tools = [FINISH_TOOL] if finish_only else [SEARCH_TOOL, EXPAND_TOOL, FINISH_TOOL]
        if not finish_only and self.tools.hyde:
            tools.insert(1, HYDE_TOOL)
        sent = messages
        if finish_only:  # 工具额度用完，只留 finish，让模型对最后一批结果打标签并给出选择
            sent = [*messages, {"role": "user", "content": "工具额度已用完，请为最后一批新片段打标签，并调用 finish 给出选择。"}]
        started = time.perf_counter()
        with self.lock:
            self.out.model_calls += 1
        reply = self.llm.generate(
            LLMRequest(messages=sent, model=self.model, tools=tools, tool_choice="auto", max_tokens=MAX_TOKENS,
                       reasoning_mode=REASONING_MODE, prompt_version=PROMPT_VERSION),
            timeout_s=self.timeout_s, idempotency_key="mimo-agent2-" + uuid.uuid4().hex)
        self._add_usage(reply.usage)
        self._last_ms = round((time.perf_counter() - started) * 1000)
        if reply.model != self.model:
            raise _AgentError("AGENT_MODEL_MISMATCH")
        if not reply.tool_calls and reply.finish_reason == "length":
            raise _AgentError("AGENT_TRUNCATED")
        return reply

    def _hyde_text(self, question: str) -> str:
        with self.lock:
            self.out.model_calls += 1
        reply = self.llm.generate(
            LLMRequest(messages=[{"role": "system", "content": HYDE_PROMPT},
                                 {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)}],
                       model=self.model, max_tokens=400, reasoning_mode="disabled", prompt_version=HYDE_PROMPT_VERSION),
            timeout_s=self.timeout_s, idempotency_key="mimo-hyde-" + uuid.uuid4().hex)
        self._add_usage(reply.usage)
        text = reply.content.strip()[:MAX_HYDE_CHARS] if reply.model == self.model else ""
        if not text:
            raise _AgentError("AGENT_HYDE_EMPTY")
        return text

    # ---- 工具 ----
    def _execute(self, call: _Call) -> tuple[list[Any] | None, dict[str, Any]]:
        """执行一次只读调用，返回 (命中片段, 步骤记录)；在线程里运行，不修改共享的检索状态。"""
        started = time.perf_counter()
        args = call.args
        if call.name == "search_kb":
            filters = self._filters(args["filters"])
            hits = list(self.tools.search(args["query"], filters))
            step: dict[str, Any] = {"tool": "search_kb", "query": args["query"], "filters": filters}
        elif call.name == "hyde_search":
            assert self.tools.hyde is not None
            hypothetical = self._hyde_text(args["question"])
            hits = list(self.tools.hyde(hypothetical, self.question))
            step = {"tool": "hyde_search", "question": args["question"], "hypothetical": hypothetical}
        else:
            hit = self.observed[args["chunk_id"]]
            step = {"tool": "expand_context", "chunk_id": args["chunk_id"]}
            text = self.tools.expand(hit)[:PARENT_CHARS]
            step["parent_chars"] = len(text)
            step["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            return None, {**step, "_text": text}
        step["chunk_ids"] = [h.chunk_id for h in hits]
        step["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return hits, step

    def _filters(self, requested: Mapping[str, Any]) -> dict[str, str]:
        """模型给的过滤条件只接受已应用的键（M10），其余丢弃；合法性由 MetadataIndex 在检索时校验。"""
        merged = dict(self.default_filters)
        merged.update({k: v.strip() for k, v in requested.items()
                       if k in APPLIED_FILTER_KEYS and isinstance(v, str) and v.strip()})
        return merged

    def _round(self, calls: list[_Call], messages: list[dict[str, Any]]) -> int:
        """执行一轮只读调用：先按顺序做重复与额度检查，再并行执行，最后按调用顺序回填 tool 消息。返回实际执行数。"""
        budget = MAX_TOOL_CALLS - self.out.tool_calls
        model_slots = MAX_MODEL_CALLS - 1 - self.out.model_calls  # 至少留一次给后续的决策调用
        replies: dict[str, dict[str, Any]] = {}
        runnable: list[_Call] = []
        for call in calls:
            reason = None
            if call.name == "search_kb":
                key: tuple[Any, ...] = ("search", call.args["query"],
                                        tuple(sorted(self._filters(call.args["filters"]).items())))
            elif call.name == "hyde_search":
                key = ("hyde", call.args["question"])
            else:
                key = ("expand", call.args["chunk_id"])
            if key in self.seen_keys:
                reason = "duplicate_query"
            elif budget <= 0:
                reason = "tool_limit"
            elif call.name == "hyde_search" and (model_slots <= 0 or self._tokens() >= MAX_TOTAL_TOKENS):
                reason = "model_call_limit"
            elif call.name == "expand_context" and call.args["chunk_id"] not in self.observed:
                reason = "unknown_chunk_id"
            if reason is not None:
                replies[call.id] = {"skipped": reason}
                self.out.steps.append({"action": "skip", "tool": call.name, "reason": reason, **_brief(call)})
                continue
            self.seen_keys.add(key)
            budget -= 1
            if call.name == "hyde_search":
                model_slots -= 1
            runnable.append(call)
        outcomes: list[tuple[_Call, tuple[list[Any] | None, dict[str, Any]] | Exception]] = []
        if len(runnable) > 1:
            self.out.parallel_rounds += 1
        with ThreadPoolExecutor(max_workers=max(1, len(runnable))) as pool:
            futures = [(call, pool.submit(self._execute, call)) for call in runnable]
            for call, future in futures:
                try:
                    outcomes.append((call, future.result()))
                except Exception as exc:  # 记录后继续收集同一轮其他调用的结果，最后统一报错降级
                    outcomes.append((call, exc))
        error: Exception | None = None
        for call, outcome in outcomes:
            if isinstance(outcome, Exception):
                error = error or outcome
                self.out.steps.append({"action": "error", "tool": call.name, "code": _code(outcome)})
                continue
            hits, step = outcome
            self.out.tool_calls += 1
            if hits is None:  # expand_context
                replies[call.id] = {"chunk_id": step["chunk_id"], "parent_text": step.pop("_text")}
            else:
                for hit in hits:
                    self.observed.setdefault(hit.chunk_id, hit)
                self.result_lists.append(hits)
                replies[call.id] = {"results": _observation(hits, self.shown)}
            self.out.steps.append({"action": "tool", **step})
        if error is not None:
            raise error
        messages.append({"role": "assistant", "content": "", "tool_calls": [c.raw for c in calls]})
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(replies[call.id], ensure_ascii=False)})
        return len(runnable)

    # ---- 主循环与收尾 ----
    def _absorb(self, calls: Sequence[_Call]) -> dict[str, str]:
        """合并本次调用给出的评估标签；只认已观察过的 chunk_id，同一片段后给的标签覆盖先给的。"""
        given: dict[str, str] = {}
        for call in calls:
            given.update({cid: label for cid, label in call.assessments.items() if cid in self.observed})
        self.labels.update(given)
        return given

    def loop(self) -> tuple[list[str] | None, str, str]:
        """返回 (模型选择的 chunk_id 或 None, verdict, 结束原因)。None 表示没有明确选择，沿用所有未被判错的片段。"""
        shown_first = _observation(self.first_hits, self.shown)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": json.dumps({"question": self.question, "first_search": shown_first},
                                                   ensure_ascii=False)}]
        while True:
            if self.out.model_calls >= MAX_MODEL_CALLS:
                return None, "sufficient", "model_call_limit"
            if self._tokens() >= MAX_TOTAL_TOKENS:
                return None, "sufficient", "token_limit"
            if len(str(messages)) > MAX_CONTEXT_CHARS:
                return None, "sufficient", "context_limit"
            finish_only = self.out.model_calls + 1 >= MAX_MODEL_CALLS or self.out.tool_calls >= MAX_TOOL_CALLS
            reply = self._decide(messages, finish_only)
            if not reply.tool_calls:
                self.out.steps.append({"action": "model", "call": self.out.model_calls, "tool_calls": [],
                                       "tokens": _total_tokens(reply.usage), "elapsed_ms": self._last_ms})
                return None, "sufficient", "model_done"
            calls = [parse_call(raw, self.names) for raw in reply.tool_calls]
            given = self._absorb(calls)
            self.out.steps.append({"action": "model", "call": self.out.model_calls,
                                   "tool_calls": [c.name for c in calls], "assessments": given,
                                   "tokens": _total_tokens(reply.usage), "elapsed_ms": self._last_ms})
            finish = next((c for c in calls if c.name == "finish"), None)
            if finish is not None:  # 同一轮里 finish 之外的调用不再执行
                return finish.args["selected_chunk_ids"], finish.args["verdict"], "finish"
            if finish_only:
                return None, "sufficient", "tool_limit"
            if self._round(calls, messages) == 0:
                return None, "sufficient", "no_new_tool"

    def select(self, selected: list[str] | None, verdict: str) -> set[str] | None:
        """CRAG 规则：返回要保留的 chunk_id；None 表示证据不足。"""
        if verdict == "insufficient":
            return None
        usable = {cid for cid in self.observed if self.labels.get(cid) != "Incorrect"}
        if selected is None or not selected:
            keep = usable  # 没有给出选择：沿用所有未被判错的已观察片段
        else:
            keep = {cid for cid in selected if cid in usable}
            dropped = [cid for cid in selected if cid not in keep]
            if dropped:
                self.out.steps.append({"action": "drop_selected", "chunk_ids": dropped})
        return keep or None

    def merged(self, keep: set[str]) -> list[Any]:
        lists = [[h for h in hits if h.chunk_id in keep] for hits in self.result_lists]
        try:
            return self.tools.merge(lists)[:self.pool_size]
        except Exception as exc:  # 重排是排序增强，失败时退回倒数排名融合
            self.out.steps.append({"action": "merge", "method": "rrf", "reason": "rerank_failed", "code": _code(exc)})
            return merge_hits(lists, self.pool_size)


def _code(exc: BaseException) -> str:
    return str(getattr(exc, "code", type(exc).__name__))


def _brief(call: _Call) -> dict[str, Any]:
    return {k: v for k, v in call.args.items() if k in ("query", "question", "chunk_id")}


def run_search_agent_v2(llm: LLMProvider, tools: AgentTools, question: str, first_hits: Sequence[Any], model: str, *,
                        default_filters: Mapping[str, str] | None = None, pool_size: int = POOL_SIZE,
                        timeout_s: float = 45) -> AgentV2Result:
    """``first_hits`` 是按原问题（带路由过滤）检索的第一批结果，不计入工具执行次数。"""
    run = _Run(llm, tools, question, first_hits, model, default_filters or {}, pool_size, timeout_s)
    out = run.out
    try:
        selected, verdict, out.stop_reason = run.loop()
        keep = run.select(selected, verdict)
        out.labels = dict(run.labels)
        if keep is None:
            out.status, out.chunks = "insufficient", []
            out.steps.append({"action": "finish", "status": "insufficient", "reason": out.stop_reason})
            return out
        out.chunks = run.merged(keep)
        out.steps.append({"action": "finish", "status": "ok", "reason": out.stop_reason,
                          "chunk_ids": [h.chunk_id for h in out.chunks]})
        return out
    except Exception as exc:  # 任何错误都不让问题失败：未被判错的已得证据加上固定链路的首轮结果
        out.status, out.code = "fallback", _code(exc)
        out.stop_reason = "error"
        out.labels = dict(run.labels)
        out.steps.append({"action": "error", "code": out.code})
        keep = {cid for cid in run.observed if run.labels.get(cid) != "Incorrect"} | {h.chunk_id for h in run.first_hits}
        out.chunks = run.merged(keep)
        out.steps.append({"action": "finish", "status": "fallback", "chunk_ids": [h.chunk_id for h in out.chunks]})
        return out


def _dense_of(retriever: Any) -> Any | None:
    """检索器的向量一路：混合检索取其 dense，单路向量检索就是它自己，BM25 单路没有。"""
    dense = getattr(retriever, "dense", None)
    if dense is not None:
        return dense
    from medidiag.rag.dense import DenseRetriever

    return retriever if isinstance(retriever, DenseRetriever) else None


@dataclass
class RetrievalAgentV2WorkflowProvider(MimoGroundedWorkflowProvider):
    """检索阶段由 Agent v2 完成，生成阶段沿用抽取式回答；不与问题改写同时使用。"""

    version: str = "retrieval-agent-v2"
    # 子块 parent_id -> 父块原文；默认从语料目录里的 parents.jsonl 懒加载，没有该文件时扩展上下文返回子块本身
    parents: Mapping[str, str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _parent_map(self) -> Mapping[str, str]:
        with self._lock:
            if self.parents is None:
                from medidiag.schemas import read_jsonl

                path = Path(self.rag_stage.corpus_path).with_name("parents.jsonl")
                self.parents = {r["chunk_id"]: r["text"] for r in read_jsonl(path)} if path.exists() else {}
            return self.parents

    def _tools(self, question: str) -> AgentTools:
        """检索与重排共用同一个模型和索引，不是线程安全的，所以并行只发生在模型调用与等待上，检索本身串行。"""
        rag = self.rag_stage
        depth = rag.reranker.candidates if rag.reranker else int(rag.retrieval_config.get("top_k", POOL_SIZE))
        top_k = int(rag.retrieval_config.get("top_k", POOL_SIZE))
        dense = _dense_of(rag.retriever)

        def search(query: str, filters: Mapping[str, str]) -> Sequence[Any]:
            with self._lock:
                return rag.search_filtered(query, filters=filters or None)[0]

        def expand(hit: Any) -> str:
            parent_id = (getattr(hit, "metadata", None) or {}).get("parent_id")
            return self._parent_map().get(parent_id or "") or hit.text

        def merge(lists: Sequence[Sequence[Any]]) -> list[Any]:
            if rag.reranker is None:
                return merge_hits(lists, POOL_SIZE)
            with self._lock:
                return list(rag.rerank_evidence(question, [h for hits in lists for h in hits], POOL_SIZE))

        def hyde(hypothetical: str, _: str) -> Sequence[Any]:
            assert dense is not None
            with self._lock:
                items = tuple(rag._evidence_item(r) for r in dense.search(hypothetical, top_k=depth) if r.chunk)
                return rag.rerank_evidence(question, items, top_k) if rag.reranker else items[:top_k]

        return AgentTools(search=search, expand=expand, merge=merge, hyde=hyde if dense is not None else None)

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        if self.llm is None:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail="未显式装配真实模型")
        rag = self.rag_stage
        route = self._route(normalized_query)
        if route is not None and route.refused:
            return {**self._refusal(normalized_query, route), "execution_mode": "mimo_grounded"}
        filters = route.filters if route is not None else {}
        with self._lock:
            first_hits, trace = rag.search_filtered(normalized_query, filters=filters or None)
        outcome = run_search_agent_v2(self.llm, self._tools(normalized_query), normalized_query, first_hits, self.model,
                                      default_filters=filters)
        if outcome.chunks:
            result = rag.freeze_evidence(normalized_query, tuple(outcome.chunks))
        else:  # 与固定流程一致：没有证据时不进入生成，也不调用模型
            result = {"query": normalized_query, "chunks": [], "top_k": 0,
                      "config_hash": rag.retrieval_config_hash, "corpus_hash": rag.corpus_hash}
        if route is not None:
            result["query_router"] = route.to_dict()
        if trace is not None:
            result["metadata_filter"] = trace.to_dict()
        return {**result, "execution_mode": "mimo_grounded", "search_agent": {
            "version": 2, "status": outcome.status, "code": outcome.code, "stop_reason": outcome.stop_reason,
            "model_calls": outcome.model_calls, "tool_calls": outcome.tool_calls,
            "parallel_rounds": outcome.parallel_rounds, "labels": outcome.labels,
            "steps": outcome.steps, "usage": outcome.usage}}
