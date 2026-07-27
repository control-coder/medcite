"""术语归一化模块。

三层归一化（PLAN.md）：
1. 轻量词典层：medical_terms.json 维护常见症状、疾病、检查项、缩写 + 同义词映射
2. 数据集派生层：从评测集 PubMedQA MESHES 字段派生术语表
3. 外部标准映射层：MeSH descriptor（mesh_synonyms.json，阶段 1 生成）

不宣称 UMLS 级能力。

归一化逻辑：
- 第 1 层提供同义词替换（synonym -> preferred_term）
- 第 2+3 层扩展术语识别范围（用于 term_overlap 计算）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from medidiag.schemas import read_jsonl


@dataclass
class MatchedTerm:
    """命中的术语信息。"""

    synonym: str
    """查询中出现的同义词/术语。"""

    preferred: str
    """归一化后的首选术语。"""

    source: str
    """来源: "lightweight" | "derived" | "mesh"。"""


@dataclass
class NormalizedQuery:
    """归一化后的查询。"""

    original: str
    """原始查询。"""

    normalized: str
    """归一化后查询（同义词已替换为首选术语）。"""

    matched_terms: list[MatchedTerm] = field(default_factory=list)
    """命中的术语列表。"""

    @property
    def coverage(self) -> float:
        """术语命中率: 命中的不同首选术语数 / 查询词数（近似）。"""
        if not self.original.strip():
            return 0.0
        words = self.original.split()
        if not words:
            return 0.0
        unique = len({m.preferred.lower() for m in self.matched_terms})
        return min(unique / len(words), 1.0)


class TerminologyNormalizer:
    """术语归一化器。

    三层归一化:
        1. 轻量词典（medical_terms.json）: 同义词替换
        2. 数据集派生（PubMedQA MESHES）: 术语识别
        3. MeSH（mesh_synonyms.json）: 术语识别
    """

    def __init__(
        self,
        medical_terms_path: str | Path | None = None,
        mesh_synonyms_path: str | Path | None = None,
        eval_set_path: str | Path | None = None,
    ) -> None:
        base = Path(__file__).resolve().parent
        self.medical_terms_path = (
            Path(medical_terms_path)
            if medical_terms_path
            else base / "medical_terms" / "medical_terms.json"
        )
        self.mesh_synonyms_path = (
            Path(mesh_synonyms_path)
            if mesh_synonyms_path
            else base / "medical_terms" / "mesh_synonyms.json"
        )
        self.eval_set_path = (
            Path(eval_set_path) if eval_set_path else None
        )

        # synonym -> preferred (第 1 层，提供替换)
        self._synonym_map: dict[str, str] = {}
        # 所有已知术语（第 2+3 层，用于术语识别）
        self._term_set: set[str] = set()
        # term -> source
        self._term_sources: dict[str, str] = {}

        self._load_lightweight()
        self._load_mesh()
        self._load_derived()

    def _load_lightweight(self) -> None:
        """加载第 1 层：轻量词典（同义词映射）。"""
        if not self.medical_terms_path.exists():
            return
        data = json.loads(self.medical_terms_path.read_text(encoding="utf-8"))

        for cat in ["symptoms", "diseases", "procedures", "medications"]:
            for preferred, syns in data.get(cat, {}).items():
                p_lower = preferred.lower()
                self._synonym_map[p_lower] = preferred
                self._term_set.add(p_lower)
                self._term_sources[p_lower] = "lightweight"
                for s in syns:
                    s_lower = s.lower()
                    self._synonym_map[s_lower] = preferred
                    self._term_set.add(s_lower)
                    self._term_sources[s_lower] = "lightweight"

        for abbr, full in data.get("abbreviations", {}).items():
            a_lower = abbr.lower()
            self._synonym_map[a_lower] = full
            self._term_set.add(a_lower)
            self._term_sources[a_lower] = "lightweight"

    def _load_mesh(self) -> None:
        """加载第 3 层：MeSH 术语表。"""
        if not self.mesh_synonyms_path.exists():
            return
        data = json.loads(self.mesh_synonyms_path.read_text(encoding="utf-8"))
        for term, syns in data.items():
            t_lower = term.lower()
            self._term_set.add(t_lower)
            if t_lower not in self._term_sources:
                self._term_sources[t_lower] = "mesh"
            for s in syns:
                s_lower = s.lower()
                self._term_set.add(s_lower)
                if s_lower not in self._term_sources:
                    self._term_sources[s_lower] = "mesh"

    def _load_derived(self) -> None:
        """加载第 2 层：从评测集 PubMedQA MESHES 派生术语表。"""
        if not self.eval_set_path or not self.eval_set_path.exists():
            return
        records = read_jsonl(self.eval_set_path)
        for rec in records:
            meshes = rec.get("metadata", {}).get("meshes", [])
            for m in meshes:
                m_lower = m.lower()
                self._term_set.add(m_lower)
                if m_lower not in self._term_sources:
                    self._term_sources[m_lower] = "derived"

    def normalize(self, query: str) -> NormalizedQuery:
        """归一化查询。

        第 1 层（轻量词典）提供同义词替换。
        第 2+3 层扩展术语识别（记录命中但不替换）。
        """
        normalized = query
        matched: list[MatchedTerm] = []
        seen_synonyms: set[str] = set()

        # 第 1 层：同义词替换（按长度降序避免短同义词先匹配）
        for syn in sorted(self._synonym_map.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(syn) + r"\b"
            if re.search(pattern, normalized, re.IGNORECASE):
                preferred = self._synonym_map[syn]
                if syn.lower() != preferred.lower():
                    normalized = re.sub(
                        pattern, preferred, normalized, flags=re.IGNORECASE
                    )
                if syn.lower() not in seen_synonyms:
                    matched.append(
                        MatchedTerm(
                            synonym=syn,
                            preferred=preferred,
                            source="lightweight",
                        )
                    )
                    seen_synonyms.add(syn.lower())

        # 第 2+3 层：术语识别（不替换，只记录命中）
        for term in self._term_set:
            if term in self._synonym_map:
                continue  # 第 1 层已处理
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, query, re.IGNORECASE):
                if term not in seen_synonyms:
                    matched.append(
                        MatchedTerm(
                            synonym=term,
                            preferred=term,
                            source=self._term_sources.get(term, "unknown"),
                        )
                    )
                    seen_synonyms.add(term)

        return NormalizedQuery(
            original=query,
            normalized=normalized,
            matched_terms=matched,
        )

    def get_term_overlap(self, query: str, chunk_text: str) -> float:
        """计算查询与 chunk 文本的术语重叠度。

        基于归一化后的术语在 chunk 中的命中比例。
        """
        nq = self.normalize(query)
        if not nq.matched_terms:
            return 0.0
        unique_preferred = list({m.preferred.lower() for m in nq.matched_terms})
        if not unique_preferred:
            return 0.0
        # 构建 preferred -> 所有变体（preferred + 同义词）的映射
        preferred_variants: dict[str, list[str]] = {}
        for syn, pref in self._synonym_map.items():
            pref_lower = pref.lower()
            if pref_lower not in preferred_variants:
                preferred_variants[pref_lower] = [pref_lower]
            preferred_variants[pref_lower].append(syn)

        hits = 0
        for term in unique_preferred:
            # 检查 preferred term 及其所有同义词是否在 chunk_text 中
            variants = preferred_variants.get(term, [term])
            found = False
            for v in variants:
                pattern = r"\b" + re.escape(v) + r"\b"
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    found = True
                    break
            if found:
                hits += 1
        return hits / len(unique_preferred)

    def preferred_form(self, term: str) -> str:
        """返回 ``term`` 的首选形式；未知术语原样返回。

        供需要把同义词折叠成同一术语的调用方使用（如 `SpecialistRouter` 的关键词
        计数，避免 `ECG` / `EKG` / `electrocardiogram` 被算成三次命中）。
        """
        return self._synonym_map.get(term.lower(), term)

    @property
    def term_count(self) -> int:
        """已知术语总数。"""
        return len(self._term_set)

    @property
    def synonym_count(self) -> int:
        """同义词映射总数（第 1 层）。"""
        return len(self._synonym_map)
