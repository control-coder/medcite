"""专科路由数据：专科池、关键词、打分权重、阈值。

确定性路由配置，不依赖 LLM，输出可复现、可消融。
"""

from __future__ import annotations

# ===== 专科池（10个）=====
SPECIALTIES: list[str] = [
    "cardiology",
    "respiratory",
    "neurology",
    "gastroenterology",
    "endocrinology",
    "oncology",
    "infectious_disease",
    "renal",
    "general_internal",
    "evidence_skeptic",
]

# 兜底组合（禁止双相同兜底角色）
FALLBACK_PAIR: tuple[str, str] = ("general_internal", "evidence_skeptic")

# 消融基线：固定心内科+呼吸科
BASELINE_PAIR: tuple[str, str] = ("cardiology", "respiratory")

# ===== 打分公式权重 =====
# 单专科总分 = 3.0*关键词分 + 2.0*归一化术语分 + 2.0*证据加权分 + 1.0*诊断规划提示分
ROUTING_WEIGHTS: dict[str, float] = {
    "keyword": 3.0,
    "normalized_term": 2.0,
    "evidence": 2.0,
    "plan_hint": 1.0,
}

# ===== Top2 筛选阈值 =====
THRESHOLDS: dict[str, float] = {
    "MIN_PRIMARY_SCORE": 2.0,
    "MIN_SECONDARY_SCORE": 1.2,
    "SCORE_GAP": 3.0,
    "LOW_CONFIDENCE": 0.45,
}

# ===== 每个专科的关键词（用于 keyword_score）=====
SPECIALTY_KEYWORDS: dict[str, list[str]] = {
    "cardiology": [
        "chest pain", "palpitation", "ECG", "EKG", "heart", "cardiac",
        "myocardial", "arrhythmia", "hypertension", "heart failure",
        "atrial fibrillation", "coronary", "ST elevation", "blood pressure",
        "syncope", "edema",
    ],
    "respiratory": [
        "cough", "dyspnea", "shortness of breath", "lung", "pulmonary",
        "asthma", "COPD", "pneumonia", "respiratory", "sputum",
        "wheezing", "hypoxia", "thoracic", "pleural", "hemoptysis",
    ],
    "neurology": [
        "headache", "seizure", "stroke", "paralysis", "numbness",
        "consciousness", "brain", "neural", "dementia", "vertigo",
        "migraine", "reflex", "motor", "sensory", "aphasia",
    ],
    "gastroenterology": [
        "abdominal pain", "nausea", "vomiting", "liver", "diarrhea",
        "constipation", "gastric", "intestinal", "bowel", "hepatitis",
        "jaundice", "ulcer", "GI bleed", "ascites", "dysphagia",
    ],
    "endocrinology": [
        "diabetes", "thyroid", "glucose", "insulin", "weight loss",
        "polyuria", "polydipsia", "hormone", "adrenal", "pituitary",
        "metabolic", "hyperglycemia", "hypoglycemia", "goiter", "cortisol",
    ],
    "oncology": [
        "tumor", "cancer", "metastasis", "chemotherapy", "biopsy",
        "malignant", "neoplasm", "carcinoma", "lymphoma", "leukemia",
        "radiation", "oncology", "mass", "lesion", "remission",
    ],
    "infectious_disease": [
        "fever", "infection", "sepsis", "antibiotic", "bacterial",
        "viral", "fungal", "parasite", "contagious", "epidemic",
        "wound infection", "UTI", "meningitis", "septicemia", "chills",
    ],
    "renal": [
        "kidney", "renal", "urinalysis", "dialysis", "creatinine",
        "urea", "nephropathy", "proteinuria", "hematuria", "oliguria",
        "renal failure", "nephritis", "BUN", "glomerular", "electrolyte",
    ],
    "general_internal": [],  # 通用，无特定关键词
    "evidence_skeptic": [],  # 证据审核，无特定关键词
}

# ===== 诊断规划系统匹配关键词（用于 plan_hint_score）=====
SYSTEM_MATCH_KEYWORDS: dict[str, list[str]] = {
    "cardiology": ["cardiovascular system", "circulatory"],
    "respiratory": ["respiratory system", "ventilatory"],
    "neurology": ["nervous system", "neurological"],
    "gastroenterology": ["digestive system", "gastrointestinal"],
    "endocrinology": ["endocrine system", "metabolic system"],
    "oncology": ["neoplastic", "oncological"],
    "infectious_disease": ["immune system", "infectious process"],
    "renal": ["urinary system", "renal system"],
}
