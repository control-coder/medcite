"""专科路由数据：专科池、关键词、打分权重、阈值。

确定性路由配置，不依赖 LLM，输出可复现、可消融。

关键词表编写约定（2026-07-26 扩表时确立）：

1. 词条来自各专科的通用临床词汇——症状、体征、常见诊断、化验、检查和治疗，
   不是按评测样本反查出来的高频词。词表**不得**为了抬高某个数据集上的路由
   命中率而增删；`eval/routing_diagnostics.py --holdout` 提供未参与编写的
   对照集，用于检查词表是不是在对 manifest 过拟合。
2. 一个词条只在它真正具有区分度的专科出现。跨专科同时出现的词条（如
   `endocarditis` 之于心内科与感染科）只保留区分度更高的一侧，否则它同时抬高
   两个专科的分数，对 Top2 选择没有贡献，却会推高歧义比值。
3. **词条必须能在归一化后的查询里存活。** `SpecialistRouter._compute_keyword_score`
   打分的是 `NormalizedQuery.normalized`，即同义词已被 `TerminologyNormalizer`
   替换为首选术语之后的文本。若某词条会被归一化器改写成另一个形式，而该形式
   不在本表内，这个词条就永远不可能命中——扩表前 `ECG` / `EKG` 正是如此
   （被改写为 `electrocardiogram`），在 100 样本 manifest 上白白丢掉 5 次命中。
   `tests/test_agents.py::TestSpecialtyKeywords` 对该不变量做了回归保护。
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
#
# 注意：`eval/runner.py` 固定以 `plan_hint=""` 调用 `route()`，因此在评测链路上
# plan_hint 分项恒为 0，8.0 的名义上限里有 1.0 不可达。归一化术语分与证据加权分
# 各自除以 `len(SPECIALTY_KEYWORDS[specialty])`，也不可能接近各自的权重。详见
# `docs/current-status.md` 的路由实测记录。
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
        # 症状与体征
        "chest pain", "chest discomfort", "palpitation", "syncope",
        "orthopnea", "claudication", "edema", "peripheral edema",
        "jugular venous distension", "murmur", "heart murmur",
        "systolic murmur", "diastolic murmur",
        # 检查
        "ECG", "EKG", "electrocardiogram", "echocardiogram",
        "echocardiography", "troponin", "cardiac catheterization",
        "ejection fraction", "ST elevation", "sinus rhythm",
        # 诊断
        "myocardial", "myocardial infarction", "angina", "angina pectoris",
        "arrhythmia", "atrial fibrillation", "tachycardia", "bradycardia",
        "heart failure", "congestive heart failure", "cardiomyopathy",
        "aortic stenosis", "aortic regurgitation", "mitral stenosis",
        "mitral regurgitation", "pericarditis", "aortic dissection",
        "coronary artery disease", "atherosclerosis", "hyperlipidemia",
        "hypertension",
        # 解剖与统称
        "heart", "cardiac", "cardiovascular", "coronary", "ventricular",
        "atrial", "valvular", "blood pressure",
    ],
    "respiratory": [
        # 症状与体征
        "cough", "dyspnea", "shortness of breath", "wheezing", "stridor",
        "crackles", "rales", "rhonchi", "tachypnea", "sputum", "hemoptysis",
        "respiratory distress", "breath sounds", "apnea",
        # 检查与治疗
        "spirometry", "chest x-ray", "oxygen saturation", "bronchodilator",
        "inhaler", "mechanical ventilation", "intubation", "nasal cannula",
        # 诊断
        "asthma", "COPD", "chronic obstructive pulmonary disease",
        "emphysema", "chronic bronchitis", "bronchitis", "bronchiectasis",
        "pneumonia", "pneumothorax", "pleural effusion", "pulmonary embolism",
        "pulmonary fibrosis", "interstitial lung disease", "sarcoidosis",
        "cystic fibrosis", "pulmonary hypertension", "atelectasis",
        "empyema", "respiratory failure", "sleep apnea", "hypoxia",
        "hypoxemia", "hypercapnia",
        # 解剖与统称
        "lung", "pulmonary", "respiratory", "thoracic", "pleural",
        "bronchial", "alveolar",
    ],
    "neurology": [
        # 症状与体征
        "headache", "migraine", "seizure", "convulsion", "paralysis",
        "hemiparesis", "hemiplegia", "numbness", "dizziness", "vertigo",
        "ataxia", "tremor", "nystagmus", "ptosis", "diplopia", "dysarthria",
        "aphasia", "altered mental status", "coma", "consciousness",
        "hyperreflexia", "hyporeflexia", "papilledema", "gait",
        # 检查
        "lumbar puncture", "cerebrospinal fluid", "electroencephalogram",
        "EEG", "cranial nerve", "reflex", "motor", "sensory",
        # 诊断
        "stroke", "transient ischemic attack", "subarachnoid hemorrhage",
        "intracerebral hemorrhage", "epilepsy", "multiple sclerosis",
        "parkinson", "alzheimer", "dementia", "neuropathy",
        "peripheral neuropathy", "myasthenia", "encephalitis",
        "encephalopathy", "meningeal", "myelopathy", "radiculopathy",
        # 解剖与统称
        "brain", "cerebral", "cerebellar", "intracranial", "spinal cord",
        "neural", "neurologic", "neurological",
    ],
    "gastroenterology": [
        # 症状与体征
        "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
        "dysphagia", "heartburn", "dyspepsia", "melena", "hematemesis",
        "hematochezia", "steatorrhea", "jaundice", "ascites",
        "hepatomegaly", "splenomegaly", "GI bleed",
        # 检查
        "colonoscopy", "endoscopy", "bilirubin", "transaminase", "stool",
        # 诊断
        "gastritis", "peptic ulcer", "ulcer", "gastroesophageal reflux",
        "GERD", "esophagitis", "colitis", "ulcerative colitis", "crohn",
        "inflammatory bowel disease", "celiac", "diverticulitis",
        "appendicitis", "pancreatitis", "cholecystitis", "cholelithiasis",
        "gallstone", "hepatitis", "cirrhosis", "portal hypertension",
        "esophageal varices", "bowel obstruction", "irritable bowel",
        # 解剖与统称
        "gastric", "intestinal", "bowel", "colon", "colorectal", "duodenal",
        "esophageal", "esophagus", "liver", "hepatic", "pancreas",
        "pancreatic", "gallbladder", "gastrointestinal",
    ],
    "endocrinology": [
        # 症状与体征
        "polyuria", "polydipsia", "weight loss", "amenorrhea",
        "gynecomastia", "hirsutism", "goiter", "thyroid nodule",
        # 检查
        "glucose", "hemoglobin a1c", "HbA1c", "TSH", "prolactin",
        "cortisol", "aldosterone", "catecholamine",
        # 诊断
        "diabetes", "diabetes mellitus", "hyperglycemia", "hypoglycemia",
        "diabetic ketoacidosis", "ketoacidosis", "hyperosmolar",
        "hyperthyroidism", "hypothyroidism", "thyrotoxicosis", "graves",
        "hashimoto", "cushing", "addison", "acromegaly", "prolactinoma",
        "hyperparathyroidism", "hypercalcemia", "hypocalcemia",
        "pheochromocytoma", "osteoporosis", "polycystic ovary", "obesity",
        # 治疗
        "insulin", "metformin", "levothyroxine", "thyroxine",
        "glucocorticoid", "corticosteroid", "growth hormone", "estrogen",
        "testosterone",
        # 解剖与统称
        "thyroid", "adrenal", "pituitary", "parathyroid", "endocrine",
        "hormone", "metabolic",
    ],
    "oncology": [
        # 症状与体征。`cachexia` 被归一化器改写为 `weight loss`（内分泌科词条），
        # 按约定 3 属于永不命中的死词条，故不收录。
        "lymphadenopathy", "night sweats", "mass", "lesion",
        # 检查
        "biopsy", "cytology", "immunohistochemistry", "tumor marker",
        "mammography", "PSA", "bone marrow", "staging",
        # 诊断
        "tumor", "cancer", "malignant", "malignancy", "neoplasm",
        "carcinoma", "adenocarcinoma", "sarcoma", "melanoma", "lymphoma",
        "hodgkin", "leukemia", "myeloma", "multiple myeloma", "metastasis",
        "metastases", "metastatic", "dysplasia", "paraneoplastic",
        "breast cancer", "lung cancer", "colon cancer", "prostate cancer",
        "cervical cancer",
        # 治疗
        "chemotherapy", "chemotherapeutic", "cytotoxic", "radiation",
        "radiotherapy", "palliative", "remission", "resection", "excision",
        # 统称
        "oncology", "oncologic", "oncogene", "tumor suppressor", "marrow",
    ],
    "infectious_disease": [
        # 症状与体征
        "fever", "febrile", "chills", "purulent", "leukocytosis",
        # 检查
        "culture", "blood culture", "gram stain", "gram-positive",
        "gram-negative",
        # 病原体
        "bacterial", "viral", "fungal", "parasite", "pathogen",
        "staphylococcus", "streptococcus", "escherichia", "pseudomonas",
        "mycobacterium", "candida", "aspergillus", "herpes", "varicella",
        "influenza", "malaria",
        # 诊断
        "infection", "sepsis", "septicemia", "abscess", "cellulitis",
        "osteomyelitis", "meningitis", "tuberculosis", "HIV", "AIDS",
        "syphilis", "gonorrhea", "chlamydia", "wound infection", "UTI",
        "urinary tract infection",
        # 治疗与流行病学
        "antibiotic", "antibiotics", "antimicrobial", "antiviral",
        "antifungal", "prophylaxis", "vaccine", "vaccination",
        "immunization", "contagious", "epidemic", "outbreak",
        "transmission", "incubation period", "immunocompromised",
        "opportunistic",
    ],
    "renal": [
        # 症状与体征
        "oliguria", "dysuria", "nocturia", "urinary frequency",
        "hematuria", "proteinuria", "albuminuria", "urine output",
        # 检查
        "urinalysis", "creatinine", "urea", "BUN", "GFR",
        "glomerular filtration rate", "urine sediment", "renal biopsy",
        # 诊断
        "renal failure", "acute kidney injury", "chronic kidney disease",
        "end-stage renal disease", "nephrotic syndrome",
        "nephritic syndrome", "glomerulonephritis", "nephritis",
        "nephropathy", "pyelonephritis", "nephrolithiasis", "kidney stone",
        "azotemia", "uremia", "polycystic kidney",
        # 电解质与酸碱
        "electrolyte", "hyperkalemia", "hypokalemia", "hyponatremia",
        "hypernatremia", "acidosis", "alkalosis", "metabolic acidosis",
        # 治疗
        "dialysis", "hemodialysis", "peritoneal dialysis", "diuretic",
        "furosemide", "ACE inhibitor",
        # 解剖与统称
        "kidney", "renal", "glomerular", "urinary", "urine", "bladder",
        "ureter", "urethra",
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
