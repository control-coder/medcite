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
3. **词条的归一化改写结果不得落在别的专科词表上。**（2026-07-27 修订）
   `SpecialistRouter._compute_keyword_score` 现在同时打分原文与归一化后的文本，
   因此「词条被改写成本表之外的形式」不再等于「永不命中」——`ECG` / `EKG` 被改写为
   `electrocardiogram` 时仍然能在原文里命中。仍然有害的是改写结果落到**别的**专科
   词表上：`cachexia` 会被改写为 `weight loss`（内分泌科词条），若只打分改写后的
   文本，一个肿瘤科词条会为内分泌科加分。`tests/test_agents.py::TestSpecialtyKeywords`
   与 `TestKeywordScoringSeesBothTexts` 对这两条性质做了回归保护。
4. **不得假设词表越长分数越低。** 三个分项现在都按固定的 `MATCH_SATURATION_COUNT`
   归一（见下），加词不会压低任何已经命中的专科的分数。修复前不是这样，
   `tests/test_agents.py::TestScoreDenominators` 守护这条不变量。
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
# 单专科总分 = 3.0*关键词分 + 2.0*归一化术语分 + 2.0*证据加权分
#
# 2026-07-27（DD-023）删除了原第四个分项 `plan_hint`（权重 1.0）：它唯一的取值来源
# 是 `route(plan_hint=...)` 的实参，而全部调用点都传空串，因此该分项恒为 0，名义
# 上限 8.0 里有 1.0 结构上不可达。删除后 7.0 既是名义上限也是可达上限。
#
# **其余三项权重刻意保持 3.0 / 2.0 / 2.0，没有按 8.0 重标定。** 对三项同乘一个常数
# 等价于把 `MIN_PRIMARY_SCORE` 除以同一个常数，那是阈值决策而不是缺陷修复；阈值
# 是否要动仍是待人工决策项（`docs/current-status.md`）。
ROUTING_WEIGHTS: dict[str, float] = {
    "keyword": 3.0,
    "normalized_term": 2.0,
    "evidence": 2.0,
}

# ===== 三个分项共用的匹配数饱和上限 =====
# 每个分项都按 `min(命中数 / MATCH_SATURATION_COUNT, 1.0)` 归一，因此分母与词表
# 长度无关。取 4 的理由：
#
# 1. 这是 `_compute_keyword_score` 自 P5 起就在用的常数，三项统一到它不引入新的
#    自由参数；统一之后 3.0 / 2.0 / 2.0 这组权重才真的表示分项之间的相对重要性。
# 2. 修复前 `normalized_term` 与 `evidence` 除以 `len(SPECIALTY_KEYWORDS[specialty])`
#    （51-59），于是**给词表加词会压低已经命中的专科的分数**——扩表与打分互相打架。
# 3. 不取更小的上限（如 2）是刻意的：更小的上限会整体抬高总分，效果等价于降低
#    `MIN_PRIMARY_SCORE`，属于阈值决策。实测分布（200 个 MedQA 样本的真实检索结果）
#    为：归一化术语命中数 p90 ≈ 2、最大 5；单条证据 chunk 的最佳专科命中数 p90 ≈ 3、
#    最大 6。4 落在两者的 p90 与 max 之间，能让强信号样本接近 1.0 而不让弱信号饱和。
MATCH_SATURATION_COUNT: float = 4.0

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
        # 症状与体征。`cachexia` 于 2026-07-27 重新收录：关键词分现在同时打分原文
        # 与归一化文本，被改写掉的词条不再是死词条（DD-023）。归一化术语分仍只看
        # 首选形式，因此该词条在那一项上仍然归到 `weight loss`（内分泌科）。
        "cachexia",
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

# ===== 已删除：SYSTEM_MATCH_KEYWORDS（原 plan_hint 分项的词表）=====
# 2026-07-27 随 plan_hint 分项一并删除（DD-023）。它只被 `_compute_plan_hint_score`
# 读取，而该函数的入参在所有调用点都是空串。持久化产物里也没有任何「系统匹配」
# 字段可以填进来：`medidiag.workflow` 的 `plan` 阶段确实产出 `objective` 文本，但
# 运行时 worker 从不调用 `SpecialistRouter`，评测链路里也没有 plan 阶段。要恢复这个
# 分项，得先有一个真实产出系统归属信息的上游阶段，那是新功能而不是修 bug。
