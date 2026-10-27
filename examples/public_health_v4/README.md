# public_health_v4 数据卡（来源部分）

v4 证据语料来源清单与采集说明，对应《RAG升级计划.md》M4。全文快照只保存在本机 `.cache/corpus/raw/`，不入库；本目录提交来源清单（`sources.json`）、采集报告（`fetch_report.json`）与本数据卡。更新日期：2026-10-27。

## 来源与数量

| 来源类型 | 发布方 | 计划数 | 纳入数 | 抓取成功 | 格式 |
| --- | --- | --- | --- | --- | --- |
| who_factsheet | 世界卫生组织 | 索引全量 242 | 242 | 242 | HTML |
| who_publication | 世界卫生组织（IRIS） | 10 | 10 | 10 | PDF |
| nhc_document | 国家卫生健康委、中国政府网 | 22 | 22 | 22 | HTML 10、PDF 8、DOC 2、DOCX 2 |
| epidemic_report | 中国疾病预防控制中心 | 24 | 24 | 24 | HTML（正文含统计表） |
| wikipedia | 中文维基百科 | 200 | 200 | 200 | JSON（MediaWiki API 响应） |
| 合计 | | 498 | 498 | 498 | HTML 277、JSON 200、PDF 17、DOC 2、DOCX 2 |

查询来源（只取问题，不作证据，本轮不下载）：Huatuo26M-Lite（FreedomIntelligence，Hugging Face），Apache-2.0，已于 2026-10-27 经其 API（`/api/datasets/FreedomIntelligence/Huatuo26M-Lite` 的 `cardData.license`）核实。

## 许可结论与依据

- **who_factsheet（纳入）**：WHO 版权页（`https://www.who.int/zh/about/policies/publishing/copyright`）允许非商业使用并注明出处。沿用 v3 做法：全文仅本机保存，展示短引并注明出处。
- **who_publication（纳入）**：逐份核实 IRIS 条目页 `dc.rights` 字段，所选 10 份均为 CC BY-NC-SA 3.0 IGO（本项目为非商业工程原型，引用时附出处与许可声明）。IRIS 为 DSpace 7，PDF 通过其 REST API（`/server/api/core/bitstreams/<uuid>/content`）获取。
- **nhc_document（纳入）**：诊疗方案、指南与政策解读是具有行政性质的官方文件，依《中华人民共和国著作权法》第五条不受著作权保护，仍注明出处与采集日期。多格式验证附件共 4 份：
  - DOCX 2 份（用户审核后补充）：门（急）诊诊疗信息页质量管理规定（试行）附件（含信息页示例表）、重点中心乡镇卫生院建设参考标准附件 1（含基本病种与 ICD 代码两个表格，按科室多级组织）；均以魔数 PK\x03\x04 且含 word/document.xml 验证为真实 DOCX。
  - DOC 2 份：安宁疗护实践指南（2025 年版）、县域慢性疾病分级诊疗技术方案附件一，魔数 D0CF11E0 为二进制 Word；本机无 LibreOffice，M5 解析时将按约定记入报告并跳过，不静默丢弃。
- **epidemic_report（纳入）**：政府公开统计数据，同依《著作权法》第五条。每条记录统计周期 `period`（YYYY-MM），覆盖 2024-09 至 2026-08 共 24 期，供时间过滤问题使用。
- **wikipedia（纳入）**：中文维基百科内容以 CC BY-SA 4.0 授权（页脚声明）。作为低权威对照来源（`level_5_other`）；M5 从中取一小份附许可声明作为可提交的测试数据。

没有「不纳入」条目；没有许可存疑条目。需要用户审核的要点：《著作权法》第五条对「官方文件」的适用范围解释（本项目按通行理解适用于诊疗方案、通知与统计数据，仅作工程演示，不构成法律意见）。

## 选取规则（确定、可复现、不看评测问题）

- **who_factsheet**：WHO 中文事实清单索引页（`https://www.who.int/zh/news-room/fact-sheets`）的全部 242 个条目，`doc_id` 为 `who_fs_` + slug 规范化。
- **who_publication**：IRIS 检索 `f.language=zh,equals & f.itemtype=Publications,equals` 按发布日期排序，人工挑选主题覆盖诊疗、公共卫生且单份小于 15 MB 的文本型 PDF 10 份，多数为指南类（`level_1_guideline`）。
- **nhc_document**：中国政府网政策文件库检索「诊疗方案 / 指南 通知」并逐页核对附件，取流行性感冒诊疗方案（2025 年版）、麻疹等传染病诊疗方案（2024 年版）、新型冠状病毒感染诊疗方案（试行第十版）与重症病例诊疗方案（试行第四版）、安宁疗护实践指南（2025 年版）、肥胖症诊疗指南（2024 年版）、县域慢性肾脏病等慢性疾病分级诊疗技术方案，以及（用户审核后补充）门（急）诊诊疗信息页质量管理规定（试行）与重点中心乡镇卫生院建设参考标准，含通知网页、附件与政策解读。附件以 `parent_doc_id` 指向对应通知网页。
- **epidemic_report**：中国疾控中心「健康数据 → 全国法定传染病疫情情况」栏目最近 24 期（列表页 `index.html`、`index_1.html`），统计周期由标题解析。
- **wikipedia**：两步确定性规则——
  - A（169 条）：WHO 事实清单 242 个主题按手工映射表对应到中文维基条目，经 `action=query` 批量验证存在（含规范化与重定向）后保留 169 条；
  - B（31 条）：从 `Category:病毒`、`Category:细菌性疾病`、`Category:寄生虫病`、`Category:各病因传染病`、`Category:性传播疾病` 按（类别顺序、条目标题）排序，剔除模板/列表/子分类后取前 31 条补足 200 条。
  - 映射表固定、与评测问题无关；复现时用 `sources.json` 中的 `wiki_selection` 字段区分 A/B 两部分。

## 证据等级约定

沿用 `src/medidiag/schemas.py` 五级：诊疗方案与指南类（国家诊疗方案、WHO 指南出版物）为 `level_1_guideline`；WHO 事实清单、政策解读、疫情月报、维基百科为 `level_5_other`。综述、原始研究、病例报告三级在本语料中没有对应来源，未使用。

## 复现命令

```powershell
conda activate medidiag
# 抓取（全量；--source-type 可只抓某一类）
python scripts/corpus_fetch.py --sources examples/public_health_v4/sources.json
# 离线测试与静态检查
python -m pytest -q tests/test_corpus_fetch.py
python -m ruff check scripts/corpus_fetch.py tests/test_corpus_fetch.py
```

快照写入 `.cache/corpus/raw/<source_type>/<doc_id>.<ext>`（原始字节，未做转换）；本机清单 `.cache/corpus/raw/manifest.json` 记录每条的 URL、sha256、字节数与抓取时间；`fetch_report.json` 为可提交的采集报告。维基百科经本机系统代理访问（脚本自动读取 Windows 注册表代理，也可用 `--proxy` 或 `MEDCITE_PROXY` 覆盖）；`action=parse`、`variant=zh-cn`、保存原始 JSON 响应，revid 已记录在 manifest。

采集限速：同一主机请求间隔不少于 1.5 秒（维基百科 3 秒，因其批量访问限速较严）；单个请求最多重试 2 次（维基百科 4 次）；User-Agent 为描述性字符串；不绕过任何访问控制。

## 采集结果与失败条目

2026-10-27 全量抓取：498/498 成功，无失败条目。格式分布：HTML 277、JSON 200、PDF 17、DOC 2、DOCX 2。总快照约 104.6 MB（WHO 事实清单 22.6 MB、WHO 出版物 27.3 MB、国家文件 4.1 MB、疫情月报 2.3 MB、维基百科 48.3 MB）。

维基百科抓取过程中曾遇到 HTTP 429（限速）与代理连接中断，通过放宽维基请求间隔到 3 秒并增加重试后全部成功，无人工干预绕过。

## 局限

- 以抓取快照与 sha256 为准；源页面会更新，报告中注明采集日期，历史引用按哈希回溯。
- WHO 事实清单索引为动态页面，条目数随时间变化；本轮快照为 2026-10-27 的 242 条。
- 疫情月报「发布日期」以页面时间为准记录在快照中，清单只填统计周期 `period`。
- 维基条目内容按 revid 定位版本，但维基百科的医学内容未经专业审核，只作低权威对照来源。
- 许可核实是工程自查，不构成法律意见；WHO 与维基内容再分发须分别遵守 CC BY-NC-SA 3.0 IGO 与 CC BY-SA 4.0 的署名/相同方式共享要求。

## 解析检查（M5）

对应《RAG升级计划.md》M5。498 份快照经 `medidiag corpus ingest` 全量解析（2026-10-27）：**成功 496、失败 0、跳过 2**（2 份 DOC 因本机无 LibreOffice 记入报告跳过，不静默丢弃）。解析产物在 `.cache/corpus/parsed/`（`documents.jsonl` 与 `report.json`，不入库）。

各格式规模：HTML 277 份（8509 元素、33 表）、JSON 200 份（7404 元素、96 表）、PDF 17 份（6867 元素、393 表、46 扫描页）、DOCX 2 份（9 元素、3 表）。疫情月报 24 期 `period` 全部进入元数据。近重复检测（字符 5-gram MinHash，Jaccard ≥ 0.85）：文档级剔除 1 篇（wiki_017 与 wiki_199 全文重复，保留先出现者），段落级跨来源重复 124 对。

每类格式抽样由 AI 对照原文件检查，记录标题层级、表格、断行的错误数：

| 格式 | 文件 | 标题层级 | 表格 | 断行 |
| --- | --- | --- | --- | --- |
| HTML | who_fs_measles / who_fs_diabetes / who_fs_coronavirus_disease_covid_19 | 0 错（h2/h3/h4 计数与原文一致） | — | 0 错 |
| JSON（维基） | wiki_010 / wiki_050 / wiki_100 | 0 错（wiki_050 的 h2=14、h3=25、h4=11 与原文完全一致） | 正确 | 0 错 |
| PDF | nhc_flu_2025_attach / nhc_measles_2024_attach1 | 0 错（「一、」「（一）」层级正确） | — | 0 错 |
| PDF | who_pub_9789240086395 | 3 错（2 处列表项合并误判为标题、1 处表格标题误为章节标题，正文段落完整） | 正确（25 个） | 0 错 |
| DOCX | nhc_township_hospital_standard_attach1 | 0 错（「一、」「二、」识别为 1 级标题） | 正确（2 个，表头 序号/病种/ICD代码 正确） | 0 错 |
| DOCX | nhc_outpatient_info_page_attach | 0 错（表单式文档，无编号标题，按段落处理） | 正确（1 个信息页示例表） | 0 错 |
| 表格（疫情月报 HTML 表） | cdc_report_202608 / 202501 / 202409 | — | 表头与数值正确（59/50/49 行） | — |

DOCX 只有 2 份真实文件，不足验收条款的 3 份抽样数，已如实列出全部 2 份。两份均无 Heading 样式，靠编号文字组织：乡镇卫生院标准按「一、（一）」编号、门诊信息页为表单。「第 X 条」类条款按约定不作标题（「第X章/节/篇/部分」作标题、「第X条」作条款正文开头）；这两份 DOCX 实际不含「第X条」，规则按通用公文约定并补了单测。2 份 DOC（安宁疗护、县域慢性病附件一）因本机无 LibreOffice 记入报告跳过。

WHO 事实清单 242 篇全量核对标题标签数与原文一致；2 篇（登革热、心血管疾病）小节标题退化为 `<p><strong>标题</strong></p>`（无 h2），已按 strong 标题模式识别为 2 级标题。疫情月报正文容器 `<div id="articleCon">` 按来源限定，导航与页脚噪声不进入正文；24 期中 23 期为标准月报（各 1 个统计表），`cdc_report_202411` 为猴痘疫情通报（无统计表），属来源内容差异而非解析错误。

v3 摘录核对（`scripts/verify_v3_excerpts.py`，按 source_url 映射到新 doc_id）：**489/498 逐字找到**（忽略空白）。未逐字找到的 9 条逐条说明：

- kb_covid_etiquette_01、kb_covid_masks_01、kb_covid_ventilation_01、kb_heat_children_01：来源为 WHO questions-and-answers 页面，不在 v4 采集范围（v4 只收 fact-sheets），快照缺失。
- kb_salt_01–04：salt-reduction 页面已被 WHO 下架（v4 清单与本机旧缓存均无），属页面更新导致的缺失。
- kb_poliomyelitis_01：摘录出自页面图片说明（图说）而非正文段落，新解析器按语义排除图片区，属历史定位差异。
- ebola 页面 slug 更新（ebola-virus-disease → ebola-disease）已登记别名映射，其 4 条摘录在新页面逐字存在，不计入缺失。

## 分块（M6）

对应《RAG升级计划.md》M6。对 M5 解析产物（495 篇，文档级去重剔除 1 篇、跳过 2 篇 DOC 不计）做两种策略分块，实现在 `src/medidiag/rag/ingest/chunking.py`，CLI 为 `medidiag corpus chunk --parsed .cache/corpus/parsed/documents.jsonl --strategy <recursive|parent_child> --out .cache/corpus/chunks/<strategy>/`。产物（`chunks.jsonl` 检索块、`parents.jsonl` 父块）只存本机，不入库。

**策略参数**

- 递归切分（recursive，基线）：512 字，重叠约 15%（76 字，字符级前缀重叠），按段落 → 句号类 → 逗号类优先级递归切分，不跨章节。
- 父子分块（parent_child）：子块 1–2 句（≤110 字）进索引，父块为同一章节内的段落（≤600 字，超长段落按句切成多个父块），子块记录 `parent_id`；父块只作送入模型的上下文，不进检索索引。
- 表格：两种策略都把表格行（序列化文本「列名：值；…」）单独成块、不与正文混切；父子分块另把整表摘要作为父块。取舍：表格行是独立的结构化事实，混切会稀释检索信号，且引用校验对比的就是序列化文本。
- 跳过 M5 文档级去重剔除的文档与段落级跨来源重复中被判剔除的一方（重跑 `dedup.dedup_paragraphs` 重建剔除清单）。

**chunk_id 规则**：`{doc_id}:{章节序号}:{策略}:{序号}`，章节序号为文档内从 0 起的全局下标；策略短码 `rc`（递归）、`pc`（父子子块）、`pp`（父子父块）；序号在每个 `(doc_id, 章节序号, 策略)` 组内从 0 起按生成顺序编号，同样输入得到同样的 chunk_id 与顺序。另存 `chunk_hash`（文本 sha256）。`source_id` 用 `doc_id`，`source` 用文档标题；新增字段（parent_id、section_path、strategy、chunk_hash、role、source_type、published_at、period、license、content_kind 等）放进 `metadata`，`KnowledgeChunk` 字段集合不变，旧 JSONL 照常加载。

**语料统计**

| 策略 | 检索块（子块） | 父块 | 子块平均字数 | 父块平均字数 | 表格行块 |
| --- | --- | --- | --- | --- | --- |
| recursive | 11985 | 0 | 207.3 | — | 3067 |
| parent_child | 43602 | 22471 | 56.8 | 104.6 | 3026 |

递归切分全部块 ≤588 字（512+76 重叠）。父子分块有 4 个父块 >600 字（单个不可分割的参考文献/长定义列表条目，22471 个父块中的 0.018%）、24 个子块为超长表格行（个别 PDF 把整段正文误抽成单行表格），属已记录的边界情况。

**标注迁移**（`scripts/migrate_v3_gold.py` → `v3_gold_mapping.json`）：v3 的 498 条摘录按包含关系（先做与 M5 清洗一致的全半角/空白归一，再去空白逐字比较）映射到 v4 子块。`exact` = 单子块逐字包含；`span` = 跨相邻子块、全部计为 gold；`failed` = 映射失败。

| 策略 | exact | span | failed |
| --- | --- | --- | --- |
| recursive | 486 | 0 | 12 |
| parent_child | 459 | 27 | 12 |

12 条失败：M5 已知的 9 条（covid_etiquette、covid_masks、covid_ventilation、heat_children、salt_01–04、poliomyelitis，快照缺失或定位差异），加 3 条内容经清洗后不再逐字一致（obesity_04、tobacco_02、campylobacter_04，含参考文献编号/数值清洗）。112 条 v3 可回答查询中 104 条完成迁移（dev 51、test 53），8 条因全部 gold 摘录失败被排除（public_q04/05/06/12/14、pv2_C_06/08/16）。

**离线对比**（`eval/chunking_matrix.py` → `artifacts/reports/retrieval/chunking-matrix-v4-20261027.json`，不产生调用费用）：在迁移后的 104 条可回答查询上，2 策略 × 2 检索（BM25 cjk_bigram、bge-small-zh 向量，GPU 编码）。检索语料为 v4 全部子块（父块不进索引）；送入模型字数按 top_k=3 计，父子分块取命中子块的父块去重、递归切分按块计。映射失败的 8 条查询从分母排除。

dev（51 条）：

| 策略 | 检索 | Hit@3 | MRR@10 | 平均送入字数 |
| --- | --- | --- | --- | --- |
| recursive | BM25 | 0.431 | 0.396 | 678 |
| recursive | 向量 | 0.373 | 0.338 | 553 |
| parent_child | BM25 | 0.294 | 0.261 | 409 |
| parent_child | **向量** | **0.490** | **0.410** | **344** |

test（53 条，只报告不调参）：

| 策略 | 检索 | Hit@3 | MRR@10 | 平均送入字数 |
| --- | --- | --- | --- | --- |
| recursive | BM25 | 0.472 | 0.409 | 789 |
| recursive | 向量 | 0.321 | 0.237 | 528 |
| parent_child | BM25 | 0.340 | 0.290 | 436 |
| parent_child | 向量 | 0.396 | 0.361 | 396 |

**基于 dev 的结论**：父子分块 + 向量检索在 dev 上 Hit@3 最高（0.490）、MRR@10 最高（0.410），且平均送入模型字数最少（344，因只取命中子块的父块），作为 M8 之后的推荐组合；递归切分 + BM25 是稳妥基线（dev Hit@3 0.431）。父子分块 + BM25 明显偏弱（子块过短、词法信号稀疏），符合小块利于语义检索、不利于词法检索的预期。test 上各组合相对关系与 dev 一致，未见反转。
