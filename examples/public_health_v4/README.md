# public_health_v4 数据卡（来源部分）

v4 证据语料来源清单与采集说明，对应《RAG升级计划.md》M4。全文快照只保存在本机 `.cache/corpus/raw/`，不入库；本目录提交来源清单（`sources.json`）、采集报告（`fetch_report.json`）与本数据卡。更新日期：2026-10-27。

## 来源与数量

| 来源类型 | 发布方 | 计划数 | 纳入数 | 抓取成功 | 格式 |
| --- | --- | --- | --- | --- | --- |
| who_factsheet | 世界卫生组织 | 索引全量 242 | 242 | 242 | HTML |
| who_publication | 世界卫生组织（IRIS） | 10 | 10 | 10 | PDF |
| nhc_document | 国家卫生健康委、中国政府网 | 18 | 18 | 18 | HTML 8、PDF 8、DOC 2 |
| epidemic_report | 中国疾病预防控制中心 | 24 | 24 | 24 | HTML（正文含统计表） |
| wikipedia | 中文维基百科 | 200 | 200 | 200 | JSON（MediaWiki API 响应） |
| 合计 | | 494 | 494 | 494 | HTML 275、JSON 200、PDF 17、DOC 2 |

查询来源（只取问题，不作证据，本轮不下载）：Huatuo26M-Lite（FreedomIntelligence，Hugging Face），Apache-2.0，已于 2026-10-27 经其 API（`/api/datasets/FreedomIntelligence/Huatuo26M-Lite` 的 `cardData.license`）核实。

## 许可结论与依据

- **who_factsheet（纳入）**：WHO 版权页（`https://www.who.int/zh/about/policies/publishing/copyright`）允许非商业使用并注明出处。沿用 v3 做法：全文仅本机保存，展示短引并注明出处。
- **who_publication（纳入）**：逐份核实 IRIS 条目页 `dc.rights` 字段，所选 10 份均为 CC BY-NC-SA 3.0 IGO（本项目为非商业工程原型，引用时附出处与许可声明）。IRIS 为 DSpace 7，PDF 通过其 REST API（`/server/api/core/bitstreams/<uuid>/content`）获取。
- **nhc_document（纳入）**：诊疗方案、指南与政策解读是具有行政性质的官方文件，依《中华人民共和国著作权法》第五条不受著作权保护，仍注明出处与采集日期。其中 2 份附件原始格式为二进制 Word（.doc，安宁疗护实践指南 2025、县域慢性疾病分级诊疗技术方案附件一），供 M5 验证 DOC/DOCX 解析。
- **epidemic_report（纳入）**：政府公开统计数据，同依《著作权法》第五条。每条记录统计周期 `period`（YYYY-MM），覆盖 2024-09 至 2026-08 共 24 期，供时间过滤问题使用。
- **wikipedia（纳入）**：中文维基百科内容以 CC BY-SA 4.0 授权（页脚声明）。作为低权威对照来源（`level_5_other`）；M5 从中取一小份附许可声明作为可提交的测试数据。

没有「不纳入」条目；没有许可存疑条目。需要用户审核的要点：《著作权法》第五条对「官方文件」的适用范围解释（本项目按通行理解适用于诊疗方案、通知与统计数据，仅作工程演示，不构成法律意见）。

## 选取规则（确定、可复现、不看评测问题）

- **who_factsheet**：WHO 中文事实清单索引页（`https://www.who.int/zh/news-room/fact-sheets`）的全部 242 个条目，`doc_id` 为 `who_fs_` + slug 规范化。
- **who_publication**：IRIS 检索 `f.language=zh,equals & f.itemtype=Publications,equals` 按发布日期排序，人工挑选主题覆盖诊疗、公共卫生且单份小于 15 MB 的文本型 PDF 10 份，多数为指南类（`level_1_guideline`）。
- **nhc_document**：中国政府网政策文件库检索「诊疗方案 / 指南 通知」并逐页核对附件，取流行性感冒诊疗方案（2025 年版）、麻疹等传染病诊疗方案（2024 年版）、新型冠状病毒感染诊疗方案（试行第十版）与重症病例诊疗方案（试行第四版）、安宁疗护实践指南（2025 年版）、肥胖症诊疗指南（2024 年版）、县域慢性肾脏病等慢性疾病分级诊疗技术方案，含通知网页、附件与政策解读。附件以 `parent_doc_id` 指向对应通知网页。
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

2026-10-27 全量抓取：494/494 成功，无失败条目。格式分布：HTML 275、JSON 200、PDF 17、DOC 2。总快照约 104.5 MB（WHO 事实清单 22.6 MB、WHO 出版物 27.3 MB、国家文件 4.0 MB、疫情月报 2.3 MB、维基百科 48.3 MB）。

维基百科抓取过程中曾遇到 HTTP 429（限速）与代理连接中断，通过放宽维基请求间隔到 3 秒并增加重试后全部成功，无人工干预绕过。

## 局限

- 以抓取快照与 sha256 为准；源页面会更新，报告中注明采集日期，历史引用按哈希回溯。
- WHO 事实清单索引为动态页面，条目数随时间变化；本轮快照为 2026-10-27 的 242 条。
- 疫情月报「发布日期」以页面时间为准记录在快照中，清单只填统计周期 `period`。
- 维基条目内容按 revid 定位版本，但维基百科的医学内容未经专业审核，只作低权威对照来源。
- 许可核实是工程自查，不构成法律意见；WHO 与维基内容再分发须分别遵守 CC BY-NC-SA 3.0 IGO 与 CC BY-SA 4.0 的署名/相同方式共享要求。
