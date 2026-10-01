# Third-party Notices

## HTMX

- Component: HTMX
- Version: 2.0.4
- Source: `https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js`
- Upstream SHA-256: `E209DDA5C8235479F3166DEFC7750E1DBCD5A5C1808B7792FC2E6733768FB447`
- Repository SHA-256: `69CAEFD0DA92269066E725D7FE175E26B9D50C962E3056459C0C477154CDB9D3` (same content with one trailing newline)
- License: BSD 2-Clause (`https://github.com/bigskysoftware/htmx/blob/v2.0.4/LICENSE`)

HTMX is a third-party UI dependency and is not claimed as a project contribution.

## 世界卫生组织中文科普短引

examples/public_health 中的 8 个片段来自世界卫生组织中文公开科普网页，仅保留用于来源追溯和工程检索演示的必要连续短引。原文标题、来源地址、页面日期、访问日期和使用边界见该目录 sources.json；不包含原文全文、图片或标志。

第三方内容不因收入本仓库而适用本项目 MIT 再许可，著作权和使用条件仍属于原权利人。公开可访问不等于可任意转载；再利用须自行核对来源条款及适用要求。摘录不表示世界卫生组织对本项目背书，不用于临床决策或医学有效性证明。

## PubMedQA

- 来源：Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering", EMNLP 2019；`https://github.com/pubmedqa/pubmedqa`
- 许可：MIT
- 使用：`eval/datasets/chunks_pubmedqa.jsonl`、`eval_set_pubmedqa.jsonl` 及 `knowledge_chunks.jsonl` 中的 PubMedQA 上下文，由 `scripts/prepare_pubmedqa.py` 转换。

## MedQA

- 来源：Jin et al., "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams", Applied Sciences 2021；`https://github.com/jind11/MedQA`
- 许可：上游仓库为 MIT。
- 使用：`eval/datasets/eval_set_medqa.jsonl` 为 MedQA US 测试题转换结果。MedQA 附带的英文教材著作权属原出版社，本仓库不分发教材语料，完整知识库须按 [eval/README.md](eval/README.md) 本地重建。
- 例外：`eval/annotations/` 中的 citation 抽样与模型预标注为审计引用关联，保留了评测时检索到的少量教材短片段（单段不超过 800 字符），仅用于研究评测；权利人如有异议请提交 Issue，将移除相应片段。

