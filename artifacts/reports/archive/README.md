# Exploratory Report Archive

本目录保存 2026-07-13 审计前的历史报告。其 A-F 实验同时改变 RAG 与 Agent topology，CitationVerifier 使用规则 fallback，dry-run 还曾把 retrieval hit 映射为 workflow success。

这些文件不可用于正式结论、简历指标或新报告输入；保留它们只为追溯旧实现与审计差异。

## 子目录

- `raw_unsplit_kb_2026-07-23/`：2026-07-23 在**未切分**知识库上产出的全部检索结果与体积测量。2026-07-27 给 chunk 加了 800 字符硬上限（DD-022），知识库内容变了，这些结果与之后的任何结果不可比。同上，只用于追溯，不得引用。详见该目录的 `README.md`。
