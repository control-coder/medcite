# 研究评测子系统

本目录保留既有评测程序、YAML 配置、公开数据集及标注。业务应用代码在 src/medidiag，本目录的研究门禁不阻塞用户应用交付。

- 根目录 Python 模块：配置、执行、诊断、指标和报告工具，保留 python -m eval.runner 等入口。
- datasets：固定公开数据和知识片段；下载原始数据不入库。仓库内 `knowledge_chunks.jsonl` 只含 1,028 段 PubMedQA 上下文，MedQA 教材原文受出版社版权约束，已从全部历史中移除。
  完整知识库须本地重建：从 [MedQA](https://github.com/jind11/MedQA) 获取数据放到 `eval/datasets/raw/medqa_data/`，执行 `python scripts/build_knowledge_base.py --kb-output eval/datasets/raw/knowledge_chunks_full.jsonl`，再在本地 `config.formal.yaml` 中指向该文件。不要用重建结果覆盖并提交仓库版本；历史研究报告基于完整知识库，仅用公开子集不能复现其数值。
- annotations：已有标注、抽样及裁决资料，遵循其研究审计与人工独立性边界。
- config.yaml：开发配置；config.formal.template.yaml：旧研究复现模板；本地 config.formal.yaml 不提交。

输出默认在 `artifacts/reports`，证据类型见 [报告索引](../artifacts/reports/README.md)。评测流程见 [研究评测协议](../docs/research/evaluation-protocol.md)。人工校准仅在需要对应研究结论时开展，不能伪造标注或修改历史无效结论。
