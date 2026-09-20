# 研究评测子系统

本目录保留既有评测程序、YAML 配置、公开数据集及标注。业务应用代码在 src/medidiag，本目录的研究门禁不阻塞用户应用交付。

- 根目录 Python 模块：配置、执行、诊断、指标和报告工具，保留 python -m eval.runner 等入口。
- datasets：固定公开数据和知识片段；下载原始数据不入库。
- annotations：已有标注、抽样及裁决资料，遵循其研究审计与人工独立性边界。
- config.yaml：开发配置；config.formal.template.yaml：旧研究复现模板；本地 config.formal.yaml 不提交。

输出默认在 `artifacts/reports`，证据类型见 [报告索引](../artifacts/reports/README.md)。研究后续记录见 [历史交接](../docs/archive/research/handoff.md)，协议见 [研究协议](../docs/archive/research/evaluation_protocol.md)，当前应用范围见 [实施计划](../项目实施计划.md)。人工校准仅在需要对应研究结论时开展，不能伪造标注或修改历史无效结论。
