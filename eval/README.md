# 评测代码与历史研究资料

业务应用代码在 `src/medidiag`，本目录的评测不阻塞应用交付。

## 现在使用的部分

- `llm_pipeline_eval.py`：真实模型端到端评测（拒答、查询改写、检索智能体、重复性、输出不合格后的重试）。每次模型调用的请求和回复存成调用记录，放在 `cassettes/`，之后可以零网络、零费用回放。方法和结果见 [docs/evaluation.md](../docs/evaluation.md)。
- `retrieval_benchmark.py`：公开中文语料（`examples/public_health_v2`）上的检索基准，对比 BM25、向量检索等方案。
- `cassettes/`：模型调用记录。文件名中带 `flash` 的是 `mimo-v2.6-flash` 的记录，其余是 `mimo-v2.5` 的记录；两者的回放测试分别是 `tests/test_llm_replay_flash.py` 和 `tests/test_llm_replay_committed.py`。文件名中带 `v3` 的是在 498 条摘录的 v3 语料（`examples/public_health_v3`）上的记录，回放测试是 `tests/test_llm_replay_v3.py`。

## 历史研究资料（只读保留）

早期的多专科 Agent 加自然语言推理（NLI）研究评测，其执行与统计代码（原 `eval/runner.py` 及配套的报告、标注核对、Kappa、费用估算、阈值敏感性、路由诊断等）已经删除，需要时从 git 历史查看。保留的只有数据和说明：

- `datasets/`：公开数据和知识片段；下载的原始数据不入库。`knowledge_chunks.jsonl` 只含 1,028 段 PubMedQA 上下文，MedQA 教材原文受出版社版权约束，已从全部历史中移除。
- `annotations/`：当时的引用核对标注、抽样及裁决资料。
- `config.yaml`：旧的多专科工作流（`medidiag run --provider deepseek_default` 等）仍然读取它，所以保留。

对应的历史结论和限制见 [研究评测协议](../docs/research/evaluation-protocol.md) 与 [NLI 语言兼容性复盘](../docs/research/nli-language-compatibility.md)；报告类型见 [报告索引](../artifacts/reports/README.md)。
