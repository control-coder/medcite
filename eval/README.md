# 评测代码

业务应用代码在 `src/medidiag`，本目录的评测不阻塞应用交付。方法、结果和限制见 [docs/evaluation.md](../docs/evaluation.md)。

## 评测入口

- `llm_pipeline_eval.py`：真实模型端到端评测（拒答、查询改写、检索 Agent、重复运行的稳定性、输出被拦截后的重试）。每次模型调用的请求和回复存成调用记录，放在 `cassettes/`，之后可以零网络、零费用回放。
- `retrieval_benchmark.py`：公开中文语料上的检索基准（BM25、向量、混合权重选择、语料规模衰减）。
- `retrieval_matrix.py`：L1 检索层矩阵，Hit/Recall/MRR/nDCG、成对比较、符号检验与自助法区间；只读报告，不调用模型。
- `generation_matrix.py`：L2/L3 矩阵，引用校验通过率、误答、过度拒答、正确作答数、回答 F1；只读报告。
- `multiturn_eval.py`：多轮追问评测，同一条追问按只用追问、拼接历史、模型补全、源查询四种方式交给向量检索加 Agent 加重排；数据集由 `scripts/build_multiturn_set.py` 生成。
- `failure_attribution.py`：把端到端报告里的每个失败归入六个错误类别之一；被拦截输出的复核结论写在代码的 `REVIEWED` 里。
- `refusal_curve.py`：拒答取舍曲线，在入库端到端结果前加证据分数门槛，按阈值 τ 重算误答、过度拒答与正确作答；本地重建证据与重排分数（不调用在线模型），门槛分数与阈值只在 v3 dev 上选。
- `regression_gate.py`：回归门槛，对照 `examples/rag_regression/thresholds.json`，CI 的 `eval-regression` 作业运行它。

## 调用记录

`cassettes/` 里 18 份记录。v2 语料下文件名带 `flash` 的是 `mimo-v2.6-flash`，其余是 `mimo-v2.5`；带 `v3` 的是 498 条摘录的 v3 语料（`examples/public_health_v3`），全部为 `mimo-v2.6-flash`。`multiturn-v3*.jsonl` 是多轮追问评测，只存新增的请求，其余读 v3 与扩充测试集的原记录。对应的回放测试是 `tests/test_llm_replay_committed.py`、`test_llm_replay_flash.py`、`test_llm_replay_v3.py`、`test_llm_replay_v3_ext.py` 和 `test_multiturn_eval.py`（最后一个的检索 Agent 部分需要本地模型，CI 中跳过）。

早期的多专科 Agent 加自然语言推理（NLI）研究评测、人工引用标注及其统计代码已经删除，需要时从 git 历史查看；入库报告类型见 [报告索引](../artifacts/reports/README.md)。
