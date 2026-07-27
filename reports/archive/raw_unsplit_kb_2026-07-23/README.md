# 未切分知识库上的检索测量（2026-07-23，已被取代）

本目录保存 2026-07-23 在**未切分**知识库上产出的全部检索结果与体积测量。它们不可
与 2026-07-27 之后的任何结果比较，也不得作为正式结论或简历指标引用。

## 为什么不可比

当时 `scripts/build_knowledge_base.py` 只按空行（`\n\n`）切分教材，没有字符上限。
18 本教材里 `Surgery_Schwartz.txt`（11.4 MB、仅 250 个换行、几乎没有空行段落边界）
因此只切出 126 个巨型块，最大单块 722,301 字符。结果是：

- 知识库 1927 chunk / 4,351,716 字符，其中最大的 15 个 chunk 占 **79.55%** 字符量；
- 21 个 chunk 超过 10,000 字符；
- embedding 模型 `all-MiniLM-L6-v2` 输入窗口为 256 token，巨块实际只有开头约
  1000 字符参与向量检索，其余内容既检索不到、又会在命中后被整块塞进 prompt。

2026-07-27 引入 `MAX_CHUNK_CHARS = 800` 后知识库变为 1928 chunk / 692,533 字符，
最大教材 chunk 800 字符。**知识库内容变了，因此 Recall@5、GoldCoverage、prompt 体积
都是在不同语料上测的数，直接对比只有"修复前 / 修复后"这一种读法，不能当作同一
设置下的性能变化。** 新旧数字的并列见 `docs/current-status.md`。

## 目录内容

| 路径 | 内容 | 已知问题 |
| --- | --- | --- |
| `20260723T072355043511Z_e5cae4bf7140/` | `rag_embedding`，Recall@5=0.0071 | FAISS 分数与 chunk ID 未按 `indices` 回填的实现缺陷，数字无效 |
| `20260723T074438809322Z_e5cae4bf7140/` | 同上，重跑 | 同上 |
| `20260723T083139002188Z_7fe2f747cc11/` | 修复回填后的单组 `rag_embedding`=0.7393 | 仅一组，未切分 KB |
| `20260723T084524335257Z_7fe2f747cc11/` | CPU 四组纯检索对照 | 未切分 KB |
| `20260723T091824063036Z_ded91c0c061e/` | GPU 四组纯检索对照（与 CPU 一致） | 未切分 KB；被 `20260727T080437012992Z_ded91c0c061e` 取代 |
| `retrieval_diagnostics_smoke.json` | 20 样本 CPU smoke，`knowledge_base_chunk_count: 1927` | 未切分 KB，且是 smoke 规模 |
| `retrieval_diagnostics_gpu_smoke.json` | 同上，GPU | 同上 |
| `prompt_size_diagnostics_unsplit_kb.json` | 未切分 KB 上的 prompt 体积（A1 阻塞项的原始证据） | 未切分 KB |

`prompt_size_diagnostics_unsplit_kb.json` 是用 `eval/prompt_size_diagnostics.py`
指向 git `HEAD:eval/datasets/knowledge_chunks.jsonl` 导出的旧知识库重测得到的，
不是 2026-07-26 那次临时脚本的原始输出。它复现了旧文档记录的 p50 与 max，但
`agent_single` 的 p90 测得 709,960 而旧文档记为 727,501，差异未追查（旧临时脚本
已不存在）。这一点在 `docs/current-status.md` 中如实记录，没有取其一了事。
