# Final Evaluation Report (Exploratory Archive)

> 已归档：该报告不满足 `SUPPLEMENT_PLAN.md` 正式门禁，不得用于简历或正式指标。

> 自动生成，请勿手动编辑。

## 锁定配置
- generation_model: `deepseek-v4-flash-free`
- temperature: 0, seed: 42
- dataset_version: v1

## 消融实验结果

| 组别 | 配置 | Recall@5 | Citation Precision | Unsupported Rate | Workflow Success | P95(ms) |
|---|---|---|---|---|---|---|
| A | 纯 embedding（基线） | 0.0400 | 0.0000 | 0.0000 | 0.0400 | 0.0 |
| B | A + BM25 | 0.3000 | 0.0000 | 0.0000 | 0.3000 | 0.0 |
| C | A + 证据等级加权 | 0.0600 | 0.0000 | 0.0000 | 0.0600 | 0.0 |
| D | A + 术语归一化 | 0.0800 | 0.0000 | 0.0000 | 0.0800 | 0.0 |
| E | A + 引用审核 | 0.0400 | 0.0000 | 0.0000 | 0.0400 | 0.0 |
| F | 全量组合 | 0.5200 | 0.0000 | 0.0000 | 0.5200 | 0.0 |

## 关键发现

### Terminology Normalization Gain
- Recall@5(A组, 无归一化): 0.0400
- Recall@5(D组, 有归一化): 0.0800
- **Gain: +0.0400**

### 多 Agent 增益
- 单 Agent (A组) Workflow Success: 0.0400
- 双专科动态路由 (C组) Workflow Success: 0.0600
- **增益: +0.0200**

### 路由覆盖率（C组动态路由）
- 非兜底路由比例: 1.0000

### 消融三组对比
| 组 | 配置 | Recall@5 | Workflow Success |
|---|---|---|---|
| A | 单 Agent baseline | 0.0400 | 0.0400 |
| B | 固定心内科+呼吸科 | 0.3000 | 0.3000 |
| C | 动态路由 Top2 | 0.0600 | 0.0600 |

## API 调用优化
- **AgentOutputCache**: 基于 input_hash 缓存，避免重复 LLM 调用（temperature=0 可复现）
- **批量 embedding**: 检索索引构建一次，所有消融组复用
- **检索结果复用**: embedding_scores 跨消融组复用（只权重不同）
- **--dry-run**: 只计算检索指标，不调 LLM（快速验证管线）

## 贡献边界
- **Upstream Reference**: edict（工程模式）、MedQA、PubMedQA、MeSH
- **Third-party Components**: DeepSeek API、sentence-transformers、cross-encoder、deberta-v3-base-mnli、FAISS、BM25
- **My Contributions**: 状态机执行器、任务租约、医学 RAG 消融、术语归一化三层、引用校验+合规管控、双专科仲裁实验

## 复现命令
```bash
python -m eval.runner --config eval/config.yaml --group all --output reports/raw/
```
