# RQ5-B Kimi 跨架构泛化最终冻结报告

最终状态：RQ5B_KIMI_FROZEN。

冻结时间：2026-09-23T01:04:08.678137+00:00。本轮只做已有证据核验、汇总和封存，未启动 GPU、未重跑实验或测试、未改动任何旧 artifact 或 FlowState 算法。

模型：`/data1/models/Kimi-Linear-48B-A3B-Instruct`；SGLang `0.5.17` / `29481685462732237d80d86076d6563e1f658102`；2 × NVIDIA H100 PCIe（各 81,559 MiB）；TP=2。

| 阶段 | 冻结依据状态 | 原始 artifact |
| --- | --- | --- |
| Phase 1 | RQ5B_KIMI_PHASE1_READY | [/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043](/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043/final_report.md) |
| Phase 2 | RQ5B_KIMI_PHASE2_READY | [/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527](/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527/final_report.md) |
| Phase 3 | RQ5B_KIMI_PHASE3_READY | [/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555](/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/final_report.md) |

Phase 1 证明 KDA tracking、MLA/KDA 驻留观测、KDA-only eviction、MLA 保持及 TP rank 一致性。smoke 和随后 3/3 独立生命周期均通过。

**前沿证据口径：**Phase 1 原报告的 `16K/16K/0 → 16K/8K/8K` 来自仅含 c1@8K、c2@16K 的逻辑视图。原始逐 rank 路径同时保留 10K、12K、14K KDA；只删除 16K 后，全祖先驻留证据给出的 E 是 14K、G 是 2K。最终冻结保留原 READY 状态和原文件，但不将该两候选视图表述为实际 8K 回退测量。Phase 1 支持组件隔离结论；真正的 `16K/8K/8K` 由 Phase 3 的实际请求匹配证明。

Phase 2 直接复用现有兼容性、H/E/G、恢复成本接口及 greedy allocator。两个 workflow、四个候选 A1/A2/B1/B2，K=2，S*={A2,B2}；两个 rank 均正确保留 A2/B2 并仅删除 A1/B1 的 KDA。smoke 和 3/3 独立生命周期通过。批量观测、只读检查、executable-state 构造、协调实现、H/E/G 预测、MLA 保持及 TP 一致性均 PASS。每个 epoch 一次逻辑批量观测、一次逻辑批量协调，每批各含两个 rank RPC，聚合验证不额外读取；不主张并发请求下的跨 rank 事务容错。

Phase 2 恢复成本数值只用于冻结接口功能验证，不是 Kimi 延迟校准。相关测试记录为 62/62 PASS；既有 5,161 文件核验无变化。

Phase 3 在 A/B 公共准备中仅删除 10K/12K/14K KDA，保留 MLA 和 8K/16K；A/B 唯一条件差异是 16K KDA 是否驻留。实际请求在两个 rank 上均确认 A=16K/16K/0、B=16K/8K/8K。五组配对采用 AB/BA/AB/BA/AB，每个条件独立 engine，使用相同历史、请求及配置。

主指标为两个 TP rank 内部恢复时延的较大者：首次实际前缀匹配至单令牌请求完成，包含相同的 256-token 后缀处理，排除启动、历史物化、观测和驱逐。该指标是实际 resume 路径时间，不是纯回放 kernel 时间。

| 配对 | A：G=0（ms） | B：G=8K（ms） | B−A（ms） |
| --- | ---: | ---: | ---: |
| 1 | 73.151000 | 821.563500 | +748.412500 |
| 2 | 70.813869 | 829.894119 | +759.080250 |
| 3 | 73.985595 | 825.251420 | +751.265825 |
| 4 | 71.011529 | 834.938277 | +763.926748 |
| 5 | 69.784901 | 848.055759 | +778.270858 |

配对差值均值 **+760.19 ms**，中位数 **+759.08 ms**；正差值 **5/5**。MLA 保持、TP 一致性、正确性门禁及 GPU 清理均为 10/10 PASS。相关测试记录 46 PASS；Phase 3 原有 5,267 文件完整性记录无变化。

本轮重新核验三份既有 manifest：Phase 1 47 项、Phase 2 100 项、Phase 3 159 项，均 PASS；同时核验历史保护清单的 5267 个文件及阶段源码对应关系。去重后共核验 5435 个既有文件，变化数为 0。既有测试仅核验记录，未重跑，不将重叠测试相加为独立测试总数。

冻结入口为 [frozen_manifest.json](frozen_manifest.json)，包含既有证据、既有保护文件、阶段源码及本次报告和完整性汇总的路径、字节数与 SHA-256。清单自身摘要见 frozen_manifest.sha256；其余核验说明见 [integrity_summary.json](integrity_summary.json)。排除各阶段 cache/ 下的编译缓存和模型权重内容；不改动旧文件权限或重写旧清单。

结论范围限于该 Kimi 模型、固定 SGLang/TP=2 环境与小规模构造工作负载。未重拟合 Φ、未新增 baseline、未复制 RQ4，且不声称 TTFT 改善比例、严格线性或完整单调性。

FlowState 的同一 executable-state allocation 与组件隔离的 batched recurrent-state realization 可在不改变核心语义的情况下迁移到 Kimi Linear 的 MLA+KDA；在固定 MLA 驻留下，真实 8K 恢复间隔在五个独立配对中均产生可测的额外恢复开销。
