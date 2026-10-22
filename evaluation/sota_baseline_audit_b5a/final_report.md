# Phase B5-A 最终报告：Marconi 与 KVFlow 源码级 Baseline 适配审计

日期：2026-10-16 ｜ 性质：只读源码审计（未实现 Baseline、未跑对比、未占 GPU、未改核心代码）

## 1. 总体判定：**PARTIAL**（审计完备；适配立约完成；实现前置 2 项评审决策）

- 源码级审计：**完成** —— 两系统论文机制均已逐条落到 `文件:函数:行号` 级证据（marconi_audit.md / kvflow_audit.md），无臆测项，未知处一律 UNVERIFIED；
- 适配契约：**完成** —— 两系统各给出 A/B/C 方案与推荐（adaptation_contract.md），A 方案均 BLOCKED（理由备案），B 方案均可 CPU-only 进算法层主表；
- 判 PARTIAL 而非 READY 的原因：B 方案各含一个**必须由任务方拍板**的条款（见 §4），在拍板前不应进入实现。

## 2. 关键发现（按影响排序）

1. **KVFlow 仓库版本归属不可建立**（kvflow_audit.md 前言）：历史压缩为 2 个 2026 年提交 + README:3 自认双项目（KVFlow+ScaleSim）混载；论文（2025-07）时点代码不可恢复。一切"论文机制在官方代码中如何实现"的问题只能回答到"当前 HEAD 事实 + UNVERIFIED"；
2. **`TreeNode.__lt__` 专项（任务点名）**：base（radix_cache.py:141）与 LoRA（lora_radix_cache.py:120）两处 `__lt__` 首行均无条件 `return last_access_time < ...`，agent priority 分支为死代码且与 `agents: dict[str,str]` 类型矛盾。可证实的影响：非分层驱逐路径当前为纯 LRU；**但分层路径的优先级保持不经过 `__lt__`**（`_evict_helper` 直接阈值判断，需 `--enable-holding`，默认关）。论文版本是否用此路径 UNVERIFIED，不作失效断言；
3. **KVFlow 优先级驱逐的可证实形态是"阈值保持"而非"排序选择"**（`_evict_helper` 中 heapify 被注释、`leaves.pop()` 无序）——适配契约采纳前者；
4. **Marconi artifact 是 CPU trace 模拟器**，无 GPU 服务路径；发现 4 处论文-代码差异（D1 MLP 公式笔误 :526、D2 调参口径、D3 mamba FLOPs 公式投稿-定稿演进、D4 V3 oracle 乐观上界），全部备案并给出实现处置；
5. **KVFlow 的 Agent Step Graph / varying-suffix 显式驱逐标记在仓库中不存在**（workflow 解析与 benchmark 客户端未随仓库发布）——适配因此定位于"steps-aware 驱逐决策层"，非 workflow 感知全栈。

## 3. 交付物清单

| 文件 | 内容 |
|---|---|
| source_lock.json | 论文/仓库/SHA/版本归属警示/审计覆盖文件清单 |
| marconi_audit.md | 7 审计项（论文§+源码行号+FlowState 差异）+ 差异表 + UNVERIFIED |
| kvflow_audit.md | 9 审计项（含 `__lt__` 专项与完整调用链）+ UNVERIFIED |
| adaptation_contract.md | A/B/C 方案、硬约束、输出接口、待评审条款 |
| baseline_comparison_matrix.md | 决策机制/信息公平性/忠实性/工程属性矩阵 |
| final_report.md | 本文件 |

## 4. 待评审决策点（进入实现前必须拍板）

1. **KVFlow-Adapt 的 steps 信息源**（adaptation_contract.md §2）：以 pending 图纪元距离作为 steps-to-execution 是否可接受（论证：当前声明状态、非未来请求、与 FlowState 同信息集故公平）。否 → 降级 C（不进主表）；
2. **Marconi-Adapt 调参口径**（差异 D2）：α 选择按论文口径（命中率）还是代码口径（FLOPs 节省）。建议论文口径 + 备注备案。

## 5. 开放问题（不阻塞实现，写作时注意）

- KVFlow 论文 camera-ready（NeurIPS 版）与 arXiv v1 是否机制有变：不可考（v2 不存在）；
- Marconi 论文轨迹数据（Google Drive）未下载；其数字引用须注明模拟口径；
- KVFlow 基准 SGLang 上游 commit 不可精确对应。

## 6. 建议下一步（待任务方指令，本轮不自动执行）

1. 拍板 §4 两项；
2. 按契约实现 `Marconi-Adapt`、`KVFlow-Adapt`（CPU、逻辑层、独立产物目录，不触 `flowstate/` 核心）；
3. 接入 g001-g004 算法层主表对比（需要时另行申请 GPU 做端到端，范围另议）。
