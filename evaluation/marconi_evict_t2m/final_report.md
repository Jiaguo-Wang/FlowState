# T2-Mini：Marconi Baseline 最小机制补强报告

日期：2026-10-27。**已完成两项核心修改与 CPU 验证；未启动 GPU；既有代码与冻结数据零修改（全部新增文件仅在本目录 `evaluation/marconi_evict_t2m/`）。**

## 1. 本次修改的代码与机制

新增 [evict_policy.py](evict_policy.py)（`MarconiEvictPolicy` + `plan_case`），在既有 `MarconiStylePolicy` 之上仅做两项修改，评分与归一化完全复用既有 `_build_marconi_metrics`（未重写、未改 `sota_policies.py`）：

1. **增量收益**：有父候选的收益代理 = `L(自身) − L(最近保留祖先)`，无父候选用 `L(自身)`（对齐官方 evict_v2 的 parent-relative savings）。内存分母使用 T2-R Census 实测等大小 **51,511,296 bytes**（16 候选逐项核验，见测试 `test_real_bytes_evidence`）。
2. **动态驱逐**（对齐官方 `evict_v2` 逐轮合同）：只有叶/单孩子节点可驱逐；每轮在**当前可驱逐集合内**重算归一化 Recency + α·归一化增量效率（α=1），驱逐最低 utility（同分按 checkpoint_id 升序）；驱逐后子节点父指针等价重接到最近保留祖先（逻辑层合并，同官方 `_evict_intermediate_node`）；重复直到满足 K。全程轨迹（每轮可驱逐集/代理/utility/受害者）随结果输出。

保留原接口输出：保留集、驱逐序（按先后）、每候选效用评分（逐轮）、最终 E/G 与 J（与冻结表同用 `legacy.epoch/common.score`）、单次调用耗时（诊断，非基准）。旧版静态 Top-K 原样保留为 **Marconi-Legacy**（`MarconiStylePolicy` 一行未动）。

## 2. 新旧 Marconi 的选集差异（A/B/C 合成声明输入 × K3/5）

| 结构/K | Legacy 选集方向 | Evict 选集方向 | 当前 J：Legacy→Evict | 未来静态 J：Legacy→Evict |
|---|---|---|---|---|
| A/3, A/5 | 最深叶 | 不同集 | 154.349→154.349（平） | 926.09→1082.89 / 617.40→779.10（更差） |
| B/3 | 最深叶 | 不同集 | 230.764→233.258（更差） | 309.08→309.08（平） |
| B/5 | 最深叶 | 不同集 | 230.764→230.764（平） | 115.19→117.66（略差） |
| C/3 | 最深叶（D7168/D7104/C6144） | 祖先链（ROOT1024/JOIN4032/JOIN4096） | 348.23→703.45（更差） | **154.35→0.00（更优）** |
| C/5 | 最深叶 | 混合 | 348.23→392.78（更差） | 0.00→0.00（平） |

6 条件选集全部不同。机制归因（以 C/K3 为例，单案例不外推）：Legacy 的累计 L 代理天然偏好最深叶节点；Evict 的父相对增量使深层节点增量小（如 D7168 相对父仅 64 Token）而优先被驱逐，最终保留覆盖声明未来调用的祖先链——**把"当前调用覆盖"换成了"未来调用覆盖"**。负向观测全部保留：当前 J 上 Evict 0 胜（3 平 3 负），未来静态 J 上 1 胜 4 负 1 平。数据为声明合成输入的 modeled ms，非实测延迟，不构成优劣结论。

## 3. 关键测试（[test_evict.py](test_evict.py)，8/8 通过）

- 小规模父子树：逐轮独立重算可驱逐集/增量代理/受害者定义全部一致；父被驱逐后子代理相对更远祖先重算（逻辑合并语义）。
- 动态驱逐重评分：两孩子中间节点在孩子均保留时不可驱逐，孩子减少后逐步可驱逐（官方保护语义）。
- 预算合同：k=0 全驱逐、k≥n 零轮、bool/负数/非整数拒绝；父缺失/父不更浅/环/重复 ID/字节不符拒绝。
- 确定性：全同分输入两次运行选集与轨迹全等（排除计时字段）。
- A/B/C×K3/5：旧版复算与冻结 `cpu_comparison.csv` 的 Marconi 行**选集/当前 J/未来 J 全等**（[run_compare.py](run_compare.py) 再证 12 项交叉全等）；新版可运行、选集合法、输入零修改；`workloads.json`/`cpu_comparison.csv` 摘要前后不变。

正式对照产物：[evict_vs_legacy.csv](evict_vs_legacy.csv)（12 行）+ [compare_summary.json](compare_summary.json) + 测试内差异记录 [abc_diff_test.json](abc_diff_test.json)。

## 4. 能否接入现有三策略 CPU/GPU 接口

**CPU 可以，GPU 未启动。** `plan_case` 输出形状对齐 `baseline_reuse_t2r/interface.plan_declared` 的 Marconi 分支（selected/per_pending/J_ms/timing），并复用其 `validate_declared_case` 冻结输入验证；接入只需将 interface.py 的 Marconi 分支替换为本模块调用（一行改动，本轮按约束未改 T2-R 目录）。GPU 对照仍按 T2-R §6 路线 1 执行：策略只输出选集，物理协调复用既有严格控制器；本轮未运行任何 GPU、未产生物理操作凭据。

**边界声明**：结果仍称 Marconi Adaptation（evict），不声称完整复现 Marconi——Recency 沿用声明访问次序（非官方 1/(t−ts)）；收益为 Token 代理非 Qwen3.5 GDN FLOPs；未实现 Radix 物理 KV 合并、Admission、Prefetch、Serving Engine。

**至此停止。**
