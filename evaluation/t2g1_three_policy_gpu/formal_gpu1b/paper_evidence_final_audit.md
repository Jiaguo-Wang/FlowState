# T2-G2 最终论文证据审计报告（CPU-only，只读）

日期：2026-10-28。审计对象：`evaluation/t2g1_three_policy_gpu/formal_gpu1b/`（GPU1 独立正式批次）。
方法：不信任既有结论文件——从 63 臂原始运行产物（summary.json / reconciliation.json / device_binding.json / 逐臂命令与清理记录）与冻结输入独立复算，再与已发布产物交叉对照。审计脚本 `paper_evidence_audit.py`，复算明细 `paper_evidence_audit_data.json`（28 项审计检查 + 40 逐对差 + 60 绝对 TTFT + 12 格归因层）。

## 判定：**READY**（28/28 审计检查全 PASS；无任何口径或数据修正）

---

## 1. 批次完整性（A1–A10，全 PASS）

- **63 臂结构**：协议 3 门禁 + 60 主臂，5 轮 × 4 Pending（C_A/C_B/C_C/C_D）× 3 策略全覆盖无重复；启动顺序 == 冻结调度；门禁 3/3、主矩阵 60/60 状态全 PASS，完成标记一致。
- **门禁不进统计（实证）**：统计产物臂域核验——formal_arms_main.csv 60 行、pair_analysis.csv 40 行（20 对 × 2 比较器）、round_equal_weight.csv 10 行，全部仅含 m\* 主臂，零 g\* 混入。
- **无批次混入**：10 项冻结输入在 gpu1b / formal_gpu1 / GPU0 formal 三方逐字节一致（当前时点复核）；GPU0 中断批次与 formal_gpu1 失败批次数据未出现在任何统计产物。
- **冻结与锁**：implementation_lock_main.json 19 文件当前 sha 全一致；c_formal runner/control/bridge 三模块 sha 与 GPU0 批次锁记录相等——**Runtime 恢复语义与测量合同自 GPU0 批次以来未变**。
- **设备与命令**：63 臂 device_binding.json 全部 GPU1（`GPU-051c9abc-…`）唯一绑定；63 条容器命令均 `--gpus device=GPU1UUID`、容器前缀 `flowstate-t2g2-g1b-`、挂载本批次 case 目录。
- **哈希链**：gate_passed.json 四项 sha 引用（验证/证据/协议/复核器）当前全部有效——GPU0 批次的覆盖瑕疵未在本批次重现。

## 2. 实验正确性（B1–B3，全 PASS；独立复算 660 + 819 项）

- **B1 合同独立重算 660 项**（60 主臂 × 11，重读原始文件，非引用验证器结论）：status/census 16=16=0/选集==冻结计划/E-G-H 逐格/prefill==len(pt)−E 且 batch_prefix==E/测量模式 audit==2/形状预热 E 且 FA 命中==E/请求窗 Triton 新增编译==0/Cold 逐 Token 独立重比/13 驱逐全 returned/TTFT 六段闭包<5ms——**全部通过**。
- **B2 原始证据子集 819 项**（63 臂 × 13）：case 目录 9 份共享输入 sha == 批次冻结值（零副本漂移）、进程退出码 0、3 次清理观察（<100MiB、无自有容器）、Engine 正常关闭、设备绑定 PASS。
- **B3 全量证据引用（只读）**：takeover/verify_evidence.py 结论 gate 180 项 + main 2928 项全 PASS（该复核器为独占写设计，不可重跑；其检查含 COW 来源/槽隔离/FA 保护/Native 匹配/实际执行形状/编译审计窗口/生命周期 barrier 唯一性）。

## 3. 统计独立复算（C1–C5，全 PASS；四个关键数字逐位复现）

从 60 臂 summary.json 原始字段重建数据格，完整重算并与已发布产物逐值对照（容差 0.002ms）：

| 口径 | 复算 | 已发布 | 一致性 |
|---|---|---|---|
| 请求级 FS−MC TTFT 差中位 | **−148.932ms**（负15/平0/正5，n=20） | −148.932 | C1 PASS |
| 请求级 FS−KV TTFT 差中位 | **−148.749ms**（负15/平0/正5，n=20） | −148.749 | C1 PASS |
| 轮级等权 FS−MC（四 Pending 先均值后差，五轮中位） | **−114.127ms**（5/5 轮负） | −114.127 | C2 PASS |
| 轮级等权 FS−KV | **−105.728ms**（5/5 轮负） | −105.728 | C2 PASS |

compute / prefill_cuda / control 三指标请求级复算同样逐值一致（C1）；40 逐对差与 pair_analysis.csv 逐值一致（C3）；8 组逐 Pending 分组一致（C4）；60 绝对 TTFT 与 formal_arms_main.csv 逐臂一致（C5）。

**口径声明（论文可直引）**：每臂为单请求 TTFT（六段分解闭合 <5ms）；"请求级"= 同轮同 Pending 配对差（FS−基线）后取 20 个差值的中位数；"轮级等权"= 策略内四 Pending 先算术平均、后配对差、再取 5 轮中位数。n=20 偶数中位为中间两数均值，不另作分布假设。

**三策略绝对 TTFT（中位 ms，五轮 min–max；每轮原始值见 audit data）**：

| Pending | FlowState-Greedy | Marconi-Evict Adaptation | KVFlow-inspired |
|---|---|---|---|
| C_A | 165.3 (163.7–169.0) | 126.0 (124.9–127.1) | **59.0** (36.7–66.2) |
| C_B | **42.2** (35.5–52.7) | 163.9 (162.3–166.3) | 127.8 (126.8–132.5) |
| C_C | **36.3** (34.6–40.0) | 209.3 (208.0–213.3) | 247.0 (244.2–247.9) |
| C_D | **48.6** (34.2–59.3) | 250.5 (250.1–251.7) | 289.8 (284.3–290.5) |

**逐 Pending 配对胜负（TTFT，5 轮符号全一致，无一轮翻转）**：

| Pending | FS−Marconi-Evict | FS−KVFlow | 驱动（E 差） |
|---|---|---|---|
| C_A | +39.2（负0/正5，FS 输） | +104.7（负0/正5，FS 输） | FS E=0 vs MC 1024 / KV 4032 |
| C_B | −124.2（负5/正0） | −84.6（负5/正0） | FS E=5120 vs 1024 / 2048 |
| C_C | −172.3（负5/正0） | −210.2（负5/正0） | FS E=6144 vs 1024 / 0 |
| C_D | −201.9（负5/正0） | −241.8（负5/正0） | FS E=7168 vs 1024 / 0 |

## 4. 性能归因（D1–D4，全 PASS；三层严格分列）

- **结构层 → 实测层单调对应**：12 格均满足 prefill_tokens == T−E（D1）；TTFT 中位对 prefill_tokens 的 OLS 拟合 **TTFT ≈ 39.0 + 0.0330·tok（R²=0.985）**，CUDA Prefill ≈ 31.5 + 0.0327·tok（R²=0.986）（D2）——TTFT 差异由恢复前沿 E 决定的重填充量近乎完全解释，残余散度主要来自 FS 32-token 格的 host 侧与首 token 开销占比变化。
- **TTFT 差 ≈ Prefill CUDA 差（差-差分解，D3）**：TTFT 差 − Prefill CUDA 差的中位仅 **−0.91ms**（FS−MC）/ **−1.04ms**（FS−KV），极差 [−5.5, +4.2]ms——**−149ms 请求级优势即 GPU 重填充计算差本身**，无其他机制贡献。
- **控制开销三层（D4）**：① 生命周期外带 apply RPC 墙钟中位 FS 11499ms / MC 11061 / KV 11123（秒级、测量窗之外，差值在对内 ±1.1–1.5s 波动内）；② 请求窗内控制近零——60 臂 TTFT 六段闭包残差 <5ms 全成立；③ 恢复在线成本亚毫秒级：COW 拷贝中位 0.21–0.52ms、Native 前缀匹配中位 0.59–0.81ms（三策略同量级）。**控制开销不折损 TTFT 净收益。**
- **模型层参照**：J（当前域恢复成本）FS 154.35 < KVFlow-inspired 627.61 < Marconi-Evict Adaptation 703.46 ms，排序与实测等权 TTFT 一致（不与实测混算，仅作结构声明）。

## 5. 新批次执行身份（E1–E5，全 PASS）

- launch_formal.py 相对 formal_gpu1 **恰两处**替换（输出目录、容器前缀），逆替换逐字节还原（E1）；verify_formal.py / analyze_formal.py 各**恰一处**目录替换（E2）；takeover/verify_evidence.py **逐字节相同**（E3）。
- device_entry.py 修正仅加 `if __name__ == '__main__'` 保护：AST 结构检查顶层无可执行语句（docstring 豁免与 test_spawn_reentry 安全形态一致）、保护存在；核心编排三要素（GPU1 UUID 校验、`open('x')` 独占写证据、runpy 接力同一冻结 Runner `evaluation.c_formal_perf_gpu.runner`）与旧版逐项相同（E4）；spawn 重归回 PASS 在案且含旧入口 RED 对照（A10）。
- 冻结计划源未变：precheck 记录的 t2g1_plans.json / policy_plans.json sha == 当前值（E5）。
- 结论：**修复 multiprocessing 入口重入问题未改变冻结策略、Runtime 恢复语义与测量合同**——formal_gpu1 失败为编排缺陷（spawn 子进程重入入口重复独占写 device_binding.json → FileExistsError → scheduler 初始化死亡），与策略/引擎/测量无关，该批次 0 有效臂已封存。

## 6. 论文结论（可直接写入 ICDE Evaluation）

基线名称规范使用 **Marconi-Evict Adaptation** 与 **KVFlow-inspired**（均为策略适配，非原论文完整系统）。

1. **总体优势**：FlowState-Greedy 对两基线的请求级配对 TTFT 差中位 −148.9ms（vs Marconi-Evict Adaptation）与 −148.7ms（vs KVFlow-inspired）（n=20，负 15/正 5）；四 Pending 等权轮级差五轮全负，中位 −114.1 / −105.7ms。绝对 TTFT：FlowState 恢复命中 Pending 为 36–49ms，两基线对应 128–290ms。
2. **优势机制**：完全由恢复前沿覆盖解释——FlowState 冻结选集与 C_B/C_C/C_D 前沿精确重合（E=T，重填充仅 32 token）；TTFT ≈ 39.0 + 0.0330·(T−E) ms（R²=0.985）；TTFT 差 − Prefill CUDA 差中位 |Δ|<1.1ms。COW 与 Native 匹配的在线恢复成本 <1ms，可忽略。
3. **负向结果如实保留（C_A）**：FlowState 选集与 C_A 前沿零交（E=0 → 4128 token 全量重填充），对 Marconi-Evict Adaptation +39.2ms、对 KVFlow-inspired +104.7ms，五轮符号一致（0/5 胜）。KVFlow-inspired 在 C_A 恢复 4032/4096 token 而取得全场最低 TTFT（59.0ms 中位）。这是当前域冻结选集在"当前成本 vs 未来覆盖"上的已声明取舍，非实现缺陷。
4. **控制开销不影响净收益**：计划应用（约 11s，三策略同量级）为生命周期外带；请求窗内控制残差 <5ms；恢复在线成本（COW+匹配）<1ms。
5. **有效性边界**：5 轮小样本，报告为中位数与符号计数，不作显著性声明；decode 双峰（>160ms 标记 12/60 臂）为 c_formal 已知测量现象，不影响首 token 处截断的 TTFT；三策略为同一 Engine/协议下的计划替换对照。

## 7. 数据来源与复算口径索引

| 结论 | 数据来源 | 复算口径 |
|---|---|---|
| −148.9 / −148.7 ms | 60 臂 run/summary.json ttft_ms | 同轮同 Pending 配对差，20 值取中位 |
| −114.1 / −105.7 ms | 同上 | 策略内四 Pending 均值 → 配对差 → 5 轮中位 |
| 绝对 TTFT 表 | 同上 + formal_arms_main.csv | 每格 5 轮中位 |
| E/G/Prefill | summary.json 与冻结计划/输入 | 逐臂恒等核验（660 项合同） |
| 拟合与差-差 | summary.json cuda_prefill_ms / ttft_ms | 12 格 OLS；40 对差-差 |
| 控制三层 | control_rpc_wall_ms / control_timing_stages / timing_decomposition | 中位 + 闭包残差 |

**最终判定：READY——formal_gpu1b 批次的正确性、配对统计与论文结论均通过独立复算核验，可冻结为 ICDE Evaluation 正式结果。**
