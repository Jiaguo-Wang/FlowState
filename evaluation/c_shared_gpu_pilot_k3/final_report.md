# C Shared/Branched K=3 真实 GPU Pilot — 最终报告

- **结论：READY（17/17 臂全部 PASS，零正确性失败）**
- 协议版本 v1.4（修订记录见 `protocol.json` amendments）；执行窗口 2026-10-21 21:17 → 2026-10-22（deadline 经 v1.4 延长，两次迭代均为检查合同/接线修正，被测系统零正确性失败）
- 固定条件：冻结 C Shared/Branched 工作负载（16 预热请求 = 8 前缀 × 2 轮；4 个 ready Pending C_A/C_B/C_C/C_D；C_JOIN 不执行）；K=3；Qwen3.5-9B；SGLang v0.5.17；H100 PCIe TP=1；策略 FlowState-Greedy / StandaloneTopK / WFC；未修改生产 FlowState、Phi、策略实现、冻结 Token 与拓扑

## 0. 执行与修正记录

| 尝试 | 结果 | 原因 | 处置 |
|---|---|---|---|
| attempt1 | strict_FlowState-Greedy_C_B BLOCKED | 我方断言"候选必须物化于 round0"过严；逐历史证据证实引擎实际时间线为末端(L)@round0、祖先(L−64)@round1 重复插入，均在冻结预热内、决策前 | v1.2：改为"物化于声明的 round0 或 round1"并记录实测轮次；证据保留于 `*.attempt1*` |
| attempt2 | 同臂 RPC 信封崩溃 `KeyError('raw_epoch')` | apply 全链路 ok=True（协调/驱逐/FA 不变性全过），e1 bridge 成功路径要求 `result['raw_epoch']` 键，pilot control 未设置 | v1.3：补键；证据保留于 `*.attempt2*` |
| attempt3 | 17/17 PASS | — | launch3 全序完成 |

## 1. 交付 1：Census + 物理预算账本

`census/run/summary.json`、`finalize_results.json:census_ledger`：

- **物理候选域 = CPU 16 候选，1:1 无残差**：physical=16，cpu=16，mapped=16，unmapped_extras=0，missing_physical=0。映射键为（预热水截断前缀逐 Token 相等, L），8 owners × {L, L−64} 全部唯一命中
- **物化轮次实测**（12 strict 臂一致）：8 个 endpoint(L) 物化于 round0；8 个 ancestor(L−64) 物化于 round1 分叉重插——全部属于冻结 16 请求预热、先于 Allocation Epoch，不利用任何未来请求（v1.2 合同）
- **槽位记账闭合**：mamba 槽 16 used + 12 free = 28 pool capacity；K=3 指真实 Recurrent 槽预算
- **四 Pending 活观察 T / FA 前沿 == CPU 冻结声明**（4/4 match）

## 2. 交付 2：三策略实际保留 / 驱逐集合

`finalize_results.json:evictions`（12 臂全量）。以 C_B 为例（物理域 16 → 每策略真实保留 3、真实驱逐 13，操作全部返回，无任何免费保留祖先，无未建模额外状态）：

| 策略 | 保留（CPU id） | 真实驱逐 |
|---|---|---|
| FlowState-Greedy | C_D_P07168, C_C_P06144, C_B_P05120 | 其余 13 个（含 C_A/C_AB/C_CD/C_ROOT/C_JOIN 全部状态） |
| StandaloneTopK | C_D_P07168, C_D_P07104, C_C_P06144 | 其余 13 个（**含 C_B 自身前缀 C_B_P05120**） |
| WFC | C_ROOT_P01024, C_CD_P03072, C_AB_P02048 | 其余 13 个（含 C_B/C_C/C_D 全部深末端） |

每臂驱逐数恒为 13；`operations_all_returned` 12/12 True；unmapped_extras 驱逐清单为空（域内无额外物理状态）。

## 3. 交付 3：逐 Pending 真实 H/E/G、Prefill Token、CUDA Prefill、TTFT

`pending_policy_matrix.csv`（全部 12 格 E/T/G 与 CPU 冻结合同逐格相等——合同认证，非推断；H=T=FA 前沿全部成立，见 §8.C。T 为规划目标=FA 前沿，输入全长=T+32）：

| Pending | 策略 | E | G | H=T | Prefill Token | CUDA Prefill | TTFT |
|---|---|---|---|---|---|---|---|
| C_A | FS-Greedy | 0 | 4096 | 4096 | 4128 | 149.7ms | 1904ms |
| C_A | Standalone | 0 | 4096 | 4096 | 4128 | 150.9ms | 2147ms |
| C_A | WFC | 2048 | 2048 | 4096 | 2080 | 82.1ms | 3085ms |
| C_B | FS-Greedy | **5120** | 0 | 5120 | **32** | 402.9ms | 3688ms |
| C_B | Standalone | 0 | 5120 | 5120 | 5152 | 192.8ms | 2105ms |
| C_B | WFC | 2048 | 3072 | 5120 | 3104 | 121.3ms | 3096ms |
| C_C | FS-Greedy | **6144** | 0 | 6144 | **32** | 373.7ms | 3595ms |
| C_C | Standalone | **6144** | 0 | 6144 | **32** | 397.2ms | 3647ms |
| C_C | WFC | 3072 | 3072 | 6144 | 3104 | 119.8ms | 3038ms |
| C_D | FS-Greedy | **7168** | 0 | 7168 | **32** | 395.0ms | 3626ms |
| C_D | Standalone | **7168** | 0 | 7168 | **32** | 393.6ms | 3584ms |
| C_D | WFC | 3072 | 4096 | 7168 | 4128 | 154.0ms | 2404ms |

Cold 对照臂（空缓存全量 Prefill）：C_B 5152→CUDA 23937ms/TTFT 25315ms；C_A 4128→23125/24544；C_C 6176→23623/25068；C_D 7200→23902/25253。

## 4. 交付 4：FA-KV / Native COW / Cold 正确性

- **FA-KV 保全**：12/12 臂协调前后 FA map 索引与内容不变（目标请求 E 之后区间的合法重算除外，保护集逐一哈希核验）
- **Native COW**：目标恢复源节点 COW 复制真实发生并逐块记录（`cow_copy_count`/`cow_copy_cuda_ms` 入 summary），源 payload 协调前后逐字节不变
- **Cold 输出一致性**：**12/12 逐 Token 完全一致**（4 Pending × 3 策略 vs 同 Pending 空缓存生命周期；`finalize_results.json:cold_comparison` 全 PASS）——严酷驱逐+恢复不改变模型输出
- 请求级合同：`fa_hit==frontier`、`executable_prefix==E`、`batch_prefix_length==E`、`extend_token_count==len−E`、输入 digest 一致、选中组件内容不变、驱逐操作全数返回

## 5. 交付 5：控制 Wall Time 与执行窗口时间

`control_overhead.csv`（RPC 墙钟 + 分段）、各臂 `summary.json:epoch_wall_ms`：

- 控制 RPC 墙钟 **10.6–12.0s/臂**，分段：final_heavy_observation 2.38–2.98s ＞ fa_content_postcheck 1.46–1.87s ＞ apply_time_freshness_preflight 1.22–1.25s ＞ fa_content_audit_before 0.92–0.95s ＞ strict_reconciliation(含后检) 0.86–0.89s ＞ 认证+计划 **0.15–0.17s** ＞ cpu_physical_mapping ~1ms
- 该墙钟为**审计增强型研究仪表**（重型观察 + 全量 payload 哈希审计约占 80%）；生产控制核心（认证+协调）<1.1s
- 完整 Epoch Wall（预热+控制+请求全窗口）53.7–59.9s，逐臂记录；请求级 warm 引擎 TTFT 1.9–3.7s、request wall 10.3–12.5s

## 6. 交付 6：原始日志 / CSV / 配置 / 清理记录

- 臂产物：`<arm>/run/`（summary / reconciliation / census-evidence / initial / history_01–16 / request / stream_chunks / output / post_request / engine_cleanup / native_trace / rpc_*）
- 宿主侧：`<arm>.runtime.log`、`_command`、`_process`、`_gpu_before`、`_cleanup.json`（每臂容器 rm + 3×GPU<100MiB 验证全 PASS）；`implementation_lock(2/3).json`、`source_identity.json`（容器内外源码哈希核验）
- 汇总：`arm_results.csv`、`pending_policy_matrix.csv`、`control_overhead.csv`、`finalize_results.json`
- 失败证据：`*.attempt1*` / `*.attempt2*`（目录与文件原样保留）

## 7. 交付 7：READY / PARTIAL / BLOCKED

**READY** — 17/17 臂 PASS（census 1 + strict 12 + cold 4），失败清单空，无未测臂。

## 8. 问题 A–E

**A. 三策略相同物理预算下合法且不同恢复前沿？——是。**
同一物理合同（16 物化状态、28 槽、K=3），三策略保留集互不相同（§2），经完整合同链（from_frozen/prepare/finish + 活观察等价 + 13 项真实驱逐全落实）认证。恢复前沿实质分化：C_B 上 E = 5120 / 0 / 2048；C_C、C_D 上 FS 与 Standalone 拉满（6144/7168）而 WFC 仅 3072；C_A 三者皆 0（无兼容共享保留）。分化方向与 CPU 复核预测逐格一致。

**B. FS 是否实际减少恢复 Token？——是（合同认证）。**
C_B：FS 32 Token vs Standalone 5152 vs WFC 3104；C_D：FS 32 vs WFC 4128。这是引擎实测 `executable_prefix==E`、`extend_token_count==len−E` 的直接读数，不是推断。FS 的跨工作负载共享推理使其在 branched 场景选到当前 Pending 最深兼容前缀；StandaloneTopK 逐工作负载 Top-K 在 C_B 上把Pending 自身前缀挤出 K=3（保留 D 的两条），恢复前沿塌缩到 0。

**C. 是否实际降低 Prefill/TTFT？——对 Cold 差距巨大但混杂编译因子；策略间差异本轮未测出（噪声主导），不做无证据推断。**
- 严酷驱逐+恢复 vs Cold：TTFT 25.3s → 2.1s（C_B Standalone 全量 mamba 重放路径），12×；CUDA Prefill 23.9s → 0.08–0.40s。但 cold 臂为全新空缓存生命周期（warmup=0），其 23–24s 与长度几乎无关（4128→23.1s，7200→23.9s），说明含大量首 extend 固定成本（编译/初始化）——**该对比是 FA 命中+已预热 双重效应的上界，不得拆归因于单一机制**
- 所有 strict 臂 H=T（本轮驱逐只动 mamba 槽，FA 全保）：策略间差异只剩 mamba 重放 G。实测 G=5120 的全量重放仅 ~193ms（Standalone C_B），与 G=0 的 32-Token prefill 403ms（FS C_B）出现倒置——单发新引擎首 extend 噪声主导 82–403ms 区间，策略间计时差异本轮不可分辨
- Token 级减少（B）确定；计时级收益需 §9 修正后的正式配对实验

**D. 控制开销是否抵消请求收益？——本轮无法判定净收益，且须区分仪表与生产路径。**
审计增强型控制 RPC 10.6–12.0s > 请求级收益差（~1–2s 量级）；但其中重型观察+内容审计 ~80% 为研究仪表，生产核心（认证+协调）<1.1s。完整 Epoch Wall 已记录（53.7–59.9s）但为单发生命周期，不构成统计样本——按协议不从未完成/单发窗口推断 E2E 净结论。

**E. 是否可进入正式多次配对 GPU 实验？——可以，带 4 项设计修正。**
1. **多次配对生命周期**（≥N 次重复 + 对内随机化）压过单发首 extend 噪声（本轮实测倒置即证据）
2. **H 与 E 分开报告**：本轮证明 mamba-only 驱逐下 FA 全保（H=T 恒成立），策略杠杆在 E（mamba 可执行前缀）；正式实验如需隔离 attention 重算成本，须增设预热引擎 + FA 驱逐臂（本轮未测，cold 对比被编译混杂）
3. **计时仪表分层**：生产控制路径（<1.1s）与研究审计（~10s）分开计时报告，避免 D 类结论被仪表污染
4. Cold/基线臂须与处理臂同引擎位置（同样预热后清缓存），消除首 extend 固定成本混杂

## 9. 产物索引

代码：`prepare.py`（输入冻结）、`control.py`（物化映射+严格协调）、`bridge.py`、`runner.py`（三模式）、`launch(2/3).py`、`finalize.py`；合同：`protocol.json` v1.4、`policy_plans.json`、`cpu_gate.json`、`prepared_inputs.json`、`cpu_candidate_prefix_mapping.json`。
