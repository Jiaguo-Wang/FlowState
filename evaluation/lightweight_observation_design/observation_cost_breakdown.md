# Runtime 观察路径成本分解（只读审计）

## 0. 证据边界声明（先行）

1. 本文件所有毫秒数均引自已有原始计时产物：B5-D2 `shadow_timing.csv`（经 final_report.md 表格复核）与 B5-D3 `control_overhead_breakdown.csv`。它们是**各自单次实验调用的实测值**，不是 P95、不是重复统计、不是线上延迟，不能直接当作生产控制开销。
2. 观察路径内部各子步（GPU 同步、单次树遍历、单次哈希、序列化、传输、解码）**没有独立计时**。本文对这些子步只给出代码结构事实（调用次数、数据量级），一律标注"未单独计量"；不提供任何拆分后的估计毫秒数，不把估计成本当实测数据。
3. B5-D2 与 B5-D3 是不同负载、不同 Engine 生命周期、不同代码路径的两轮实验，两表数字不可互相加减得出"净收益"。
4. 本轮只读审计：不运行 GPU、不修改任何生产或实验代码。

## 1. 已有实测计时总表

### 1.1 B5-D2 Shadow Planning 单次分段（表源：`exact_runtime_shadow_gate_b5d2/final_report.md` §分段耗时，原始 CSV `shadow_timing.csv`）

| 阶段 | 归属 | 实测 ms | 计时语义（代码依据） |
|---|---|---:|---|
| runtime_observation_freeze_rpc | 共同 | 1102.325 | 客户端 `perf_counter` 包住一次 `client._call({'action':'b5c3_observe','final':True})` 往返（`exact_runtime_shadow_gate_b5d2/runner.py:49-51`），含入队等待 Scheduler idle、Scheduler 侧全部观察工作、JSON 序列化、本机 TCP 传输、客户端解码 |
| epoch_construction_authentication | 共同 | 110.831 | 客户端 CPU：`blob(strict_epoch)`+`blob(全部物理观察)` 序列化与 SHA256、逐候选来源检索（读 `history_XX.json`）、`from_frozen`+`prepare`（含 `StrictAllocationEpoch.validate()` 全量验证与内容摘要重算）、快照写出（`exact_runtime_shadow_gate_b5d2/shadow.py:10-21`） |
| benefit_matrix | 共同 | 0.041 | `planner.benefits` 一次 |
| solve Greedy / Exact | 各自 | 0.187 / 0.225 | `planner.solve` |
| plan_construction_validation | Greedy / Exact | 13.553 / 13.747 | `planner.finish`→`validate_plan`（含 `epoch.validate()` 全量重验与 `frontiers` 重算，`flowstate/strict_control.py:94-109`） |
| independent_oracle | 审计 | 0.404 | Decimal 独立穷举 |

D2 关键结构事实：观察 RPC（重型）与 CPU 规划是分离的——**CPU 规划在前观察 RPC 返回后、于 Scheduler 控制窗口之外执行**（shadow.py 全程不访问 Runtime）。重型成本集中在观察 RPC 与 Epoch 认证两段。

### 1.2 B5-D3 真实 Reconciliation 全量分段（表源：`exact_reconciliation_correctness_gate_b5d3/control_overhead_breakdown.csv`，category 列为该轮自证归属）

| 阶段 | 实测 ms | category（原文） |
|---|---:|---|
| heavy_observation_gpu_sync | 1203.035102 | 重型实验审计 |
| strict_epoch_authentication | 347.979142 | 必要规划或安全控制与当前审计混合，未验证轻量生产成本 |
| exact_solve_plan_validation | 8.874514 | 必要规划或安全控制与当前审计混合，未验证轻量生产成本 |
| independent_oracle | 0.257195 | 重型实验审计 |
| apply_time_freshness_preflight | 551.672320 | 必要规划或安全控制与当前审计混合，未验证轻量生产成本 |
| fa_content_audit_before | 153.653532 | 重型实验审计 |
| strict_reconciliation_with_postchecks | 170.587878 | 必要规划或安全控制与当前审计混合，未验证轻量生产成本 |
| fa_content_postcheck_and_resume_baseline | 288.040879 | 重型实验审计 |
| native_run_batch_0_1 … 7_2 | 351.45 / 6659.46 / 73.56 / 47.45 / 47.22 / 47.12 / 48.27 / 59.48 | 真实请求宿主计时，含审计或异步执行边界；非纯GPU核时间 |
| strict_request_wall_with_observation | 8307.721369 | 真实请求宿主计时，含审计或异步执行边界；非纯GPU核时间 |
| adapter_eviction_0…4（5 次） | 1.380 / 1.040 / 1.045 / 1.025 / 1.007 | 必要规划或安全控制与当前审计混合，未验证轻量生产成本 |

## 2. 观察路径的代码级分解（结构事实，非独立计时）

入口链：`ControlClient._call`（`motivation/artifacts/wp3b_gate_20260820/targeted_probe.py:569-586`，TCP+逐行 JSON）→ `ProbeState.submit`/`on_idle` 单命令排空（targeted_probe.py:482-522，命令在 **Scheduler 线程**执行）→ `observe_control`（`evaluation/multishare_runtime_gate_b5c3/bridge.py:25-91`）。

`observe_control(final=True)` 内部（代码结构，子步均未单独计量）：

| # | 子步 | 代码位置 | GPU 同步 | 每次观察执行次数 |
|---|---|---|---|---|
| 1 | 安全范围检查（idle/无插入/无传输/tp=1 等） | `runtime_control_closure_r2c0/bridge.py:41-58` + `targeted_probe.py:277-311` | 否（纯属性读取） | 入口 1 次 + 每次 `observe` 内 1 次 |
| 2 | 显式设备同步 | `multishare_runtime_gate_b5c3/bridge.py:39` `torch.cuda.synchronize()` | **是** | 1 次 |
| 3 | 全树遍历 + **逐节点 FA 张量整体内容哈希** | `_global_maps`（targeted_probe.py:106-140）：每个 FA 节点 `_tensor_sha256(fcd.value)` 即 GPU→CPU 拷贝 + 全内容 SHA256 | **是**（隐式，`.cpu()`） | 每次 `state_snapshot` 内 2 次（前后非干涉比对，`strict_compatibility_b1c/runner.py:82,102`） |
| 4 | 逐节点 census：父链重建全 Token、`token_digest`、引用读取 | b1c/runner.py:85-98 | 否（`tensor_ids` 小型 D2H） | 每次 `observe` 1 轮 |
| 5 | 逐槽 Recurrent 组件**内容哈希**（conv+temporal 每槽 GPU 拷贝+SHA256） | `payload`（b5c3/bridge.py:15-23） | **是**（`t[:,slot].contiguous().cpu()`） | final 模式 2 次（before/after） |
| 6 | 组件切片身份（指针区间摘要） | `slice_identity`（r2c0/bridge.py:27-35） | 否 | 每个驻留槽 × 每张量 |
| 7 | 逐 Pending FA 前沿 radix 游走（无内容哈希） | `inspect_resident_fa_frontier`（`barrier_fa_frontier_control.py:96-128`） | 否 | 每个 Pending 1 次 |
| 8 | 第二次完整 `observe`（after）+ 深度相等比对 | b5c3/bridge.py:60-63 | 同 3/5 | 1 次（即再触发 2 次全树哈希） |
| 9 | 巨型结果 JSON 序列化→TCP→客户端解码 | targeted_probe.py:543-562, 574-586 | 否 | 1 次（结果含 before/after 视图、两份审计、两份逐槽哈希、strict_epoch 全量 token；D3 落盘同名 census ≈1.1–1.6 MB/份，可作数据量级参照） |

由此可得的结构事实（可复核的乘数，不是耗时拆分）：

- 一次 `observe_control(final=True)` = **2 次完整 observe = 4 次全树 FA 内容哈希 + 2 轮逐槽 Recurrent 内容哈希**。
- `reconcile_strict`（`flowstate/strict_control.py:204-221`）中 `runtime.observe()` 调用数 = 2（bind 前 + preflight）+ 2×驱逐数（每项驱逐前后 postcheck 各 1 次）+ 1（final）。B5-D3 为 5 项驱逐 → **13 次完整 observe ≈ 26 次全树哈希**（D4a/D3 包装器还会每次 `asdict` 追加审计记录，`exact_reconciliation_correctness_gate_b5d3/control.py:111-112`）。
- `apply_time_freshness_preflight`（D3 control.py:94-102）含：又一次完整 `prior.observe`（2 次全树哈希）+ 又一轮 `capture.payload` 逐槽哈希 + 全 Pending FA 前沿复核 + `bind`+`preflight`（`preflight` 内部再次 `bind` 全量重验，strict_control.py:155-163）+ 大字典深度相等比对。
- `fa_content_audit_before/postcheck`（D3 control.py:16-53, 103, 120-127）：`fa_map` 全树 Token 重建+运行摘要（CPU）+ `fa_payload` 对全部驻留 FA 槽按 8 个 FA 层 × K/V 做 `index_select`+D2H+全内容 SHA256；postcheck 段再完整重复一次并另加受保护子集哈希。
- `strict_epoch_authentication`（D3 347.979 ms / D2 110.831 ms，口径不同：D3 含 Journal 逐阶段写盘）：对**整份物理观察 JSON**（而非仅 strict_epoch）再序列化+SHA256 两次（`encode(raw)`、`encode(initial)`，D3 control.py:59, 67），再走 `from_frozen` 全量验证。

## 3. 归属结论（对应 D3 category 列的代码依据）

- **重型实验审计**（可从决策路径剥离而不削弱安全语义的）：全树 FA 内容哈希（§2.3）、逐槽 Recurrent 内容哈希（§2.5）、FA K/V 内容审计（fa_content_*）、before/after 双观察与深度相等比对（§2.8）、独立 Oracle、Journal 逐阶段写盘、census 巨型 JSON 传输。这些验证的是"观察本身无副作用/内容未变"这一实验命题，不是分配决策的输入。
- **必要安全与规划控制**（不可删除，只可精简重复）：scope/静止性检查、槽位记账闭合与别名检查（`validate_view`，strict_control.py:112-138）、候选唯一绑定与 Keep/Evict 不相交（bind/preflight）、每次驱逐前后结构 postcheck、Adapter 逐项驱逐的安全重验（实测 1.0–1.4 ms/次）、`from_frozen`/`validate_plan` 身份验证。其中"内容未变"类检查当前以内容哈希实现，是可替换的实现选择而非语义本体（见 minimal_state_view_contract.md §新鲜度）。
- **规划本体**（CPU，量级毫秒以内至十几毫秒）：benefit_matrix、solve、finish/validate_plan；已在 Scheduler 控制窗口之外（D2 结构事实）。

## 4. 尚不存在的数据（明确缺失，不用估计填充）

- 观察路径内 GPU 同步本身、单次全树哈希、单轮逐槽哈希、JSON 编解码、TCP 传输各自的独立耗时：**未计量**。
- 移除重型审计后的轻量观察耗时、重复多次的分布、P95、端到端 TTFT 影响：**不存在任何实测**。本轮不通过相减或抽样外推生成这类数字。
