# ICDE 投稿前剩余证据定向审计报告

日期：2026-10-31 ｜ 性质：只读审计（零 GPU、零代码修改、零历史数据修改、未重跑任何历史实验）
原则：本任务以证据闭环为目标，不以获得 READY 为目标。所有数字均为本审计从原始文件独立重算或逐位核对的结果，不引用历史汇总报告作为证据（历史报告仅用于定位目标）。

---

## 1. 审计范围与版本锁定

### 1.1 审计对象（按任务 spec）

| 任务 | 优先级 | 对象 | 版本锁定结果 |
|---|---|---|---|
| Task 1 | P0 | VI-B 五轮 GPU 主实验（p0_control_cycle_gpu）TTFT 计时边界与控制开销归因 | 15 臂 m{1..5}_ck3_{fs,mc,kv}，入口 runner.py SHA `b6cee2d3…`、control.py SHA `b1c3a64a…`、统计脚本 five_round_stats.py SHA `c7597587…`；镜像 `lmsysorg/sglang:v0.5.17-cu129-runtime` sha256 `1dfce903…`；ledger 断言 15 PASS + 身份哈希一致 |
| Task 2.1 | P1 | VI-F OpenHands 24 快照实验 | `rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/`，协议 FROZEN（seed 20260907），source_integrity 7 文件前后 SHA 一致，216/216 run 完成 |
| Task 2.2 | P1 | VI-F Kimi-Linear-48B-A3B-Instruct | `rq5b_kimi_final_output/rq5b_frozen_20260923_090154/`（冻结）+ Phase1/2/3 原始目录；镜像同上 sha `1dfce903…` |
| Task 3 | P1 | AgentX Formal/Raw、VI-D 764 条件与 Oracle 统计 | `agentx_runtime_final_audit_20260906_194000/`（B2.3 差异审计）、`rq3_agentx_policy_eval_20260906_214956/`（Formal 策略评估）、`evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/`（K1–4 预算敏感性，PASS） |

### 1.2 版本锁定方法

- 五轮矩阵：runner.py / control.py / five_round_stats.py 当前工作区 SHA 与 ledger 身份断言一致（five_round_stats.py 装载时强制校验 15 臂 PASS 与实现身份哈希）；无历史版本混入（arms/ 下另有 `discarded_assembly_*` 废弃目录，已排除，未参与统计）。
- OpenHands 24 快照：frozen_protocol.json 状态 FROZEN；source_integrity.json 记录 7 个执行前后源文件 SHA（含当时 `flowstate/optimizer.py` = `eabf84c7…`，为 hybrid 集成前版本——该实验 FS 臂使用的是当时 Marginal Greedy 入口，与当前生产 `plan_strict`（`7dfa9aec…`）版本不同，引用时必须标注历史版本身份）。
- Kimi：冻结报告 frozen_manifest.json + sha256；Phase3 逐条件 record.json / engine_args.json 原始在档。
- AgentX/764：B2.3 审计（2026-09-06）、策略评估（2026-09-06）、预算敏感性（2026-10-30，裁决 PASS）各自带 source_manifest 与前后 SHA。

---

## 2. Task 1（P0）：VI-B 五轮主实验 TTFT 计时边界与控制开销归因

### 2.1 TTFT 计时边界（从原始代码逐行确认，非报告推断）

- **起点**：`evaluation/p0_control_cycle_gpu/runner.py:247` `start_ns = time.monotonic_ns()` 紧贴 `engine.generate(...)` 调用之前；**终点**：流式返回中首个携带输出内容的 chunk（runner.py:257-259）；`ttft_ms = first_token_ns − start_ns`（runner.py:336）。同进程 CLOCK_MONOTONIC，客户端侧计时。
- **TTFT 窗口内包含**：请求排队 + Prefill（含恢复重算 G 的计算）+ 首个 decode 步 + 交付。
- **TTFT 窗口外（关键结论）**：每臂执行顺序为 引擎启动 → 初始观察（空断言）→ 预热重放 1 → flush → 预热重放 2（重建候选）→ 单次 b5d4_apply RPC（含 T0–T3 全部控制阶段）→ T0–T3 整数纳秒恒等式校验 → 形状预热块（v1.1，4 Pending 种子，不进 control_cycle、在请求之前）→ 4 个串行流式请求 → T4。**观察、选择、协调、日志序列化、形状预热全部发生在请求起点之前，不在任何 TTFT 内。**
- **六段分解闭环**：runner.py 对每请求记录 queue/pregap/execwin/postgap/delivery 五段 + compute_cuda/first_decode_cuda，分解求和与 ttft_ms 闭环断言（<5ms）。m1_ck3_fs 的 C_A 实测：queue 4.78 + pregap 2.81 + execwin 155.24（compute_cuda 152.14，first_decode 24.3）+ postgap 0.19 + delivery 2.52 = 162.55 ms。

### 2.2 控制开销指标是否具备独立原始计时（全部具备，同一执行过程）

control.py apply() 内 `timed()` 装饰同时写入 `clock['stages']`（时钟阶段，T0=首阶段起点）与 `result['timing']`。以 m1_ck3_fs 原始 summary/reconciliation 为例：

| 指标 | 原始记录位置 | 实测（m1 FS） | 计时性质 |
|---|---|---|---|
| Observation（最终重观察） | clock stage `final_heavy_observation` | 3383.3 ms | 单次批量（T0 起） |
| 物理映射（census 16=16） | `cpu_physical_mapping` | 0.7 ms | 单次 |
| Strict Epoch 认证 | `strict_epoch_authentication` | 171.0 ms | 单次 |
| Selection（候选构建/收益/求解） | `policy_online_select`（整段）+ `selection.selection_call_ms`（纯求解调用，meta 内） | 35.0 / 17.9 ms | 两级：整段含候选构建；selection_call 仅 plan_strict 调用（单次在线） |
| 新鲜度预检（observe+bind+preflight） | `apply_time_freshness_preflight` | 1645.7 ms | 单次 |
| 完成后审计与基线恢复 | `fa_content_postcheck_and_resume_baseline`（T3=终点） | 1875.1 ms | 单次 |
| Reconciliation（物理保留/驱逐/完成验证） | `result['timing'].strict_reconciliation_with_postchecks`（六子阶段 + 总时长） | 907.7 ms（另有 fa_content_audit_before 1283.5） | **有独立原始计时**，位于 T0–T3 窗口内阶段间 |
| Control Cycle | `cycle.t3_ns − t0_ns`（整数纳秒） | 10448.181 ms | T0→T3，仅控制 |
| Full Cycle（含预热） | `cycle.t4_ns − t0_ns` | 25023.406 ms | T0→T4（含形状预热+请求），**直接计时，非相加推导** |
| 请求执行总时长 | `cycle.request_exec_total_ms` | 848.731 ms | 4 请求串行窗口 |
| 日志 IO | `journal_io.in_window_serialize_ms` / `after_window_flush_ms` | 窗口内序列化计入间隙；磁盘 fsync 在 T3 后 | DeferredJournal 设计 |

口径区分：求解器"0.22 ms 中位"等出自 CPU 微基准（paper_solver_tradeoff，独立实验），与本五轮在线 `selection_call_ms`（FS 中位 16.38 / MC 2.05 / KV 4.40 ms，15 臂重算）不是同一计时，不得混同。

### 2.3 15 臂三口径重算（本审计第一手，全部来自各臂 summary.json）

| 口径 | FS | MC | KV | 配对差中位 FS−MC | 配对差中位 FS−KV |
|---|---|---|---|---|---|
| control_cycle_ms（T3−T0）中位 | 10138.9 | 8123.1 | 9596.0 | **+2015.8**（逐轮全正 +841~+2359） | **+436.9**（逐轮全正 +148~+2288） |
| cycle_ms_full_incl_warm（T4−T0）中位 | 24107.2 | 21379.7 | 22840.3 | **+1704.0**（含 +3126/+3648 两大正轮） | **+781.3**（符号混合） |
| request_exec_total_ms 中位 | 784.0 | 1240.9 | 1191.9 | −456.9（五轮全负） | −445.4（五轮全负） |
| ΣTTFT（4 请求和）中位 | 261.9 | 732.2 | 684.4 | **−470.208** | **−429.711** |

ΣTTFT 逐轮配对差：FS−MC = [−457.4, −468.9, −471.9, −470.2, −481.2]，FS−KV = [−434.9, −437.2, −417.8, −422.5, −429.7]；中位 **−470.208 / −429.711**，与论文数字逐位一致（五轮全负）。

### 2.4 Task 1 四问回答

1. **VI-B TTFT 是否包含控制开销？** 不包含。全部控制阶段（观察/选择/协调/审计/预热）在请求 start_ns 之前完成；TTFT 仅请求窗口（排队+Prefill+首 decode+交付）。
2. **470.208 / 429.711 代表什么？** 五轮实验中，同一轮内 FS 臂与 MC/KV 臂的 **ΣTTFT（4 个 Pending 请求 TTFT 之和）配对差**，再对 5 轮取中位。是**请求窗口恢复收益**的度量。
3. **支持请求恢复改善还是完整控制周期净收益？** 仅支持前者。控制周期配对差五轮全正（FS 控制更贵：FS−MC 中位 +2015.8 ms、FS−KV +436.9 ms），含预热全周期中位亦为正（+1704.0 / +781.3 ms）。**本实验不支持"完整控制周期含入后净收益为正"的系统级结论**；若论文如此表述即超出证据。
4. **VI-B 应如何描述计时边界？** 见 §10 修改建议（明确"TTFT=控制完成后的请求窗口"、控制开销独立成列、不将 −470/−430 表述为净收益、注明 n=5 配对中位与无显著性检验）。

**Task 1 状态：VERIFIED**（版本锁定、代码级计时边界、全部控制指标独立原始计时、四问全部有证据回答）。

---

## 3. Task 2.1：VI-F OpenHands 24 快照实验证据链

### 3.1 完整来源链（原始快照 → 配置 → 执行 → 原始日志 → 汇总）

- **原始快照**：源种群 `rq3_openhands_main_formal_20260904_001017`（168 快照）；frozen_protocol 记录选择规则（round 固定配额 2/3/4/5 各 6；round 内按 4 个 Pending anchor_pos 总和排序分 6 个等频 strata；每层取 SHA256('flowstate-rq4-a|seed|round|stratum|snapshot_id|snapshot_digest') 最小者，seed=20260907）。24 个快照 ID 逐一枚举在案。输入数据集 `/home/wjg/data/agentic_coding_trajectories/sessions.parquet`（SHA `e261707f…`）。
- **实验配置**：策略 FS/LRU/Marconi（policy_blind，seeded Latin-square 顺序）；预算 K = max(1, floor(0.25·|C|))，实测 |C|∈{8,12,16,20} → K∈{2,3,4,5}（每快照恒定，逐 run 校验 `K==max(1,cand//4)` 全成立）；重复 3 次；216 计划 run；每 run 独立进程+独立引擎（manifest `fresh_process_and_engine_per_run=true`）；`future_trajectory_used=false`、`infrastructure_retry_limit=0`。
- **执行与原始日志**：`runs/` 216 目录（每 run 含 correctness.json、pending_requests.jsonl、reconcile.json、selection.json、runtime.log 等）+ `workers/` 216 个 worker 日志。correctness 216/216 PASS；G=H−E 恒等式 864/864；future_leakage=0、truncation=0、oom=0、fa_preserved=true、跨 run 污染=false。
- **环境**：SGLang v0.5.17-cu129 镜像，Qwen3.5 hybrid GDN 模型（Qwen3_5ForConditionalGeneration），tp_size=1，单 GPU（physical_gpu_index 0）；请求 completion_tokens=1。
- **TTFT 边界**：请求在状态协调完成后发出；`ttft_ms` 取引擎 runtime metrics 的 first_token_latency；`recovery_latency_telemetry=PARTIAL`（recovery_latency_ms=null，**未用 TTFT 代填**）；控制开销以 run 级 timings 存档（engine_init 均值 35.94 s、pending_resume 1.91 s、reconcile 6.49 s、replay 7.52 s），不在 TTFT 窗口内。

### 3.2 历史数字重新关联原始记录核验（本审计从 runs.jsonl 864 条原始请求重算）

| 数字 | 汇总表值 | 本审计原始重算 | 一致 |
|---|---|---|---|
| 请求级 TTFT 均值 FS / LRU / Marconi | 94.02458 / 164.70500 / 129.09717 ms | 94.02458018 / 164.70500181 / 129.09717031 ms（n=288 各） | ✅ 逐位 |
| 请求级 TTFT 中位 | 55.2306 / 170.6719 / 81.4150 ms | 同 | ✅ |
| 逐快照相对改善均值 vs LRU | 43.058% | 43.058%（24 快照，范围 +3.38%~+76.96%） | ✅ |
| 逐快照相对改善均值 vs Marconi | 24.264% | 24.264%（范围 −26.52%~+56.43%） | ✅ |
| 胜/负 vs LRU | 24/0/0 | 24/0/0 | ✅ |
| 胜/负 vs Marconi | 22/0/2 | 22/0/2（负例：g015-round5 FS 54.2 vs 42.8 ms；g184-round2 FS 212.2 vs 191.9 ms） | ✅ |
| 快照级 CSV | — | 逐快照三策略均值与 G tokens 与 analysis_snapshot_level/snapshot_level_metrics.csv 最大差 5.7e-14 / 0.0 | ✅ |

独立旁证：`evaluation/legacy_runtime_evidence_p0_audit/`（更早的独立复算）给出 43.057902% / 24.264401%、24/0/0 与 22/0/2，与本审计第三条独立路径一致。

### 3.3 判定：**保留（限定保留）**

必须随文的限定（缺一须改写）：(1) 单输出 Token（completion=1）的受控请求级 TTFT，协调完成后发起；(2) 旧版协议与旧版求解入口（当时 optimizer `eabf84c7…`，非当前 hybrid `plan_strict`），不具当前 Strict Cold 逐请求认证身份；(3) LRU 为全局 LRU 基线、Marconi 为本文明确实现的 Adaptation（非官方完整系统）；(4) 逐快照改善是"快照内相对改善的平均"，不是两组均值之比；(5) 控制开销未计入 TTFT（恢复遥测 PARTIAL，如实披露）；(6) 负例 2 快照必须保留披露；(7) 与五轮矩阵/B5-C4 是不同实验（模型规模、预算、协议均不同），数字不得混同引用。

**Task 2.1 状态：VERIFIED**（来源链完整、全部汇总数字从原始记录重算证实、限定条款明确）。

---

## 4. Task 2.2：VI-F Kimi-Linear-48B-A3B-Instruct 证据链与 A/B/C/D 判定

### 4.1 版本与环境（原始文件核验）

- 模型：`/data1/models/Kimi-Linear-48B-A3B-Instruct`（容器内 /model 只读挂载；Phase1 报告记录模型配置/分词器/20 个权重分片映射一致，不含全权重内容哈希——如实记录）。
- SGLang 0.5.17（构建提交 `29481685…`），镜像 `lmsysorg/sglang:v0.5.17-cu129-runtime`（sha256 `1dfce903…`，environment.json 原始在档）；2 × NVIDIA H100 PCIe（各 81,559 MiB）；**TP=2**（逐条件 engine_args.json `tp=2` 实测）。
- 状态组织：MLA Cache（attention 侧）+ KDA recurrent state（MambaComponent/slot）；物化 = 历史重放后按授权驱逐构造；恢复 = 真实请求前缀匹配 + 实际 extend 重算。

### 4.2 三阶段证据（原始记录核验）

- **Phase 1**（3/3 独立生命周期）：KDA tracking、MLA/KDA 驻留观测、KDA-only eviction、MLA 保持、TP rank 一致。**口径修正已冻结**：原报告 `16K/16K/0→16K/8K/8K` 来自两候选逻辑视图；全祖先驻留证据下 E=14K、G=2K；该两候选视图**不表述为实际 8K 回退测量**。
- **Phase 2**（3/3）：复用生产兼容性/H-E-G/恢复成本接口与 greedy allocator；A1/A2/B1/B2 四候选 K=2，S*={A2,B2}，两 rank 正确保留 A2/B2、仅删 A1/B1 KDA；批量观测/只读检查/协调/MLA 保持/TP 一致 PASS；恢复成本数值仅作接口功能验证（**非 Kimi Φ 校准**）。
- **Phase 3**（5 配对，AB/BA/AB/BA/AB，逐条件独立引擎、同历史/请求/配置）：原始 record.json 逐条件核验——A：H/E/G=16384/16384/0，两 rank extend=[256]；B：H/E/G=16384/8192/8192，两 rank extend=[2048×4, 256]（Σ=8448=8192 重算+256 后缀）；期望==实际逐 rank 相等；MLA digest `88224e99…` 跨 rank、跨条件一致（MLA 不变性直接证据）；14 项 checks（actual_prefill_work、only_expected_evictions、one_output_token、fresh_lifecycle 等）逐条件全 PASS；驱逐事件 4 项全部 authorized。
- **主指标定义（原始说明字段原文口径）**：两 TP rank 调度器内部恢复时延的较大者——"首次实际前缀匹配至单令牌请求完成，含相同 256-token 后缀处理，排除启动、历史物化、观测和驱逐"；客户端时延与纯 CUDA 时间另有记录但**不冒充主指标**。
- 配对差逐对核验：[748.413, 759.080, 751.266, 763.927, 778.271] ms，均值 **760.19**、中位 **759.08**、正差 **5/5**（与冻结报告逐位一致）；CUDA forward 配对差 740.65~766.83 ms 同记录。

### 4.3 A/B/C/D 判定

| 结论类 | 判定 | 依据 | 限定 |
|---|---|---|---|
| A 跨架构可行性 | **支持** | Phase1-3：同一 executable-state allocation 与组件隔离在 MLA+KDA 上迁移，核心语义不变 | 固定模型/版本/TP=2/小规模构造负载 |
| B 组件级控制正确性 | **支持** | 授权驱逐逐事件在档、MLA digest 不变、TP 两 rank 一致、期望==实际驻留 | 每 epoch 一次逻辑批量观测/协调；不主张并发跨 rank 事务容错 |
| C 原生恢复正确性 | **支持** | 实际请求匹配 A=16K/16K/0、B=16K/8K/8K（两 rank），真实 extend 执行、one_output_token、same_output | 单令牌完成口径；Phase1 两候选视图已修正为 14K/2K 口径 |
| D 跨架构请求性能改善 | **不支持** | Phase3 是受控 A/B（唯一差异=16K KDA 驻留），非策略对比；760.19 ms 是**恢复窗口差**，既非 TTFT 服务收益、也非完整控制周期净收益；无任何策略间请求性能测量 | 不得因"实现+恢复成功"认定 D |

**判定：保留（限定为 A/B/C 类结论）**。VI-F 若将 Kimi 写成跨架构性能改善（D）即超出证据；760.19 ms 必须按"固定 MLA 驻留下真实 8K 恢复间隔的额外恢复开销（5 独立配对）"表述。

**Task 2.2 状态：VERIFIED**（环境/组织/物化/恢复/不变性全部有原始记录；A/B/C/D 分类明确）。

---

## 5. Task 3.1-3.2：AgentX Formal↔Raw 差异与 VI-D 版本锁定

### 5.1 原始轨迹与转换过程

- 原始语料：冻结 AgentX Weka corpus `semianalysisai/cc-traces-weka-062126`（393 traces；B0 可行性审计 PASS：296/393 有 d(c)≥2 分支、312/393 有 ≥2 同时 pending、max d(c)=734、上下文长度 WARN max 989,824 tokens）。
- 转换链：B0.1 在线兼容审计 → B1 人口 census（84 条 online-safe shared traces，其中 23 条含 ≥1 上下文兼容 epoch；runtime context limit 131,200）→ **23 个 runtime 快照、568 个候选、Formal max d_t(c)=2**。
- **Formal 权威定义**（compatibility_exact_set_audit.json 原文）："仅 logical lineage 严格祖先、validated FORK inherited、candidate token position 不超过 active target；禁止 radix/token LCP/物理驻留定义兼容性"。

### 5.2 Formal↔Raw 差异记录（非"快照数一致即等价"——逐边差异在档）

| 维度 | 结果 |
|---|---|
| 候选宇宙 | 568==568，`candidate_universe_exact_match=true`，无额外/缺失候选，metadata 不匹配 0（**无物理身份别名**） |
| Pending 宇宙 | 23/23 快照完全一致 |
| **兼容边（兼容集合逐候选精确匹配）** | **仅 234/568 一致；334 不一致** |
| Raw 多计 | extra_raw_pair **413** 条（token-prefix/LCP 把同 conversation SPAWN/非继承/非逻辑祖先的物理前缀重合计入兼容） |
| Raw 少计 | missing_logical_pair **76** 条（逻辑祖先成立但独立合成请求的 token 序列不是 pending 字面前缀） |
| 度数 | B2 raw max d_t(c)=4 vs Formal 2；27 个 d>2 候选全部为 raw 多计（逐候选 root_cause 在档） |
| 离线修正 | corrected == B1 逐字一致（`corrected_exactly_matches_b1=true`） |

（注：交接文档 §7.1 曾写"70 条 Formal-only 边"，权威审计文件为 missing_logical_pair_count=76；以本差异文件为准，交接文档数字系口径笔误，不影响证据。）

### 5.3 VI-D 采用版本与可支持结论

- **VI-D 采用版本 = Formal Allocation Snapshot**（CPU Φ 模型化分配）：策略评估 `rq3_agentx_policy_eval_20260906_214956` 明示"使用 B2 raw compatibility：**否**"；764 预算敏感性同样基于 Formal 去预算输入。`paper_allocation_structure_legacy573.csv`：OpenHands 504 条与 AgentX 69 条 **strict_runtime_certified 均为 0**。
- 三层结论边界：
  1. **Formal 模型离线分配质量——支持**：AgentX 69 条件（23×3 比例预算 25/50/75%），Exact tractable 48、FS==Exact 48/48；相对改善随预算饱和递减（vs LRU 13.86%→5.18%→3.39%；vs Marconi 5.00%→0.25%→0.07%），胜/平/负（三预算合计）vs LRU 16/53/0、vs LFU 12/57/0、vs Marconi 16/53/0，**0 负**。
  2. **原始轨迹真实执行关系——不无损**：334/568 候选兼容集差异（413 多计/76 少计）已固化在案；Raw token-prefix 转换**不得**作为 Formal 等价证据；Formal 定义为权威。
  3. **Native Runtime 实际可执行恢复——无证据**：AgentX 无 GPU 恢复性能实验（strict_runtime_certified=0）；B2.3 为正确性/census 级审计。**不得把 Formal Allocation 写成 AgentX GPU 恢复性能证据。**
- 补充（764 内 AgentX 结构性发现，已独立复核）：AgentX K1 有 9/23 快照四策略 J 分歧（恰为 9 个 K_min=2 共享快照；g000/g007/g014 FS 选共享项胜 PCB/LDR，g005/g006/g010/g017/g018/g021 FS 选深专属项胜 WFC；FS 0 负）；K2 起 23/23 同集同 J。

**Task 3.1-3.2 状态：VERIFIED**（差异记录完整、版本锁定明确、三层结论边界清晰）。

---

## 6. Task 3.3-3.4：764 条件与 Oracle 统计核查

### 6.1 条件构成（CSV 独立复核）

- `exact_oracle_results.csv` 764 行 = **191 快照 × K∈{1,2,3,4}**；OpenHands 672（168×4）+ AgentX 92（23×4）✅。
- 与旧 573 的关系：573 = 自然比例预算条件（OH 504 + AX 69，K 为各快照 25/50/75% 比例预算）；764 = 统一 K1–4 扫描。**两套条件集不同**，论文不得混用分母（573 的"greedy 最优 573/573"与 764 的"FS 396/396 门内最优"是不同命题）。
- 第一门禁（573/573 复现）与第二门禁（191/191 K1-4）均 PASS（budget_sensitivity_audit_report.md，2026-10-30）。

### 6.2 Oracle 统计边界

- **门内/门外**：PASS **396** / N/A_SIZE_GATE **368** ✅（CSV status 列逐行复核）。门禁=候选数 n≤12（PASS 行 n∈[6,12]；N/A 行 n∈[6,149]，如 n=149 的 K4 枚举 20,271,451 子集——**未偷偷扩大枚举**）。396 = 99 快照（OH 90 + AX 9）× 4 预算。
- **N/A 处理**：N/A 行 J_ms='N/A'、selected='N/A'、oracle_gap_ms='N/A'（字符串，非 0）——**不存在 N/A 当零值**；任何均值/统计只能以 396 为分母，**以 764 为分母即口径错误**。
- **Gap 公式区分**：
  - **Solver Optimality Gap** = oracle_gap_ms = J(策略) − J(Oracle argmin)（容差 1e-9 ms；Oracle=|S|≤K 全子集枚举含空集，tie-break 最少项→ID 字典序，不补零收益项）；
  - **Recovery Gap** = G = T − E（逐 Pending 恢复 token 数，Φ 的输入）。两者量纲与含义不同，VI-D 引用时必须区分命名。
- **最优性统计复核**：FS 门内 **396/396 gap==0**；PCB/LDR 395/396（最大 gap 1476.849 ms）；WFC 394/396（最大 1953.251 ms）✅。
- **多解披露**：门内最优集可多解（OH 28/360、AX 25/36 条件多解，含预算富余时加零边际候选的不同基数集合）——不得解释为不同恢复质量。
- **其余复核**：764 中 754 同集同 J、10 条选集+J 双差异（AgentX K1 9 + OH g051-round5/K3 一例 0.0477 ms，n20 门外不扩 Oracle）；相邻 K 无非嵌套替换（2292 行核对）。
- **CSV↔论文一致性**：CSV 侧全部数字复核完毕；**论文正文（v6.5 tex）不在本仓库**（搜索范围：仓库根及 4 级深度 `*.tex`/paper 主稿均无），故"论文是否误以 764 为分母/是否混同两种 Gap"无法在仓库内逐句核对——列为未闭合项（见 §9），投稿前须按 §10 清单人工核对正文。

**Task 3.3-3.4 状态：PARTIALLY_VERIFIED**（证据/CSV 侧全 VERIFIED；正文侧核对因稿件不在仓库而无法完成）。

---

## 7. 原始文件及 SHA-256 清单（均为本审计实际读取并现场计算的哈希）

| 文件 | SHA-256（前 16 位） |
|---|---|
| evaluation/p0_control_cycle_gpu/runner.py | b6cee2d3ffa04e8f |
| evaluation/p0_control_cycle_gpu/control.py | b1c3a64a61720e69 |
| evaluation/p0_control_cycle_gpu/five_round_stats.py | c7597587dfe71396 |
| evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs/run/summary.json | a1d87802b2d0fa3f |
| evaluation/p0_control_cycle_gpu/arms/m1_ck3_fs/run/reconciliation.json | da6f5541f5795770 |
| （另 14 臂 summary.json 同批读取重算，路径模式 arms/m{1..5}_ck3_{fs,mc,kv}/run/summary.json） | — |
| rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/frozen_protocol.json | 5f7dfeef5f8b8ac3 |
| …/final_report.md | a4ea59bf1f0d49cb |
| …/collection_summary.json | b875b42d2ca78aa8 |
| …/runs.jsonl（216 行原始记录） | dc3f1b78d778f05b |
| …/source_integrity.json | 801f9cb96a2a9988 |
| …/runtime_correctness.json | 9a6040e63719cd89 |
| …/manifest.json | 90853519a0d7801d |
| …/analysis_snapshot_level/snapshot_level_metrics.csv | 8816e8b244d6f56e |
| …/runs/g005_round3_flowstate_rep01/pending_requests.jsonl（含运行级原始请求记录） | —（目录内逐 run 在档） |
| evaluation/rq5b_kimi_final_output/rq5b_frozen_20260923_090154/final_report.md | bf8d8abd2b76b350 |
| evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/paired_measurements.json | 698eb3ae85f023ef |
| …/summary.json | cb0816fc9b93c8f4 |
| …/environment.json | 63965f7d3833ad19 |
| …/runs/pair01_A/record.json | 90969b7dc167357c |
| …/runs/pair01_B/record.json | f13ad874e01d75ba |
| agentx_runtime_final_audit_20260906_194000/final_report.md | eafac49b81baaf87 |
| agentx_runtime_final_audit_20260906_194000/compatibility_exact_set_audit.json | f65ee1457b59c613 |
| rq3_agentx_policy_eval_20260906_214956/final_report.md | 21affa53f131e8c0 |
| rq3_agentx_policy_eval_20260906_214956/aggregate_results.json | c51e60a297efcc13 |
| STEP_13G_B0_AGENTX_FINAL_REPORT.md | 28355c76b41af749 |
| STEP_13G_B1_AGENTX_RUNTIME_POPULATION_FINAL_REPORT.md | 5801c4e57530c994 |
| evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/budget_sensitivity_audit_report.md | cf8397875e09ef4f |
| …/exact_oracle_results.csv | 4c67757fb3fea244 |
| …/budget_sensitivity_strategy_results.csv | 58cca4c35028013b |
| evaluation/paper_allocation_structure_legacy573.csv | bf50acf2f48c3941 |
| evaluation/legacy_runtime_evidence_p0_audit/final_report.md | f55d932209c8ca1d |

---

## 8. 已验证论文数字（本审计独立重算/逐位核对通过）

| 论文位置 | 数字 | 验证方式 |
|---|---|---|
| VI-B | ΣTTFT 配对差中位 FS−MC **−470.208** / FS−KV **−429.711** ms（五轮全负） | 15 臂 summary.json 重算，逐位一致 |
| VI-B（口径义务） | control_cycle 配对中位 +2015.8/+436.9、full cycle +1704.0/+781.3（FS 更慢，须随文披露） | 同上重算 |
| VI-F OpenHands | TTFT 均值 94.025/164.705/129.097 ms；中位 55.231/170.672/81.415；逐快照改善 43.058%/24.264%；24/0/0 与 22/0/2 | runs.jsonl 864 请求重算，逐位一致 |
| VI-F Kimi | 配对差均值 760.19 / 中位 759.08 ms、5/5 正；A=16K/16K/0、B=16K/8K/8K（两 rank） | paired_measurements.json + 逐条件 record.json 重算 |
| VI-D/AgentX | 69 条件、FS==Exact 48/48、预算饱和递减、0 负；K1 分歧 9/23（FS 全部最小或并列最小） | aggregate_results + strategy CSV 重算 |
| VI-D/764 | 191×4=764（OH 672/AX 92）、Oracle 396 PASS/368 N/A、FS 396/396 gap=0、PCB/LDR 395、WFC 394、N/A 非零值 | exact_oracle_results.csv + strategy CSV 逐行复核 |

---

## 9. 未闭合证据及原因（不补实验，如实列示）

1. **论文正文（v6.5 tex）不在本仓库**——已搜索（根目录及 4 级深度无任何 .tex/主稿）。影响：§6.2 的正文分母/命名核对、§10 建议的逐句落改无法在仓库内完成，须投稿前人工对照。此项为本次审计唯一阻断项。
2. **rq4 请求级控制遥测 PARTIAL**（recovery_latency_ms=null）：控制开销仅有 run 级 timings（reconcile 均值 6.49 s 等）；对"24 快照实验控制含入净收益"不存在可计算口径——VI-F 不得推导该数字。
3. **Kimi D 类（跨架构请求性能改善）无实验**：Phase3 为受控 A/B 恢复窗口差；如需 D 须新实验（本轮不做）。
4. **AgentX Raw 运行时兼容非无损**（413 多计/76 少计）：Formal 为唯一权威口径；Raw 侧无法升级为执行等价证据（除非未来重写判据，本轮不做）。
5. **rq4 FS 臂为历史版本入口**（optimizer `eabf84c7…`，pre-hybrid）：与当前 `plan_strict`（`7dfa9aec…`）不同版本；数字有效但版本身份必须随文标注。
6. **OpenHands 24 快照 TTFT（引擎 runtime metrics）与五轮矩阵 TTFT（客户端 monotonic）计时实现不同**：两实验绝对值不可直接比较，只能各自口径内引用。

---

## 10. 论文修改建议（VI-B / VI-D / VI-F）

### VI-B（五轮主实验）
1. 明确写出：**"TTFT 为控制周期完成后的请求窗口时延（排队+Prefill 恢复重算+首 decode+交付）；观察、选择、协调与预热不计入 TTFT。"**
2. −470.208/−429.711 表述为"**请求窗口 ΣTTFT 配对差中位**（同轮 4 Pending 之和先配对、5 轮取中位；n=5，无显著性检验）"；删除任何"净收益/系统级收益"字样。
3. 增列（或脚注）控制开销三口径：control_cycle（FS 更慢，配对中位 +2015.8/+436.9 ms）、full cycle 含预热（+1704.0/+781.3 ms）、selection_call（16.38/2.05/4.40 ms 中位）——如实呈现"请求窗口改善、控制周期更贵"的权衡，并指向控制成本优化结论（生产核心 1.0-1.2 s、apply 占墙钟 10.5% 等已有审计）。
4. 不将 CPU 微基准求解时耗（0.22 ms）与在线 selection_call_ms 混写。

### VI-D（AgentX 与预算敏感性）
1. AgentX 全部结果标注 **"Formal Allocation（CPU 模型化）"**；不得出现"AgentX GPU 恢复性能"表述（strict_runtime_certified=0）。
2. 随文披露 Formal↔Raw 差异事实（候选/Pending 宇宙一致，但 334/568 候选兼容集差异；Raw token-prefix 判据非权威）——建议一句话+引用审计产物，防止审稿人质疑转换等价性。
3. Oracle 统计分母写 **396**（门内），368 N/A 如实说明（n≤12 门禁，N/A 非零值）；"Solver Optimality Gap"与"Recovery Gap (G=T−E)"分开命名。
4. 573（自然比例预算）与 764（K1–4 扫描）明确为两套条件集，分别给分母。
5. AgentX K1 9 快照分歧与 OH g051/K3 一例 0.048 ms 差异按审计报告双向呈现（FS 0 负；两组相反竞争机制）。

### VI-F（OpenHands + Kimi，两实验不得合并为同一种验证）
1. 分开小节：OpenHands=同架构真实轨迹的**请求级 TTFT 策略对照**（历史版本入口、单输出 Token、24 快照/3 重复/比例预算、负例 2 个保留）；Kimi=**跨架构组件可行性与恢复正确性**（A/B/C 类），两者证据类型互补、不可叠加为同一主张。
2. OpenHands 数字使用：均值/中位 + 逐快照改善 43.058%/24.264%（注明"逐快照相对改善的平均"）+ 胜/负 24/0/0、22/0/2；注明 Marconi 为 Adaptation 实现。
3. Kimi 数字使用：760.19/759.08 ms 表述为"固定 MLA 驻留下、真实 8K 恢复间隔相对 G=0 的额外恢复开销（5 独立配对，主指标=两 rank 调度器内部恢复时延最大者）"；不写 TTFT 改善、不写控制净收益、不写 Φ 校准。
4. 环境标注：OpenHands（SGLang 0.5.17、Qwen3.5 hybrid、tp=1、单 GPU）；Kimi（SGLang 0.5.17、Kimi-Linear-48B-A3B、2×H100 PCIe、TP=2）——模型不同，禁止跨实验绝对值比较。

---

## 11. 各 Task 独立状态与整体状态

| Task | 状态 | 说明 |
|---|---|---|
| Task 1（VI-B 计时边界与控制归因，P0） | **VERIFIED** | 版本锁定、代码级边界、独立原始计时、四问全答；请求窗口改善成立、全周期净收益不成立（须按 §10 收窄表述） |
| Task 2.1（OpenHands 24 快照） | **VERIFIED** | 来源链 5 环完整；全部汇总数字从 864 条原始请求重算证实；判定=保留（限定） |
| Task 2.2（Kimi） | **VERIFIED** | 环境与三阶段原始记录完整；判定=A/B/C 支持、D 不支持；保留（限定） |
| Task 3.1-3.2（AgentX Formal/Raw） | **VERIFIED** | 差异记录完整（234/568、413/76、修正==Formal）；VI-D=Formal Allocation，无 GPU 恢复证据 |
| Task 3.3-3.4（764/Oracle） | **PARTIALLY_VERIFIED** | CSV/证据侧全部复核通过；论文正文不在仓库，正文侧分母/命名核对无法完成 |

**整体状态：PARTIAL。**

理由：三个 Task 的证据侧全部闭合（版本锁定、原始记录、独立重算、差异记录），无一项需要补实验；唯一未闭合项是论文正文（v6.5 稿件）不在本仓库，导致 spec 中"核对论文及 CSV 一致性"的正文半边与 §10 建议的逐句落改无法在仓库内执行。按"不以获得 READY 为目标、不为 READY 补数据"的原则，如实给出 PARTIAL，并将在仓库外完成正文核对后即可闭合（无新实验需求）。

---

**审计执行声明**：本审计为只读；除本报告文件外未创建/修改任何文件；未启动 GPU；未重跑任何历史实验或门禁；未修改 Runtime/Optimizer/Controller/实验脚本与历史数据；所有 SHA-256 均为现场对实际读取文件计算。完成本报告后停止，等待人工审查，不自动修改论文正文。
