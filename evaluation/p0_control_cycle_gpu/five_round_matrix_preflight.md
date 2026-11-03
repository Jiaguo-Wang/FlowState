# P0-4 五轮三策略正式 GPU 对照实验 · 第一阶段只读审计与执行准备报告

日期:2026-10-30 | 实验包:`evaluation/p0_control_cycle_gpu/` | 依据:五轮矩阵任务书(spec id 82ee)
本阶段零代码修改、零 GPU 启动;r1 Pilot 三臂证据零触碰。

## 0. 结论:PASS(审计通过;执行待人工确认后开始)

八项核查全部通过;需新增的仅为**与策略实现隔离的编排代码**(轮次装配 + 定卡启动器 + 追加式总账,全增量文件,不改 runner/observation/control/select_entry 等测量与策略路径),方案见 §4,实施放在确认之后作为第二阶段第 0 步。

## 1. 八项审计逐项结果

| # | 核查项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 三策略 runner/observation/Runtime/镜像一致 | ✅ | 三臂 source_identity 的 `image_id`(sha256:1dfce903…)与 `runtime_sources`(镜像内 16 文件哈希)完全一致;runner.py 当前 `b6cee2d3ffa0…f9f84` == fs 臂登记(attempt5 装配)== mc/kv 臂 identity_update 链终点;observation.py 当前 `7c1dcefcb5b4…392837` 同样三方一致;control/bridge/protocol/persistence 四接线文件三臂登记一致且==当前源 |
| 2 | 冻结输入/请求集合/16 候选/四 Pending/K=3 一致 | ✅ | 8 个 NAMES 文件 + t2g1_plans.json + workloads_case_C.json 三臂**逐字节一致**(sha256 见 §3);census 16=16;K=3;四 Pending C_A..C_D 固定序 |
| 3 | 三策略均用真实在线选择内核 | ✅ | control.py 分派:FS→生产 plan_strict;MC→`MarconiEvictPolicy.select`;KV→`kvflow_inspired_retention_cpu.policy.select`;kernel_entry 已在两臂 summary.selection 落盘 |
| 4 | 无读取 Oracle/冻结选集/未来轨迹 | ✅ | 控制路径不读 t2g1_plans/policy_plans(runner 仅 T4 后窗口外读冻结对照);三策略 T0 仅见各自 epoch/case 快照,无未来信息 |
| 5 | prepare.py 能否每轮安全建独立目录 | ⚠️→方案 | 现有 `ARM` 映射硬编码 r1 三臂(prepare.py:35),不支持轮次参数;但 `build_arm` 自带 `assert not out.exists()`+write_once 拒绝覆盖纪律,可安全复用。方案:新增 `prepare_round.py` 全增量装配器(§4.1) |
| 6 | Attempt 编号/覆盖风险 | ✅ | r1 编号链已在任务 A 审计核验;五轮用全新目录名(m1..m5,与 r1 零冲突),装配即拒绝已存在目录,历史证据不可覆盖 |
| 7 | GPU/磁盘/容器/launcher 门禁 | ✅ | 双卡空闲(14/17 MiB、0%、零计算进程)、无 p04 容器与 launcher 残留、盘余 68G(15 臂×约1G,逐臂 >20G 门禁余量充足)、冻结镜像本地在位(lmsysorg/sglang:v0.5.17-cu129-runtime);launch.py 现有 pick_gpu/df/串行 fail-loud 语义沿用 |
| 8 | 计时边界与记录方式一致 | ✅ | 同一 runner.py 跑三策略,POLICY 仅用于 env 校验/断言/记录/预热 seed/T4 后对照,无行为分叉;双口径(control_cycle_ms=T3−T0 与 cycle_ms_full_incl_warm=T4−T0)定义同一 `definition_note`;五窗口 triton 扫描、Journal 窗内/出窗、恒等式口径完全相同 |

**arms_status.json 覆盖问题(任务书点名)**:确认存在——launch.py:128 每次调用整文件覆盖,现仅存任务 B 两臂条目。处置:五轮**不依赖**该文件;新启动器写逐轮 `arms_status_m{R}.json` + 追加式总账 `five_round_ledger.jsonl`(§4.3);arms_status.json 与 r1 证据原样保留不改写。

## 2. 源码与镜像 SHA256(冻结身份)

| 项 | 完整 SHA256 / 标识 |
|---|---|
| 容器镜像 | `sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`(=lmsysorg/sglang:v0.5.17-cu129-runtime,--pull never) |
| runner.py | `b6cee2d3ffa04e8f85e9658bf166a37cfb063ad21983c9488b37fb8b4eaf9f84` |
| observation.py | `7c1dcefcb5b456144a70908d7b48cba2b815584a5f0332c3e9f2586445392837` |
| 镜像内 Runtime | 三臂 source_identity.runtime_sources 16 文件哈希一致(scheduler/memory_pool/unified_tree_core 等,沿用 g1_CB_FS 冻结值) |
| 接线文件 | control.py / bridge.py / protocol.py / persistence.py / select_entry.py:三臂登记一致且==当前源(装配时重采集进新臂 identity) |
| GPU | GPU-bfff4d31-0b94-7de3-9946-f593437ae566(r1 三臂同卡,五轮**固定此卡**) |

## 3. 冻结输入身份(三臂逐字节一致,新臂按同源复制)

prepared_inputs `b6c5203bb099b497…`、cpu_candidate_prefix_mapping `dab16d223398672a…`、cpu_gate `f8e04a5a8888bb0f…`、cold_reference_C_A/B/C/D `e9b20839…`/`0096bf25…`/`dc7e8307…`/`19c78d32…`、device_binding `3ea21058f835f2d2…`、t2g1_plans `e13c8c446e87e958…`、workloads_case_C `af8a7508a585e8f7…`(完整值装配时写入各新臂 source_identity.frozen_input_hashes)。

## 4. 五轮目录组织方案(新增编排,与策略实现隔离)

### 4.1 装配:`prepare_round.py`(新文件,导入 prepare.py 的 NAMES/SRC/T2G1/write_once/cpu_prechecks/collect_entry_sources,复用其拒绝覆盖纪律)

- 臂目录命名:`m{R}_ck3_{fs|mc|kv}`,R=1..5,共 15 个全新目录,与 r1_* 及历史归档零冲突;
- identity 沿用同一 schema,新增 `'round': R` 字段;attempt 语义=干净起点装配(不涉 r1 retry 链);
- 逐轮写 `prepare_record_m{R}.json`;cpu_prechecks(census/E 向量/oracle schema/串行安全)每轮重跑。

### 4.2 启动:`launch_round.py`(新文件,复用 launch.py 的 docker_cmd;run 循环自持以支持定卡)

- **固定物理 GPU**=GPU-bfff4d31(与 r1 三臂同卡):每臂启动前实查该卡空闲(<100MiB 且无计算进程),被占→**停止等待,不自动换卡、不杀他人进程**;
- 调用:`python3 -m evaluation.p0_control_cycle_gpu.launch_round {R}`,按任务书固定顺序串行,一臂非 PASS 即终止该轮(fail-loud,不静默重试、不跳过);
- 每臂沿用现有按臂名键控的 `{arm}_command.json`/`{arm}_process.json`/`{arm}_cleanup.json`/`{arm}.runtime.log`,零覆盖;
- 逐轮写 `arms_status_m{R}.json`,**不触碰 arms_status.json**。

### 4.3 追加式实验总账:`five_round_ledger.jsonl`

每臂完成后追加一行(fsync append,只增不改):`ts, round, order_idx, policy, arm, gpu_uuid, image, runner_sha256, observation_sha256, returncode, status, wall_s, disk_free_gb, control_cycle_ms, cycle_ms_full_incl_warm, request_exec_total_ms, sum_ttft_ms, sum_cuda_prefill_ms, J_ms, evidence_dir`。第四阶段统计直接由 ledger+各臂 summary 生成 CSV/JSON(交付 #6)。

## 5. 执行顺序(预先固定,不随中途结果调整)

| 轮次 | 第一 | 第二 | 第三 |
|---|---|---|---|
| Round 1 (m1) | FlowState | Marconi | KVFlow |
| Round 2 (m2) | Marconi | KVFlow | FlowState |
| Round 3 (m3) | KVFlow | FlowState | Marconi |
| Round 4 (m4) | FlowState | KVFlow | Marconi |
| Round 5 (m5) | Marconi | FlowState | KVFlow |

每轮三臂全部成功才进入下一轮;某轮中断→该轮作废重跑(不得拼接零散结果),作废目录按既有归档纪律处置。

## 6. 计时与统计口径

- **双口径严禁混用**:control_cycle_ms(T3−T0,不含预热/请求)与 cycle_ms_full_incl_warm(T4−T0)并列报告;request_exec_total、预热块、RPC 尾段、取证开销(Journal 出窗/audit/oracle 复核/cold 重放)分列归属;
- 正确性门禁沿用 Pilot 全集(§任务书五-E),不降门槛:cold 逐 Token、E/G/Prefill/COW/FA 合同、admission==E、census 无 unmapped、oracle 三检、T0–T4 恒等式、audit=2、请求窗口 triton 0、源码身份、唯一 summary+rc=0;
- 统计:逐轮配对差(FS−MC、FS−KV)对 ΣTTFT/逐 Pending TTFT/Σcuda_prefill/prefill tokens,先算每轮配对值再报五轮配对中位数;各指标报中位数/最小/最大;J 与实测一致性、选择时间与控制周期变化按轮列示;**样本量 5,不做显著性声称、不外推普适结论**;
- r1 Pilot 数字仅作参考基准,不进五轮统计样本。

## 7. GPU 启动前置条件(每轮发射前逐项实查)

1. GPU-bfff4d31 空闲(<100MiB、零计算进程;他人占用→停止等待);
2. 盘余 >20G(当前 68G;预期每臂约 1G、15 臂合计约 15G);
3. 无 flowstate-p04-* 残留容器、无 launcher 残留进程;
4. 当前 runner/observation 哈希 == §2 冻结身份(每轮复核);
5. 该轮 m{R}_* 三臂目录为装配新建(prepare_record_m{R}.json 在位、run/ 为空);
6. 冻结输入哈希与 §3 一致(装配时 cpu_prechecks+identity 自动把守)。

## 8. 边界与风险(如实)

- 单臂墙钟 120–240s+引擎启动波动,15 臂串行预计 40–70 分钟纯 GPU 时间;轮间与臂间均有 GPU 实查,他人任务出现即停等;
- 已知运行间波动:重观察段 20–30%(r1 两轮对照),配对分析正是为吸收该波动设计;
- 任一臂失败→立即停该轮、保留现场、报告根因,重跑须人工显式发起(沿用 r1 纪律);
- 本任务不改核心 Runtime/Solver/Controller/策略定义/冻结输入/论文正文;不扩大预算;不混入 27B。

## 9. 待人工确认事项

1. 是否批准 §4 三个新增编排文件(prepare_round.py / launch_round.py / five_round_ledger.jsonl 机制)——全部为增量,不改任何现有文件与实验语义;
2. 是否批准臂目录命名 m1..m5_ck3_* 与 §5 固定顺序;
3. 确认后执行序列:实施 §4 编排 → CPU 侧自检(compileall+装配+账簿演练)→ 逐轮 GPU(R1→R5,每轮前 §7 门禁)→ 第四阶段统计与 `five_round_matrix_audit_report.md`。

**第一阶段到此停止,等待人工确认;未确认前不启动任何 GPU 实验。**
