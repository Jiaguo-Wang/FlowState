# G1 只读审计（实验前，2026-10-25）

任务：Hybrid Solver Exact/Greedy 双路径 GPU 闭环验收。本记录为装配前只读审计结论，全部只读，未改动任何冻结目录。

## 1. 代码位置与接口

- 生产入口：`flowstate/optimizer.py` `GlobalOptimizer.plan_strict()`（hybrid：`_exact_admission` 门限 n≤12 候选、≤4 Pending、K≤4、枚举子集≤794 → Exact 穷举；否则 Marginal Greedy；决策记入 `last_strict_decision`；fail-stop 无 fallback）。
- 协调入口：`flowstate/strict_control.py`（`validate_plan` / `reconcile_strict`），经 b5c3/b5d4 控制面在 Scheduler 进程内调用。
- 现有可复用程序：`evaluation/paper_hybrid_gpu_fix_a`（w1/K2 单臂骨架，B5-C4 目标 Pending 硬编码）、`evaluation/paper_hybrid_gpu_fix_b`（C/K3 骨架，`PILOT_PENDING` 已参数化，Cold 参照按 `cold_reference_{continuation_id}.json` 命名）。
- git：HEAD=8cba094d（2026-10-22），工作区 `flowstate/optimizer.py`、`tests/test_strict_epoch.py` 有未提交修改（hybrid solver 集成任务的已知状态）；被测代码身份以逐文件 SHA-256 台账锁定，不依赖 git 清洁。

## 2. 冻结参考

- **实验A w1/K2**：Epoch=E2/w1 快照（7 候选 L={N18:3392,N20:2432,N21:3712,N22:3648,N24:3712,N25:4096,N27:4160}，4 Pending IMW1_{A,B,C,D}_TURN_003 T=3648/3712/4096/4160，12 容许边）。e3 w1/K2 Exact 冻结行：选集 {N18,N21}，J=53.77075886917578，E/G=(3392,256)/(3392,320)/(3712,384)/(3712,448)。输入与 Cold 取自 E4-1A（prepared_inputs 按其 execution_lock 逐哈希核验；4×Cold 经 validated_resumes 逐 Token 验证）。Exact 门内枚举子集=1+7+21=29。
- **实验B C/K3**：16 候选、K=3、4 Pending。冻结 FS={C_D_P07168,C_C_P06144,C_B_P05120} J=154.34895561599998，E=(0,5120,6144,7168)、G=(4096,0,0,0)、Prefill=G+32；SA/WFC 冻结行只作背景。输入取自 c_formal_perf_gpu/g1_CB_FS（其运行记录 census 16=16、J 一致）；4×Cold 与 m1{A,B,C,D}_FS 正式臂冻结输出逐 Token 一致。
- 宿主 CPU 冒烟（实验A装配时）：build_epoch(E2/w1, K=2)→生产 plan_strict→Exact {N18,N21} J=53.77075886917578，与 e3 冻结行逐位一致（cpu_gate.json）。

## 3. 选源可达性（实验A）

Exact 选集 {N18,N21} 下四 Pending 均有唯一最深选中共享状态：A/B→N18(L=3392)、C/D→N21(L=3712)（宿主按容许边复核），pkg_exact 的 G1_PENDING 选源对四臂均可唯一命中。

## 4. GPU 与镜像

- 两块 H100 PCIe，审计时全空闲（14/17 MiB，无 compute apps）；共享环境 → 每臂前查空闲、后查清理。
- 冻结镜像 sha256:1dfce903… 在位；E4-1A 与 g1_CB_FS 的 runtime_sources（镜像内 sglang 源码哈希）沿用为容器内核验台账。

## 5. 需要的适配（不触核心语义）

- pkg_exact=fix_a 复制＋3 处手术改动：G1_PENDING 参数化目标选源（原硬编码 B5-C4 目标）、Cold 参照按 continuation_id 命名、scenario 名。求解/协调/安全断言零改动。
- fix_b 原计划零改动复用；后因 E=0 目标（C_A）暴露 runner 无源分支缺失，改为 pkg_greedy=fix_b 复制＋2 处最小修复（见 `pkg_greedy/build_record.json`，语义对齐 c_formal 生产 runner 的同款分支；E>0 断言面不变）。

## 6. 判定口径

- 8 臂（两路径×4 Pending）全 PASS 且 9 行核验全 PASS → 总体 PASS；仅一路径 → PARTIAL；安全/接线/正确性门禁失败 → BLOCKED。旧 CPU 验收与历史 GPU 冻结选集执行不作为本次验收结果。
