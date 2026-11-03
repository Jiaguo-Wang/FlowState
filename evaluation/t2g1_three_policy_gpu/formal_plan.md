# T2-G1 附：三策略正式 GPU 对照执行方案（未执行）

前置：GPU Smoke Test 尚未运行（本轮 GPU 被占，见 gpu_gate.json）。正式实验启动前必须先补 smoke（三策略各 1 臂，验证本方案的 runner 消费路径）。

## 1. 实验域与选集来源（全部冻结）

- 域：C/K3 冻结域——16 个 CPU 候选 ↔ 物理候选（T2-R 逐 Token 映射证据）、四 Pending C_A..C_D（与 CPU ready 调用逐 Token 全等）、K=3。
- 选集：`t2g1_plans.json`（与 pilot policy_plans.json 同格式）。FS 计划已逐位复现已通过 GPU 验证的 pilot 臂；Marconi-Evict/KVFlow 新增两键，由 runner 以 `PILOT_POLICY` 消费。
- 决策输入全部为请求前冻结声明（访问次序/ASG/固定前缀）；C_JOIN（CPU 独有 future）不进入决策。

## 2. 臂设计与调度

- 3 策略 × 4 Pending × 5 轮 = **60 主臂**，对齐 `c_formal_perf_gpu` 的 60 主臂结构与 C/K3 冻结五轮独立生命周期协议；每臂独立 Docker/Engine/Triton 缓存生命周期。
- 策略执行顺序按轮轮换（奇数轮 FS→MC→KV、偶数轮反转）以对冲时间漂移；schedule 预固定，launch 拒绝跳序/重跑（沿用 G2 实践）。
- 每臂控制路径零修改复用 pilot `control.py`：census 对账（16=16）→ 冻结认证 → E/T/G 逐格 == t2g1_plans 预期 → bind/preflight → `reconcile_strict`（按 logical_eviction_ids 真实驱逐非保留状态）→ FA 索引与内容保护 → 目标 Pending 请求。

## 3. 测量口径与有效性（沿用既有合同，不另行定义）

- 三层口径：CUDA 事件（计算）/ 请求 TTFT 六段分解（闭合残差合同）/ 控制分段（含策略选择与协调）。
- 有效性：请求窗口零审计零新增 Triton 编译（快照差分）、形状预热先于计时、输出与独立 Cold 参照逐 Token 一致、Prefill 合同沿用 `c_formal_perf_gpu` 定义。
- 统计：配对差中位数为请求级主口径，轮级四 Pending 等权为集合级口径；两种中位口径并列报告杜绝混用；五轮样本不使用"显著"表述；负向观测全保留。

## 4. 预期结构差异（CPU 层已知，供实验设计解读）

| 策略 | 选集 | 四 Pending E | 当前域 J (ms) |
|---|---|---|---|
| FlowState-Greedy | D7168/C6144/B5120 | 0/5120/6144/7168 | 154.35 |
| KVFlow-inspired | AB1984/AB2048/A4032 | 4032/2048/0/0 | 627.61 |
| Marconi-Evict | JOIN4032/JOIN4096/ROOT1024 | 1024×4 | 703.46 |

三策略选集两两不同、覆盖结构互补（末端叶 / 分支前缀 / 根+JOIN 链），正式对照将给出真实 Prefill/TTFT 差异；Marconi-Evict 的当前域 J 最差但其声明未来覆盖最优（T2-Mini：C/K3 未来静态 154.35→0），该权衡是实验要测的假设而非预设结论。

## 5. 停止与产出

- 失败臂保留全部证据，不剔除负向/平局观测；60/60 PASS 才判整体 PASS。
- 产出：raw_results.csv（逐臂 30+ 列）+ 配对分析 + 三层结论分列（模型 J / 结构 E·G·Prefill Token / 实测 TTFT·CUDA）+ 复算脚本。
