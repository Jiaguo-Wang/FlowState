# 优化候选（≤2 项；本轮仅设计，不实现）

通用约束：不修改生产 Scheduler/Optimizer/StateController/Runtime Bridge；不删除严格兼容、安全检查或新鲜度要求；FA-KV 观察语义保留、控制仍仅限 Recurrent State；所有"预期减少"均指**操作清单的移除**（哪些已实测阶段不再出现在决策路径），不折算成毫秒收益——轻量路径的实际耗时在 GPU 测量完成前不存在任何数字。原型一律写在新实验目录，CPU 先行。

---

## 候选 1：变更点增量维护的最小状态视图（对应合同文档全文）

**思路**：把 §二类"全量 census+重建"与 §三/四类"内容哈希"逐出每轮决策路径，代之以 Scheduler 线程在状态变更点维护的登记表（字段 1/2/3/5/6），决策时只做 O(N) 内存读取 + 每 Pending 一次 FA 前沿游走 + apply 时逐选中项定点重验。

**修改模块（未来授权轮；本轮零改动）**：
- 新增 runtime 侧视图维护器（挂接点=现有 instrumentation 同点：`match_prefix`、`copy_from`、`evict_component`、`_free_values`、树 insert/split 内部路径；MonkeyPatch 子进程安装方式沿用 tests/runtime/wp3b_end_to_end_transport.py:147-181 模式，不改 SGLang 磁盘源码）；
- 新增决策侧观察入口（替代 `b5c3_observe` 的轻量动作，返回 KB 级视图）；
- `flowstate/strict_control.py` 不改语义：`validate_view/bind/preflight/postcheck` 在小视图上照常执行（PhysicalView 字段即登记表内容，结构不变）；
- `flowstate/adapters/sglang.py` 不改：apply 时定点重验复用 `_validate_target`。

**预期减少的操作**（对照 observation_cost_breakdown.md §2；均为结构事实级移除，非耗时承诺）：
- 决策路径上消失：每次 observe_control 的 4 次全树 FA 内容哈希、2 轮逐槽 Recurrent 内容哈希、before/after 双 observe 深度比对、巨型 census JSON 序列化/传输/解码、O(N²) 前缀包含扫描、逐节点父链 Token 重建+全量 token_digest；
- reconcile 期间消失：13 次完整 observe（D3 结构）→ 每次驱逐前后仅做小视图 postcheck（结构/槽位记账比对）+ 既有 Adapter 精确释放范围断言；
- 保留不变：一次 `torch.cuda.synchronize()` 静止点、scope/记账闭合/唯一性检查、bind/preflight/最终 postcheck、逐选中项物理重验、fail-closed 隔离。

**安全风险与对策**：
1. *挂接点遗漏* → 登记表漂移而 generation 不 bump。对策：变更点清单穷举（合同 §3）+ apply 定点重验兜底 + 周期性全量 census 对账（漂移即冻结控制）；
2. *COW/引用别名* → 候选域污染。对策：候选槽引用==0 断言保留，COW 目的槽标记为请求占用；
3. *分裂后旧身份复活* → 定点重验 node_id+prefix_digest 必失败，fail-closed；
4. *与重型路径语义漂移* → R3-B 对拍门禁（两路径候选集/严格边/T 必须一致）。

**CPU 负面测试**（全部可在 fake cache 上执行，无 GPU）：
- 绕过登记接口直接改底层 fake 树（模拟漏 hook）→ 周期对账必须报漂移；
- 计划与 apply 之间插入一次驱逐/分裂/COW → generation 变化，bind 必须拒绝；
- 构造引用非零/host_copy 存在/槽重复/记账不闭合 → validate_view/preflight 必须拒绝；
-驱逐半途失败 → ledger 记录已完成项并隔离（StrictReconciliationError 合同不变）；
- FA 前沿回退（FA 组件被驱逐）→ T 重算或拒绝，且不得触发任何 FA 控制动作（语义红线回归测试）。

**GPU 测量计划（需另行授权；本轮不运行）**：复用 B5-D4b 纪律——专用容器、同流 CUDA events、成对引擎 A/B（重型全审计路径 vs 最小视图路径）、同负载同回放、逐阶段宿主计时、资源连续性检查；报告多次重复的分布（非单次点值）。指标：观察 RPC 墙钟、Scheduler 控制窗口占用、逐驱逐 adapter 耗时、请求 wall。**测量完成前不作任何线上收益表述。**

---

## 候选 2：分层观察与验证复用（Tiered observation + validation reuse）

**思路**：不动数据来源，先消除**重复**：(a) 同一静止窗口内 scope/validate 的多次执行合并为"窗口令牌+一次检查"；(b) `validate_plan`/`epoch.validate()` 的全量重验在"不可变 Epoch 身份一致 + generation 未变"时复用首次结果；(c) `preflight` 内部的二次 `bind` 全量重验改为对首次绑定结果的差异核对；(d) 观察分两层——T0 结构层（登记表/结构摘要，纯 CPU）每轮执行；T1 内容层（全量内容哈希/语义对拍）仅审计模式或周期对账执行。

**修改模块（未来授权轮；本轮零改动）**：`flowstate/strict_control.py` 增加"已验证结果缓存"（键：epoch_digest+plan_digest+generation；不可变对象，纯函数语义）；观察入口增加层级开关；其余生产文件零改动。可独立于候选 1 先行（在现重型路径上即可做去重）。

**预期减少的操作**：
- reconcile/apply 流程中 validate_plan·epoch.validate 的多次全量重跑（D2 实测 plan_construction_validation 13.553/13.747 ms、D3 exact_solve_plan_validation 8.875 ms 为"含重验"的现值；复用后重验次数从 O(流程步数) 降为 O(1)，实际节省以测量为准）；
- scope 检查从每 observe 2+ 次降为每窗口 1 次；
- 审计内容层按开关移出决策路径（同候选 1 的 T1）。

**安全风险与对策**：
1. *缓存键设计错误导致陈旧复用* → 键含 generation+三摘要，任一变即失效；负面测试专测"generation 变化后复用必须失效"；
2. *验证次数减少削弱实验保证* → 保留首次与最终两次全量验证不变，只去除中间重复；审计模式可开关回全重复；
3. *层级开关误用（生产误开内容层）* → 开关默认关，开启需显式配置并记录在证据链。

**CPU 负面测试**：缓存失效正确性（epoch/plan/view/generation 任一变化）；篡改不可变对象内部（构造对抗样本）→ 摘要不匹配必须拒绝；层级开关下 T0 与 T1 结果一致性（同 fake 状态两層输出必须相等）。

**GPU 测量计划**：同候选 1 协议；可与候选 1 合并为同一 A/B 矩阵的四臂（重型基线 / +去重 / +最小视图 / +两者），按 D4b 成对纪律分批执行。

---

## 不做的方向（明确排除）

- 不开发完整状态管理框架、不引入后台线程/异步观察服务（保持"Scheduler 线程内静止窗口"语义）；
- 不以删除安全检查换取速度（七类分析中"生产必需"项全部保留）；
- 不在无 GPU 测量情况下宣称任何性能改进数字。
