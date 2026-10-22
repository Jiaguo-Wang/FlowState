# 实现与验证计划（本轮不实现；供审查后授权）

总原则：CPU 原型先行、只写新实验目录、不改生产源码（`flowstate/`、既有 evaluation 目录、B5-D4b 进行中的代码与产物一律不动）；GPU 阶段需单独授权并复用 B5-D4b 纪律；每阶段设明确门禁与停止点。

## R3-A：CPU 原型——最小视图维护器 + 负面测试（新目录 `evaluation/minimal_view_prototype_r3a/`）

范围：
1. Fake cache 对象实现 SGLang 统一缓存的可观测面（`tree_core` arena/父子/key、component_data 双组件、LRU、mamba 池槽、allocator 计数、ongoing 标志），只覆盖已审计的 0.5.17 接口子集（sglang.py `validate_runtime_capabilities` 所列能力清单）；
2. 视图维护器：在 fake 的 insert/split/merge、`evict_component`（单组件+叶集合更新）、`copy_from`（COW）、`_free_values` 挂点更新登记表并 bump `mutation_generation`；
3. 轻量观察入口：返回合同字段 1–7 的 KB 级视图 + epoch 身份；
4. 严格边推导与 `StrictAllocationEpoch`/`validate_plan`/`bind`/`preflight`/`postcheck` 在小视图上的适配层（语义不变，输入格式对齐现快照）。

测试（unittest，`/tmp/flowstate-test-venv/bin/python` 或 uv312 运行）：
- 正向：固定 8 节点 fake 树上构造 4 候选/4 Pending/16 边，全流程（观察→epoch→规划→绑定→preflight）通过，与手工期望一致；
- 负面清单（optimization_candidates.md §候选1 所列 6 项逐一落实为断言）；
- 性质测试（随机变更序列）：任意合法变更序列后，登记表与"全量重算 census"必须一致（对账不变式）。

门禁 R3-A：全部测试通过；负面测试每项独立可复现；未触碰生产文件（git status 核对）。

## R3-B：对拍门禁——轻量视图 vs 重型路径（仍在 CPU）

同一 fake 状态上运行：现重型观察逻辑（提取自 b5c3/r2c0/b1c 路径，指到 fake 对象）与轻量视图，断言候选集、逐边三态、每 Pending T、槽位记账、引用状态**逐项相等**；随机变更序列下持续相等。任何不一致即 FAIL 并阻塞后续阶段。此门禁是"轻量化不改变决策语义"的唯一证据。

## R3-C：验证复用（候选 2 独立部分，CPU）

在现重型 CPU 路径上实现"已验证结果缓存"（键=epoch_digest+plan_digest+generation），测试覆盖缓存失效与对抗样本；报告去重前后 `validate_plan` 调用次数对比（调用次数是代码事实，可声称；耗时变化不声称）。

## R3-D：GPU 测量（需单独授权；本轮不运行）

协议继承 B5-D4b：专用容器、≤11 引擎配对、同流 CUDA events、逐阶段宿主计时、产物逐阶段 fsync 仅用于实验证据、GPU 清理与连续性检查。四臂矩阵：重型基线 / +验证复用 / +最小视图 / +两者；每臂同负载重复 ≥10 次，报告分布（中位数/P90/最大值），不报单次点值。指标：观察 RPC 墙钟、控制窗口占用、逐驱逐 adapter 耗时、请求 wall、TTFT（仅当请求级配对成立时）。

停止点：R3-D 报告只陈述实测分布；任何"线上收益/P95/净改进"结论必须以该报告为准，此前一律不写。

## 交付与审查

- 每阶段产物留在各自新目录（`evaluation/minimal_view_prototype_r3a/` 等），含测试日志、负向用例清单与结果、对拍结果；
- 本目录（lightweight_observation_design/）六份文件为本轮全部交付，不随 R3 阶段自动更新；
- 本轮完成后停止，等待审查。
