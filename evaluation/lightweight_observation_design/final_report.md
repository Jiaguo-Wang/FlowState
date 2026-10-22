# FlowState 轻量级 Runtime 状态观察路径审计——最终报告

**Gate：READY（仅评价审计完整性与设计可行性；不代表轻量观察性能已验证）**

## 1. 本轮做了什么

只读审计了 FlowState 当前 Runtime 观察与严格状态认证的耗时来源，并提出可实现的轻量在线状态视图设计。未运行 GPU、未修改任何生产或实验代码、未触碰 B5-D4b 进行中的代码与产物。

证据基础（全部既有原始产物与源码，本轮零新增运行）：

- B5-D2 Shadow Planning：`exact_runtime_shadow_gate_b5d2/final_report.md`+`shadow_timing.csv`+`shadow.py`+`runner.py`——观察 RPC 1102.325 ms、Epoch 构建认证 110.831 ms、plan 构建验证 13.553/13.747 ms 等单次分段，及"CPU 规划在控制窗口外"的结构事实；
- B5-D3 真实 Reconciliation：`exact_reconciliation_correctness_gate_b5d3/control_overhead_breakdown.csv` 全 23 行（含自证 category 归属）+`control.py`——重型观察 1203.035、apply-time 重验 551.672、协调含 postcheck 170.588、FA 内容审计 153.654+288.041、逐项 Adapter 驱逐 1.0–1.4 ms；
- 生产源码：`flowstate/strict_epoch.py`、`flowstate/strict_control.py`、`flowstate/controller.py`、`flowstate/adapters/sglang.py`；
- 观察实现链：`multishare_runtime_gate_b5c3/bridge.py`、`runtime_control_closure_r2c0/bridge.py`、`strict_compatibility_b1c/runner.py`、`barrier_fa_frontier_control.py`、`motivation/artifacts/wp3b_gate_20260820/targeted_probe.py`、`tests/runtime/wp3b_end_to_end_transport.py`；
- B5-D4b 纪律参照：`exact_greedy_paired_gpu_gate_b5d4b/experiment_protocol.md` 与 D4a `control.py`/`observation.py`/`bridge.py`（只读）。

## 2. 核心结论

1. **成本归属清晰可分**：观察路径的 dominating 成本是实验审计性质的操作——全树 FA 内容哈希（一次 final 观察 4 次、B5-D3 协调全程约 26 次，乘数为代码结构事实）、逐槽 Recurrent 内容哈希、before/after 双观察深比对、巨型 census JSON 序列化传输（MB 级）、Journal 逐阶段写盘与独立 Oracle。这些不是分配决策的输入：决策只用元数据、驻留标志与 FA 前沿。
2. **不可移出的部分同样明确**：静止性/范围检查、槽位记账闭合与别名检查、bind/preflight/postcheck、apply 时逐选中项物理重验、逐项驱逐的精确释放范围断言（实测 1.0–1.4 ms/项）——这是严格合同与安全的本体，设计予以全量保留。
3. **七类逐项分析完成**（critical_path_analysis.md）：每类均给出代码位置、数据来源、GPU 同步需求、是否 Scheduler 关键路径、是否生产必需、独立耗时证据有无、可否移出与新鲜度替代机制。观察子步无独立计时，本审计不以估计拆分实测值。
4. **最小状态视图合同成立**（minimal_state_view_contract.md）：8 类字段按"runtime 维护 / 决策边界读取 / 审计数据"划分责任；generation+身份摘要+定点重验作为新鲜度载体替代内容对拍；并发修改/驱逐/COW/Radix 变化的处理与防错误绑定均有既有点位可挂接（与现有 instrumentation 同点，不改 SGLang 磁盘源码）；FA-KV 观察语义完整保留、控制仍仅限 Recurrent State。
5. **两个优化候选均可实现且风险可控**（optimization_candidates.md）：候选 1（增量最小视图）移除决策路径上的全部内容哈希与全量 census；候选 2（分层观察+验证复用）消除重复验证，可独立先行。二者均有成套 CPU 负面测试设计与 GPU 测量计划（复用 B5-D4b 纪律，四臂矩阵，报分布不报点值）。

## 3. 边界与不主张

- 所有毫秒数为各自单次实验实测值，不是 P95、不是线上延迟；D2 与 D3 不可互相加减得出净收益。
- 轻量路径的实际耗时、重复分布、TTFT 影响：**没有任何实测数据**；本轮不通过相减或外推生成数字，不把"移除审计后的理论耗时"称作已验证收益。
- 设计可行性基于代码结构事实（挂接点存在、合同字段可推导、验证语义可复用），未经任何运行验证；R3-A/R3-B 原型与对拍通过前，不应视作可实现性已证明。
- 本轮不实现、不上线、不运行 GPU 测试；完成后停止，等待审查。

## 4. 交付物清单

| 文件 | 内容 |
|---|---|
| observation_cost_breakdown.md | 实测计时总表（D2/D3 原样引用）+观察路径代码级分解（乘数结构）+归属结论+数据缺失声明 |
| critical_path_analysis.md | 七类逐项分析（统一八栏目）+汇总表 |
| minimal_state_view_contract.md | 语义红线、8 字段责任划分、决策时序、并发/驱逐/COW/Radix 处理、防错误绑定、与重型路径的对拍关系 |
| optimization_candidates.md | 候选 1（增量最小视图）与候选 2（分层观察+验证复用）：模块、预期移除操作、风险、CPU 负面测试、GPU 测量计划；明确排除方向 |
| implementation_and_validation_plan.md | R3-A/B/C/D 分阶段计划与门禁（CPU 原型→对拍→复用→GPU 测量），均在新目录 |
| final_report.md | 本文件 |
