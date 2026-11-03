# T1 数据来源与冻结身份

审计日期：2026-10-26。仅补充离线统计；未运行策略实验或 GPU。审计门禁与历史 Runtime 认证门禁不是同一个概念。

## 来源及口径

| 来源 | 实际行数/条件 | 身份与用途 |
|---|---:|---|
| P1 `per_case_results.jsonl` | 573 | OpenHands 504、AgentX 69；包含强基线及明确收益矩阵，不依据 Workflow 标签重造关系 |
| 原 OpenHands `per_snapshot_results.jsonl` | 504 | 按 P1 source_file/source_line 核对 snapshot_id、k、snapshot_digest、FlowState 选集与成本 |
| 原 AgentX `per_case_results.jsonl` | 69 | 同上；非 Strict Epoch |
| Gate1 `matrices.jsonl` | 573 | 独立交叉核对每格收益、等大小成本与预算 |
| E3 `condition_results.csv` | 140 | 5 个快照 × 4 个预算 × 7 策略；CSV 行号包含表头，JSONL 从 1 开始 |
| E3 `selection_traces.json` | 50 | 各预算逐轮轨迹；5 个 K4 路径共20步，较小 K 重复其前缀 |
| E2 w1–w4 / B5-C3 Strict Epoch | 5 | 每个7候选、4 Pending、28三态关系；逐准入边核验原始Token前缀、Epoch和配置摘要 |
| G2 `formal/raw_results.csv` | 40 | 单个 B5-C4/K2；5轮 × 4 Pending × 2策略；只读对应，不计入 Strict20 CPU 条件 |

## 校验与版本

- 四份冻结清单共检查 200 个文件条目，SHA-256 全部吻合；未发现缺失或版本不一致。完整 expected/actual 见 [verification.json](verification.json)。
- 本次 8946 项断言全部通过；包括逐格收益、逐 Pending E/G/J、冻结边际轨迹、原始行身份、K_min 独立证书与读取前后 SHA 一致性。
- Phi 使用冻结 `position_aware_quadratic_v1`，系数 a=37.82815、b=0.345974143、c=−0.156201917；G/T 先除1024，输出 modeled ms。容差为绝对 1e−9 ms。
- 本次不导入当前生产 Optimizer，不因当前代码变化重算策略选集。历史策略版本由冻结清单认证；E3 首改善 1-Swap 规则来自其 protocol.json。
- 先前存在的 `t1_extract.py` 未作为本次统计入口或证据使用；本次入口为 [audit_frozen.py](audit_frozen.py) 与 [write_reports.py](write_reports.py)。

## 不可混同的身份

- 历史573：只证明给定 Formal 矩阵上的代数性质与策略结果。真实 Token、完整同边界状态认证缺失，Strict 支持、Strict K_min、实际 Runtime 收益均为 UNKNOWN。本表给出的历史结构数值明确是 Formal 值。
- AgentX 既有 Gate1 Step3 审计记录70条 Formal-only 新增关系（对应该次原始/形式化域比较，不能乘以预算当成独立边）；不能把历史 Formal 最优性作为真实可恢复性。
- Strict20：5 个受控来源快照，20 个预算条件不是20个独立工作流；EPOCH_ADMISSIBLE 不意味着逐边 RESUME_VALIDATED。所有Node编号均为各快照内部身份。
- K 为等大小 Recurrent Checkpoint 逻辑单位。E2冻结预算合同要求独占等大小槽、无别名；B5-C4记录每项51,511,296 bytes。原采集Epoch的 logical_budget="未定义；本轮不分配"，后续CPU协议明确K=1..4，不反向修改采集记录。

## 源文件索引与内容摘要

| 文件 | SHA-256 | 字节数 |
|---|---|---:|
| [evidence_manifest.json](/home/wjg/code/FlowState/evaluation/independent_runtime_allocation_e2/evidence_manifest.json) | `2b14e2604cecec71ef12207cbf151d3db3760b0bd744b84e697080af10666b6d` | 14040 |
| [evidence_manifest.json](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/evidence_manifest.json) | `233877acc8a82c9034648ea9e38c1cb3f29a609fadc765d4df2ca1cbfb4eef33` | 2276 |
| [artifact_manifest.json](/home/wjg/code/FlowState/evaluation/p1_strong_baseline_validation/artifact_manifest.json) | `f96bf7a6cfe5d4bbb2c5547c6b4efee0a5b5677847ae2a4ec1eee449bc6a67c1` | 1954 |
| [evidence_manifest.json](/home/wjg/code/FlowState/evaluation/multishare_allocation_gate_b5c4/evidence_manifest.json) | `1ee284e4a2f5dbf0f31902a5e6e8befa3aeffd9e9f605cf612ef256637d1f815` | 1286 |
| [per_case_results.jsonl](/home/wjg/code/FlowState/evaluation/p1_strong_baseline_validation/per_case_results.jsonl) | `cf010df47b4007b6c93f457d4be1905ba6aa89ebd7e65c3321aabf0fcf294116` | 12864443 |
| [matrices.jsonl](/home/wjg/code/FlowState/evaluation/gate1_step1_allocation_equivalence/matrices.jsonl) | `9aa81ef02257a01f45a9aedc1357a48b98d601ab13cc80e6ba76e9b9520a64e6` | 1081858 |
| [per_snapshot_results.jsonl](/home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl) | `3008226b49fb0e27265fac77204ab43ede2c6470c1b5e529eb76e4da888f2dbf` | 13144283 |
| [per_case_results.jsonl](/home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl) | `a22e18fa42cdcaa8490ea98230c70398d98df877442f843cd5a1d9dc3bb14866` | 3273971 |
| [condition_results.csv](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/condition_results.csv) | `195b2ed1bbde3609a95209da8d7d4cff326b7e56185e3698667fb5054ee054f9` | 81765 |
| [selection_traces.json](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/selection_traces.json) | `4a334ecda6058c50700a5759dac8a435561bf0d2410f29d8b7b8b3a5b7c3073a` | 97693 |
| [swap_traces.json](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/swap_traces.json) | `7267ff16df8372027a40fdc119bdc0e3224a0a60ff7ebe397b8d42c7a4d65459` | 14488 |
| [local_trap_diagnostics.json](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/local_trap_diagnostics.json) | `51fa49a7a83472fb36ed56607216f0b801d6a6591fbbc38ff436c949f4e1f2d1` | 1685 |
| [results.json](/home/wjg/code/FlowState/evaluation/multishare_allocation_gate_b5c4/results.json) | `3ec9f90ce457fc8a0eb7f4ec09a8474ce3041e17477f03a2f33881bd864a390f` | 105352 |
| [strict_epoch_snapshot.json](/home/wjg/code/FlowState/evaluation/independent_runtime_allocation_e2/w1/strict_epoch_snapshot.json) | `970cd9680c9c7f79430c51e85d7657ed1644ae62557778c5b3a52f853754b246` | 617163 |
| [strict_epoch_snapshot.json](/home/wjg/code/FlowState/evaluation/independent_runtime_allocation_e2/w2/strict_epoch_snapshot.json) | `ec63f3c064c4808ac92618507256d9ed3ba6c5197ec1c3543d5b47dad7e1cff6` | 605231 |
| [strict_epoch_snapshot.json](/home/wjg/code/FlowState/evaluation/independent_runtime_allocation_e2/w3/strict_epoch_snapshot.json) | `d4de3c0cce210e5afc61e135a07a694e2de6ce5e0becf7afb91c2228004b6eac` | 597131 |
| [strict_epoch_snapshot.json](/home/wjg/code/FlowState/evaluation/independent_runtime_allocation_e2/w4/strict_epoch_snapshot.json) | `a2013b190f6575acd5a90797bd61c865941d95fbcdd44b87f0d2bf93110b0f5c` | 655353 |
| [strict_epoch_snapshot.json](/home/wjg/code/FlowState/evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json) | `3b41f232b4ee4d79aab081643c6c2adad1671d0364841617e849f52b6c16ac7a` | 629912 |
| [raw_results.csv](/home/wjg/code/FlowState/evaluation/b5c4_formal_gpu_g2/formal/raw_results.csv) | `fd45b2d028fb9df6252bb4df4e8004871899ed5fc7dae1cc03885956d6055eac` | 11041 |
| [final_report.md](/home/wjg/code/FlowState/evaluation/b5c4_formal_gpu_g2/final_report.md) | `7f0dd5cb1dd46d12113a1a6783f76835dc4a89ef6cea4e792cb3f9c4b10b8cb8` | 4993 |
| [protocol.json](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/protocol.json) | `5a7a87cdbe24a8e7e1452af3e8fcae82905995ac5a098f269141604859569f4e` | 1073 |
| [greedy_wfc_rule_audit.md](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/greedy_wfc_rule_audit.md) | `c30d329f839780958925ed85251437779c8a468ba0cd05767a8f7ebdf0869904` | 1666 |
| [audit.py](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/audit.py) | `1fcbc090675d4a796114219aac90be4a67a559d50f443ad17c7dbce7089bd792` | 8068 |
| [diagnostics.py](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/diagnostics.py) | `595afa88e38bfa7b937c2d594d5f4b761a4cbff65f19140522bb542196e465f1` | 4182 |
| [local_optimum_neighbors.csv](/home/wjg/code/FlowState/evaluation/greedy_swap_structure_e3_step1/local_optimum_neighbors.csv) | `1751a6a34f1bfa41c7af4019440fabd2fae92a4ca05d6dd23cf6a10270b979fd` | 3077 |
| [measurement_and_validity.md](/home/wjg/code/FlowState/evaluation/independent_runtime_allocation_e2/measurement_and_validity.md) | `aca64c6c11fe4706afc98b3b2b2a4a4d13315f96cb218e8104d1c10fd44daa87` | 2225 |
| [summary.json](/home/wjg/code/FlowState/evaluation/gate1_step3_compatibility_retention_contract_audit/summary.json) | `8b8e3931660888df05fa26ec8a261aecd89873404f5eb3fb7c0d97b04bd9ed81` | 12809 |
| [audit_report.md](/home/wjg/code/FlowState/evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md) | `105db52eabc0d73c1b07c7d50bd574410673787564c2ac0f26274105dab89598` | 17284 |
