# G2 最终报告：B5-C4/K2 严格多共享结构 GPU 性能对照

**verdict：G2-A PASS / G2-B PASS**（40/40 正式臂 PASS，失败 0；判定取两阶段全 PASS 才为整体 PASS）

## 1. 研究问题与设计
- 相同严格兼容关系与 K=2 预算下，公共祖先状态（Greedy 选集 {N20,N18}，J=130.8030838298517）替换为两个深层分支状态（Exact 选集 {N18,N27}，J=48.228728759089606），能否稳定减少实际 Prefill 开销与请求 TTFT。
- Exact 臂＝生产 `plan_strict` 门禁内自动（7 候选/4 Pending/K=2 满足门禁）；Greedy 臂＝`greedy_adapter` 复用生产 Marginal Greedy 内部路径（control 断言域满足门禁，未篡改候选/门禁/预算）；两臂同走 validate_plan→Bind/Preflight→reconcile_strict。
- 每臂独立 Docker/GPU 生命周期；G2-B 40 臂按 schedule.json 预固定顺序（奇数轮 Exact 先、偶数轮 Greedy 先），launch.py 强制顺序执行。

## 2. G2-A 门禁（8 臂，preflight/gate/verify.json）
| 臂 | 判定 |
|---|---|
| exact_a | PASS |
| exact_b | PASS |
| exact_c | PASS |
| exact_d | PASS |
| greedy_a | PASS |
| greedy_b | PASS |
| greedy_c | PASS |
| greedy_d | PASS |

- D4b 专项（ΔCUDA Prefill 与 ΔTTFT 方向，±3ms 噪声带）：未发现反向
  - Pending A：Δprefill=-2.44ms Δttft=-2.70ms（噪声带内）
  - Pending B：Δprefill=-0.05ms Δttft=0.43ms（噪声带内）
  - Pending C：Δprefill=34.55ms Δttft=35.41ms
  - Pending D：Δprefill=38.22ms Δttft=38.92ms
- Cold 参照：C 沿用 b5d4b 冻结；A/B/D 由 preflight/cold/ 三臂补采（全新 Engine、空缓存断言、FA命中==E==0）经 promote_cold.py 核验落位。

## 3. G2-B 正式结果（40 臂，formal/）
### 请求级：每 Pending 五轮中位（ms）
| Pending | Exact TTFT 中位 | Greedy TTFT 中位 | ΔTTFT | Exact Prefill 中位 | Greedy Prefill 中位 | ΔPrefill |
|---|---|---|---|---|---|---|
| B5C2_A_TURN_003 | 53.3 | 53.21 | -0.09 | 46.28 | 46.16 | -0.12 |
| B5C2_B_TURN_003 | 53.15 | 53.84 | 0.69 | 46.49 | 46.51 | 0.02 |
| B5C2_C_TURN_003 | 47.48 | 82.18 | 34.7 | 40.57 | 74.34 | 33.77 |
| B5C2_D_TURN_003 | 48.98 | 85.67 | 36.69 | 42.23 | 78.24 | 36.01 |

### 轮级：四 Pending 等权均值 → 五轮中位（ms，正值=Exact 更快）
| 轮 | Exact TTFT | Greedy TTFT | ΔTTFT_r | Exact Prefill | Greedy Prefill | ΔPrefill_r |
|---|---|---|---|---|---|---|
| 1 | 50.66 | 68.4 | 17.74 | 43.48 | 61.14 | 17.66 |
| 2 | 56.66 | 68.19 | 11.53 | 48.47 | 61.3 | 12.82 |
| 3 | 58.86 | 70.26 | 11.4 | 50.25 | 61.74 | 11.5 |
| 4 | 50.21 | 71.1 | 20.89 | 43.63 | 62.26 | 18.63 |
| 5 | 50.64 | 69.78 | 19.13 | 43.89 | 62.24 | 18.35 |

- 五轮中位汇总：{"Exact_ttft": 50.66, "Greedy_ttft": 69.78, "Exact_prefill": 43.89, "Greedy_prefill": 61.74, "delta_ttft_median": 17.74, "delta_prefill_median": 17.66}
- 保留全部有效结果：Exact 单臂 TTFT 更慢的轮次 [('r1_exact_a', 'A', '54.4'), ('r2_exact_a', 'A', '53.3'), ('r2_exact_b', 'B', '68.1'), ('r3_exact_b', 'B', '66.9'), ('r5_exact_a', 'A', '54.1')]（明细见 raw_results.csv）；失败臂：无；重试臂：无
- 结构对照：A/B 两策略 E 相同（3712）为近零差对照；C/D 为结构差异主效应（Exact E=3520 vs Greedy E=2432）。

## 4. 三层结论（不得跨层推导）
1. **模型层**：在线 J 与 e3 冻结参照全精度一致（Exact 48.228729 vs Greedy 130.803084，全部 40 臂核验=True）。
2. **结构层**：Native Gap（G_model=T−E）Exact C/D=[46, 384] vs Greedy C/D=[1134, 1472]；实际 Prefill Token 数 Exact C/D=[915, 966] vs Greedy C/D=[2003, 2054]（Prefill=len(pt)−E）。
3. **实测层**：CUDA Prefill / TTFT 配对差见 §3；方向与结构层一致。
- 第 1/2 层为模型与结构事实，不构成第 3 层的证明；实测层以配对差报告，五轮样本不宣称统计显著性。

## 5. 有效性威胁与限制
- 共享单卡 H100 PCIe、未锁频：clocks.sm/power.draw 逐臂留档（gpu_before/cleanup）；轮内策略顺序均衡以对冲慢漂移。
- CUDA 事件是设备时间线区间而非纯内核之和；控制口径完整 Wall 含审计成本（分段单独报告）。
- Cold 参照与正式臂同 Engine 配置同 Tokenizer 模板；输出逐 Token 一致只证恢复语义等价，不证血缘。
- N18/N20/N27 为逻辑标签，物理身份逐生命周期重建；未跨容器复用 Node/Slot ID。

## 6. 产物清单
- 冻结与门禁：preflight/{prepare,cpu_preflight,stage_inputs,promote_cold}.py、preflight/{prepared_inputs,cpu_gate,source_identity,g2_reference,cold_reference_*}.json、preflight/cold/、preflight/gate/（含 verify.json）
- 正式：formal/{schedule.json,launch.py,driver.sh,analyze.py,raw_results.csv,paired_analysis.csv,checks.json,arm_status.json} + 40 臂 run 目录
- 代码包：pkg_g2/（bridge/runner/control/greedy_adapter/source_contract/persistence + build_record.json，差异全部文档化）
- 顶层：protocol.json、implementation_lock.json、本报告
