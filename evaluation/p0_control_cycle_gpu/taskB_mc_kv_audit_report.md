# P0-4 任务 B · Marconi-Evict 与 KVFlow-inspired 完整控制周期 GPU 对照审计报告

日期:2026-10-30 | 实验包:`evaluation/p0_control_cycle_gpu/` | 本轮臂:r1_ck3_mc、r1_ck3_kv
参照臂:r1_ck3_fs(FlowState-Greedy attempt5,已冻结,本轮零改动、零重跑)

## 1. 两臂退出状态与门禁

**执行序**:单一 launcher(`launch Marconi-Evict,KVFlow-inspired`)串行 fail-loud——MC 无 summary.json 则 KV 不启动;实际 MC 先 PASS,KV 后 PASS,满足"Marconi 完整 PASS 后才启动 KVFlow-inspired"。

| 门禁 | Marconi-Evict (r1_ck3_mc) | KVFlow-inspired (r1_ck3_kv) |
|---|---|---|
| 退出码 / 结论文件 | rc=0;run/summary.json 唯一,**零 failure 文件** | rc=0;run/summary.json 唯一,**零 failure 文件** |
| status / oracle_match / identity_exact | PASS / True / True | PASS / True / True |
| 四 Pending 完成 | C_A..C_D 各 8 Token、cold_output_match 4/4、residuals 末项=0 | 同左 4/4 |
| E/G/Prefill/COW/FA 物理合同 | E=1024×4;prefill=T−E+32(3104/4128/5152/6176);batch_prefix=1024=E;FA 命中=T;census 16=16 无 unmapped | E=4032/2048/0/0;prefill=96/3104/6176/7200;batch_prefix=E;FA 命中=T;census 16=16 |
| admission==E 逐请求 | ✅(request_plan 期望 E/T/G 与实际逐项相等) | ✅ |
| 测量模式 | audit_entries_total=2;请求窗口 triton 新编译=0 | 同左 |
| T0–T4 恒等式 | identity_exact=true(整数 ns) | 同左 |
| Oracle 三检(runner + 本报告独立复核) | 选集/J 逐位/E·T·G 逐 Pending 相等 | 同左 |
| Journal 口径 | 窗内仅同步序列化 247.2ms;写盘+fsync(935.7+387.5ms)全在 T3 后 | 窗内 257.2ms;出窗 914.5+384.5ms |
| 源码身份 | after_attempt4 登记 sha_after == 当前 runner.py/observation.py(完整哈希);容器挂载仓库即运行该版本 | 同左 |
| GPU / 盘 / 墙钟 | GPU-bfff4d31(与 FS 同卡);wall 180.2s;盘 67→66G | 同卡;wall 120.1s;盘 66→65G |

**形状预热不污染恢复条件(专项核查)**:预热块后状态满足物理驻留与恢复合同——逐请求 admission==E 通过、physical_fa_hit=T(可执行前缀完整)、post_request.ok=true(请求后 selected_payload/FA 保护不变)、census 16=16、请求窗口 triton 新编译=0(形状已预热且未引入新物化)。

**登记事项(非失败)**:`arms_status.json` 为单次 launcher 调用的账簿,本次内容为 MC/KV 两臂,FS 条目因此不在该文件中;FS attempt5 权威证据(r1_ck3_fs/run/summary.json、r1_ck3_fs_cleanup.json、launch 日志)完好未动。

## 2. 三策略真实调用链与输入可见性

| 策略 | 在线选择入口(control.py 分派) | 内核 | 后续物化路径 |
|---|---|---|---|
| FlowState-Greedy | `select_flowstate({'epoch'})` | 生产 `GlobalOptimizer.plan_strict`(域内 Marginal Greedy) | 生产 apply 直连 |
| Marconi-Evict | `select_marconi({'case': case})` | `evaluation.marconi_evict_t2m.evict_policy.MarconiEvictPolicy.select`(真实字节/recency/父相对增量+evict_v2 动态驱逐) | census cpu→phys 映射 → 生产 `planner.finish` |
| KVFlow-inspired | `select_kvflow({'case': case})` | `evaluation.kvflow_inspired_retention_cpu.policy.select` | 同上 |

- 输入可见性:三策略在 T0 只见各自的 epoch/case 快照(真实状态观察产物),**无任何未来轨迹信息**;`t2g1_plans.json`/冻结选集仅由 runner 在 T4 之后读取(对照用,不进控制路径)。
- runner.py 零策略行为分叉:POLICY 仅用于 env 校验、结果断言、request_plan/预热块记录、形状预热 seed(各策略按自身 E 向量预热)、T4 后 oracle 对照、summary/打印。
- 公平性:同模型、同物理 GPU(三臂同为 GPU-bfff4d31)、同 16 候选、同预算 K=3、同四 Pending 与顺序、同重置与测量窗口;差异仅保留集(各策略规则产物)+ 共同物理合同把守。

## 3. 控制周期分段计时对照(ms,单轮,双口径禁混用)

| 分段 | FS(attempt5) | MC | KV |
|---|---|---|---|
| final_heavy_observation | 3084.6 | 3133.9 | 3119.6 |
| cpu_physical_mapping | 0.6 | 0.65 | 0.63 |
| strict_epoch_authentication | 150.3 | 168.1 | 154.3 |
| policy_online_select(整段) | 31.9 | 35.5 | 41.9 |
| —其中内核调用 | 31.9(plan_strict) | 2.4 | 4.9 |
| apply_time_freshness_preflight | 1560.3 | 1595.7 | 1583.6 |
| fa_postcheck & resume_baseline | 1880.5 | 1349.0 | 1417.9 |
| 六段合计 | 6708.2 | 6282.8 | 6317.9 |
| 段间隙合计(含 Journal 窗内序列化) | 3174.9 | 3240.9 | 3237.0 |
| **control_cycle_ms(T3−T0)** | **9883.1** | **9523.7** | **9555.0** |
| RPC 尾段(rpc_tail) | 4280.7 | 4031.2 | 3786.4 |
| 形状预热块合计 | 7984.8 | 7388.8(C_A 首编译 6720.5) | 7346.7(C_A 首编译 6643.3) |
| request_exec_total | 756.3 | 1252.8 | 1184.7 |
| 窗口首隙(rpc→warm) | 1365.7 | 954.8 | 964.8 |
| **cycle_ms_full_incl_warm(T4−T0)** | **24273.2** | **23159.5** | **22848.1** |
| triton 新编译(五窗口) | [41,0,0,9,0] | [41,0,0,8,0] | [41,0,0,8,0] |

三臂 control_cycle_ms 差 <4%(9883/9524/9555),其中五段共用控制段量级一致,差异集中在策略选择段与 fa_postcheck 段;单轮样本,不做统计推断(重观察段已知存在 20–30% 运行间波动)。

## 4. 四 Pending TTFT / Prefill 对照(单轮)

| Pending | FS: E / TTFT / cuda_prefill / wall | MC: E / TTFT / cuda_prefill / wall | KV: E / TTFT / cuda_prefill / wall |
|---|---|---|---|
| C_A (T=4096) | 0 / 159.5 / 152.0 / 283.0 | 1024 / 123.8 / 116.2 / 253.3 | 4032 / **38.2** / 31.2 / 159.8 |
| C_B (T=5120) | 5120 / **32.9** / 27.9 / 156.3 | 1024 / 161.3 / 153.9 / 289.2 | 2048 / 124.2 / 116.4 / 256.0 |
| C_C (T=6144) | 6140→6144 / **32.3** / 26.7 / 154.6 | 1024 / 205.2 / 196.6 / 335.7 | 0 / 240.6 / 231.4 / 363.1 |
| C_D (T=7168) | 7168 / **32.0** / 26.9 / 157.7 | 1024 / 242.8 / 233.7 / 370.4 | 0 / 279.1 / 269.4 / 401.0 |

- 汇总:ΣTTFT = FS 256.7 / MC 733.0 / KV 682.1 ms;Σcuda_prefill = 233.5 / 700.4 / 648.5 ms。
- TTFT 分解(queue/pregap/execwin/postgap/delivery)三臂同构:queue 2.5–3.4、pregap 0.9–3.0、postgap ≈0.1–0.3、delivery 2.6–3.1、首 token decode CUDA ≈17ms;TTFT 差异几乎全部由 cuda_prefill(=恢复量)驱动。
- COW:MC 每请求 4 次(共享 C_ROOT/C_JOIN 驻留,每次 1.16ms);KV C_A/C_B 各 1 次(0.30/0.27ms)、E=0 者零 COW;FS 为 0/1/1/1(0.25–0.26ms)。
- 逐 token 解码 stream TPOT 三臂一致(17.3–18.8 ms/token)。

## 5. 开销归属(共用 / 策略特有 / 实验取证 / 请求执行 / 未归属)

- **策略计算开销(控制周期内)**:online select 段 FS 31.9 / MC 35.5 / KV 41.9 ms,其中内核调用 31.9(优化求解)/ 2.4(驱逐启发式)/ 4.9(评分启发式)——MC/KV 内核便宜但包装(映射+finish+决策记录)摊薄后整段与 FS 同量级。
- **策略后果开销(请求/预热域,不计入控制周期)**:E 向量差异驱动的 prefill/COW/预热形状差异(§4、预热块三臂 7984.8/7388.8/7346.7 ms)。
- **共用控制开销**:观察 3084.6/3133.9/3119.6、映射 0.6/0.65/0.63、认证 150.3/168.1/154.3、preflight 1560.3/1595.7/1583.6、postcheck 1880.5/1349.0/1417.9,加段间隙 3174.9/3240.9/3237.0(含 Journal 窗内序列化 239.0/247.2/257.2)。
- **实验取证开销(全在测量窗外)**:Journal 出窗序列化+写盘 1285.9/1323.2/1299.0 ms(计入 RPC 尾段)、audit×2、triton 五窗口扫描、oracle 复核 0.15/0.14 ms、cold reference 重放。
- **请求执行开销**:756.3/1252.8/1184.7 ms(§4 明细)。
- **未归属**:rpc_tail 4280.7/4031.2/3786.4 ms(apply RPC 墙钟−控制周期,含交接与服务端;前轮审计已定位取证 RPC 占主导)与窗口首隙 1365.7/954.8/964.8 ms。

## 6. 三策略选择结果与模型化 J(对照冻结 oracle 逐位一致)

| 策略 | 在线选集(K=3) | 模型化 J(ms) | Σ(T−E) tok | E 向量 | Σ实测 cuda_prefill(ms) |
|---|---|---|---|---|---|
| FlowState-Greedy | C_B_P05120, C_C_P06144, C_D_P07168 | 154.34895561599998 | 4096 | 0/5120/6144/7168 | 233.5 |
| Marconi-Evict | C_JOIN_P04032, C_JOIN_P04096, C_ROOT_P01024 | 703.45464601 | 18432 | 1024×4 | 700.4 |
| KVFlow-inspired | C_AB_P01984, C_AB_P02048, C_A_P04032 | 627.6149768490118 | 16448 | 4032/2048/0/0 | 648.5 |

- MC/KV 实测 prefill 合计与模型化 J 偏差 −0.4% / +3.3%;FS 差额来自每请求固定 32-token prefill 地板(≈27ms×3)。单轮,仅报告方向与量级。
- **结构性记录(serial_safety worst_case 注册表,与计时无关)**:在"假想中间物化"最坏情形下(lcp≤E_later 才安全),FS 的 E 向量对全部请求对安全,MC 有 2 对不安全(C_A→C_B、C_C→C_D),KV 有 5 对不安全(E=0 各对);实际引擎语义为终端 donate(物化点≥T_i>两两 LCP),实际执行序 terminal_checks 全安全,并由逐请求 admission==E 门禁实测把守。该注册表为 FS 选集的次序鲁棒余量提供结构性证据。

## 7. 五轮正式矩阵启动条件(本轮不启动)

1. 三臂首轮全 PASS,协议/冻结输入/身份链(runner `b6cee2d3…`、observation `7c1dcefc…`)已冻结,launcher 串行入口可用;
2. 每轮×每臂须由 prepare 重新装配新臂目录(冻结输入+identity;不再复用 r1_* 目录),GPU 空闲窗口实查后发射(单卡串行,每臂 2–4 分钟,三臂五轮约 30–60 分钟纯运行);盘余 65G 充足(每臂约 1G);
3. 统计口径预先声明:逐轮全指标落盘,五轮后按配对中位数陈述集中趋势;已知单轮重观察段 20–30% 波动,不得以单轮差值下结论;
4. 账簿改进建议(只建议,不实施):arms_status.json 应改追加式或逐轮归档 launch 日志,避免单次调用覆盖历史臂条目。

## 8. 最终裁决:PASS

- 两臂均达成三态判定的最高级"单臂完整 GPU 门禁通过":rc=0、结论文件唯一、九类门禁全过、oracle 三检逐位、证据链完整;
- 三策略首轮(1 轮 × 3 策略)全部成立:FS( attempt5,已冻结) + MC + KV(本轮);
- 未验证事项(如实):五轮重复性与配对统计(未启动,见 §7);单轮数值不外推;
- 失败项:无。

**已停止:不启动五轮矩阵;不改核心 Runtime/Solver/Controller/冻结数据/论文正文;不做性能优化。**
