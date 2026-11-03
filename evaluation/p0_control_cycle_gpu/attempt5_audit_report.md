# P0-4 第一轮 GPU 验证 · Attempt 5 装配、门禁与运行审计报告

日期：2026-10-29/30 | 实验包：`evaluation/p0_control_cycle_gpu/` | 策略臂：r1_ck3_fs（FlowState-Greedy）

## 0. 最终结论：PASS（单臂完整 GPU 门禁通过——干净、唯一的结论文件）

| 阶段 | 状态 |
|---|---|
| 一、安全装配 | PASS（含一次记录层缺陷修复，如实归档） |
| 二、CPU 预检（6 项） | PASS 6/6 |
| 三、GPU 发射前门禁 | 首次 FAIL（两卡被他人 VLLM 占用，按规则停止）；2026-10-30 复核 PASS 后发射 |
| 四、正式 GPU 实验（attempt5） | **PASS：rc=0、summary 唯一落盘、无任何 failure 文件、全部门禁通过** |

## 1. Attempt 5 装配结果及目录位置

- 臂目录：`evaluation/p0_control_cycle_gpu/arms/r1_ck3_fs/`；装配记录：`arms/retry_record_r1_ck3_fs_attempt5.json`（attempt=5，archived_dir 注明"无：干净起点装配"）
- 编号推导：archived_runs=4（failed×3+passed×1）+pending_ran=0+1=**5**，与运行序列 1→4 正确衔接
- 装配缺陷披露：首次装配后写 retry_record 时 `archive` 未初始化（臂目录不存在分支）抛 UnboundLocalError；修复 `archive=None` 语义后重新装配。缺陷装配（未运行）归档为 `discarded_assembly_r1_ck3_fs_…_record_write_bug`，零覆盖

## 2. 新旧 Runner 身份及 MC/KV 登记结果（完整 sha256）

| 项 | 值 |
|---|---|
| attempt5 运行版 runner.py | `b6cee2d3ffa04e8f85e9658bf166a37cfb063ad21983c9488b37fb8b4eaf9f84` |
| 臂 identity 记录 == 当前文件（发射前后两次复核） | 逐字节相等 PASS |
| attempt4 运行版 runner（passed_print_bug 归档） | `7a09a5ba3649b3c3…`（差异=两处收尾修复：打印行字段名、except 结论保护） |
| mc/kv `identity_update_after_attempt4_fix.json` | 两臂均登记 runner/observation 完整 before→after 哈希，after 与当前源一致 |

## 3. CPU 预检逐项结果（6/6 PASS）

1. 全包 compileall PASS
2. identity attempt==5、fix_note/serial_safety 完整 PASS
3. runner/observation 完整哈希三方一致（臂 identity==当前源==mc/kv 登记）PASS
4. 冻结输入 8 文件与 g1_CB_FS 逐字节一致、t2g1_plans 与源一致、历史归档 9 目录在位 PASS
5. 正式入口链（launch.py 冻结镜像→device_entry 单卡校验→runner）、输出目录干净 PASS
6. 打印/oracle 键位精确核查（孤立 `['cycle_ms']`、`frozen['recovery_cost_ms']` 零匹配；打印↔构造键一一对应）PASS（初检 FAIL 为检查脚本对 plan 侧合法字段的子串误报，已澄清）

## 4. 发射前 GPU 状态

- 2026-10-29 16:1x：两卡被他人 VLLM 任务占用（73G/52G+5.5G，3 进程）→ 按规则停止发射（不杀他人进程）
- 2026-10-30 发射前复核：两卡 14/17 MiB、0 计算进程、盘 67G、无残留容器/launcher；预检第 3/4 项重跑一致（身份/冻结输入/输出干净、attempt=5）→ 发射

## 5. 实际启动的实验臂与运行记录

- 唯一臂：FlowState-Greedy（attempt5）。launcher pid 2728717，容器 `flowstate-p04-r1_ck3_fs`（GPU-bfff4d31…，冻结镜像 `sha256:1dfce903…`、--pull never），启动参数存 `arms/r1_ck3_fs_command.json`
- 原始日志：`arms/r1_ck3_fs.runtime.log`；输出 `arms/r1_ck3_fs/run/`（summary/四请求 output/流式 chunks/reconciliation/shape_warm_block/post_request 等 17 类文件 + rpc 存档）
- 退出：`r1_ck3_fs_cleanup.json` **returncode=0、status=PASS**、wall 240.2s、盘 delta 0

## 6. 运行门禁逐项核验（全部通过）

| 门禁 | 结果 |
|---|---|
| 四 Pending 完成 | ✅ C_A..C_D 各 8 Token、零回撤（residuals 末项=0） |
| Cold Reference 逐 Token | ✅ cold_output_match 4/4 |
| E/G/Prefill/COW/FA/物理映射 | ✅ C_A E=0：prefill 4128=全长、batch_prefix=0、FA 命中 4096=T、无 COW；C_B/C_C/C_D E=5120/6144/7168：prefill=32、batch_prefix=E、FA 命中=T、COW 各 1 次（0.25–0.26ms）；census 16=16 无 unmapped |
| T0–T4 恒等式 | ✅ identity_exact=true（整数 ns） |
| Oracle 三检 | ✅ oracle_match=true：选集 [C_B_P05120,C_C_P06144,C_D_P07168]、J=154.34895561599998 逐位、E/T/G 逐 Pending 相等（本报告独立复算 reconciliation↔t2g1_plans 亦一致） |
| 测量模式 | ✅ audit_entries_total=2（apply 基线+请求后）；请求窗口 triton 新编译=0（形状预热生效） |
| Journal 口径 | ✅ 窗内仅同步序列化 239.0ms（计入段间隙）；写盘+fsync（908.2+377.7ms）全部出窗 |
| 退出与结论唯一性 | ✅ rc=0；summary 唯一 PASS；**无 failure.json / failure_after_summary.json** |
| 源码身份一致 | ✅ 发射前后两次复核 identity==当前文件；运行期间零改动 |

## 7. 计时结果（双口径并列，禁止混用）

- **control_cycle_ms（T3−T0，不含预热/请求）= 9883.07**：final_heavy_observation 3084.6 + cpu_mapping 0.6 + epoch 认证 150.3 + 在线 select 31.9（plan_strict，Marginal Greedy）+ freshness preflight 1560.3 + FA postcheck&resume baseline 1880.5 = 6708.2ms，段间隙 3174.9ms（最大 2771.3，含 Journal 窗内序列化）
- **cycle_ms_full_incl_warm（T4−T0）= 24273.23**：含 RPC 尾段 4280.7（apply RPC wall 14171.7）+ 形状预热块 7984.8（C_A 首编译 7338.2，余 514.6/76.1/55.9）+ 请求执行 756.3 + 窗口间隙（rpc→warm 1365.7；段间 0.34–1.58）
- **请求段（独立）**：TTFT C_A 159.5ms（无恢复，全长 prefill）vs C_B/C_C/C_D 32.9/32.3/32.0ms（恢复后仅 32 Token prefill）；request_wall 283.0/156.3/154.6/157.7ms；cuda_prefill 152.0/27.9/26.7/26.9ms；decode ~117–120ms；请求间隙残余 1.54–1.58ms
- 冷启动背景：engine_boot 78.1s、warmup1 40.9s/warmup2 5.2s（隔夜页缓存冷；不影响 T0–T4 计时域）

## 8. 与 Attempt 4 的差异（口径零变化）

- 测量协议、冻结输入、门禁集合、计时段定义完全一致（identity 三方一致链）；差异仅在收尾：attempt4 打印行 KeyError→rc=1+failure 并存；attempt5 干净收尾 rc=0
- 数值对照：control_cycle 8052.5→9883.1ms（波动集中在观察/认证段：final_heavy_observation 2393→3085、preflight 1234→1560、postcheck 1486→1881）；请求段 750.9→756.3ms、TTFT 159/33/32/34→159/33/32/33ms、恒等式与 oracle 结论一致——控制周期分段与请求级指标稳定，重观察段存在 ~20–30% 运行间波动（如实记录，不外推）

## 9. 已验证事实 / 尚未验证 / 失败项

- **已验证**：attempt5 单臂全链（装配→门禁→运行→核验）完整通过；三态判定的最高级"单臂完整 GPU 门禁通过"成立
- **尚未验证**：Marconi-Evict、KVFlow-inspired 两臂（本轮未启动，mc/kv 臂仍为"已装配未运行"）；五轮重复性（本轮 1 轮）
- **失败项**：无

## 10. 环境与证据位置

- 证据根：`evaluation/p0_control_cycle_gpu/arms/`（r1_ck3_fs=attempt5 PASS、passed_print_bug=attempt4、failed_attempt1–3、discarded_assembly×3、arms_status.json={FlowState-Greedy: PASS}、launch 日志×5）
- 盘余 67G；容器 --rm 自清理；本报告：`evaluation/p0_control_cycle_gpu/attempt5_audit_report.md`

**已停止：不启动 MC/KV 两臂与五轮矩阵，等待人工审查。**
