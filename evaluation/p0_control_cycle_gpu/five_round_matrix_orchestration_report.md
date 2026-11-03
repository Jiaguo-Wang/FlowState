# P0-4 五轮矩阵 · 第二阶段编排实施与 CPU 自检报告

日期:2026-10-30 | 依据:五轮矩阵任务书确认件(spec id 82ee)+ `five_round_matrix_preflight.md` §4/§9
本阶段零 GPU 启动、零修改现有文件、零触碰 r1 证据。

## 1. 修改文件清单及代码差异

**新增 2 个编排文件(与策略实现隔离;不改 prepare.py/launch.py/runner.py/observation.py/control.py/select_entry.py/冻结输入)**:

| 文件 | 行数 | 职责 |
|---|---|---|
| `prepare_round.py` | 96 | 轮次装配器:复用 prepare.py 原语(write_once 拒绝覆盖/cpu_prechecks/collect_entry_sources/serial_safety_check),装配 m{R}_ck3_{fs,mc,kv};identity 同 schema 新增 `'round'` 字段;逐轮写 prepare_record_m{R}.json(一次性) |
| `launch_round.py` | 252 | 定卡串行启动器:复用 launch.py 的 docker_cmd/df_root;固定 GPU-bfff4d31(被占即停等,不换卡不杀他人进程);拉丁方串行 fail-loud;追加式总账 five_round_ledger.jsonl(失败臂同样记账);逐轮 arms_status_m{R}.json(一次性,不触碰 arms_status.json);--dry-run 与 --ledger-selftest 模式 |

**对现有文件的修改:无**(git 状态:整个包为未跟踪新目录;launch_round 身份门禁每轮发射前断言 runner/observation 完整 sha256==冻结值,反向证明测量路径零改动)。

**实施过程中发现并修复的 2 处自体缺陷(均在自检阶段暴露,如实记录)**:
1. `prepare_round.build_round_arm` 首版漏引 `OUTBASE` 别名 → 装配即 NameError(失败点在 mkdir 之前,零残留;write_once 亦有兜底),补别名后 15 臂一次装配成功;
2. `residue_gate` 首版沿用 `pgrep -f` 模式 → 调用链 bash 包装与进程自身被误判为残留 launcher(任务 A 同款踩坑,记忆有档)→ 改为 /proc 枚举 + 祖先链(PPid 链)+自身 PID 排除,五轮 dry-run 全过。

## 2. CPU 自检结果(全部通过)

| 自检项 | 结果 |
|---|---|
| compileall(两新文件) | ✅ |
| cpu_prechecks(装配前置:census 16=16、三策略 E 向量、oracle schema、串行安全) | ✅(15 臂装配前执行,断言全过) |
| 15 臂装配(m1..m5 × fs/mc/kv) | ✅ 一次性成功;装配序=执行序(拉丁方逐轮轮转可见) |
| 15 臂冻结输入逐字节 == r1_ck3_fs | ✅ 9 个装配输入(8 NAMES+t2g1_plans;workloads_case_C 为抽取副本含断言)全部 sha256 相等 |
| 15 臂 identity | ✅ round 字段正确;image_id/runtime_sources(镜像内 16 文件)== r1;hybrid_entry_sources 中 runner/observation == 冻结身份 b6cee2d3…/7c1dcefc… |
| run/ 目录 | ✅ 15 臂全部不存在(干净起点,拒绝覆盖) |
| 执行顺序核验 | ✅ ROUND_ORDER == 任务书§四.2 逐轮逐位;位次分布 FS 2/2/1、MC 2/1/2、KV 1/2/2(首位/次位/末位) |
| 源码身份门禁(launch_round.identity_gate) | ✅ dry-run 中五轮全过 |

**口径澄清(核验中发现)**:`device_binding.json` 是容器内 device_entry.py 的**运行时输出**,不在装配输入集(prepare.NAMES)内;preflight §3 所列其哈希为 r1 三臂运行产物巧合一致,五轮各臂将由自身运行生成,以 device_binding.status==PASS 为准。

## 3. 15 个实验臂目录与启动命令

- 目录:`evaluation/p0_control_cycle_gpu/arms/m{R}_ck3_{fs|mc|kv}`,R=1..5,共 15 个(与 r1_* 及历史归档零冲突);每臂内 9 个装配输入 + source_identity.json(round 字段)+ prepare_record_m{R}.json(轮级)。
- 启动命令:每臂已构造并存证 `{arm}_command_dryrun.json`(15 份;正式运行时写 `{arm}_command.json`,同一构造函数)。
- 样例(m1_ck3_fs):`docker run --rm --pull never --name flowstate-p04-m1_ck3_fs --gpus device=GPU-bfff4d31-0b94-7de3-9946-f593437ae566 … --env PILOT_POLICY=FlowState-Greedy … <冻结镜像 sha256:1dfce903…> -B …/device_entry.py`;臂目录→/b5out rw、仓库与模型只读挂载、TRITON_CACHE_DIR=/b5out/triton_cache,与 r1 完全一致,仅臂目录/容器名/PILOT_POLICY 随臂变化。
- 正式执行入口(逐轮,人工发起):`python3 -m evaluation.p0_control_cycle_gpu.launch_round {R}`,R=1→5;每轮前自动执行 §5 门禁。

## 4. Ledger 写入测试

- 自检账簿 `arms/five_round_ledger_selftest.jsonl`:追加 2 行(PASS 样例含 control_cycle/cycle_full/request_exec/ΣTTFT/Σprefill/J/选集;FAIL 样例 status=BLOCKED、error 摘要随账、指标为 null)→ 回读校验行数/状态/指标聚合/错误字段全部一致;**只增不改**语义(fsync append)。
- 正式总账 `arms/five_round_ledger.jsonl`:当前不存在,首臂真实完成时创建;每臂一行,字段=轮次/序位/策略/臂名/定卡/镜像/runner+observation 完整哈希/退出码/状态/墙钟/盘余/双口径计时/ΣTTFT/Σprefill/J/选集/证据目录(失败臂含 error)。
- 防覆盖纪律:prepare_record_m{R}.json、arms_status_m{R}.json、source_identity.json 全部 write_once(已存在即拒写);臂目录存在即拒装配。

## 5. 最终 GPU 执行门禁报告(发射前状态)

R1..R5 五轮 dry-run 全部通过以下门禁(正式运行时每轮、每臂重复执行):

| 门禁 | 内容 | 当前状态 |
|---|---|---|
| 磁盘 | df 根分区 >20G | ✅ 68G(15 臂预计 ~15G) |
| 定卡 | GPU-bfff4d31 在位、<100MiB、零计算进程 | ✅ 14MiB/0%(他人占用→停止等待,不换卡不杀进程) |
| 残留 | 无 flowstate-p04-* 容器、无 launcher 进程(排除自身调用链) | ✅ 零残留 |
| 源码身份 | runner/observation 完整 sha256 == 冻结身份 | ✅ 三方一致(preflight §2) |
| 臂目录 | 该轮三臂已装配且 run/ 不存在 | ✅ 15/15 |
| 执行纪律 | 拉丁方固定顺序串行;一臂非 PASS 即终止该轮;不重试/不跳过/不覆盖;中断轮作废重跑(不拼零散结果) | 机制就绪(launch_round fail-loud + write_once + 总账) |

**已验证**:编排实现、15 臂装配与身份/冻结输入一致性、五轮 dry-run 门禁与命令构造、总账追加与失败记账、顺序表与位次平衡。
**尚未验证(须真实 GPU)**:15 臂实际运行、九类正确性门禁逐臂复核、五轮配对统计。
**失败项**:无(两处自体缺陷已在自检阶段修复并记录于 §1)。

## 6. 停止声明

**本阶段到此停止,等待审查。未启动任何 GPU 实验**;审查通过后按 `launch_round 1→5` 逐轮人工发起执行。
