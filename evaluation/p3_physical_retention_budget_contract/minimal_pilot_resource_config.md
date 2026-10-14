# 最小 Pilot 资源配置（仅设计）

本轮只读审查与 CPU 整数容量推导，未启动 GPU、Engine、Agent、Pilot、replay 或策略比较。日期：2026-10-09。

源码约定：`S/` 为 `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`，使用 P3-G0 与停止的正式实验容器逐字节核实的版本；本轮重新核验其 16 个文件摘要。其余相对路径以 `/home/wjg/code/FlowState/` 为根。新扩展仅为设计，没有实施；UNKNOWN 不当作零。

**推荐先做非紧预算的 instrumentation Pilot，不预设它必须产生容量竞争。** 这是新增独立设计，不修改 P3-G0 文件，不改变任何旧正式配置。

## 配置表

| 项目 | 建议值/规则 | 来源或性质 |
|---|---|---|
| 模型 | `/home/wjg/models/qwen3.5-9b`，Qwen3.5-9B | 已有 config 与旧完整状态证据 |
| Runtime | SGLang v0.5.17，P3-G0 核实版本；单 H100，TP=PP=DP=1 | 继承既有范围，未来启动需记录 resolved args |
| cache | UnifiedRadixCache + FULL/MAMBA，extra_buffer，track_interval=256，path cap=-1 | 冻结 optimized_formal/protocol.json:24–27 |
| 模式约束 | no-overlap、非 lazy、无 speculative/ReplaySSM、无 int8、无 HiCache/session radix/Unified Memory；FP32 temporal、BF16 conv | 保持已核实的单池大小与控制安全范围 |
| 环境锁定 | SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=false，pool layout 与旧普通模式一致 | environ.py:982；改变此开关后必须重审容量公式，不能继续声称4Q依据相同 |
| N | max_mamba_cache_size=28 | 直接复用旧正式物理池大小，未为 K_min 缩小 |
| Q | max_running_requests=2，外部 admission 同时最多2个；保证 resolved cap仍≥2 | 两分支验证设计，不是旧四请求串行协议 |
| 其他已有参数 | chunked_prefill_size=45056、context_length=131200、mem_fraction_static=0.4、disable_cuda_graph=true | 旧配置参考，不能认为这些数值保证新任务 FA headroom；若用 frozen Phi 还须 T≤131072 |
| b | 51,511,296 bytes=49.125 MiB | 旧 raw fingerprint 验证；未来 tensor/layout 不同则拒绝沿用 |
| 基本 Working/Tracking | 至多2个 working + 2个 tracking 的基础角色；COW destination 已算 working | 由 no-overlap extra_buffer 源码推导，不是全部执行峰值 |
| R_exec 设计基线 | 4Q=8 槽，包容 working/tracking 与其余 admission/临时余量 | 采用 runtime 默认四槽请求规划因子作为保守服务预留。未证明所有长序列/多次插入都被8覆盖；必须补峰值门禁 |
| O | 独占 Engine 的合同要求0；观测到未知/其他 allocated slot 时失败 | 不是把未测占用填0；这是未来应满足的资格门禁 |
| K | 在 O=0 且8槽包络验证通过时，K=N−R_exec=20；budget_bytes=20b | 由旧 N 和事前两请求服务包络派生，不按 candidate_count 设定 |
| 精确 GPU 实测峰值 | UNKNOWN | 本轮未启动 GPU，不能声称新两分支配置已通过 |
| 长活动阶段峰值/所需额外余量 | UNKNOWN | 未绑定具体任务、长度和插入上界前禁止执行，不能采集后再改 reserve |

8槽预留是本轮建议的服务合同，**不是 SGLang 已经替我们强制留空的8槽**。剩余4槽不能再分别作为 COW、native、donation 各算4槽；需逐事件按并集账目核验。若未来事前静态界要求更多空间，应在第一条 workload 生成前修订并冻结新合同；本轮不填凭空计算出的“实际峰值”。

## CPU 容量计算

- 可分配 payload：28b=1,442,316,288 bytes。
- 包括 slot0 的物理状态 tensor payload：29b=1,493,827,584 bytes=1424.625 MiB。
- Retention 权限：20b=1,030,225,920 bytes=982.5 MiB。
- 执行 reservation：8b=412,090,368 bytes=393 MiB。

见 capacity_calculation.json。这些是 tensor payload 算术，不是进程/GPU 总显存。没有额外分配两个物理池，也没有在释放 checkpoint 时释放393MiB等量的驱动内存。

## 捕获与先后关系

P3-G0 的真实两分支准备下一阶段请求 → 两个输入已就绪且未提交 → 全 ingress gate 关闭 → 前序任务完成、GPU fence、scheduler idle → 一次读取 C/R/ownership/free/lease → 校验资源合同 → 后续若另获授权才执行 recurrent-only reconciliation → 全局 retained≤20、free≥8、lease=0 → 才允许下一批 match/COW/alloc。

本 Pilot 当前只设计原生事件与只读状态验证，不执行优化器或策略对比。若 C≤20，本就没有 quota 超额，无须驱逐；若实际超额但该阶段没有获得 reconciliation 执行授权，则保持 gate 关闭并结束验证，不能偷偷调用 native eviction 达标。

每个阶段记录：N/resolved并发上限、conv/temporal shape/dtype/b、全部物理 slot→owner/generation、canonical checkpoint/lineage/L、FA R/T、native eviction计数和原因、COW源目的、每次分配前后 free、tracking donation、lease到期、selected映射（若未来执行保留操作）、全局 persistent数量、GPU完成与idle证据、FA保存、OOM/retraction/truncation、cleanup。检查 slot0 不入可分配账目，重复/未知所有权直接失败。

必须对未来请求的最大输入/输出与可能的中间 checkpoint 插入次数先给容量界；不能用“8槽看起来够”替代。不能在策略执行后统计够用再倒称预留已证明。

## 是否会产生所需竞争

这套合理的起始配置**不会在仅两个有效 pending 时产生 K<K_min**：K=20，而 K_min≤2。即使 C>20（例如多层历史），也不代表最深前沿必须取舍；冗余 checkpoint 多不是紧约束。

当前没有足够的独立资源依据推荐两分支 K=1 的正式配置。因此本轮不为了得到正证书改成小 N 或夸大 W/H。两种可诚实的后续结果是：先完成事件/state证据且报告容量非紧；或另行获得真实部署内存/并发服务合同，在采集前据此绑定新 N/R/K。不是允许采集后调参。没有合同前，“最小可区分容量配置”的数值为 UNKNOWN。
