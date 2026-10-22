# Marconi 输入可行性：PARTIAL

当前输入不是“无历史信息”，但尚不足以直接宣称忠实 Marconi 主实验就绪。可进入有限的CPU输入合同补齐工作；原生公式适配、父节点语义和独立Bootstrap必须先冻结，不能把现有代理改名为完整Marconi。

证据根：`evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01`（下称B4根）。源码根：`evaluation/sota_baseline_audit_b5a/raw/marconi_repo`。只读核验commit为08016617b1524e6bf6ac29b680641cc945bda7f0，工作树干净。

|必要输入|状态|真实依据与边界|
|---|---|---|
|候选创建历史|AVAILABLE|每组snapshot.history_census保存逐请求新增节点/Slot；history_request_rows有实际request_ordinal；候选creation_order经collector.py:465–510绑定已物化节点。是完成后观测，不是精确GPU插入时刻事件|
|历史命中/恢复顺序|AVAILABLE（受控执行）|snapshot.history[].native_match_and_copy.records有实际rid、monotonic_ns、best_match_node、源槽和E；必须按history_request_rows关联实际序号，不能按history数组顺序排序：capture.py:58按requests字典收集，与执行顺序不同|
|原生Marconi Recency|PARTIAL|现成last_access_order不是完整访问语义。collector.py:484–489只更新同Workflow、同E位置；跨Workflow源命中会漏记。实际历史另有充分部分事件可规范化，但不是原生Marconi时钟序列的现成等价物|
|真实Token前缀父子结构|AVAILABLE（静止树）|barrier_start.nodes完整Token与节点身份；history_census.raw_tree.structure_rows含node,parent,children,key长度。结构不能被同Workflow父候选代替|
|相对父节点FLOPs|PARTIAL|radix_cache_hybrid.py:518–529按原树父节点累计长度计算增量。现有metadata.marconi_flop_saved来自sota_metadata.py:107起，同Workflow父候选的Token跨度代理，不是该公式|
|模型层数、维度和组件容量|AVAILABLE；计算语义PARTIAL|实际config.json SHA与四组配置一致；32层=24 GatedDeltaNet+8 full attention，hidden4096、intermediate12288、KV heads4/head_dim256；原论文Mamba公式不能不加论证移植为GatedDeltaNet FLOPs|
|隔离测试的Bootstrap|MISSING（已冻结协议）|未找到独立训练窗口清单、窗口起始树、原生容量、α选择输出等锁。四组各8/12/16/20条同快照预历史不是已预注册的独立调参总体；不能用本轮测试结果挑α|
|未命中/新候选/初始时间戳规则|PARTIAL|有创建序号；官方TreeNode初始化时间及match/insert双时间推进可读，但与当前collector单请求ordinal不同，必须固定映射，避免now−last=0；不能临时按结果填0或任意rank|
|内存分母和驱逐资格|PARTIAL（语义不等价）|原生效用分母含SSM状态和Attention KV，evict_v2仅叶/单子节点并会改变树；本任务固定候选+recurrent-only K不包含这些动作|

## 可利用的原始历史，不应丢弃

`native_access_evidence.json`按当前候选Node、源Slot和E一致条件联接已有Native Match。四组各观察到C1、D1两个请求命中B1，共8个跨Workflow历史源匹配事件。例如g001实际ordinal=3的`rq3e-g001-c-turn-001`及ordinal=4的`rq3e-g001-d-turn-001`命中B1。它们被同Workflow更新规则遗漏，但g001 B1随后在ordinal6又被本Workflow访问，所以不能直接声称最终last_access_order一定不同。问题是事件历史语义不完整，而非凭两个漏记事件推断所有最终排序错误。

这些证据支持规范化“受控已知历史的候选访问”，不自动代表Marconi自身插入/分裂过程中发生的所有访问。原生代码radix_cache_hybrid.py:134、234分别推进insert和match逻辑时间，:256–261更新最后命中节点；初始化见:33–41。应明确哪些事件计入控制适配，绝不能按未来Pending输出回填过去的访问时间。

## 父节点的具体冲突

g001 A1为Node18，L=3520；实际父节点Node19只到2437且无Recurrent状态，其上B1 Node20到2432。当前代理值为3520（同Workflow没有更早候选），既不是实际树边长度1083，也不是最近可恢复祖先到A1的1088。证据：snapshot候选metadata与history_census末次raw_tree.structure_rows、barrier_start.nodes。

因此“无需运行时树信息”的B5-A断言不成立。需要明确选择原生结构父节点、可恢复父节点还是固定候选父节点；后两种均是适配差异，不能悄悄声称官方原始语义。驱逐改变父子关系时是否更新效用、是否允许多子节点驱逐，也必须在固定候选合同中说明。

## 容量和模型不能照抄

B4节点实际组件形状：Conv [24,29,8192,3] bf16，Temporal [24,29,32,128,128] fp32。每个非哨兵Slot的逻辑组件字节为24×(8192×3×2+32×128×128×4)=51,511,296，与metadata.memory_bytes一致。28为可分配池槽，29含哨兵；不是K=1–4下实际HBM压力。

Marconi原始 `get_mamba1_flops`、MLP和Attention公式针对其模型假设；Qwen使用GatedDeltaNet、GQA及实际MLP宽度，维度已知不代表公式已等价。B5-A记录的radix_cache_hybrid.py:526 MLP减项调用Attention公式问题仍需显式处理。原代码FLOPs口径优先指α调参评价目标，不授权用Phi/J替代，也不授权把Mamba公式无说明应用到GDN。

## Bootstrap和进入B5-B1的边界

用户已选定α按官方ConfigTuner FLOPs Savings选择（config_tuner.py:84–110）；论文Hit Rate仅后续敏感性分析。本轮无需再次请求选择口径。代码需窗口前树和过去(input,output)对，原树首次驱逐/窗口规则见radix_cache_hybrid.py:113–118、137–154。当前snapshot.history保存输入和匹配记录，history_request_rows保存输出数量但非完整历史生成Token；现存pending输出属于Barrier之后，不能充当过去Bootstrap。

可以审查现有独立历史来源以补齐输入，但g000发现组、g005/006配额外组、旧573条件均不能自动转成独立训练集；必须预先定义隔离、时间范围、状态生命周期和参数冻结规则，不能按本批J优劣选择训练窗口或α。未满足前，固定α=1只能标明旧受控代理或预设参数敏感性，不是已经执行官方Bootstrap。

**建议：有条件进入B5-B1的CPU输入规范化与合同验证，暂不进入正式效用比较。** 完整官方系统复现仍不在固定候选合同内。只有父节点、真实访问时序、GDN/GQA计算与资源分母、Bootstrap证据逐项闭合，才可标记具有明确忠实性边界的Marconi Eviction Adaptation READY。
