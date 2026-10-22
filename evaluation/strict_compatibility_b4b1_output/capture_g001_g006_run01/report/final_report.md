# Phase B4-B1：严格Compatibility Runtime采集报告

## A. Final Gate

**READY：g001–g006六组均成功完成受控采集，六张准入矩阵完整。** 没有失败组、重试、补位或额外生命周期。按主层A最多4个的预注册配额，**只有g001–g004进入正式B4-C入选子集**；g005–g006完整保存，因层额满排除，不转派B/C，也不是因UNKNOWN或不利策略结果排除。

实验为Controlled Known-input Replay，不认证自然在线Agent Ready、原生Root/Fork/Join或并发Workflow。未计算任何Allocation Objective、策略选择、K_min或策略胜负。

## B. 执行账本与Compatibility覆盖

指定启动命令执行一次，退出码0；g000未重采。来源Session新24个，加发现组配额4个共28，剩余2个未拼组。下面的“准入/拒绝/未知”为EPOCH_ADMISSIBLE/REJECTED/UNKNOWN。

| Group | Round | Engine数 | 历史请求 | Pending核对 | 候选 | 全配对 | 准入/拒绝/未知 | 跨Workflow准入 | 屏障空闲槽 | 配额结论 | 清理 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| g001 | 3 | 1 | 12 | 4 | 12 | 48 | 15/33/0 | 3 | 16 | 入选 | PASS |
| g002 | 4 | 1 | 16 | 4 | 16 | 64 | 19/45/0 | 3 | 12 | 入选 | PASS |
| g003 | 5 | 1 | 20 | 4 | 20 | 80 | 23/57/0 | 3 | 8 | 入选 | PASS |
| g004 | 2 | 1 | 8 | 4 | 8 | 32 | 11/21/0 | 3 | 20 | 入选 | PASS |
| g005 | 3 | 1 | 12 | 4 | 12 | 48 | 15/33/0 | 3 | 16 | 主层配额已满，不转派 | PASS |
| g006 | 4 | 1 | 16 | 4 | 16 | 64 | 19/45/0 | 3 | 12 | 主层配额已满，不转派 | PASS |

合计84次历史请求、24次Pending身份核对、84个候选、336条配对：102准入、234拒绝、0未知。跨Workflow准入18条，占全部准入18/102=17.647059%；跨Workflow全部配对252条，准入18/252=7.142857%。不得将不同分母混用。正式入选g001–g004子集合计56候选、224配对（68准入、156拒绝、0未知），其中12条跨Workflow准入；配额外g005/g006不计入该正式分母。准入计数不等于逐边恢复认证：本轮新增RESUME_VALIDATED为0，所有resume_validated字段保持false。

原始Session列表、屏障ID、T和组件形状见summary.json/details；每个Checkpoint的Node、Slot、Prefix Digest与访问元数据见checkpoint_identity_index.json；全部配对条件与拒绝原因见compatibility_index.json及原snapshot.json。数量均来自实际采集，不从旧Formal矩阵移植。

## C. 结构覆盖

六组全部具有非互斥标签A、B、C，主层均为A。观察到的标签计数A=6、B=6、C=6；正式入选主层A=4、B=0、C=0。**主层B/C为0源于固定优先级，不是这些结构未出现。** 无标签组为0；g005/g006没有被删去或伪装为未采集。入选子集四组也均保留三个附加标签。

分层由冻结Collector使用L/T及准入边计算，未读取Phi、收益、算法输出。不同结构的存在尚不证明算法决策或目标有差异；本轮不判断区分力胜负。

## D. Runtime完整性

1. **预检查**：B4-B0最终11个交付文件摘要通过；最终执行锁244文件（包括完整Parquet、模型权重、Tokenizer、配置、协议、源码）通过，启动器及各容器重复核验。未使用旧开发锁或重新生成锁。结束后再次验证相同锁与B4-B0清单。
2. **环境**：Qwen3.5-9B、SGLang0.5.17、TP1、单H100 PCIe、28 recurrent slots及原Cache配置保持不变。Docker镜像ID与B1-C一致。配置与软件版本及核心Runtime源码摘要在启动门禁核对。
3. **Token身份**：84个历史请求的提交Token与原始Runtime match记录摘要一致；24个Pending实际接收的完整输入长度和摘要与冻结缓冲一致。候选Token Prefix逐项核对，准入边均L≤T。未重新解释Workflow ID来认证共享。
4. **屏障一致性**：6/6在静止边界采集，running/waiting均0；采集开始/结束nodes、tree、accounting完全一致，观测非干扰通过。全树resident recurrent节点集合恰由候选映射覆盖，没有观测到候选外驻留recurrent节点。
5. **物理状态**：84候选映射与Token位置一致；各屏障slot无别名，conv与temporal组件及索引有效，保护引用符合准入条件。保护槽位均0；空闲槽为16/12/8/20/16/12。Slot Generation仍UNKNOWN，不能宣称代际验证；组件张量内容没有独立全字节认证。
6. **Snapshot不可变**：每组先完成snapshot.json写盘及SHA-256记录，随后才提交4个Pending；冻结事件在所有Pending提交之前。所有Snapshot结束时哈希仍与原summary及audit一致。Pending产生的新E/Node/Slot只写在独立记录，未回填原屏障。
7. **Native Eviction与错误**：历史逐步census未推断到Native recurrent eviction或FA cascade；Pending前后原有prefix→slots映射保持不变。没有OOM、回撤、输入截断、身份冲突或门禁失败。该结论基于现有census/状态差分门禁，不宣称拥有每个底层瞬时事件的完整追踪器。
8. **清理**：6/6引擎shutdown返回，6/6独立容器退出，g001/g002后GPU0为14MiB，g003–g006后为17MiB，均通过冻结的GPU0<1000MiB门禁。g003/g005后GPU0瞬时利用率为4%，其他后置记录为0%；后四组GPU1占用7379MiB，其进程来源在本批日志中UNKNOWN，不能声称当时两卡均空闲或把它归因于本任务。本任务容器仅分配GPU0，未操作其他GPU任务。最终额外查询无本批容器、无计算进程，两卡均14MiB。日志中的“no such container”来自--rm已自动删除后再次清理，并非残留容器；GPU IPC及可选torchcodec警告未触发执行门禁，日志原样保留。

物理identity索引是本次生命周期观测值，不沿用B1的Node/Slot编号；不存在新增跨生命周期代际主张。

## E. Marconi元数据与后续就绪

**在冻结Marconi Adaptation定义下，元数据足够用于后续公平比较。** 84候选均保存creation_order、last_access_order、memory_bytes、lineage_path、marconi_flop_saved、alpha及原始访问证据。本报告独立核对last_access为原collector规则下的历史匹配更新，增量span与同Workflow较浅候选一致，alpha=1；没有执行Marconi选择。

这保留的是既有受控代理：Last Access按实际历史match E、同Workflow/同位置更新，不是原生全树LRU；跨Workflow真实访问事件在原始match记录中保留，但本轮没有改写冻结策略将其加入新的打分规则。contributing_request_ids也不能冒称逐节点真实命中次数。后续应明确称Marconi Adaptation，不称完整原生Marconi。不得根据不同收益重新选择元数据口径。

B4-C就绪数：**4个正式入选新Snapshot（g001–g004）**。另外2个技术证据完整但配额外，仅描述性保留，不能静默纳入正式比较。因UNKNOWN/证据冲突需排除数为0。六组均无必要矩阵字段UNKNOWN；未观测的Slot Generation与原生Agent事件仍属范围限制，不被升级为认证能力。

现有入选证据足以进入按协议的CPU-only比较，不需要为补满主层B/C或追求胜例追加GPU采集。K=1…min(4,候选数)及K_min留给B4-C，逻辑预算不能称真实物理容量压力。本轮结束后停止，不运行任何策略或新增实验。

## F. 原始证据与完整性

原始根：`/home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01`。

- audit.jsonl、batch_summary.json：固定顺序、配额决策、清理及整批成功证据。
- gNNN/snapshot.json：不可变屏障Token、物理状态、矩阵及历史事件；summary.json保存摘要。
- gNNN/pending_identity_checks.json及pending_*.json：冻结后输入核对，post_pending_state.json为独立后状态。
- gNNN/manifest.json与根manifest.json：启动器生成的原始文件哈希。本报告先核验其全部条目，没有重写原manifest。
- gNNN.runtime.log、gpu_before/after.csv、command.json：完整运行日志与命令。
- execution_lock.json、image_identity.json、source.json、configuration.json：来源及执行身份。

报告及只读汇总脚本均置于新增report/子目录，不属于原启动器manifest的写入时点；由独立report_manifest.json登记。execution_coverage.csv提供逐组表，summary.json提供完整计数。复现汇总只读取本次证据，不调用Runtime或策略。命令为 `/home/wjg/uv312/bin/python -B -m evaluation.strict_compatibility_b4b1_output.capture_g001_g006_run01.report.summarize`（从项目根运行）。

执行时间以原audit中的Unix纳秒为准，保留原记录，不因客户端跨日更新改写时间。无失败组，无自动重试，无旧文件覆盖。
