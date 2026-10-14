# Phase B4-B0：Capture-only实现与CPU门禁

## A. Final Gate

**READY：独立采集实现与CPU合同门禁就绪。** 此状态只表示可提交下一阶段执行审查，不表示g001–g006已完成Runtime采集或已认证。实际环境、状态驻留、组件完整性与新边界准入仍须B4-B1运行后判断。本轮没有启动Docker、SGLang、GPU或历史请求回放。

适用范围严格限定为受控已知输入回放；30限额为来源Session配额，非已认证的原生Roots。没有策略比较、收益计算、GPU性能或自然Agent Ready主张。

## B. 新增与复用文件

全部新增文件位于 `evaluation/strict_compatibility_b4b0/`：

- `contracts.py`：SHA-256核对、固定group/round合同、UNKNOWN/REJECTED/准入检查、结构标签、不可变Snapshot及审计清单。
- `capture.py`：单组独立生命周期入口，延迟导入Engine，只在显式执行时运行。先冻结屏障证据，再核验Pending实际接收身份。
- `launch.py`：固定g001–g006顺序的批次驱动，每组新Docker/Engine、只读源挂载、清理检查、配额与失败账本。不根据结构或胜负跳到后续新组。
- `lock_inputs.py` / `execution_lock.json`：B0一次性锁定完整来源Parquet、模型权重及配置、协议、相关Python源码；已有锁不允许静默覆盖。未来执行前在主机和容器中逐项核验。
- `test_contracts.py` / `cpu_tests.log`：纯CPU静态证据及模拟合同测试。
- `evidence_manifest.json`：本轮交付哈希。

复用B1的 `_load_frozen_group`、`_path_evidence`，不调用B1的main或reference/evicted流程；复用B1-C的`scheduler_process`，直接观察原始match和COW调用；复用旧Collector的materialize_group_requests、replay_group_to_barrier和SGLangGroupRuntime；复用原Marconi元数据构造函数，但不调用任何选择器。

不改动生产Optimizer、Controller、SGLang、Baseline、旧采集脚本或B1–B4-A产物。

## C. Capture-only完整流程

1. 输出目录必须不存在，创建独立目录后先记录启动事件。检查协议、完整数据和源码锁；非法组或摘要变化写失败账本并停止。
2. 读取既有manifest对应组及原round，不重新分组；核对镜像身份、SGLang/Transformers版本、Runtime关键源码与B1-C一致，验证模型及Tokenizer文件摘要、Chat Template身份。
3. 用既有Collector准备截至round的历史输入与round+1的Pending，保存完整Token数组、来源及future_boundary_audit，记录`controlled_input_ready`。不得写成原生Agent Ready。
4. 新Engine回放既定历史；逐请求保存原始输入与实际match输入摘要、请求序号、访问记录和逐步census。失败时保存partial_replay。
5. 到达静止屏障，保存全树节点Token路径、物理槽位、conv/temporal形状、引用、headroom与保护状态。核查无活动/排队/chunked请求、无Native Eviction/FA级联、观测非干扰。
6. 逐候选核对逻辑ID、L、Token Prefix Digest、Node ID与完整状态映射，核对候选覆盖全部驻留recurrent节点。发现候选外状态则停止，避免模型漏算其他恢复位置。
7. 准备完整Candidate×Pending矩阵：Token身份、L≤T、同配置、同边界、完整组件、FA路径、引用和slot alias检查。明确违例REJECTED；必要字段缺失UNKNOWN；只有全部条件满足才EPOCH_ADMISSIBLE。不使用Workflow ID替代Token判断。
8. 重新读取静止状态，与采集开始的nodes/tree/accounting逐项比较；冻结`snapshot.json`并保存内容SHA-256。结构分层仅使用位置、目标和准入边，按A>B>C，不读取Phi或策略结果。本轮**不生成收益矩阵**；未来收益只能派生自这个不可变输入。
9. 冻结后按原A/B/C/D顺序提交4个Pending，各生成1个确定性Token，仅核对实际Runtime接收Token摘要和长度。输出、match E和Handle存入独立`pending_*.json`及`pending_identity_checks.json`，绝不回填`snapshot.json`。每次提交后重新核对Snapshot SHA。已存在recurrent节点非预期驱逐或替换时失败。
10. 保存summary，正常或异常均执行shutdown、审计日志和manifest。主机启动器清理本次命名容器并检查GPU显存回落；任何失败停止整批，记录后续所有组未执行，不自动重试或补位。

安全字段不足时不靠推测补齐。例如Slot Generation没有接口仍明确UNKNOWN；本轮继承B1受控静止边界合同，不据此声称跨时刻代际安全。候选组件池检查加节点槽位映射不等于逐张量内容认证；实际恢复也仍须独立证据，矩阵默认resume_validated=false。

## D. CPU门禁

**最终16/16项CPU测试通过**（12.531秒，含完整内容哈希复核）；`cpu_tests.log`包含以下测试：

1. 已有静态正例准入。
2. 固定g001–g006及原round，拒绝g000/g007和round改变。
3. 文件内容改变使哈希门禁失败。
4. 已存在文件不可覆盖。
5. Candidate Token Identity及不等前缀拒绝。
6. L>T拒绝。
7. 组件缺失、保护引用非零、slot alias不能错误认证。
8. 必要字段缺失产生UNKNOWN。
9. 非静止屏障、Native Eviction失败。
10. 冻结后内存对象变化不会回写Snapshot，文件污染被检出。
11. 模拟g003失败后只执行g001–g003，其余逐项记未执行，不补位。
12. 主层A满4后不转派B，不改变六组检查顺序。
13. 分层不依赖策略收益，注入胜负或收益字段不改变标签。
14. 提取并执行原Collector的输入构造函数，在模拟Token化/毒化未来消息下证明不消费Pending输出或后续消息。
15. 实际执行锁覆盖数据、协议、实现并逐文件校验。
16. 导入capture模块不导入或创建SGLang Engine。

测试使用静态B1/B1-C证据、临时文件和模拟执行器。uv312中Transformers与huggingface-hub依赖不兼容，不能直接导入整个旧Collector；本轮未安装或修改依赖，而是AST提取原函数体执行纯CPU输入边界测试。此测试不验证GPU容器的完整导入链。B4-B1使用旧Docker环境，不使用uv312启动Engine；如容器接口或依赖出现变化，执行时失败并保存证据。

所有源文件通过CPU语法编译；模型与Parquet哈希采用流式读取，不创建GPU上下文。测试不能代替新的Runtime验证，尤其不预判6组都会采集成功。

## E. 固定采样与元数据规则

- g000只引用为发现案例，占4个来源Session，不重新采集。
- g001–g006共24个来源Session，连同g000计28；保留原round3/4/5/2/3/4。剩余2个Session额度不拼组。
- 主层优先级A>B>C，保留全部标签；主层满4后只记录不入选，不改派或补位。无结构或有UNKNOWN均保存，不计算策略来决定去留。
- 每组一个生命周期、原turn后A/B/C/D顺序；最多84次历史请求与24次Pending身份核对。实际失败可提前停止；没有reference/evicted三臂流程。
- 保留逻辑K=1至min(4,候选数)及K_min协议，但本阶段不计算策略或收益；由未来B4-C使用。
- Creation Order和Last Access Order直接来自既有Collector的历史trace，保存对应原始match事件供审计。Last Access是既有同Workflow、同恢复位置更新规则，不冒称完整原生LRU或跨Workflow访问计数。额外保存memory_bytes、lineage_path、增量FLOP代理和alpha；原构造规则未改，B4-C仍须评估其可比性。

## F. B4-B1运行命令、产物与清理

以下为**下一阶段授权后**的命令，本轮没有执行：

```bash
cd /home/wjg/code/FlowState
/home/wjg/uv312/bin/python -B -m evaluation.strict_compatibility_b4b0.launch \
  --execute-b4b1 \
  --out /home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01
```

主机uv312只执行控制、哈希与Docker命令；SGLang在冻结的`lmsysorg/sglang:v0.5.17-cu129-runtime`容器运行。GPU0、TP1、单H100、模型目录及原B1引擎配置保持不变，28 recurrent slots不自动扩容。源仓库、模型和数据只读挂载，只有新输出根可写。输出根存在则拒绝运行。

批次根预期包含execution_lock、镜像身份、audit.jsonl、各组命令/runtime.log/GPU前后CSV、batch_summary、manifest。每组包含source、configuration、prepared_inputs、history_input_evidence、barrier_capture、不可变snapshot、Pending独立记录、post_pending_state、summary、audit及manifest；失败另有failure.json和部分回放证据。

正常或异常结束都调用Engine shutdown；主机对本组唯一容器名执行清理，再检查GPU0显存低于1000MiB。该阈值是安全可继续的门禁，不等于保证恢复到精确14MiB；实际值写CSV供审查。清理或显存检查失败则停止后续组。整个批次不会为失败自动重启、换组或修改配置。

完整内容锁每组再次验证，包含约19GB模型文件，增加CPU I/O但不会改变Runtime实验语义。锁文件本身由本交付manifest摘要固定；正式执行前应核对交付manifest，不能以重新生成锁绕过变更。

## G. 仍不能认证的语义

- 自然在线Ready、Fork、原生Root和跨Session真实因果关系未认证。
- 重放历史来自冻结轨迹，不是目标模型驱动的闭环Agent工具执行。
- 新6组的实际Token接收、物化、驻留、别名、容量headroom与静止状态待Runtime核验。
- EPOCH_ADMISSIBLE不等于每条边都实际恢复；Pending身份检查不用于逐边恢复归因。
- Slot Generation、任意并发下原子性、真实物理Retention Quota及全体请求输出正确性不由本轮证明。
- 本轮没有J、策略胜负、Runtime性能或收益矩阵比较。

完成代码与CPU门禁后停止；等待B4-B1授权，不自动运行以上命令。

## 最终内容锁记录

最终execution_lock.json覆盖244个文件。Parquet完整SHA-256为`3dd8ec3546cf771ce4ab2ac6c51ccefdd621197fa997a2cefb430b50df808fb6`。CPU复核后补严了缺少候选Token和部分引用字段的UNKNOWN处理，旧开发锁和旧测试日志以pre_missing_field_fix/before_missing_field_fix文件名归档；最终代码重新锁定并16/16复测通过。未更改协议、候选或策略规则，不能使用旧开发锁执行正式采集。
