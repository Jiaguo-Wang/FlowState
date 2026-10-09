# Serving-path 实验门禁资格报告

状态：BLOCKED_BY_CORRECTNESS_GATE。

本轮创建独立协议、原计划的逐字副本和 CPU 门禁资格检查；没有创建可执行的 serving E2E runner，没有运行 GPU、history replay 或正式生命周期。没有改变旧实验结果。阻塞只说明当前替代证据未满足门禁，不说明被动日志方案在原则上不可能实现。

## 阻塞原因与证据

原 runner 在每请求前通过 inspect_candidate_states 检查精确候选驻留，在每请求后用 census 比较已有循环状态节点及槽位。若把这些检查改为只看 epoch 末态，会丢失中间边界证据。

独立 CPU 资格检查直接提取旧 compact_census 与 reconstruct_node_positions 函数体，构造两种合成反例：中途丢失循环状态，以及中途换槽；二者均被旧门禁拒绝。但恢复原槽后的末态比较均无法识别。这不是正式 workload 的异常报告，也不是新实验数据。

从已停止的 flowstate-control-profile 容器读取固定运行时源码，以 CPU 对象执行原 MambaComponent.evict_component 函数体，确认循环状态已释放但原生事件队列没有对应记录。因此仅复用现有 H/E/G 记录、原生事件队列和最终 census，不能证明无 native replacement contamination。

涉及源码：evaluation/control_path_optimized_e2e.py 第205—215行；evaluation/openhands_4workflow_occupancy_calibration.py 的 compact_census；固定运行时 components/mamba_component.py 的 evict_component。运行时源码哈希见 gate_qualification.json。

## 待执行协议与 deferred 边界

完全复用旧 plan.json：24 个快照、三策略、三次重复、216 个独立生命周期。Qwen3.5-9B、SGLang 0.5.17、单 H100、TP=1、25% logical recurrent-state budget、四个串行单令牌请求不变。

t0 在 history replay 完成且 pending 就绪后；真实观测、FlowState frontier、输入构造、选择、recurrent-only reconciliation、必要句柄和原生安全检查、请求提交及响应传输均计时，c4 为第四次流结束。engine shutdown 与 GPU cleanup 在 c4 后执行并单独作为门禁。

frozen-reference 对照、已记录 H/E/G 和输出元数据核对、诊断输出及写盘可以延后。请求间精确驻留与无驱逐门禁当前只有同步采样证据；尚未实现并验证可等价替代它们的完整被动变更日志。不能把同步 census 换成空断言，也不能把它改名后继续放在 serving path。

解除阻塞需要覆盖全部相关缓存变更路径、保留请求边界与节点/槽位身份、检测丢失/换槽/恢复等中间事件的被动日志；需要通过故障注入验证不会漏掉旧门禁会拒绝的案例，并证明不会改变分配、LRU、recency、容量或请求调度语义。当前没有完成这项证明，因此按用户规定在 GPU 正式运行前停止。

## 正确性、统计与论文结论

正式生命周期：0/216；正式请求：0/864。两项 CPU 资格检查成功验证了证据缺口；这不是 serving 实验 correctness PASS。

三策略 latency breakdown、四项 paired statistics、95% CI、W/T/L、CV、round breakdown、全部负例 IDs：均为未测量，不适用。不能将空负例列表解释为全胜。

selected-set 与 H/E/G：冻结 reference 仍保留；没有本轮新运行的一致性证据。不得把旧实验的216/216、864/864迁移为本轮通过。

旧 probe-inclusive 数据保留原 measurement scope；本轮未从旧值扣除探针时间，也未拼接 profiling。统计协议仍为先聚合三次重复，以24个快照配对，按轮次分层bootstrap 10000次。

论文可用结论：在保持原逐请求正确性门禁的前提下，现有被动证据尚不足以完成 serving-path 测量，因此本轮不能建立 serving-path 性能收益结论。

## 完整性与复现

保护文件11228个均未修改或缺失，见 integrity_summary.json。新增源码为 evaluation/serving_path_gate_qualification.py；只提取原函数体并使用 CPU 合成状态，不导入或启动 SGLang 引擎。

资格检查复现：先从停止容器读取 /sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache 到临时目录，再用 Python 3.12 执行 serving_path_gate_qualification.py，传入 --runtime-source 临时目录和 --output 新结果路径。不得覆盖本目录中的资格记录。
