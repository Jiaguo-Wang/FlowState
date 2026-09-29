# FlowState控制路径瓶颈profiling

状态：PASS；72/72独立生命周期全部通过原有E2E门禁。
冻结实现未修改，诊断仅包装函数计时并增加响应元数据。所有状态操作及检查仍调用冻结函数。
本表是新profiling的每epoch累计值，不是历史293.093ms与211.337ms的未记录内部计时。

| 阶段 | 总计 | 提交入口前 | 队列等待 | 服务端 | 事件交接 | 返回与解码 | 客户端本地 |
|---|---:|---:|---:|---:|---:|---:|---:|
| common | 202.067 | 176.786 | 0.168 | 17.745 | 1.610 | 1.531 | 4.227 |
| policy | 299.434 | 203.944 | 0.247 | 9.890 | 73.754 | 4.557 | 7.043 |
| reconciliation | 213.673 | 165.752 | 0.082 | 39.550 | 0.527 | 2.994 | 4.768 |

每项平均/P50/P95、调用次数与完整嵌套路径见profile_summary.json；下表展示各子步骤独占时间，单位ms。

| 阶段与子步骤 | 平均 | P50 | P95 | 平均调用次数/epoch |
|---|---:|---:|---:|---:|
| common/rq6_batched_control_transport._batch_view | 0.031139 | 0.029448 | 0.041691 | 1.000 |
| common/rq6_batched_control_transport._batch_view/rq6_batched_control_transport._canonical_digest | 0.201409 | 0.193806 | 0.274321 | 1.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._accounting_snapshot | 0.015035 | 0.013971 | 0.019548 | 1.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._accounting_snapshot/targeted_probe._tensor_ids | 0.017749 | 0.017194 | 0.021180 | 1.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._global_maps | 0.099275 | 0.099196 | 0.126748 | 1.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._global_maps/targeted_probe._tensor_ids | 0.204331 | 0.204229 | 0.264674 | 14.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._global_maps/targeted_probe._tensor_sha256 | 0.379368 | 0.339692 | 0.754108 | 16.542 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._global_maps/targeted_probe._tensor_sha256/targeted_probe._tensor_ids | 0.002459 | 0.002378 | 0.003187 | 1.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot | 3.782998 | 3.477959 | 6.372371 | 14.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot/targeted_probe._find_exact_node | 3.603393 | 3.373179 | 6.212819 | 14.000 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot/targeted_probe._tensor_ids | 0.797384 | 0.747575 | 1.277894 | 58.500 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot/targeted_probe._tensor_sha256 | 1.794651 | 1.648459 | 3.490518 | 74.292 |
| common/rq6_batched_control_transport._batch_view/targeted_probe._validate_runtime_scope | 0.013035 | 0.012280 | 0.015445 | 1.000 |
| common/rq6_batched_control_transport._parse_handles | 6.381922 | 5.853900 | 10.777925 | 1.000 |
| policy/targeted_probe._accounting_snapshot | 0.098517 | 0.096455 | 0.103381 | 8.000 |
| policy/targeted_probe._accounting_snapshot/targeted_probe._tensor_ids | 0.139312 | 0.137179 | 0.145510 | 8.000 |
| policy/targeted_probe._global_maps | 0.766432 | 0.780773 | 0.954267 | 8.000 |
| policy/targeted_probe._global_maps/targeted_probe._tensor_ids | 1.551905 | 1.545932 | 2.159549 | 112.000 |
| policy/targeted_probe._global_maps/targeted_probe._tensor_sha256 | 2.991863 | 2.832223 | 4.410868 | 132.333 |
| policy/targeted_probe._global_maps/targeted_probe._tensor_sha256/targeted_probe._tensor_ids | 0.018639 | 0.016856 | 0.022473 | 8.000 |
| policy/targeted_probe._validate_runtime_scope | 0.084157 | 0.071066 | 0.132329 | 8.000 |
| policy/wp3b_end_to_end_transport.inspect_resident_fa_frontier | 2.400070 | 2.228542 | 3.267194 | 4.000 |
| policy/wp3b_end_to_end_transport.semantic_cache_snapshot | 0.215671 | 0.197855 | 0.358732 | 8.000 |
| policy/wp3b_end_to_end_transport.semantic_snapshot_differences | 1.293577 | 1.334626 | 1.634983 | 4.000 |
| reconciliation/SGLangAdapter.evict_mamba_only | 0.108411 | 0.106340 | 0.164056 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter._capture_target_snapshot | 0.171845 | 0.163923 | 0.259840 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter._evict_mamba_component_only | 0.581747 | 0.536798 | 0.900837 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter._evict_mamba_component_only/UnifiedRadixCache.sanity_check | 0.673103 | 0.632846 | 1.217157 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter._find_exact_node | 1.984783 | 1.852251 | 3.363141 | 21.500 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter._validate_postconditions | 0.176159 | 0.171652 | 0.270616 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter._validate_target | 2.597596 | 2.424216 | 4.440909 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter.validate_runtime_capabilities | 0.225547 | 0.224774 | 0.344375 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter.validate_runtime_capabilities/SGLangAdapter._validate_runtime_version | 0.028351 | 0.028140 | 0.042649 | 10.750 |
| reconciliation/SGLangAdapter.evict_mamba_only/SGLangAdapter.validate_runtime_capabilities/SGLangAdapter._validate_runtime_version/SGLangAdapter._read_sglang_version | 4.593928 | 4.450943 | 6.965608 | 10.750 |
| reconciliation/SGLangAdapter.validate_runtime_capabilities | 0.056167 | 0.052243 | 0.089941 | 1.000 |
| reconciliation/SGLangAdapter.validate_runtime_capabilities/SGLangAdapter._validate_runtime_version | 0.004114 | 0.003883 | 0.006503 | 1.000 |
| reconciliation/SGLangAdapter.validate_runtime_capabilities/SGLangAdapter._validate_runtime_version/SGLangAdapter._read_sglang_version | 0.950641 | 0.808882 | 1.759464 | 1.000 |
| reconciliation/UnifiedRadixCache.sanity_check | 0.048456 | 0.049665 | 0.059481 | 1.000 |
| reconciliation/rq6_batched_control_transport._batch_view | 0.049746 | 0.049820 | 0.065764 | 2.000 |
| reconciliation/rq6_batched_control_transport._batch_view/rq6_batched_control_transport._canonical_digest | 0.341012 | 0.345725 | 0.455725 | 2.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._accounting_snapshot | 0.028063 | 0.026983 | 0.035452 | 2.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._accounting_snapshot/targeted_probe._tensor_ids | 0.036106 | 0.035436 | 0.039165 | 2.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._global_maps | 0.182730 | 0.183590 | 0.228493 | 2.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._global_maps/targeted_probe._tensor_ids | 0.247848 | 0.237109 | 0.315142 | 17.250 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._global_maps/targeted_probe._tensor_sha256 | 0.717875 | 0.648644 | 1.208377 | 33.083 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._global_maps/targeted_probe._tensor_sha256/targeted_probe._tensor_ids | 0.004683 | 0.004540 | 0.005074 | 2.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot | 7.545952 | 7.212420 | 12.768361 | 28.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot/targeted_probe._find_exact_node | 7.167094 | 6.803202 | 12.400694 | 28.000 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot/targeted_probe._tensor_ids | 0.902424 | 0.893844 | 1.402543 | 75.750 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._path_snapshot/targeted_probe._tensor_sha256 | 3.458553 | 3.477247 | 5.385288 | 148.583 |
| reconciliation/rq6_batched_control_transport._batch_view/targeted_probe._validate_runtime_scope | 0.022456 | 0.020996 | 0.029145 | 2.000 |
| reconciliation/rq6_batched_control_transport._parse_handles | 6.295415 | 6.061951 | 10.745338 | 1.000 |
| reconciliation/rq6_batched_control_transport._proof | 0.032602 | 0.031818 | 0.047324 | 1.000 |
| reconciliation/rq6_batched_control_transport._validate_identity_and_residency | 0.009580 | 0.009612 | 0.013118 | 1.000 |

## 调用与安全边界

每epoch公共读取1次RPC、frontier读取4次串行RPC、协调1次RPC。TP/PP/DP均为1，无跨rank/NCCL调用；调用跨worker与scheduler进程，经本机TCP、服务线程、队列及事件交接。
每个frontier调用读取前后各一次全树、分配器和语义状态，共8份快照；四次查询间没有请求执行或状态变更。公共view读取一次全部候选，协调再读取驱逐前后两份view；候选路径逐checkpoint串行读取，共享祖先索引被重复摘要。
同一scheduler安全点内，相邻frontier查询的前一后态可以作为后一前态，但每个查询仍必须独立比较；不能仅检查整批首尾而遗漏中间状态变化。
不可移出的检查：空闲及运行模式、句柄精确节点/前缀、决策view摘要匹配、观察非干扰、每候选驻留、每次驱逐目标/后置条件、组件隔离、独立容量、全局仅目标状态改变以及最终sanity。
可移出的仅是日志落盘、结果汇总、参考oracle重算及审计展示；冻结E2E已经把这些放在区间外。诊断性输出不等于用于安全验证的状态读取，不能因其位于测试目录而删除。
可安全复用候选token序列、节点预期身份和当前epoch句柄对象；不得跨状态变更复用recurrent驻留、allocator、LRU、view digest或frontier。只读view内重复tensor索引摘要可局部复用，离开该只读操作即失效。

## 历史数值归因限制

历史reconciliation可严格分成逐驱逐和、服务端其他工作、服务端外余量；历史frontier没有内部时钟，无法唯一追溯293.093ms的各子步骤。所有可核实历史统计保存在frozen_measured。新测各阶段闭合误差见max_closure_error_ms。
提交入口前包含连接/编码、服务线程读取与解码及调度影响；事件交接包含唤醒和重新获得执行权。连接/编码/解码另有计时，但没有把剩余时间武断归因于GIL或操作系统。tensor.cpu/tolist可能隐式同步，未新增GPU synchronize，不能宣称测得每次设备同步时间。
