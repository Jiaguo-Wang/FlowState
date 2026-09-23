# RQ6-E 批量运行时控制优化

最终状态：RQ6_BATCHED_CONTROL_READY。

24个冻结快照的72/72个独立正式运行全部通过，正确性门禁全部为PASS。本次收尾仅整理现有结果并生成最终冻结清单，没有重新运行实验或测试，也没有启动Kimi。

## 性能

| 指标 | 优化前均值（ms） | 优化后均值（ms） | 均值降低 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 总控制路径 | 8398.847 | 629.827 | 92.501% | 13.335 |
| 状态批量读取（introspection） | 1913.301 | 322.573 | 83.141% | 5.931 |
| 状态批量协调（reconciliation） | 6483.256 | 304.785 | 95.299% | 21.272 |

优化后总控制路径P50为215.229 ms、P95为2708.593 ms、最大值为6305.152 ms；introspection的P95为1612.845 ms，reconciliation的P95为1237.127 ms。

同步RPC总数由3708降至144，每个epoch均值由51.500降至恰好2次：一次批量读取、一次批量协调。协调后的统一验证另行记录，不计入上述计时RPC口径。

性能结果来源：[聚合结果](aggregate_results.json)与[逐轮原始记录](../raw/per_epoch_overhead.jsonl)。

## 语义与正确性

selected-set等价性：72/72 PASS；H/E/G等价性：288/288三元组PASS；FA preservation：72/72 PASS。

快照摘要与循环状态驻留全部保持等价；无原生循环状态驱逐、意外重驻留、FA级联、未来信息或跨运行污染。每个epoch只在批量协调后执行一次统一重型验证。

正式运行结果来源：[运行收集状态](../correctness/collection_status.json)、[聚合正确性门禁](aggregate_results.json)与[独立运行记录](../runtime_runs/)。

## 测试

- RQ6-E相关测试：80 passed / 0 failed / 0 skipped / 2 warnings，耗时10.82秒。来源：[相关测试日志](../logs/rq6e_related_tests.log)。两条警告均为只读环境下pytest缓存写入警告。
- 完整CPU测试套件：888 passed / 0 failed / 0 skipped / 2 warnings，耗时3876.64秒（1:04:36），已正常完成。来源：[完整CPU测试日志](../full_cpu_suite.log)。两条警告分别为未注册的slow与timeout标记。

## 完整性与冻结

源码完整性：11/11 PASS，当前文件字节数及SHA-256与运行后源码清单一致；运行前后源码完整性状态为PASS。来源：[源码完整性记录](../correctness/source_integrity.json)与[运行后源码清单](../provenance/source_after.json)。

冻结基线artifact完整性：476/476 PASS，逐文件核对字节数与SHA-256。本次仅只读核对既有RQ6基线清单，没有修改任何冻结RQ2/RQ3/RQ4/RQ6基线artifact。

本次RQ6-E artifact路径：`evaluation/rq6_batched_control_output/rq6e_20260911_085820`。

最终冻结清单：[frozen_manifest.json](../correctness/frozen_manifest.json)。清单覆盖当前artifact内全部普通文件，排除清单自身，逐文件记录相对路径、字节数与SHA-256。

最终收尾核对记录：[final_freeze_validation.json](../correctness/final_freeze_validation.json)。
