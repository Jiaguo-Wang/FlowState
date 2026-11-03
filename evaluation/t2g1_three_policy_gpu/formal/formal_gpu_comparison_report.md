# T2-G2 正式GPU实验断点接管报告

**最终状态：BLOCKED（执行期间出现外部GPU占用，已按冻结协议停止）。**

## 接管与门禁

原 verify_formal.py 的字符串/字典类型错误在接管前已经修复保存，本次没有再次修改。三个既有 Gate 未重跑；原CPU检查器33项与补充原始证据173项通过后，检查程序生成 gate_passed.json。门禁覆盖选集、T/E/G、Prefill、FA保护、13项Recurrent-only驱逐、Native来源/COW、Cold逐Token、形状预热、请求窗口编译和生命周期收尾。详见 formal_verification_gate.json 与 takeover/evidence_gate.json。

接管时主矩阵尚无产物，未发现活动实验launcher或Engine，GPU0为H100 PCIe、14MiB、无计算进程。核对冻结协议、来源、模型镜像及实现摘要后，按授权启动原 launch_formal.py main；没有新增GPU预实验，没有修改策略、调度、选集或原生控制语义。结束后再次核验 implementation_lock_main.json，所有文件摘要一致。

## 实际执行与停止原因

正式主矩阵尝试4/60臂：3臂完整通过，1臂收尾资源门禁失败，56臂未执行。尚未完成任何完整四Pending轮次，更未完成五轮正式配对。

|臂|状态|H/E/G|Prefill Tokens|原始TTFT ms|原始CUDA Prefill ms|
|---|---|---|---:|---:|---:|
|m1A_FS|PASS|4096/0/4096|4128|159.498027|151.381210|
|m1A_MC|PASS|4096/1024/3072|3104|128.270915|117.393570|
|m1A_KV|PASS|4096/4032/64|96|39.933084|33.020863|
|m1B_FS|RESOURCE_GATE_FAIL|5120/5120/0|32|49.989876|42.964577|

上述数字只保留已经发生的尝试，不据此计算正式性能收益。第四臂不能因 summary.status=PASS 而升级为完整通过。

m1B_FS 进程返回0，Engine shutdown_returned=true，专用容器已不存在。其三次清理观察显存依次为14、17、1569MiB；第三次出现外部PID 1303777，程序路径为 /home/yangshuo/robot-learning/.venv/bin/python，占1546MiB。随后只读核查仍有该外部进程，占1652MiB，GPU总用量1675MiB；没有本轮专用容器或launcher残留。未终止或干预该用户任务。

证据支持“收尾时失去独占空卡条件”，不支持“恢复计算错误”或“本轮Engine清理泄漏”。外部进程准确启动时刻及是否与第四臂计时窗口重叠为UNKNOWN；该臂保留为资源门禁失败，不用于正式收益统计。

## 证据与统计边界

- Gate三臂仅作正确性/有效性门禁，全部排除性能统计。
- 已尝试四臂的局部只读复核见 takeover/interrupted_main_verification.json；唯一失败项为第四臂三次空卡收尾。原始文件不覆盖。
- 全60臂账本见 takeover/interrupted_main_ledger.csv；缺测不填零、不估计、不以Gate替代。未调用要求完整60臂的分析程序，未生成五轮配对或四Pending等权总体结论。
- 模型J、实际E/G/Prefill与实测TTFT分开保留；控制RPC及epoch_wall原始值见账本，不与其子区间重复相加，也不解释为生产净收益。
- Marconi-Evict为Marconi-Evict Adaptation，KVFlow为KVFlow-inspired；不称完整原论文系统。当前实验应用冻结选集，不证明在线策略求解。跨Workflow物理血缘仍为PROVENANCE_UNVERIFIED。

## 断点与保全

最后完整通过：m1A_KV。失败对象：m1B_FS。原计划下一未执行臂：m1B_MC。本次没有重试或自动等待空卡。

原launcher会独占创建 implementation_lock_main.json、execution_started_main.json 和逐臂目录，并不支持跳过已有主臂；因此不能直接再次执行 launch_formal.py main。任何后续恢复都需要先明确处理既有失败臂和不可覆盖的执行账本，不能绕过冻结测量条件。本轮不擅自实现续跑入口或补跑。

关键证据：launch_main_failure.json、arm_status_main.json、m1B_FS_cleanup.json、m1B_FS_process.json、m1B_FS/run/{summary,output,engine_cleanup,reconciliation,post_request}.json、runtime_main.log、main_takeover_host.log、takeover/recovery_inventory.json。

本轮已停止。没有修改冻结输入、生产代码或原有GPU产物。
