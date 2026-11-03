# 实际运行与复算入口

所有命令在仓库根目录执行。没有依赖安装、镜像构建或模型下载。

CPU：冻结镜像sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926，无GPU、无网络，仓库只读挂载；`python3 -B -m unittest evaluation.control_path_separation_20261101.test_separation tests.test_strict_control tests.test_strict_epoch -v`，65项通过，cpu_tests.log。

首次GPU0仅做资源预检，GPU已被其他进程占用，没有模型启动。随后首次GPU运行前锁定GPU1，原锁和修订锁都保留。

实际GPU入口：

```bash
python3 -B -m evaluation.control_path_separation_20261101.execute_gpu1 freeze
python3 -B -m evaluation.control_path_separation_20261101.execute_gpu1 smoke
python3 -B -m evaluation.control_path_separation_20261101.execute_gpu1 first_pair
python3 -B -m evaluation.control_path_separation_20261101.execute_gpu1 remaining
python3 -B -m evaluation.control_path_separation_20261101.execute_gpu1 three_policy
```

以上均已完成，**不要直接重跑**；现有目录使用x模式拒绝覆盖。每臂完整Docker参数见arms/*/command.json，父级日志分别为smoke_launcher.log、first_pair_launcher.log、remaining_launcher.log、three_policy_launcher.log；每臂GPU、返回码、清理与完整日志在各自目录。

GPU全部结束后执行离线报告生成：

```bash
python3 -B -m evaluation.control_path_separation_20261101.build_report
```

报告生成也拒绝覆盖CSV；如需复算，先在独立审阅副本中调整输出路径，不修改本冻结结果包。analyze.py给出全部统计定义，build_report.py给出完整物理/来源/响应复核。五轮中间物理复核为five_pair_physical_equivalence.json，未与本轮性能测量同时执行。

原T0/T3/T4定义不变；新增T5审计完成时间。Root源码差异见implementation.diff；精确身份见execution_lock_gpu1.json、baseline_source_identity.json与final_source_manifest.json。旧execute.py/execution_lock.json对应未启动模型的GPU0预检计划，本轮有效入口及协议是execute_gpu1.py/execution_lock_gpu1.json。
