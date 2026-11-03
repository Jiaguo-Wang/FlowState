# 固定K扫描复现

本目录只执行CPU，不调用SGLang或GPU，不更改候选/兼容/Phi，不修改旧结果。依赖使用项目已有环境，不安装新包。

```bash
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/scripts/run_budget.py --out evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/reproduce_01
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/scripts/write_reports.py --out evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/reproduce_01
```

请使用没有既有结果的新目录。脚本拒绝覆盖已有JSON/CSV/Markdown。第一步先核对来源锁和三预算去重，再复现573条件×4策略；任何失败保存门禁原因并停止，不扫描K。全部通过后才按固定K1–4执行，Exact仅在12/4/4/794门禁内枚举。

FlowState调用真实生产`_select_candidates` Greedy核心，使用固定Formal评分闭包；不调用会混入Exact的`plan_strict`，不伪造Workflow或Lineage。PCB/LDR/WFC直接导入原锁定模块。

计时协议先于策略执行写入protocol.json；输入准备与外部评价不计时，真实函数内部的诊断轨迹构造仍计时。15批×5次，报告的是批均单次时间中位和P95。禁止解释为线上GPU开销或生产尾延迟。

第二步只读本轮结果，用独立max-benefit恒等式在同一门内复核Oracle并生成两份报告及九个AgentX详细实例。不会启动后续实验。
