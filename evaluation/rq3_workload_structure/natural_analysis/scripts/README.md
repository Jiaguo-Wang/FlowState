# 独立分析复现

仅使用已经安装的 `/home/wjg/uv312/bin/python`、标准库和 pyarrow。无需GPU、Docker、生产求解器、网络或新依赖。原始数据保持只读。

三个入口依次运行：

```bash
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/natural_analysis/scripts/analyze.py --out evaluation/rq3_workload_structure/natural_analysis/reproduce_01
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/natural_analysis/scripts/make_report.py --input-dir evaluation/rq3_workload_structure/natural_analysis/reproduce_01
/home/wjg/uv312/bin/python -B evaluation/rq3_workload_structure/natural_analysis/scripts/validate_outputs.py --input-dir evaluation/rq3_workload_structure/natural_analysis/reproduce_01
```

`reproduce_01`必须不存在。不要把复现输出指向原结果目录。统计入口记录全部源文件前后SHA并拒绝内容变化；报告入口拒绝覆盖既有报告；恒等式核验入口对已有结果只读比较，不覆盖。

- `analyze.py`：直接重读原始逐预算评估、canonical物理快照、Formal定义与四策略逐条件输出；独立实现Phi、结构统计、覆盖证书、轨迹再评价及预算比较，不导入旧汇总或调用策略选择函数。
- `make_report.py`：从新计算结果生成中文报告和来源清单，补独立收益/位置推荐检查及旧内容锁交叉核验。
- `validate_outputs.py`：对四策略全部2292份结果复核 `F(S)=J(空集)−J(S)`，记录历史内容锁来源。

K_min位掩码状态数不超过256；只计算必要的离线覆盖统计，不对大候选域进行无界Exact枚举。原始调用容器与消息只能用于其实际提供的来源字段，不能用于伪造真实Token分支关系或Strict认证。
