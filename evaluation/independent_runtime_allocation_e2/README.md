# E2产物索引与复现边界

- `protocol.json`：GPU开始前冻结的顺序、预算、规模及失败规则。
- `prepare.py`、`source_identity.json`：验证冻结来源，装配已有Token；不重新分词。
- `runner.py`、`launch.py`：复用C3只读采集入口，每组独立容器/Engine。只接受空闲GPU，不覆盖既有执行目录。
- `execution_lock.json`：执行前代码与协议摘要。
- `cpu_regression.json`：B5-C4的24个冻结策略条件回归。
- `w1`–`w4`：每组原始输入、命令、日志、历史实际接收记录、最终完整物理Census、Strict Epoch、清理和采集摘要。
- `evaluate.py`：无Runtime调用的严格CPU评价，函数`evaluate(raw)`可直接只读重算单组；返回结果，不写冻结文件。
- `summarize.py`、`report.py`、`mechanism_details.py`：派生表格与中文报告。产物多数使用独占创建，防止覆盖；不要对既有输出重新启动Launcher。
- `validation_summary.json`：逐组完整性、逐行目标/E/G和独立枚举验证。
- `final_integrity.json`：全部432项来源以及GPU执行代码保持不变。

仅CPU只读复核示例：

```bash
/home/wjg/uv312/bin/python -B - <<'PY'
from evaluation.independent_runtime_allocation_e2.evaluate import evaluate, OUT
for group in ('w1', 'w2', 'w3', 'w4'):
    result = evaluate(OUT / group)
    print(group, result['n'], result['K_min'], len(result['rows']))
PY
```

本轮已结束。不自动重跑GPU，不扩充工作负载，不进入下一阶段。所有模型化ms均非GPU时延。
