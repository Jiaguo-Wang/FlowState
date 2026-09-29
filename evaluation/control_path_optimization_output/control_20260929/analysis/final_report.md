# FlowState控制路径优化与同协议E2E复测

状态：PASS。新216/216生命周期通过；逐run选择集与旧实验一致，864/864请求的H/E/G及长度一致。
统计单位仍为24个snapshot，三次重复先平均，再做10000次round分层bootstrap。
稳定净收益（同时相对LRU与Marconi）：否。

| 策略 | 旧epoch | 新epoch | 旧总控制 | 新总控制 | 旧请求执行均值 | 新请求执行均值 |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 2130.623 | 2952.253 | 232.892 | 149.455 | 163.375 | 163.191 |
| Marconi | 1888.372 | 1875.991 | 267.758 | 97.957 | 128.157 | 127.987 |
| FlowState | 2078.838 | 2425.641 | 656.771 | 131.904 | 92.537 | 91.152 |

## 新实验配对结果

| 对比 | 平均改善 | 中位改善 | 95%CI | 胜/平/负 |
|---|---:|---:|---|---|
| FlowState相对LRU | 526.612 | 156.510 | [-1057.440, 2214.943] | 15/0/9 |
| FlowState相对Marconi | -549.649 | -134.416 | [-1784.913, 699.396] | 8/0/16 |

## 前后控制分解

| 策略/版本 | 公共观测 | 专属观测 | 输入构造 | 选择 | 协调 |
|---|---:|---:|---:|---:|---:|
| LRU/旧 | 100.963 | 0.005 | 0.276 | 0.095 | 131.554 |
| LRU/新 | 94.956 | 0.005 | 0.290 | 0.092 | 54.113 |
| Marconi/旧 | 119.702 | 0.005 | 0.348 | 0.192 | 147.511 |
| Marconi/新 | 54.511 | 0.005 | 0.387 | 0.215 | 42.839 |
| FlowState/旧 | 150.926 | 293.093 | 0.299 | 1.115 | 211.337 |
| FlowState/新 | 66.225 | 24.985 | 0.240 | 0.929 | 39.525 |

## 全部epoch负例与平局

LRU：rq3-openhands-main-g005-round3、rq3-openhands-main-g037-round3、rq3-openhands-main-g050-round4、rq3-openhands-main-g094-round4、rq3-openhands-main-g119-round5、rq3-openhands-main-g122-round4、rq3-openhands-main-g167-round5、rq3-openhands-main-g177-round3、rq3-openhands-main-g184-round2。
Marconi：rq3-openhands-main-g015-round5、rq3-openhands-main-g016-round2、rq3-openhands-main-g037-round3、rq3-openhands-main-g050-round4、rq3-openhands-main-g058-round4、rq3-openhands-main-g094-round4、rq3-openhands-main-g112-round2、rq3-openhands-main-g119-round5、rq3-openhands-main-g122-round4、rq3-openhands-main-g156-round2、rq3-openhands-main-g167-round5、rq3-openhands-main-g171-round5、rq3-openhands-main-g172-round2、rq3-openhands-main-g177-round3、rq3-openhands-main-g184-round2、rq3-openhands-main-g199-round5。

## 语义边界与证据

优化仅涉及批量frontier、不可变句柄引用、无损消息编码和epoch连接复用。每个frontier分别通过非干扰检查；原动态view验证、逐驱逐adapter及请求间检查均保留。
不修改objective、T/E/G、online boundary、snapshot、预算、4请求/1token或engine配置。
profiling报告：../profiling/analysis/profiling_report.md；每个子步骤的平均/P50/P95与调用数：../profiling/analysis/profile_summary.json。
全部三项主指标及总控制的逐snapshot差值、负例、round分层与CV见comparison.json和../optimized_formal/analysis/summary.json。
这些结果仅适用于control-inclusive resume-epoch latency，不代表多token continuation或完整workflow完成时间。
