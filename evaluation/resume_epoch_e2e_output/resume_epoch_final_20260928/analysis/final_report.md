# 包含控制路径的单令牌恢复 epoch 实验

状态：PASS。216/216 独立运行与864/864请求通过正确性及清理门禁。

统计单位为24个快照；每种策略先对三次重复取均值。正配对差表示基线减FlowState，置信区间为10000次round分层bootstrap。

| 策略 | epoch(ms) | 平均boundary至完成(ms) | 平均请求执行(ms) | 总控制(ms) |
|---|---:|---:|---:|---:|
| LRU | 2130.623 | 1489.458 | 163.375 | 232.892 |
| Marconi | 1888.372 | 1283.345 | 128.157 | 267.758 |
| FlowState | 2078.838 | 1519.909 | 92.537 | 656.771 |

## 配对结果

相对 LRU：平均改善 51.785 ms；95% CI [-226.75770578318176, 384.20600833826575]；胜/平/负 8/0/16。
负例及平局：rq3-openhands-main-g005-round3、rq3-openhands-main-g015-round5、rq3-openhands-main-g016-round2、rq3-openhands-main-g037-round3、rq3-openhands-main-g069-round3、rq3-openhands-main-g070-round4、rq3-openhands-main-g119-round5、rq3-openhands-main-g122-round4、rq3-openhands-main-g156-round2、rq3-openhands-main-g157-round3、rq3-openhands-main-g167-round5、rq3-openhands-main-g171-round5、rq3-openhands-main-g172-round2、rq3-openhands-main-g176-round2、rq3-openhands-main-g177-round3、rq3-openhands-main-g184-round2。
相对 Marconi：平均改善 -190.466 ms；95% CI [-384.9105204297437, 35.20921325816038]；胜/平/负 11/0/13。
负例及平局：rq3-openhands-main-g005-round3、rq3-openhands-main-g015-round5、rq3-openhands-main-g037-round3、rq3-openhands-main-g050-round4、rq3-openhands-main-g069-round3、rq3-openhands-main-g070-round4、rq3-openhands-main-g119-round5、rq3-openhands-main-g157-round3、rq3-openhands-main-g167-round5、rq3-openhands-main-g171-round5、rq3-openhands-main-g172-round2、rq3-openhands-main-g176-round2、rq3-openhands-main-g199-round5。

## 分段均值

| 策略 | 公共观测 | 专属观测 | 输入构造 | 选择 | 协调 | 请求间验证及派发 |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 100.963 | 0.005 | 0.276 | 0.095 | 131.554 | 1244.232 |
| Marconi | 119.702 | 0.005 | 0.348 | 0.192 | 147.511 | 1107.986 |
| FlowState | 150.926 | 293.093 | 0.299 | 1.115 | 211.337 | 1051.919 |

## 分层与重复稳定性

round 2：LRU 1193.157 ms；Marconi 1366.672 ms；FlowState 1349.301 ms。
round 3：LRU 1793.007 ms；Marconi 1649.117 ms；FlowState 1868.541 ms。
round 4：LRU 2856.588 ms；Marconi 2288.995 ms；FlowState 2097.198 ms。
round 5：LRU 2679.741 ms；Marconi 2248.704 ms；FlowState 3000.312 ms。
LRU 的epoch重复CV：中位数 0.2943，P95 0.6843，最大 1.2774。
Marconi 的epoch重复CV：中位数 0.3724，P95 0.5552，最大 0.6049。
FlowState 的epoch重复CV：中位数 0.4515，P95 0.6721，最大 0.7444。

## 盈亏平衡外推

LRU：{"total_control_difference_ms": 423.87926366594104, "policy_specific_control_difference_ms": 294.13270825000006, "common_observation_difference_ms": 49.96286775, "reconciliation_difference_ms": 79.78368779166664, "request_saving_ms": 70.83789152777776, "validation_dispatch_difference_ms": -192.31287232200611, "policy_specific_break_even_count": 5, "total_control_break_even_count": 6, "validation_adjusted_break_even_count": 4, "说明": "仅外推：控制成本固定、每请求节省及验证差可重复；空值表示正节省条件不满足，非实测新请求数。"}
Marconi：{"total_control_difference_ms": 389.01303523116644, "policy_specific_control_difference_ms": 293.9630510416667, "common_observation_difference_ms": 31.22372379166667, "reconciliation_difference_ms": 63.82626045833331, "request_saving_ms": 35.61993483333332, "validation_dispatch_difference_ms": -56.067317142941874, "policy_specific_break_even_count": 9, "total_control_break_even_count": 11, "validation_adjusted_break_even_count": 8, "说明": "仅外推：控制成本固定、每请求节省及验证差可重复；空值表示正节省条件不满足，非实测新请求数。"}

完整逐快照配对差、各指标负例、分层置信区间及CV见 summary.json；未删除负例、平局或异常值。
盈亏平衡仅使用本轮测量，属于固定控制成本与平均单请求节省可外推的假设，不代表新增请求数量的实测。
