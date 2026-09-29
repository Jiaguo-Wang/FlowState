# 包含控制路径的单令牌恢复 epoch 实验

状态：PASS。216/216 独立运行与864/864请求通过正确性及清理门禁。

统计单位为24个快照；每种策略先对三次重复取均值。正配对差表示基线减FlowState，置信区间为10000次round分层bootstrap。

| 策略 | epoch(ms) | 平均boundary至完成(ms) | 平均请求执行(ms) | 总控制(ms) |
|---|---:|---:|---:|---:|
| LRU | 2952.253 | 1899.797 | 163.191 | 149.455 |
| Marconi | 1875.991 | 1259.647 | 127.987 | 97.957 |
| FlowState | 2425.641 | 1498.533 | 91.152 | 131.904 |

## 配对结果

相对 LRU：平均改善 526.612 ms；95% CI [-1057.4400107006231, 2214.9434228754703]；胜/平/负 15/0/9。
负例及平局：rq3-openhands-main-g005-round3、rq3-openhands-main-g037-round3、rq3-openhands-main-g050-round4、rq3-openhands-main-g094-round4、rq3-openhands-main-g119-round5、rq3-openhands-main-g122-round4、rq3-openhands-main-g167-round5、rq3-openhands-main-g177-round3、rq3-openhands-main-g184-round2。
相对 Marconi：平均改善 -549.649 ms；95% CI [-1784.9132109125455, 699.3964541832602]；胜/平/负 8/0/16。
负例及平局：rq3-openhands-main-g015-round5、rq3-openhands-main-g016-round2、rq3-openhands-main-g037-round3、rq3-openhands-main-g050-round4、rq3-openhands-main-g058-round4、rq3-openhands-main-g094-round4、rq3-openhands-main-g112-round2、rq3-openhands-main-g119-round5、rq3-openhands-main-g122-round4、rq3-openhands-main-g156-round2、rq3-openhands-main-g167-round5、rq3-openhands-main-g171-round5、rq3-openhands-main-g172-round2、rq3-openhands-main-g177-round3、rq3-openhands-main-g184-round2、rq3-openhands-main-g199-round5。

## 分段均值

| 策略 | 公共观测 | 专属观测 | 输入构造 | 选择 | 协调 | 请求间验证及派发 |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 94.956 | 0.005 | 0.290 | 0.092 | 54.113 | 2150.033 |
| Marconi | 54.511 | 0.005 | 0.387 | 0.215 | 42.839 | 1266.087 |
| FlowState | 66.225 | 24.985 | 0.240 | 0.929 | 39.525 | 1929.129 |

## 分层与重复稳定性

round 2：LRU 1597.991 ms；Marconi 874.818 ms；FlowState 1092.742 ms。
round 3：LRU 1459.120 ms；Marconi 1686.276 ms；FlowState 1468.001 ms。
round 4：LRU 2160.012 ms；Marconi 3166.646 ms；FlowState 4702.516 ms。
round 5：LRU 6591.890 ms；Marconi 1776.226 ms；FlowState 2439.303 ms。
LRU 的epoch重复CV：中位数 0.2600，P95 1.3372，最大 1.4979。
Marconi 的epoch重复CV：中位数 0.3389，P95 0.7774，最大 1.4027。
FlowState 的epoch重复CV：中位数 0.3641，P95 1.2797，最大 1.4736。

## 盈亏平衡外推

LRU：{"total_control_difference_ms": -17.55117193857828, "policy_specific_control_difference_ms": 25.766941777777777, "common_observation_difference_ms": -28.730845986111106, "reconciliation_difference_ms": -14.587267791666669, "request_saving_ms": 72.03937509375001, "validation_dispatch_difference_ms": -220.9036464935998, "policy_specific_break_even_count": 1, "total_control_break_even_count": 1, "validation_adjusted_break_even_count": 1, "说明": "仅外推：控制成本固定、每请求节省及验证差可重复；空值表示正节省条件不满足，非实测新请求数。"}
Marconi：{"total_control_difference_ms": 33.947070611847764, "policy_specific_control_difference_ms": 25.546207222222222, "common_observation_difference_ms": 11.714065611111117, "reconciliation_difference_ms": -3.3132021944444574, "request_saving_ms": 36.83494245486111, "validation_dispatch_difference_ms": 663.0421487812196, "policy_specific_break_even_count": 1, "total_control_break_even_count": 1, "validation_adjusted_break_even_count": null, "说明": "仅外推：控制成本固定、每请求节省及验证差可重复；空值表示正节省条件不满足，非实测新请求数。"}

完整逐快照配对差、各指标负例、分层置信区间及CV见 summary.json；未删除负例、平局或异常值。
盈亏平衡仅使用本轮测量，属于固定控制成本与平均单请求节省可外推的假设，不代表新增请求数量的实测。
