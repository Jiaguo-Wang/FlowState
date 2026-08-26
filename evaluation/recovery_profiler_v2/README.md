# Recovery Profiler v2

## 目的

本目录保存独立 Recovery Profiler v2。它在与 Step 9B 相同的 SGLang 0.5.17、Qwen3.5-9B、H100 PCIe、TP=1、流式单 token 输出和 client-side TTFT 边界下，测量真实 recovery gap 对 incremental TTFT 的影响。

Step 9B 是 evaluation data；Recovery Profiler v2 是独立 calibration data。两者严格分离。Step 9B 的 latency samples 不进入本目录的任何模型拟合，只在模型评估结束后使用 Step 9C 已冻结的四个汇总值做外部对照。

## 冻结数据划分

- Calibration gaps：0、4096、8192、16384、32768。
- Held-out validation gaps：2048、6144、12288、24576。
- 每个 gap 有 2 次 warmup 和 12 次 measured。
- validation gaps 不参与线性模型或分段线性模型拟合。
- 固定顺序 seed：20260826。

每个 case 都从 fresh/flush cache 开始，在相同 32K physical FA frontier 下，只改变可执行 recurrent frontier。target request 的 token shape、单 token 输出、runtime-ready 检查与 TTFT 计时函数直接复用 Step 9B 的正式实现。

## 模型比较

离线分析比较三个候选：

1. 当前冻结旧 Phi；
2. calibration points 上通过原点的线性回归；
3. calibration knots 上的单调分段线性插值。

三个模型均满足 `Phi(0)=0` 和成本单调非减。模型只按 held-out validation 的 MAE、MAPE 和最大绝对误差比较。本步骤不会写回 `flowstate/recovery_model.py`。

## Artifacts

- `profile_runner.py`：单次 GPU profiling runner。
- `analyze.py`：纯 CPU 离线汇总和模型比较。
- `raw_samples.jsonl`：126 个逐 case 原始样本。
- `calibration_summary.csv`：calibration gap 统计。
- `validation_summary.csv`：held-out gap 统计与三个模型预测。
- `model_comparison.json`：拟合参数、held-out 误差、最佳模型和 Step 9C 对照。
- `run_metadata.json`：冻结环境、顺序、运行状态和完整性信息。

## 数据质量门

- runtime H/E/G 必须精确等于 target H/E/G。
- warmup 不进入统计。
- 每个 gap 必须恰有 12 个 measured 样本。
- calibration 与 validation gap 集合必须不重叠。
- 任一 correctness failure 会停止整个 profiler，不会静默删除样本。
- 分段 knots 若不单调会直接失败，不对数据做静默修正。
