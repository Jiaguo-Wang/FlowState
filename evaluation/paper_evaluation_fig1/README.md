# VI-B 主实验统计与可复现绘图（C/K3 正式 GPU 实验）

本目录把 `evaluation/c_formal_perf_gpu/` 的 60 个正式 GPU 测量臂转化为论文 VI-B
（Effectiveness of Executable-State Allocation）可直接使用的统计 CSV 与图表。
**不重新做实验**：全部数字只读自冻结原始产物，可一键复现。

## 输入数据来源

- 原始文件：`evaluation/c_formal_perf_gpu/m{1-5}{A-D}_{FS,SA,WFC}/run/summary.json`（60 臂，
  每臂一个生命周期的全精度计时与恢复证据）；完整性哈希台账见 `provenance.json`。
- 冻结选集合同：各臂 `policy_plans.json`（FlowState-Greedy / StandaloneTopK / WFC 的
  selected_checkpoint_ids 与 expected_E_G）。
- 参考聚合值（仅用于一致性校验，不用于绘图）：`c_formal_perf_gpu/paper_pending_aggregate_audit.md`
  的等权 TTFT 中位 77.829 / 111.960 / 125.262 ms。
- **范围**：仅 60 正式臂；g1/g2 门禁臂、旧 Pilot、C/K3 CPU 模拟与其他计时协议一律排除。

## 统计定义

- **Per-Pending TTFT（A1）**：每策略 × 每 Pending 的五轮 TTFT 中位数。
- **Aggregated TTFT（A2）**：每轮先对四个 Pending 的 TTFT 取等权算术平均，再对五轮均值取中位数。
  不将四个 Pending 的中位数直接平均。
- **配对比较（A3）**：同轮编号下 FS−SA、FS−WFC 的等权均值之差。百分比两种口径分列：
  `pct_paired_vs_base`＝同轮配对差 / 该轮基线等权均值；`pct_aggregate_median_ratio`＝聚合中位数比值。
  两者口径不同，不可混用。
- **恢复行为（A4）**：T/E/G、Prefill Tokens 逐臂核验（E/G 必须等于冻结选集合同，
  Prefill Tokens 必须等于 G+32）。
- **完整性门禁（A5）**：样本数恰 60、12 格各五轮、全 PASS、Cold 参照逐 Token 一致、
  无重复缺失、聚合值与冻结参考一致（±0.001ms）。任一失败即拒绝出图。

## 依赖

- 统计：Python 3（仅标准库）。
- 绘图：matplotlib ≥3.5、numpy（本机用 `/home/wjg/uvgood/bin/python`，matplotlib 3.11 / numpy 2.4）。

## 一键复现

```bash
cd <仓库根>
python3 evaluation/paper_evaluation_fig1/scripts/extract_stats.py
/home/wjg/uvgood/bin/python evaluation/paper_evaluation_fig1/scripts/plot_fig_vi_b.py
```

两脚本均接受可选参数 `<输出目录>`（默认本目录），可在干净目录中整链重生成；
已验证：CSV 与 PNG 逐字节一致，SVG 仅嵌入的 dc:date 时间戳不同（图形内容一致）。

## 文件用途

| 文件 | 用途 |
|---|---|
| `scripts/extract_stats.py` | 只读提取 60 臂 → 统计 CSV + 门禁 checks + provenance |
| `scripts/plot_style.py` | 统一样式（单栏 3.35in、Okabe-Ito 色盲安全配色、矢量文字、跨图一致），供 VI-C/D/E 复用 |
| `scripts/plot_fig_vi_b.py` | 只读 data/*.csv → (a) Per-Pending 分组柱状图、(b) 聚合柱状图+真实五轮散点 |
| `data/raw_arms.csv` | 60 臂原始行（TTFT/Prefill/T/E/G/Cold/来源路径+sha256） |
| `data/pending_ttft.csv` | A1：12 格五轮中位与逐轮值 |
| `data/aggregate_ttft.csv` | A2：聚合中位与五轮等权均值 |
| `data/paired_comparison.csv` | A3：逐轮配对差+中位行+两种百分比口径 |
| `data/recovery_behavior.csv` | A4：T/E/G/Prefill/TTFT 表（Table VI-B） |
| `figures/` | fig_vi_b_pending_ttft 与 fig_vi_b_aggregate_ttft 各 pdf/svg/png |
| `checks.json` / `provenance.json` | 门禁结果 / 输入与臂文件哈希台账 |

## 测量边界

- 图为"固定四 Pending 需求集合上的等权独立请求 TTFT"，非连续 Workflow E2E、并发完成时间或 Goodput。
- C/K3 的 J 为模型数值；GPU 实验使用冻结选集执行（PILOT_POLICY 读 policy_plans.json），
  不在线求解。五轮重复不构成统计显著性检验，图上不做显著性标注。
- J/TTFT 的口径与 b5d4b、E4/E5 等研究仪表协议不可混算（本目录只用 c_formal 生产路径 TTFT）。

## 验收结论

Task D 九项检查全部通过（2026-10-25）：可溯源、60 臂门禁 PASS、聚合与冻结参考一致、
柱高即 CSV 值、三格式齐备、干净目录整链重生成一致、脚本无绘图用硬编码数值
（extract 中的 77.829/111.960/125.262 仅作校验参照）、单栏字号可辨、跨图配色与命名一致。
