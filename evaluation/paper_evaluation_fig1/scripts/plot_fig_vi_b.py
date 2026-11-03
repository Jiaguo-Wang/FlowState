"""VI-B 正式实验绘图：只读 data/ 统计 CSV 生成两张图，不出现任何硬编码实验数值。

- Figure VI-B(a) 分组柱状图：各策略 × 各 Pending 的五轮 TTFT 中位数（含对 FlowState 不利的格子）；
- Figure VI-B(b) 聚合柱状图：每轮四 Pending 等权均值 → 五轮中位；叠加真实五轮等权均值散点，
  不虚构误差线、不做显著性标注、不截断纵轴。
"""
import csv,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as st

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DATA = OUT / 'data'
FIGS = OUT / 'figures'
ROUND_COLS = ['round_mean_r1', 'round_mean_r2', 'round_mean_r3', 'round_mean_r4', 'round_mean_r5']


def read_csv(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def main():
    st.apply()
    FIGS.mkdir(parents=True, exist_ok=True)
    pending = read_csv('pending_ttft.csv')
    aggregate = read_csv('aggregate_ttft.csv')
    checks = __import__('json').loads((OUT / 'checks.json').read_text())
    if not checks.get('all_pass'):
        raise SystemExit('extract_stats.py 门禁未通过，拒绝生成正式图表')

    # ---------- Figure VI-B(a)：Per-Pending TTFT 分组柱状图 ----------
    fig, ax = plt.subplots(figsize=(st.SINGLE_COL_IN, 2.4))
    n_pol = len(st.POLICY_ORDER)
    width = 0.8 / n_pol
    xs = np.arange(len(st.PENDING_ORDER))
    for i, pol in enumerate(st.POLICY_ORDER):
        vals = [float(r['ttft_median_ms']) for q in st.PENDING_ORDER
                for r in pending if r['policy'] == pol and r['pending'] == q]
        bars = ax.bar(xs + (i - (n_pol - 1) / 2) * width, vals, width * 0.92,
                      label=st.POLICY_LABEL[pol], color=st.POLICY_COLOR[pol],
                      edgecolor='black', linewidth=0.4, hatch=st.POLICY_HATCH[pol])
        # 柱顶数值标注（小字竖排），保证单栏下可读
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.1f}', (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 1.5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=5.5, rotation=90, color='#333333')
    ax.set_xticks(xs)
    ax.set_xticklabels([st.PENDING_LABEL[q] for q in st.PENDING_ORDER])
    ax.set_xlabel('Pending request')
    st.tidy_axes(ax, y_label='TTFT (ms)')
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.18),
              columnspacing=1.0, handlelength=1.2)
    st.save_all(fig, str(FIGS / 'fig_vi_b_pending_ttft'))
    plt.close(fig)

    # ---------- Figure VI-B(b)：Aggregated TTFT（含真实五轮散点） ----------
    fig, ax = plt.subplots(figsize=(st.SINGLE_COL_IN, 2.2))
    xs = np.arange(len(st.POLICY_ORDER))
    vals = [float(next(r['aggregate_ttft_ms'] for r in aggregate if r['policy'] == pol))
            for pol in st.POLICY_ORDER]
    bars = ax.bar(xs, vals, 0.55, color=[st.POLICY_COLOR[p] for p in st.POLICY_ORDER],
                  edgecolor='black', linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.annotate(f'{v:.1f}', (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=st.SMALL_FONT_SIZE, color='#333333')
    # 真实五轮等权均值散点（离散程度展示，非虚构误差线）
    rng = np.random.default_rng(20261025)
    for i, pol in enumerate(st.POLICY_ORDER):
        row = next(r for r in aggregate if r['policy'] == pol)
        pts = [float(row[c]) for c in ROUND_COLS]
        ax.scatter(np.full(len(pts), i) + rng.uniform(-0.13, 0.13, len(pts)), pts,
                   s=7, color='black', zorder=3, marker='o', facecolors='none', linewidths=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([st.POLICY_LABEL[p] for p in st.POLICY_ORDER])
    st.tidy_axes(ax, y_label='TTFT (ms)')
    ax.set_ylim(bottom=0)
    ax.annotate('bars: median of per-round equal-weight means\ncircles: the 5 per-round means',
                xy=(0.02, 0.98), xycoords='axes fraction', va='top', ha='left',
                fontsize=5.5, color='#555555')
    st.save_all(fig, str(FIGS / 'fig_vi_b_aggregate_ttft'))
    plt.close(fig)
    print('已生成 6 个图片文件于', FIGS)


if __name__ == '__main__':
    main()
