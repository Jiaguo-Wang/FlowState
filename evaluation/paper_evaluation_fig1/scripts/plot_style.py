"""VI 系列论文图统一样式（供 VI-B 及后续 VI-C/VI-D/VI-E 复用）。

面向 ICDE 双栏论文单栏宽度（约 3.35 英寸）：
- 白色背景、无装饰；Okabe-Ito 色盲安全配色（灰度打印下亮度可分）；
- PDF 内嵌 TrueType（fonttype=42）、SVG 文字保留为文本（fonttype='none'），保证矢量可编辑；
- 统一策略命名、配色与图例样式，跨图一致。
"""
import matplotlib as mpl

# 单栏宽度（英寸）与字号基准（正文 8pt，可按刊物要求微调）
SINGLE_COL_IN = 3.35
BASE_FONT_SIZE = 8
SMALL_FONT_SIZE = 6.5

# 策略统一命名、配色与 Hatch（灰度打印备用纹理；色相已可分，Hatch 默认关闭）
POLICY_ORDER = ['FlowState-Greedy', 'StandaloneTopK', 'WFC']
POLICY_LABEL = {'FlowState-Greedy': 'FlowState', 'StandaloneTopK': 'StandaloneTopK', 'WFC': 'WFC'}
POLICY_COLOR = {'FlowState-Greedy': '#0072B2', 'StandaloneTopK': '#E69F00', 'WFC': '#009E73'}
POLICY_HATCH = {'FlowState-Greedy': '', 'StandaloneTopK': '', 'WFC': ''}

# Pending 展示顺序与标签
PENDING_ORDER = ['C_A', 'C_B', 'C_C', 'C_D']
PENDING_LABEL = {'C_A': 'A', 'C_B': 'B', 'C_C': 'C', 'C_D': 'D'}


def apply():
    """应用全局 rcParams；绘图脚本在 import matplotlib 后、建图前调用一次。"""
    mpl.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': BASE_FONT_SIZE,
        'axes.labelsize': BASE_FONT_SIZE,
        'xtick.labelsize': BASE_FONT_SIZE,
        'ytick.labelsize': BASE_FONT_SIZE,
        'legend.fontsize': SMALL_FONT_SIZE,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        # 矢量输出：PDF 内嵌 TrueType、SVG 保留可编辑文字、PNG 高分辨率
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'svg.hashsalt': 'flowstate-vi',  # 固定散列盐：SVG 内部 path id 可复现（仅 dc:date 时间戳随运行变化）
        'savefig.dpi': 600,
        'figure.dpi': 150,
    })


def tidy_axes(ax, y_label=None):
    """统一坐标轴外观：去掉上/右边框，仅保留 y 向网格。"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    if y_label:
        ax.set_ylabel(y_label)


def save_all(fig, out_prefix):
    """按论文要求同时输出 PDF/SVG/PNG 三种格式（矢量优先）。"""
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(f'{out_prefix}.{ext}', bbox_inches='tight', pad_inches=0.02)
