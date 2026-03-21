"""
实验结果绘图脚本 — 使用真实实验数据生成报告图表

生成图表:
  report/figs/01_training_curves.png    训练曲线（三模型 log-scale 对比）
  report/figs/02_main_results_bar.png   最终结果柱状图（merge+2opt，附论文参考值）
  report/figs/03_inference_time.png     推理速度对比（步数 + 每步耗时 + Pareto）
  report/figs/04_bug_fix_comparison.png Bug 修复前后对比
  report/figs/05_gap_distribution.png   Gap 完整分布箱线图

用法:
  cd final_project
  python experiments/plot_results.py             # 生成所有图
  python experiments/plot_results.py --plot convergence
  python experiments/plot_results.py --plot main
  python experiments/plot_results.py --plot all
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, 'results')
CKPT_DIR    = os.path.join(ROOT, 'checkpoints')
FIG_DIR     = os.path.join(ROOT, 'report', 'figs')

# ── 颜色/标签方案 ─────────────────────────────────────────────────────────
MODE_STYLES = {
    'flow_matching':   {'color': '#2196F3', 'label': 'Flow Matching',       'marker': 'o'},
    'discrete_ddpm':   {'color': '#4CAF50', 'label': 'Discrete DDPM (D3PM)', 'marker': 's'},
    'continuous_ddpm': {'color': '#FF9800', 'label': 'Continuous DDPM',     'marker': '^'},
}
COLORS = {k: v['color'] for k, v in MODE_STYLES.items()}
LABELS = {k: v['label'] for k, v in MODE_STYLES.items()}

# ── 文件名映射（使用整理后的文件名） ───────────────────────────────────────
RESULT_FILES = {
    'flow_matching':   'fm_merge2opt.json',
    'discrete_ddpm':   'd3pm_merge2opt.json',
    'continuous_ddpm': 'ddpm_merge2opt.json',
}
CKPT_DIRS = {
    'flow_matching':   'flow_matching_gated_gcn',
    'discrete_ddpm':   'discrete_ddpm_gated_gcn',
    'continuous_ddpm': 'continuous_ddpm_gated_gcn',
}


def _load_result(mode_key):
    path = os.path.join(RESULTS_DIR, RESULT_FILES[mode_key])
    with open(path) as f:
        return json.load(f)

def _load_history(mode_key):
    path = os.path.join(CKPT_DIR, CKPT_DIRS[mode_key], 'history.json')
    with open(path) as f:
        return json.load(f)

def _load_archive(fname):
    path = os.path.join(RESULTS_DIR, 'archive', fname)
    with open(path) as f:
        return json.load(f)

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    size_kb = os.path.getsize(out) // 1024
    print(f'  ✓  {name}  ({size_kb} KB)')
    return out


# =============================================================================
# 图1：训练曲线（三模型 log-scale 对比）
# =============================================================================

def plot_convergence():
    """Train/Val loss log-scale 三模型对比 + LR 曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Training Curves — TSP-50 (50 Epochs, RTX 4090)', fontsize=13)

    for mode, style in MODE_STYLES.items():
        hist = _load_history(mode)
        epochs = list(range(1, len(hist['train_loss']) + 1))
        c, lbl = style['color'], style['label']

        axes[0].plot(epochs, hist['train_loss'], color=c, label=lbl, lw=2)
        axes[1].plot(epochs, hist['val_loss'],   color=c, label=lbl, lw=2, ls='--')
        if 'lr' in hist:
            axes[2].plot(epochs, hist['lr'],     color=c, label=lbl, lw=2)

    for ax, title, ylabel in [
        (axes[0], 'Training Loss',   'Loss'),
        (axes[1], 'Validation Loss', 'Loss'),
        (axes[2], 'LR Schedule (Cosine Decay to 0)', 'Learning Rate'),
    ]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if title != 'LR Schedule (Cosine Decay to 0)':
            ax.set_yscale('log')

    plt.tight_layout()
    _save(fig, '01_training_curves.png')


# =============================================================================
# 图2：主结果柱状图（merge+2opt，附论文参考值）
# =============================================================================

def plot_main_comparison():
    """最终结果柱状图，附 DIFUSCO 论文参考线"""
    modes  = ['flow_matching', 'discrete_ddpm', 'continuous_ddpm']
    labels = [LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]

    gaps = [_load_result(m)['avg_gap'] for m in modes]
    stds = [_load_result(m)['std_gap'] for m in modes]

    # DIFUSCO 论文 TSP-50 参考值
    paper_ref = {'discrete_ddpm': 0.10, 'continuous_ddpm': 0.25}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(modes))
    bars = ax.bar(x, gaps, color=colors, width=0.5, edgecolor='white',
                  linewidth=0.8, yerr=stds, capsize=5,
                  error_kw={'elinewidth': 1.5, 'ecolor': 'gray'})

    for bar, g, s in zip(bars, gaps, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.15,
                f'{g:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    for m, ref in paper_ref.items():
        i = modes.index(m)
        ax.hlines(ref, i - 0.35, i + 0.35, colors='red', lw=2, ls='--', zorder=5)
        ax.text(i + 0.37, ref + 0.06, f'Paper: {ref:.2f}%', color='red', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('TSP-50 Optimality Gap — merge_tours + 2-opt  (n=1000)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(gaps) * 1.4)
    ax.plot([], [], color='red', ls='--', lw=2, label='DIFUSCO Paper (ref)')
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    _save(fig, '02_main_results_bar.png')


# =============================================================================
# 图3：推理速度：步数 + 每步耗时 + Pareto
# =============================================================================

def plot_speed_comparison():
    """三合一推理速度图"""
    modes  = ['flow_matching', 'discrete_ddpm', 'continuous_ddpm']
    steps  = {'flow_matching': 20, 'discrete_ddpm': 50, 'continuous_ddpm': 50}

    data = {m: _load_result(m) for m in modes}
    times = [data[m]['avg_infer_ms'] for m in modes]
    gaps  = [data[m]['avg_gap']      for m in modes]
    per_step = [times[i] / steps[modes[i]] for i in range(len(modes))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Inference Efficiency Comparison — TSP-50', fontsize=13)
    colors = [COLORS[m] for m in modes]
    labels = [LABELS[m] for m in modes]
    x = np.arange(len(modes))

    # 子图1：总时间
    bars = axes[0].bar(x, times, color=colors, width=0.5, alpha=0.85, edgecolor='white')
    for bar, t, m in zip(bars, times, modes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f'{t:.0f} ms\n({steps[m]} steps)',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('Avg Inference Time (ms)', fontsize=11)
    axes[0].set_title('Total Inference Time', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0, max(times) * 1.45)

    # 子图2：每步耗时
    bars2 = axes[1].bar(x, per_step, color=colors, width=0.5, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars2, per_step):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{v:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Time per Step (ms)', fontsize=11)
    axes[1].set_title('Per-Step Computation Cost', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(per_step) * 1.45)

    # 子图3：Pareto (时间 vs gap)
    ax3 = axes[2]
    for m, t, g in zip(modes, times, gaps):
        ax3.scatter(t, g, color=COLORS[m], s=180, zorder=5,
                    edgecolors='white', lw=1.5, label=LABELS[m])
        ax3.annotate(LABELS[m], (t, g),
                     textcoords='offset points', xytext=(8, 4),
                     fontsize=8, color=COLORS[m])

    # 标注 FM 的速度优势
    fm_t, d3pm_t = times[0], times[1]
    speedup = d3pm_t / fm_t
    ax3.annotate(f'FM is {speedup:.1f}× faster\nthan D3PM',
                 xy=(fm_t, gaps[0]), xytext=(fm_t + 50, gaps[0] + 1.2),
                 fontsize=8.5, color='#1565C0',
                 arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    ax3.set_xlabel('Avg Inference Time (ms)', fontsize=11)
    ax3.set_ylabel('Optimality Gap (%)', fontsize=11)
    ax3.set_title('Speed–Quality Pareto', fontsize=11)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(50, 500)

    plt.tight_layout()
    _save(fig, '03_inference_time.png')


# =============================================================================
# 图4：Bug 修复前后对比
# =============================================================================

def plot_bug_fix_comparison():
    """展示两个关键 bug 修复：FM 值域 bug 和 DDPM valid_rate"""
    fig = plt.figure(figsize=(13, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
    fig.suptitle('Effect of Bug Fixes — TSP-50', fontsize=14)

    # ── A：FM 值域 bug ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    # 修复前：旧 FM checkpoint（{0,1} 空间，greedy）= 28.09%
    # 修复后：重训 FM（{-1,+1} 空间，merge+2opt）= 3.45%
    old_fm = _load_archive('pre_fix_fm_greedy.json')
    new_fm = _load_result('flow_matching')
    stages = ['Before Fix\n({0,1} domain,\ngreedy)', 'After Fix\n({-1,+1} retrain,\nmerge+2opt)']
    vals   = [old_fm['avg_gap'], new_fm['avg_gap']]
    clrs   = ['#EF9A9A', COLORS['flow_matching']]

    bars = ax1.bar(stages, vals, color=clrs, width=0.4, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Avg Optimality Gap (%)', fontsize=11)
    ax1.set_title('Flow Matching\n(Value Domain Bug Fix)', fontsize=11)
    ax1.set_ylim(0, 35)
    ax1.grid(axis='y', alpha=0.3)

    improvement = vals[0] - vals[1]
    ax1.annotate('', xy=(1, vals[1] + 0.5), xytext=(0, vals[0] - 0.5),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax1.text(0.5, 15, f'↓ {improvement:.1f}%\nimprovement',
             ha='center', color='green', fontsize=12, fontweight='bold',
             transform=ax1.transData)

    # ── B：DDPM valid_rate bug ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    old_ddpm = _load_archive('pre_fix_ddpm_greedy.json')
    new_ddpm = _load_result('continuous_ddpm')

    x = np.arange(2)
    before_valid = old_ddpm['valid_rate'] * 100  # 4%
    after_valid  = new_ddpm['valid_rate']  * 100  # 100%
    before_gap   = old_ddpm['avg_gap']            # 5.92% (only 40 instances!)
    after_gap    = new_ddpm['avg_gap']            # 6.21% (all 1000)

    ax2b = ax2.twinx()
    b1 = ax2.bar(x - 0.2, [before_valid, after_valid], 0.35,
                 color=['#EF9A9A', COLORS['continuous_ddpm']], alpha=0.75,
                 edgecolor='white', label='Valid Tour Rate (%)')
    b2 = ax2b.bar(x + 0.2, [before_gap, after_gap], 0.35,
                  color=['#EF9A9A', COLORS['continuous_ddpm']], alpha=0.4,
                  hatch='//', edgecolor='gray')

    for bar, v in zip(b1, [before_valid, after_valid]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    for bar, v in zip(b2, [before_gap, after_gap]):
        label = f'{v:.2f}%\n(n={40 if v == before_gap else 1000})'
        ax2b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                  label, ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Before Fix\n(DDIM+greedy)', 'After Fix\n(DDPM post.\n+merge)'], fontsize=9)
    ax2.set_ylabel('Valid Tour Rate (%)', fontsize=11)
    ax2b.set_ylabel('Avg Gap on Valid Instances (%)', fontsize=10, color='gray')
    ax2.set_title('Continuous DDPM\n(DDIM→DDPM Posterior + merge decoder)', fontsize=11)
    ax2.set_ylim(0, 130)
    ax2b.set_ylim(0, 12)
    ax2.grid(axis='y', alpha=0.2)

    solid = mpatches.Patch(color='gray', alpha=0.75, label='Valid Rate')
    hatch = mpatches.Patch(color='gray', alpha=0.4, hatch='//', label='Avg Gap')
    ax2.legend(handles=[solid, hatch], fontsize=8, loc='upper center')

    fig.subplots_adjust(top=0.88, wspace=0.38)
    _save(fig, '04_bug_fix_comparison.png')


# =============================================================================
# 图5：Gap 完整分布箱线图
# =============================================================================

def plot_gap_distribution():
    """1000 个实例的 gap 完整分布箱线图"""
    modes  = ['flow_matching', 'discrete_ddpm', 'continuous_ddpm']
    labels = [LABELS[m] for m in modes]
    colors = [COLORS[m] for m in modes]
    all_gaps = [_load_result(m)['all_gaps'] for m in modes]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bp = ax.boxplot(
        all_gaps, tick_labels=labels, patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 2.5},
        flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.35},
        whiskerprops={'linewidth': 1.5},
        capprops={'linewidth': 1.5},
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, (m, gaps) in enumerate(zip(modes, all_gaps)):
        mean_val = np.mean(gaps)
        ax.scatter(i + 1, mean_val, marker='D', color='white',
                   edgecolors='black', s=50, zorder=6)
        ax.text(i + 1.28, mean_val, f'μ={mean_val:.2f}%',
                va='center', fontsize=9.5, color='dimgray')

    # DIFUSCO 论文参考线
    for m, ref in [('discrete_ddpm', 0.10), ('continuous_ddpm', 0.25)]:
        i = modes.index(m)
        ax.hlines(ref, i + 0.55, i + 1.45, colors='red', lw=1.8, ls='--', zorder=5)
        ax.text(i + 1.48, ref + 0.05, f'Paper {ref:.2f}%', color='red', fontsize=7.5)

    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('Gap Distribution — TSP-50  merge_tours+2opt  (n=1000)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.3, None)

    handles = [mpatches.Patch(facecolor=c, alpha=0.6, label=l)
               for c, l in zip(colors, labels)]
    handles.append(plt.scatter([], [], marker='D', color='white',
                               edgecolors='black', s=40, label='Mean'))
    handles.append(mpatches.Patch(color='red', label='DIFUSCO Paper (ref)'))
    ax.legend(handles=handles, fontsize=9, loc='upper right')

    plt.tight_layout()
    _save(fig, '05_gap_distribution.png')


# =============================================================================
# 图6：推理步数扫描 — Steps vs Gap 折线图（核心创新 Figure）
# =============================================================================

def plot_steps_sweep():
    """Steps vs Gap 折线图 + Steps vs 推理时间双轴图"""
    steps_list = [5, 10, 20, 50, 100]
    modes = ['flow_matching', 'discrete_ddpm', 'continuous_ddpm']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Inference Steps vs Quality & Speed — TSP-50 (greedy decoder)', fontsize=13)

    for mode in modes:
        gaps_m, times_m, steps_ok = [], [], []
        for s in steps_list:
            fpath = os.path.join(RESULTS_DIR, f'steps_{mode}_s{s}.json')
            if os.path.exists(fpath):
                with open(fpath) as f:
                    d = json.load(f)
                gaps_m.append(d['avg_gap'])
                times_m.append(d['avg_infer_ms'])
                steps_ok.append(s)

        if not steps_ok:
            continue

        style = MODE_STYLES[mode]
        axes[0].plot(steps_ok, gaps_m, color=style['color'], marker=style['marker'],
                     label=style['label'], lw=2, markersize=8)
        axes[1].plot(steps_ok, times_m, color=style['color'], marker=style['marker'],
                     label=style['label'], lw=2, markersize=8)

    axes[0].set_xlabel('Inference Steps', fontsize=12)
    axes[0].set_ylabel('Avg Optimality Gap (%)', fontsize=12)
    axes[0].set_title('Steps vs Quality', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_xticks(steps_list)
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())

    axes[1].set_xlabel('Inference Steps', fontsize=12)
    axes[1].set_ylabel('Avg Inference Time (ms)', fontsize=12)
    axes[1].set_title('Steps vs Speed', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_xticks(steps_list)
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    _save(fig, '06_steps_sweep.png')


# =============================================================================
# 图7：跨规模泛化 — Scale vs Gap
# =============================================================================

def plot_generalization():
    """Scale vs Gap 折线图（TSP-20/50/100 泛化测试）"""
    scales = [20, 50, 100]
    modes = ['flow_matching', 'discrete_ddpm', 'continuous_ddpm']
    ckpt_dirs_map = {
        'flow_matching': 'flow_matching_gated_gcn',
        'discrete_ddpm': 'discrete_ddpm_gated_gcn',
        'continuous_ddpm': 'continuous_ddpm_gated_gcn',
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mode in modes:
        gaps_m, scales_ok = [], []
        for s in scales:
            fpath = os.path.join(RESULTS_DIR, f'gen_{ckpt_dirs_map[mode]}_tsp{s}.json')
            if os.path.exists(fpath):
                with open(fpath) as f:
                    d = json.load(f)
                gaps_m.append(d['avg_gap'])
                scales_ok.append(s)

        if not scales_ok:
            continue

        style = MODE_STYLES[mode]
        ax.plot(scales_ok, gaps_m, color=style['color'], marker=style['marker'],
                label=style['label'], lw=2.5, markersize=10)

        # 标注数值
        for x, y in zip(scales_ok, gaps_m):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                        xytext=(8, 5), fontsize=9, color=style['color'])

    ax.set_xlabel('TSP Instance Size (N)', fontsize=12)
    ax.set_ylabel('Avg Optimality Gap (%)', fontsize=12)
    ax.set_title('Cross-Scale Generalization — Trained on TSP-50, merge+2opt', fontsize=13)
    ax.set_xticks(scales)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, '07_generalization.png')


# =============================================================================
# 图8：解码策略消融 — Pareto 图
# =============================================================================

def plot_decoding_ablation():
    """解码策略对比：时间 vs Gap 的 Pareto 图"""
    decode_methods = ['greedy', 'beam3', 'beam5', 'beam10', 'greedy2opt']
    decode_labels  = ['Greedy', 'Beam k=3', 'Beam k=5', 'Beam k=10', 'Greedy+2opt']
    markers = ['o', 's', 'D', '^', '*']

    # 查找有解码消融数据的模型
    ckpt_dirs_list = [
        ('flow_matching_gated_gcn', 'FM'),
        ('discrete_ddpm_gated_gcn', 'D3PM'),
    ]

    fig, axes = plt.subplots(1, len(ckpt_dirs_list), figsize=(7 * len(ckpt_dirs_list), 5.5))
    if len(ckpt_dirs_list) == 1:
        axes = [axes]
    fig.suptitle('Decoding Strategy Comparison — TSP-50', fontsize=13)

    for ax_idx, (ckpt_dir, model_label) in enumerate(ckpt_dirs_list):
        ax = axes[ax_idx]
        for dm, dl, mk in zip(decode_methods, decode_labels, markers):
            fpath = os.path.join(RESULTS_DIR, f'{ckpt_dir}_decode_{dm}.json')
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                d = json.load(f)
            ax.scatter(d['avg_infer_ms'], d['avg_gap'], s=120, marker=mk,
                       label=dl, zorder=5, edgecolors='white', lw=1)
            ax.annotate(f'{d["avg_gap"]:.1f}%', (d['avg_infer_ms'], d['avg_gap']),
                        textcoords='offset points', xytext=(8, 3), fontsize=8.5)

        ax.set_xlabel('Avg Inference Time (ms)', fontsize=11)
        ax.set_ylabel('Avg Optimality Gap (%)', fontsize=11)
        ax.set_title(f'{model_label}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, '08_decoding_ablation.png')


# =============================================================================
# 主入口
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate report figures from experiment results')
    parser.add_argument('--plot', type=str, default='all',
                        choices=['convergence', 'main', 'speed', 'bugfix', 'distribution',
                                 'steps_sweep', 'generalization', 'decoding', 'all'],
                        help='Which figure to generate (default: all)')
    args = parser.parse_args()

    print(f'Saving figures to: {FIG_DIR}\n')

    plot_fns = {
        'convergence':     plot_convergence,
        'main':            plot_main_comparison,
        'speed':           plot_speed_comparison,
        'bugfix':          plot_bug_fix_comparison,
        'distribution':    plot_gap_distribution,
        'steps_sweep':     plot_steps_sweep,
        'generalization':  plot_generalization,
        'decoding':        plot_decoding_ablation,
    }

    to_run = list(plot_fns.keys()) if args.plot == 'all' else [args.plot]
    for name in to_run:
        try:
            plot_fns[name]()
        except Exception as e:
            print(f'  ✗  {name}: {e}')

    print(f'\nAll figures saved to: {FIG_DIR}')
