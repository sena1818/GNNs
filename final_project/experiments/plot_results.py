"""
实验结果绘图脚本

用法:
  python experiments/plot_results.py --plot convergence    # 训练收敛曲线
  python experiments/plot_results.py --plot steps_sweep    # 推理步数 vs gap
  python experiments/plot_results.py --plot all            # 所有图
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# 统一颜色/标签
MODE_STYLES = {
    'flow_matching':   {'color': '#e74c3c', 'label': 'Flow Matching', 'marker': 'o'},
    'discrete_ddpm':   {'color': '#3498db', 'label': 'D3PM (Discrete)', 'marker': 's'},
    'continuous_ddpm': {'color': '#2ecc71', 'label': 'Gaussian DDPM', 'marker': '^'},
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def plot_convergence():
    """绘制训练收敛曲线 (train_loss, val_loss, lr)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for mode, style in MODE_STYLES.items():
        hist_path = os.path.join(CKPT_DIR, f'{mode}_gated_gcn', 'history.json')
        if not os.path.exists(hist_path):
            print(f'  Skip {mode}: {hist_path} not found')
            continue

        hist = json.load(open(hist_path))
        epochs = list(range(1, len(hist['train_loss']) + 1))

        axes[0].plot(epochs, hist['train_loss'],
                     color=style['color'], label=style['label'], linewidth=1.5)
        axes[1].plot(epochs, hist['val_loss'],
                     color=style['color'], label=style['label'], linewidth=1.5)
        if 'lr' in hist:
            axes[2].plot(epochs, hist['lr'],
                         color=style['color'], label=style['label'], linewidth=1.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('LR Schedule (Cosine Decay)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'convergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def plot_steps_sweep():
    """绘制推理步数 vs gap 曲线 (核心 Figure)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    steps_list = [5, 10, 20, 50, 100]

    for mode, style in MODE_STYLES.items():
        gaps = []
        times = []
        valid_steps = []

        for s in steps_list:
            f = os.path.join(RESULTS_DIR, f'steps_{mode}_s{s}.json')
            if os.path.exists(f):
                d = json.load(open(f))
                gaps.append(d['avg_gap'])
                times.append(d.get('avg_infer_ms', 0))
                valid_steps.append(s)

        if valid_steps:
            ax.plot(valid_steps, gaps,
                    color=style['color'], label=style['label'],
                    marker=style['marker'], linewidth=2, markersize=8)

    ax.set_xlabel('Inference Steps', fontsize=12)
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('Inference Steps vs Solution Quality', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(steps_list)
    ax.set_xticklabels([str(s) for s in steps_list])

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'steps_vs_gap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def plot_main_comparison():
    """绘制主结果柱状图 (greedy vs greedy+2opt)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    modes = list(MODE_STYLES.keys())
    labels = [MODE_STYLES[m]['label'] for m in modes]
    colors = [MODE_STYLES[m]['color'] for m in modes]
    x = np.arange(len(modes))
    width = 0.6

    for ax_idx, (decode_name, decode_label) in enumerate([
        ('greedy', 'Greedy'), ('greedy2opt', 'Greedy + 2-opt')
    ]):
        gaps = []
        stds = []
        for mode in modes:
            f = os.path.join(RESULTS_DIR, f'main_{mode}_{decode_name}.json')
            if os.path.exists(f):
                d = json.load(open(f))
                gaps.append(d['avg_gap'])
                stds.append(d.get('std_gap', 0))
            else:
                gaps.append(0)
                stds.append(0)

        bars = axes[ax_idx].bar(x, gaps, width, yerr=stds,
                                color=colors, capsize=5, alpha=0.85)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels, fontsize=10)
        axes[ax_idx].set_ylabel('Optimality Gap (%)', fontsize=11)
        axes[ax_idx].set_title(f'TSP-50 Results ({decode_label})', fontsize=12)
        axes[ax_idx].grid(True, alpha=0.3, axis='y')

        # 在柱上标注数值
        for bar, gap in zip(bars, gaps):
            if gap > 0:
                axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  f'{gap:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'main_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def plot_speed_comparison():
    """绘制推理速度对比 (Pareto: time vs gap)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    steps_list = [5, 10, 20, 50, 100]

    for mode, style in MODE_STYLES.items():
        times = []
        gaps = []

        for s in steps_list:
            f = os.path.join(RESULTS_DIR, f'steps_{mode}_s{s}.json')
            if os.path.exists(f):
                d = json.load(open(f))
                times.append(d.get('avg_infer_ms', 0))
                gaps.append(d['avg_gap'])

        if times:
            ax.scatter(times, gaps, color=style['color'], label=style['label'],
                       marker=style['marker'], s=100, zorder=5)
            ax.plot(times, gaps, color=style['color'], alpha=0.5, linewidth=1)

            # 标注步数
            for t, g, s in zip(times, gaps,
                               [s for s in steps_list
                                if os.path.exists(os.path.join(RESULTS_DIR, f'steps_{mode}_s{s}.json'))]):
                ax.annotate(f'{s}', (t, g), textcoords='offset points',
                            xytext=(5, 5), fontsize=8, color=style['color'])

    ax.set_xlabel('Inference Time (ms/instance)', fontsize=12)
    ax.set_ylabel('Optimality Gap (%)', fontsize=12)
    ax.set_title('Speed-Quality Pareto Front', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'speed_quality_pareto.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=str, default='all',
                        choices=['convergence', 'steps_sweep', 'main', 'speed', 'all'])
    args = parser.parse_args()

    if args.plot in ('convergence', 'all'):
        print('Plotting convergence curves...')
        plot_convergence()

    if args.plot in ('steps_sweep', 'all'):
        print('Plotting steps sweep...')
        plot_steps_sweep()

    if args.plot in ('main', 'all'):
        print('Plotting main comparison...')
        plot_main_comparison()

    if args.plot in ('speed', 'all'):
        print('Plotting speed comparison...')
        plot_speed_comparison()
