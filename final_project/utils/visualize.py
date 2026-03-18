"""
可视化工具 — 扩散过程 GIF、路径对比、训练曲线、消融实验图

函数列表:
  save_diffusion_gif(model, coords, output_path, n_frames, device)
  plot_tour_comparison(coords, model_tour, opt_tour, save_path, title)
  plot_heatmap(coords, heatmap, title, save_path)
  plot_training_curve(history, save_path, title)
  plot_ablation_bar(results_dict, save_path, metric, title)
  plot_generalization_curve(sizes, gaps_dict, save_path, title)

依赖: matplotlib, imageio, numpy, torch
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无头环境下不弹窗
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Dict

import torch


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _coords_to_np(coords) -> np.ndarray:
    """统一转成 (N, 2) numpy array。"""
    if isinstance(coords, torch.Tensor):
        return coords.detach().cpu().numpy()
    return np.asarray(coords)


def _draw_tour(ax, xy: np.ndarray, tour: List[int], color: str, lw: float = 1.5, alpha: float = 0.8):
    """在 ax 上画 tour 路径（闭合回路）。"""
    N = len(tour)
    segments = []
    for k in range(N):
        i, j = tour[k], tour[(k + 1) % N]
        segments.append([xy[i], xy[j]])
    lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], s=25, c='black', zorder=5)


def _tour_cost(tour: List[int], xy: np.ndarray) -> float:
    """计算 tour 总路径长度。"""
    total = 0.0
    N = len(tour)
    for k in range(N):
        i, j = tour[k], tour[(k + 1) % N]
        diff = xy[i] - xy[j]
        total += float(np.linalg.norm(diff))
    return total


# ---------------------------------------------------------------------------
# 1. 扩散过程 GIF
# ---------------------------------------------------------------------------

def save_diffusion_gif(
    model,
    coords: torch.Tensor,          # (1, N, 2) 或 (N, 2)
    output_path: str,
    n_frames: int = 20,
    device=None,
    fps: int = 5,
):
    """
    可视化 Flow Matching 的去噪过程，保存为 GIF。

    每帧对应推理的一个时刻 t (1.0 → 0.0)，
    热力图颜色越深表示该边在当前时刻概率越高。

    Args:
        model:       TSPDiffusionModel，mode='flow_matching'
        coords:      (1, N, 2) 或 (N, 2)，单个实例的城市坐标
        output_path: 输出路径，如 'report/figs/diffusion.gif'
        n_frames:    帧数（t 均匀采样 n_frames 个时刻）
        device:      torch.device，None 则自动检测
        fps:         GIF 播放帧率
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("请安装 imageio: pip install imageio")

    if device is None:
        device = next(model.parameters()).device

    # 统一 shape → (1, N, 2)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    coords = coords.to(device)

    model.eval()
    xy = _coords_to_np(coords[0])       # (N, 2)

    # t 从 1.0 均匀降到 0.0 共 n_frames 帧
    t_values = np.linspace(1.0, 0.0, n_frames)
    frames = []

    with torch.no_grad():
        for t_val in t_values:
            heatmap = model.get_intermediate_heatmap(
                coords, target_t=float(t_val), total_steps=n_frames
            )
            hm = heatmap[0].cpu().numpy()   # (N, N)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.suptitle(f'Flow Matching Denoising  (t = {t_val:.2f})', fontsize=13)

            # 左图：热力图
            ax0 = axes[0]
            im = ax0.imshow(hm, vmin=0, vmax=1, cmap='RdYlGn', origin='upper')
            plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            ax0.set_title('Edge Probability Heatmap')
            ax0.set_xlabel('Node j')
            ax0.set_ylabel('Node i')

            # 右图：在城市坐标上叠加高概率边
            ax1 = axes[1]
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_aspect('equal')
            ax1.set_title('High-Prob Edges on City Map')

            # 只画概率 > 0.5 的边
            N = hm.shape[0]
            thresh = 0.5
            segs, alphas = [], []
            for i in range(N):
                for j in range(i + 1, N):
                    p = hm[i, j]
                    if p > thresh:
                        segs.append([xy[i], xy[j]])
                        alphas.append(p)
            if segs:
                colors = plt.cm.RdYlGn(np.array(alphas))
                lc = LineCollection(segs, colors=colors, linewidths=1.5)
                ax1.add_collection(lc)
            ax1.scatter(xy[:, 0], xy[:, 1], s=30, c='black', zorder=5)

            plt.tight_layout()

            # 渲染成 RGB numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            frame = buf.reshape(h, w, 4)[..., :3]   # drop alpha → RGB
            frames.append(frame)
            plt.close(fig)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f'GIF saved → {output_path}  ({len(frames)} frames, {fps} fps)')


# ---------------------------------------------------------------------------
# 2. 路径对比图
# ---------------------------------------------------------------------------

def plot_tour_comparison(
    coords,
    model_tour: List[int],
    opt_tour: List[int],
    save_path: str,
    title: str = 'Tour Comparison',
):
    """
    并排展示最优解（蓝色）和模型解（橙色），下方标注 cost 和 gap。

    Args:
        coords:     (N, 2) 城市坐标
        model_tour: 模型预测的 tour（0-indexed）
        opt_tour:   最优 tour（0-indexed）
        save_path:  图片保存路径，如 'report/figs/tour_cmp.png'
        title:      图标题
    """
    xy = _coords_to_np(coords)
    opt_cost   = _tour_cost(opt_tour, xy)
    model_cost = _tour_cost(model_tour, xy)
    gap = (model_cost - opt_cost) / opt_cost * 100.0 if opt_cost > 1e-10 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=13)

    for ax, tour, color, label in [
        (axes[0], opt_tour,   '#2196F3', f'Optimal  cost={opt_cost:.4f}'),
        (axes[1], model_tour, '#FF9800', f'Model    cost={model_cost:.4f}  gap={gap:.2f}%'),
    ]:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=10)
        _draw_tour(ax, xy, tour, color=color)
        # 节点编号标注（城市数 ≤ 30 时才标）
        if len(tour) <= 30:
            for idx, (x, y) in enumerate(xy):
                ax.annotate(str(idx), (x, y), textcoords='offset points',
                            xytext=(4, 4), fontsize=7, color='dimgray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Tour comparison saved → {save_path}')


# ---------------------------------------------------------------------------
# 3. 边概率热力图
# ---------------------------------------------------------------------------

def plot_heatmap(
    coords,
    heatmap,
    title: str = 'Edge Probability Heatmap',
    save_path: str = None,
):
    """
    可视化 N×N 边概率矩阵，并在右侧叠加城市坐标散点图。

    Args:
        coords:    (N, 2) 城市坐标
        heatmap:   (N, N) 边概率矩阵，值域 [0, 1]
        title:     图标题
        save_path: 保存路径；None 则直接 plt.show()
    """
    xy = _coords_to_np(coords)
    if isinstance(heatmap, torch.Tensor):
        hm = heatmap.detach().cpu().numpy()
    else:
        hm = np.asarray(heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=13)

    # 左：热力图
    ax0 = axes[0]
    im = ax0.imshow(hm, vmin=0, vmax=1, cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title('N×N Edge Probability Matrix')
    ax0.set_xlabel('Node j')
    ax0.set_ylabel('Node i')

    # 右：城市地图 + 高概率边
    ax1 = axes[1]
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title('High-Prob Edges (p > 0.4)')

    N = hm.shape[0]
    segs, probs = [], []
    for i in range(N):
        for j in range(i + 1, N):
            p = hm[i, j]
            if p > 0.4:
                segs.append([xy[i], xy[j]])
                probs.append(p)
    if segs:
        colors = plt.cm.viridis(np.array(probs))
        lc = LineCollection(segs, colors=colors, linewidths=1.5, alpha=0.8)
        ax1.add_collection(lc)
    ax1.scatter(xy[:, 0], xy[:, 1], s=35, c='red', zorder=5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Heatmap saved → {save_path}')
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 4. 训练曲线
# ---------------------------------------------------------------------------

def plot_training_curve(
    history: dict,
    save_path: str,
    title: str = 'Training Curve',
):
    """
    绘制训练/验证 loss 曲线以及学习率变化。

    Args:
        history:   train.py 输出的 history.json 对应的 dict，
                   包含 'train_loss', 'val_loss', 'lr' 三个 list
        save_path: 图片保存路径
        title:     图标题
    """
    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss', [])
    lr         = history.get('lr', [])
    epochs     = list(range(1, len(train_loss) + 1))

    has_lr = bool(lr)
    n_rows = 2 if has_lr else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    # Loss
    ax = axes[0]
    if train_loss:
        ax.plot(epochs, train_loss, label='Train Loss', color='#2196F3', lw=2)
    if val_loss:
        ax.plot(epochs[:len(val_loss)], val_loss, label='Val Loss',
                color='#F44336', lw=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss vs Epoch')

    # 学习率
    if has_lr:
        ax2 = axes[1]
        ax2.plot(epochs[:len(lr)], lr, color='#4CAF50', lw=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Learning Rate Schedule')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curve saved → {save_path}')


# ---------------------------------------------------------------------------
# 5. 消融实验柱状图
# ---------------------------------------------------------------------------

def plot_ablation_bar(
    results_dict: Dict[str, float],
    save_path: str,
    metric: str = 'avg_gap',
    title: str = 'Ablation Study',
    ylabel: str = 'Optimality Gap (%)',
):
    """
    绘制消融/对比实验的柱状图。

    Args:
        results_dict: {实验名称: 数值} 字典
                      例如 {'FM-GatedGCN': 2.3, 'FM-GAT': 3.1, 'D3PM-GatedGCN': 2.8}
                      也可以是 {name: result_json_dict} 格式，会自动提取 metric 字段
        save_path:    图片保存路径
        metric:       当 value 是 dict 时，提取的指标名，默认 'avg_gap'
        title:        图标题
        ylabel:       Y 轴标签
    """
    # 统一展开：支持直接传 float 或传 result dict
    names, values = [], []
    for name, val in results_dict.items():
        names.append(name)
        if isinstance(val, dict):
            values.append(val.get(metric, float('nan')))
        else:
            values.append(float(val))

    # 颜色映射（按模式区分）
    palette = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63',
               '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.8)

    # 在柱子上方标注数值
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9
            )

    ax.set_xlabel('Method / Configuration')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.25 + 0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Ablation bar chart saved → {save_path}')


# ---------------------------------------------------------------------------
# 6. 跨规模泛化曲线
# ---------------------------------------------------------------------------

def plot_generalization_curve(
    sizes: List[int],
    gaps_dict: Dict[str, List[float]],
    save_path: str,
    title: str = 'Generalization across TSP Sizes',
):
    """
    绘制不同方法在不同 TSP 规模下的 Optimality Gap 曲线。

    Args:
        sizes:      X 轴，TSP 城市规模列表，如 [20, 50, 100]
        gaps_dict:  {方法名: [gap_20, gap_50, gap_100, ...]} 字典
                    例如 {'Flow Matching': [1.2, 2.4, 4.1], 'D3PM': [1.5, 2.9, 5.3]}
        save_path:  图片保存路径
        title:      图标题
    """
    colors  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (name, gaps) in enumerate(gaps_dict.items()):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        x = sizes[:len(gaps)]
        ax.plot(x, gaps, color=c, marker=m, lw=2, markersize=7, label=name)
        # 标注数值
        for xi, gi in zip(x, gaps):
            ax.annotate(f'{gi:.2f}%', (xi, gi),
                        textcoords='offset points', xytext=(4, 5),
                        fontsize=8, color=c)

    ax.set_xlabel('TSP Problem Size (N cities)', fontsize=11)
    ax.set_ylabel('Optimality Gap (%)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(sizes)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Generalization curve saved → {save_path}')


# ---------------------------------------------------------------------------
# 快速单元测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json

    print('=== visualize.py 单元测试 ===')
    os.makedirs('test_figs', exist_ok=True)

    N = 15
    np.random.seed(42)
    coords = np.random.rand(N, 2)
    opt_tour   = list(range(N))
    model_tour = list(range(N))
    model_tour[3], model_tour[7] = model_tour[7], model_tour[3]   # 故意弄差一点

    # 2. plot_tour_comparison
    plot_tour_comparison(coords, model_tour, opt_tour,
                         save_path='test_figs/tour_comparison.png',
                         title=f'TSP-{N} Tour Comparison')

    # 3. plot_heatmap
    heatmap = np.random.rand(N, N)
    heatmap = (heatmap + heatmap.T) / 2
    plot_heatmap(coords, heatmap,
                 title=f'TSP-{N} Edge Heatmap',
                 save_path='test_figs/heatmap.png')

    # 4. plot_training_curve
    history = {
        'train_loss': [0.9 - i * 0.015 + np.random.rand() * 0.02 for i in range(50)],
        'val_loss':   [0.92 - i * 0.014 + np.random.rand() * 0.02 for i in range(50)],
        'lr':         [1e-3 * (0.95 ** i) for i in range(50)],
    }
    plot_training_curve(history, save_path='test_figs/training_curve.png',
                        title='Flow Matching Training Curve')

    # 5. plot_ablation_bar
    ablation = {
        'FM-GatedGCN':   2.31,
        'FM-GAT':        3.05,
        'FM-GCN':        3.87,
        'D3PM-GatedGCN': 2.58,
        'DDPM-GatedGCN': 4.12,
    }
    plot_ablation_bar(ablation, save_path='test_figs/ablation_bar.png',
                      title='Architecture Ablation (TSP-50)')

    # 6. plot_generalization_curve
    sizes = [20, 50, 100]
    gaps_dict = {
        'Flow Matching': [1.2, 2.3, 4.5],
        'Discrete DDPM': [1.5, 2.8, 5.3],
        'Continuous DDPM': [2.1, 3.9, 7.2],
    }
    plot_generalization_curve(sizes, gaps_dict,
                              save_path='test_figs/generalization.png',
                              title='Generalization across TSP Sizes')

    print('\n所有图已保存到 test_figs/ 目录。')
    print('注意: save_diffusion_gif 需要训练好的模型，此处跳过。')
    print('visualize.py OK')
