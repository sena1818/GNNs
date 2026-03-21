"""
扩散过程可视化 — 生成 GIF 动画 + 路径对比静态图 + 热力图

产出:
  report/figs/diffusion_fm.gif         FM 去噪过程动画
  report/figs/diffusion_d3pm.gif       D3PM 去噪过程动画
  report/figs/diffusion_ddpm.gif       DDPM 去噪过程动画
  report/figs/tour_comparison.png      路径对比（模型 vs 最优）
  report/figs/heatmap_evolution.png    热力图中间步截图

用法:
  python visualize_diffusion.py
  python visualize_diffusion.py --mode flow_matching   # 只做 FM
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(__file__))
from models.tsp_dataset import TSPDataset, collate_fn
from models.tsp_model import TSPDiffusionModel
from models.diffusion_schedulers import FMInferenceSchedule, InferenceSchedule
from utils.tsp_utils import merge_tours
from utils.decode import two_opt_improve, tour_length

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not installed, GIF generation will be skipped.")
    print("  Install with: pip install imageio[ffmpeg]")


# ============================================================================
# 核心：带中间步记录的采样函数
# ============================================================================

@torch.no_grad()
def sample_with_intermediates(model, coords, steps=None, record_every=1):
    """
    采样并记录中间步的 heatmap，用于可视化。

    Returns:
        final_heatmap: (N, N) 最终热力图
        intermediates: list of (t_value, heatmap_NxN) 中间步快照
    """
    steps = steps or model.inference_steps
    device = coords.device
    B, N, _ = coords.shape
    assert B == 1, "visualization only supports batch_size=1"

    intermediates = []

    if model.mode == 'flow_matching':
        x = torch.randn(B, N, N, device=device)
        for idx, (t_val, dt) in enumerate(FMInferenceSchedule(steps)):
            # 记录中间步
            if idx % record_every == 0:
                h = (x[0] * 0.5 + 0.5).clamp(0, 1)
                h = (h + h.T) / 2
                intermediates.append((t_val, h.cpu()))

            t_tensor = torch.full((B,), t_val, device=device)
            v = model.encoder(coords, x, t_tensor).squeeze(1)
            x = x - dt * v

        heatmap = (x * 0.5 + 0.5).clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        intermediates.append((0.0, heatmap[0].cpu()))

    elif model.mode == 'discrete_ddpm':
        xt = torch.randn(B, N, N, device=device)
        xt = (xt > 0).long()

        schedule = InferenceSchedule(
            inference_schedule=model.inference_schedule_type,
            T=model.T, inference_T=steps,
        )

        x0_pred_prob = None
        for i in range(steps):
            t1, t2 = schedule(i)

            if i % record_every == 0:
                if x0_pred_prob is not None:
                    h = x0_pred_prob[0, :, :, 1]
                    h = (h + h.T) / 2
                    intermediates.append((t1 / model.T, h.cpu()))
                else:
                    intermediates.append((t1 / model.T, xt[0].float().cpu()))

            xt_input = xt.float() * 2 - 1
            xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input))
            t_tensor = torch.tensor([int(t1)], dtype=torch.float, device=device)
            x0_pred = model.encoder(coords.float(), xt_input.float(), t_tensor)
            x0_pred_prob = x0_pred.permute(0, 2, 3, 1).contiguous().softmax(dim=-1)
            xt = model._categorical_posterior(int(t2), int(t1), x0_pred_prob, xt)

        heatmap = x0_pred_prob[..., 1]
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        heatmap = heatmap.clamp(0, 1)
        intermediates.append((0.0, heatmap[0].cpu()))

    elif model.mode == 'continuous_ddpm':
        xt = torch.randn(B, N, N, device=device)

        schedule = InferenceSchedule(
            inference_schedule=model.inference_schedule_type,
            T=model.T, inference_T=steps,
        )
        diffusion = model.scheduler

        for i in range(steps):
            t1, t2 = schedule(i)

            if i % record_every == 0:
                h = (xt[0] * 0.5 + 0.5).clamp(0, 1)
                h = (h + h.T) / 2
                intermediates.append((t1 / model.T, h.cpu()))

            t_tensor = torch.tensor([int(t1)], dtype=torch.float, device=device)
            pred = model.encoder(coords.float(), xt.float(), t_tensor)
            pred = pred.squeeze(1).clamp(-10.0, 10.0)

            xt = model._gaussian_posterior_ddpm_tensor(
                int(t2), int(t1), pred, xt,
                diffusion.alpha_torch, diffusion.alphabar_torch, diffusion.beta_torch,
            )

        heatmap = (xt.detach() * 0.5 + 0.5).clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        intermediates.append((0.0, heatmap[0].cpu()))

    return heatmap[0].cpu(), intermediates


# ============================================================================
# 可视化函数
# ============================================================================

def plot_heatmap_with_tour(ax, coords, heatmap, tour=None, title="", show_edges=True):
    """在 axes 上绘制热力图 + 可选路径。"""
    N = coords.shape[0]
    c = coords.numpy()

    if show_edges and heatmap is not None:
        h = heatmap.numpy()
        # 绘制所有边，颜色深浅表示概率
        segments = []
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if h[i, j] > 0.05:  # 只绘制概率 > 5% 的边
                    segments.append([c[i], c[j]])
                    colors.append(h[i, j])
        if segments:
            lc = LineCollection(segments, cmap='YlOrRd', alpha=0.6, linewidths=0.8)
            lc.set_array(np.array(colors))
            lc.set_clim(0, 1)
            ax.add_collection(lc)

    # 绘制路径
    if tour is not None:
        tour_closed = list(tour) + [tour[0]]
        tour_coords = c[tour_closed]
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=1.8, alpha=0.9, zorder=2)

    # 绘制城市节点
    ax.scatter(c[:, 0], c[:, 1], c='black', s=25, zorder=3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def make_diffusion_gif(coords, intermediates, tour_gt, out_path, fps=4):
    """生成去噪过程 GIF 动画。"""
    if not HAS_IMAGEIO:
        print(f"  Skipping GIF (imageio not installed): {out_path}")
        return

    c = coords.numpy()
    N = c.shape[0]
    frames = []

    # 选取关键帧（不超过 30 帧）
    n_inter = len(intermediates)
    if n_inter > 30:
        indices = np.linspace(0, n_inter - 1, 30, dtype=int)
        intermediates = [intermediates[i] for i in indices]

    for t_val, heatmap in intermediates:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
        h = heatmap.numpy()

        # 绘制热力图边
        segments = []
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if h[i, j] > 0.03:
                    segments.append([c[i], c[j]])
                    colors.append(h[i, j])
        if segments:
            lc = LineCollection(segments, cmap='hot_r', alpha=0.7, linewidths=1.0)
            lc.set_array(np.array(colors))
            lc.set_clim(0, 1)
            ax.add_collection(lc)

        ax.scatter(c[:, 0], c[:, 1], c='black', s=30, zorder=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_val:.3f}', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)

    # 最后一帧停留更久
    if frames:
        for _ in range(3):
            frames.append(frames[-1])

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {out_path} ({len(frames)} frames)")


def make_heatmap_evolution_figure(coords, intermediates, mode_name, out_path):
    """绘制热力图演化截图（选 6 个关键步）。"""
    n = len(intermediates)
    n_show = min(6, n)
    indices = np.linspace(0, n - 1, n_show, dtype=int)
    selected = [intermediates[i] for i in indices]

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 3.5))
    if n_show == 1:
        axes = [axes]

    for ax, (t_val, heatmap) in zip(axes, selected):
        plot_heatmap_with_tour(ax, coords, heatmap, title=f't = {t_val:.3f}')

    fig.suptitle(f'{mode_name} — Denoising Process', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Evolution figure saved: {out_path}")


def make_tour_comparison_figure(
    coords_list, tours_model, tours_opt, gaps, mode_names, out_path
):
    """路径对比图：模型预测 vs 最优解。"""
    n_instances = len(coords_list)
    n_models = len(mode_names)

    fig, axes = plt.subplots(
        n_instances, n_models + 1,
        figsize=(3.5 * (n_models + 1), 3.5 * n_instances),
    )
    if n_instances == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_instances):
        c = coords_list[row]
        opt = tours_opt[row]
        opt_len = tour_length(opt, c)

        # 最优解列
        ax = axes[row, 0]
        plot_heatmap_with_tour(ax, c, None, tour=opt,
                               title=f'Optimal ({opt_len:.3f})')

        # 各模型列
        for col, mname in enumerate(mode_names):
            ax = axes[row, col + 1]
            pred = tours_model[row][col]
            if pred is not None:
                pred_len = tour_length(pred, c)
                gap = gaps[row][col]
                plot_heatmap_with_tour(
                    ax, c, None, tour=pred,
                    title=f'{mname}\n{pred_len:.3f} (gap={gap:.1f}%)',
                )
            else:
                ax.set_title(f'{mname}\nN/A')
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Tour comparison saved: {out_path}")


# ============================================================================
# 主函数
# ============================================================================

def load_model(ckpt_path, device):
    """加载模型（复用 evaluate.py 逻辑）。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})

    mode = ckpt.get('mode') or saved_args.get('mode', 'flow_matching')
    encoder_type = saved_args.get('encoder_type', 'gated_gcn')
    n_layers = saved_args.get('n_layers', 12)
    hidden_dim = saved_args.get('hidden_dim', 256)
    T = saved_args.get('T', 1000)
    diffusion_schedule = saved_args.get('diffusion_schedule', 'linear')
    inference_schedule = saved_args.get('inference_schedule', 'cosine')
    inference_steps = 20 if mode == 'flow_matching' else 50

    model = TSPDiffusionModel(
        mode=mode,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        T=T,
        diffusion_schedule=diffusion_schedule,
        inference_schedule=inference_schedule,
        inference_steps=inference_steps,
    ).to(device)

    state_key = 'ema_state' if 'ema_state' in ckpt else 'model_state'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/tsp50_test.txt')
    parser.add_argument('--instance_idx', type=int, default=0,
                        help='which test instance to visualize')
    parser.add_argument('--n_instances', type=int, default=3,
                        help='number of instances for tour comparison')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['flow_matching', 'discrete_ddpm', 'continuous_ddpm'],
                        help='only visualize one mode (default: all)')
    parser.add_argument('--out_dir', type=str, default='report/figs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # 加载数据
    dataset = TSPDataset(args.data_file)
    print(f"Dataset: {args.data_file} ({len(dataset)} instances)")

    # 模型配置
    model_configs = {
        'flow_matching': {
            'ckpt': 'checkpoints/flow_matching_gated_gcn/best.pt',
            'label': 'Flow Matching',
            'short': 'fm',
        },
        'discrete_ddpm': {
            'ckpt': 'checkpoints/discrete_ddpm_gated_gcn/best.pt',
            'label': 'D3PM',
            'short': 'd3pm',
        },
        'continuous_ddpm': {
            'ckpt': 'checkpoints/continuous_ddpm_gated_gcn/best.pt',
            'label': 'Continuous DDPM',
            'short': 'ddpm',
        },
    }

    if args.mode:
        model_configs = {args.mode: model_configs[args.mode]}

    # ── 1. 扩散过程 GIF + 热力图演化 ──

    # 固定随机种子，让三个模型从同一个噪声起步
    seed = 42
    idx = args.instance_idx
    coords_tensor, adj_gt, tour_gt = dataset[idx]
    coords_batch = coords_tensor.unsqueeze(0).to(device)
    tour_gt_list = tour_gt.tolist()

    for mode_name, cfg in model_configs.items():
        ckpt_path = cfg['ckpt']
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {mode_name}: {ckpt_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"  {cfg['label']} — Generating visualizations")
        print(f"{'='*50}")

        model, _ = load_model(ckpt_path, device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        steps = model.inference_steps
        record_every = max(1, steps // 20)  # ~20 个快照

        heatmap, intermediates = sample_with_intermediates(
            model, coords_batch, steps=steps, record_every=record_every
        )

        # GIF
        gif_path = os.path.join(args.out_dir, f'diffusion_{cfg["short"]}.gif')
        make_diffusion_gif(coords_tensor, intermediates, tour_gt_list, gif_path, fps=4)

        # 热力图演化
        evo_path = os.path.join(args.out_dir, f'heatmap_evolution_{cfg["short"]}.png')
        make_heatmap_evolution_figure(
            coords_tensor, intermediates, cfg['label'], evo_path
        )

    # ── 2. 路径对比图 ──

    print(f"\n{'='*50}")
    print(f"  Tour Comparison")
    print(f"{'='*50}")

    n_inst = min(args.n_instances, len(dataset))
    coords_list = []
    tours_model_all = []
    tours_opt_all = []
    gaps_all = []
    active_modes = []

    for mode_name, cfg in model_configs.items():
        if os.path.exists(cfg['ckpt']):
            active_modes.append((mode_name, cfg))

    for inst_idx in range(n_inst):
        coords_i, adj_gt_i, tour_gt_i = dataset[inst_idx]
        coords_list.append(coords_i)
        tours_opt_all.append(tour_gt_i.tolist())

        row_tours = []
        row_gaps = []
        for mode_name, cfg in active_modes:
            model, _ = load_model(cfg['ckpt'], device)
            cb = coords_i.unsqueeze(0).to(device)

            torch.manual_seed(seed + inst_idx)
            heatmap = model.sample(cb)
            h = heatmap[0].cpu()
            pred_tour = merge_tours(h, coords_i)
            pred_tour = two_opt_improve(pred_tour, coords_i)

            pred_len = tour_length(pred_tour, coords_i)
            opt_len = tour_length(tour_gt_i.tolist(), coords_i)
            gap = (pred_len - opt_len) / opt_len * 100.0

            row_tours.append(pred_tour)
            row_gaps.append(gap)

        tours_model_all.append(row_tours)
        gaps_all.append(row_gaps)

    mode_labels = [cfg['label'] for _, cfg in active_modes]
    tour_cmp_path = os.path.join(args.out_dir, 'tour_comparison.png')
    make_tour_comparison_figure(
        coords_list, tours_model_all, tours_opt_all, gaps_all,
        mode_labels, tour_cmp_path,
    )

    print(f"\nAll visualizations saved to: {args.out_dir}/")


if __name__ == '__main__':
    main()
