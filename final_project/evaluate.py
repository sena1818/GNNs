"""
评估脚本 — 加载 checkpoint，推理，计算 Optimality Gap

支持三种模式（mode 自动从 checkpoint 读取，也可手动覆盖）:
  flow_matching   — 20-step Euler ODE
  discrete_ddpm   — 50-step D3PM 后验采样
  continuous_ddpm — 50-step DDIM 确定性采样

用法:
    # 自动识别 mode（推荐）
    python evaluate.py --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
                       --data_file data/tsp20_train.txt

    # 指定解码策略 + 2-opt
    python evaluate.py --checkpoint checkpoints/discrete_ddpm_gated_gcn/best.pt \
                       --data_file data/tsp50_train.txt \
                       --decode beam_search --beam_k 5 --use_2opt

    # 跨规模泛化测试
    python evaluate.py --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
                       --data_file data/tsp100_test.txt --inference_steps 20

    # 保存 JSON 供绘图
    python evaluate.py --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
                       --data_file data/tsp50_train.txt \
                       --save_result results/fm_tsp50_greedy.json

输出示例:
    Mode: flow_matching | Encoder: gated_gcn | Epoch: 50
    ======================================================
    tsp50_train Results (5000 instances, decoder=greedy):
      Avg Optimality Gap : 2.34%
      Best Gap           : 0.12%
      Worst Gap          : 8.76%
      Valid Tour Rate    : 100.0%
      Avg Inference Time : 45.3 ms/instance
    ======================================================
"""

import argparse
import os
import sys
import time
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from models.tsp_dataset import TSPDataset, collate_fn
from models.tsp_model import TSPDiffusionModel
from utils.decode import batch_decode, tour_length, is_valid_tour


# ---------------------------------------------------------------------------
# Optimality Gap 计算
# ---------------------------------------------------------------------------

def compute_gap(pred_tour, opt_tour, coords):
    """
    gap = (pred_cost - opt_cost) / opt_cost * 100%
    返回 gap (%)
    """
    pred_cost = tour_length(pred_tour, coords)
    opt_cost  = tour_length(opt_tour, coords)
    if opt_cost < 1e-10:
        return 0.0
    return (pred_cost - opt_cost) / opt_cost * 100.0


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------

def evaluate(args):
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})

    # mode / encoder 优先从 checkpoint 读取，允许手动覆盖
    mode = args.mode or ckpt.get('mode') or saved_args.get('mode')
    if mode is None:
        mode = 'flow_matching'
        print('WARNING: mode not found in checkpoint, defaulting to flow_matching. '
              'Use --mode to override if this is wrong.')
    elif args.mode is None:
        print(f'INFO: mode inferred from checkpoint: {mode}')

    encoder_type = saved_args.get('encoder_type', 'gated_gcn')
    n_layers     = saved_args.get('n_layers', 12)
    hidden_dim   = saved_args.get('hidden_dim', 256)
    T            = saved_args.get('T', 1000)

    # 推理步数：命令行优先，否则按 mode 使用默认值
    if args.inference_steps is not None:
        inference_steps = args.inference_steps
    else:
        inference_steps = 20 if mode == 'flow_matching' else 50

    print(f'Mode: {mode} | Encoder: {encoder_type} | Epoch: {ckpt.get("epoch", "?")}')
    print(f'Inference steps: {inference_steps} | Device: {device}')

    # 构建模型
    model = TSPDiffusionModel(
        mode=mode,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        T=T,
        inference_steps=inference_steps,
    ).to(device)

    # 优先使用 EMA 权重（训练稳定性更好）
    state_key = 'ema_state' if 'ema_state' in ckpt else 'model_state'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    print(f'Loaded {state_key} from checkpoint.')

    # 数据集（全量测试，不做 val split）
    dataset = TSPDataset(args.data_file)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f'Dataset: {args.data_file} ({len(dataset)} instances)')

    # 推理与评估
    all_gaps         = []
    all_valid        = []
    total_infer_time = 0.0

    for batch_idx, (coords, adj_0, tours_gt) in enumerate(loader):
        coords = coords.to(device)
        B, N, _ = coords.shape

        # 扩散推理 → 热力图 (B, N, N)
        t_start = time.time()
        with torch.no_grad():
            heatmaps = model.sample(coords, inference_steps=inference_steps)
        total_infer_time += time.time() - t_start

        # 解码：热力图 → 合法 tour
        pred_tours = batch_decode(
            heatmaps, coords,
            method=args.decode,
            beam_k=args.beam_k,
            use_2opt=args.use_2opt,
        )

        # 逐实例计算 optimality gap
        for i in range(B):
            c    = coords[i].cpu()
            pred = pred_tours[i]
            opt  = tours_gt[i].tolist()   # 数据集中的最优 tour（Concorde / OR-Tools 解）

            valid = is_valid_tour(pred, N)
            all_valid.append(valid)

            if valid:
                gap = compute_gap(pred, opt, c)
                all_gaps.append(gap)
            else:
                # 无效 tour 不参与 gap 统计，但会在最终报告中单独列出
                pass

        # 进度打印（每 10 batch 一次）
        if (batch_idx + 1) % 10 == 0:
            done = (batch_idx + 1) * args.batch_size
            print(f'  [{done}/{len(dataset)}] running avg gap: '
                  f'{sum(all_gaps)/len(all_gaps):.2f}%' if all_gaps else '')

    # 汇总统计
    n_total    = len(dataset)
    n_valid    = sum(all_valid)
    n_invalid  = n_total - n_valid
    avg_gap    = sum(all_gaps) / len(all_gaps) if all_gaps else float('nan')
    best_gap   = min(all_gaps) if all_gaps else float('nan')
    worst_gap  = max(all_gaps) if all_gaps else float('nan')
    avg_ms     = total_infer_time / n_total * 1000

    import statistics
    std_gap = statistics.stdev(all_gaps) if len(all_gaps) > 1 else 0.0

    decode_str = args.decode + ('+2opt' if args.use_2opt else '')
    data_name  = os.path.basename(args.data_file).replace('.txt', '')

    print(f'\n{"="*55}')
    print(f'{data_name} Results ({n_total} instances, decoder={decode_str}):')
    print(f'  Mode               : {mode}')
    print(f'  Encoder            : {encoder_type}')
    print(f'  Avg Optimality Gap : {avg_gap:.2f}% ± {std_gap:.2f}%')
    print(f'    (computed on {n_valid} valid tours only)')
    if n_invalid > 0:
        print(f'  INVALID Tours      : {n_invalid}/{n_total} ({n_invalid/n_total*100:.1f}%) — EXCLUDED from gap')
    print(f'  Best Gap           : {best_gap:.2f}%')
    print(f'  Worst Gap          : {worst_gap:.2f}%')
    print(f'  Valid Tour Rate    : {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)')
    print(f'  Avg Inference Time : {avg_ms:.1f} ms/instance')
    print(f'{"="*55}')

    # 保存 JSON（供 visualize.py 画图使用）
    result = {
        'data_file':       args.data_file,
        'checkpoint':      args.checkpoint,
        'mode':            mode,
        'encoder_type':    encoder_type,
        'decoder':         decode_str,
        'inference_steps': inference_steps,
        'n_total':         n_total,
        'n_valid':         n_valid,
        'n_invalid':       n_invalid,
        'avg_gap':         avg_gap,
        'std_gap':         std_gap,
        'best_gap':        best_gap,
        'worst_gap':       worst_gap,
        'valid_rate':      n_valid / n_total,
        'avg_infer_ms':    avg_ms,
        'all_gaps':        all_gaps,
    }
    if args.save_result:
        os.makedirs(os.path.dirname(args.save_result) or '.', exist_ok=True)
        with open(args.save_result, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Results saved to: {args.save_result}')

    return result


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate TSP Diffusion Model (FM / D3PM / DDPM)',
    )
    p.add_argument('--checkpoint',      type=str, required=True,
                   help='Path to .pt checkpoint file')
    p.add_argument('--data_file',       type=str, required=True,
                   help='Test data file (DIFUSCO format .txt)')
    p.add_argument('--mode',            type=str, default=None,
                   choices=['flow_matching', 'discrete_ddpm', 'continuous_ddpm'],
                   help='Override mode (default: read from checkpoint)')
    p.add_argument('--batch_size',      type=int, default=32)
    p.add_argument('--inference_steps', type=int, default=None,
                   help='Diffusion/ODE steps; default: 20 for FM, 50 for DDPM')
    p.add_argument('--decode',          type=str, default='greedy',
                   choices=['greedy', 'beam_search'],
                   help='Decoding strategy')
    p.add_argument('--beam_k',          type=int, default=5,
                   help='Beam width for beam_search decoding')
    p.add_argument('--use_2opt',        action='store_true',
                   help='Apply 2-opt local search after decoding')
    p.add_argument('--save_result',     type=str, default=None,
                   help='Save result JSON to this path (e.g. results/fm_tsp50.json)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
