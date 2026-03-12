"""
评估脚本 — 加载 checkpoint，推理，计算 Optimality Gap

用法:
    python evaluate.py --checkpoint checkpoints/fm_gatedgcn/best.pt \\
                       --data_file data/tsp50_train.txt \\
                       --decode greedy

    python evaluate.py --checkpoint checkpoints/fm_gatedgcn/best.pt \\
                       --data_file data/tsp100_test.txt \\
                       --decode beam_search --beam_k 5 --use_2opt

输出示例:
    TSP-50 Test Results (1000 instances, decoder=greedy):
    Avg Optimality Gap : 2.34%
    Best Gap           : 0.12%
    Worst Gap          : 8.76%
    Valid Tour Rate    : 100.0%
    Avg Inference Time : 45.3 ms/instance
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
from models.tsp_model import TSPFlowMatchingModel
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

    model = TSPFlowMatchingModel(
        n_layers=saved_args.get('n_layers', 4),
        hidden_dim=saved_args.get('hidden_dim', 128),
        encoder_type=saved_args.get('encoder_type', 'gated_gcn'),
        inference_steps=args.inference_steps,
    ).to(device)

    # 优先使用 EMA 权重（质量更好）
    state_key = 'ema_state' if 'ema_state' in ckpt else 'model_state'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    print(f'Loaded {state_key} from epoch {ckpt.get("epoch", "?")}')

    # 数据集（全量测试）
    dataset = TSPDataset(args.data_file)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 推理与评估
    all_gaps   = []
    all_valid  = []
    total_infer_time = 0.0

    for coords, adj_0, tours_gt in loader:
        coords  = coords.to(device)
        B, N, _ = coords.shape

        # 推理：Flow Matching 采样
        t_start = time.time()
        with torch.no_grad():
            heatmaps = model.sample(coords, inference_steps=args.inference_steps)
        total_infer_time += time.time() - t_start

        # 解码
        pred_tours = batch_decode(
            heatmaps, coords,
            method=args.decode,
            beam_k=args.beam_k,
            use_2opt=args.use_2opt,
        )

        # 计算每个实例的 gap
        for i in range(B):
            c = coords[i].cpu()
            pred = pred_tours[i]
            opt  = tours_gt[i].tolist()

            valid = is_valid_tour(pred, N)
            all_valid.append(valid)

            if valid:
                gap = compute_gap(pred, opt, c)
                all_gaps.append(gap)

    # 汇总
    n_total   = len(dataset)
    n_valid   = sum(all_valid)
    avg_gap   = sum(all_gaps) / len(all_gaps) if all_gaps else float('nan')
    best_gap  = min(all_gaps) if all_gaps else float('nan')
    worst_gap = max(all_gaps) if all_gaps else float('nan')
    avg_ms    = total_infer_time / n_total * 1000

    decode_str = args.decode + ('+2opt' if args.use_2opt else '')
    data_name  = os.path.basename(args.data_file).replace('.txt', '')

    print(f'\n{"="*55}')
    print(f'{data_name} Results ({n_total} instances, decoder={decode_str}):')
    print(f'  Avg Optimality Gap : {avg_gap:.2f}%')
    print(f'  Best Gap           : {best_gap:.2f}%')
    print(f'  Worst Gap          : {worst_gap:.2f}%')
    print(f'  Valid Tour Rate    : {n_valid/n_total*100:.1f}%')
    print(f'  Avg Inference Time : {avg_ms:.1f} ms/instance')
    print(f'{"="*55}')

    # 保存结果 JSON（供画图使用）
    result = {
        'data_file': args.data_file,
        'checkpoint': args.checkpoint,
        'decoder': decode_str,
        'inference_steps': args.inference_steps,
        'n_total': n_total,
        'n_valid': n_valid,
        'avg_gap': avg_gap,
        'best_gap': best_gap,
        'worst_gap': worst_gap,
        'valid_rate': n_valid / n_total,
        'avg_infer_ms': avg_ms,
        'all_gaps': all_gaps,
    }
    if args.save_result:
        with open(args.save_result, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Results saved to: {args.save_result}')

    return result


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate TSP Flow Matching model')
    p.add_argument('--checkpoint',      type=str, required=True)
    p.add_argument('--data_file',       type=str, required=True)
    p.add_argument('--batch_size',      type=int, default=32)
    p.add_argument('--inference_steps', type=int, default=20,
                   help='Euler integration steps for Flow Matching')
    p.add_argument('--decode',          type=str, default='greedy',
                   choices=['greedy', 'beam_search'])
    p.add_argument('--beam_k',          type=int, default=5)
    p.add_argument('--use_2opt',        action='store_true',
                   help='Apply 2-opt local search after decoding')
    p.add_argument('--save_result',     type=str, default=None,
                   help='Save result JSON to this path')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
