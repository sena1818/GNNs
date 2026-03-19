"""
训练入口脚本 — 三种扩散框架统一训练 (对齐 DIFUSCO 超参数)

DIFUSCO 官方超参数 (reproducing_scripts.md):
  ✅ optimizer:     AdamW (不是 Adam)
  ✅ learning_rate: 2e-4
  ✅ weight_decay:  1e-4
  ✅ lr_scheduler:  cosine-decay
  ✅ batch_size:    64 (TSP50), 32 (TSP100)
  ✅ num_epochs:    50
  ✅ inference_schedule: cosine
  ✅ inference_steps:    50

用法:
  # TSP-20 快速调试
  python train.py --mode flow_matching --data_file data/tsp20_train.txt --epochs 5

  # TSP-50 正式训练 (对齐 DIFUSCO)
  python train.py --mode discrete_ddpm --data_file data/tsp50_train.txt

  # 三方对比
  python train.py --mode flow_matching   --data_file data/tsp50_train.txt
  python train.py --mode discrete_ddpm   --data_file data/tsp50_train.txt
  python train.py --mode continuous_ddpm --data_file data/tsp50_train.txt
"""

import argparse
import copy
import os
import random
import sys
import time
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from models.tsp_dataset import TSPDataset, collate_fn
from models.tsp_model import TSPDiffusionModel


# ---------------------------------------------------------------------------
# EMA (与 DIFUSCO nn.py update_ema 一致)
# ---------------------------------------------------------------------------

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999):
    """指数移动平均: ema_params = decay * ema_params + (1-decay) * params"""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# 学习率调度 (对齐 DIFUSCO cosine-decay)
# ---------------------------------------------------------------------------

def get_cosine_decay_scheduler(optimizer, total_steps: int, warmup_steps: int = 0):
    """
    Cosine decay with optional warmup.
    DIFUSCO 官方用 cosine-decay via get_schedule_fn。
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(args):
    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    print(f'Mode:   {args.mode}')
    print(f'Encoder:{args.encoder_type}')

    # 数据集 (90/10 划分)
    dataset = TSPDataset(args.data_file)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,  # DIFUSCO 官方: drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f'Train: {n_train} | Val: {n_val} | Batches/epoch: {len(train_loader)}')

    # 模型
    model = TSPDiffusionModel(
        mode=args.mode,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        T=args.T,
        diffusion_schedule=args.diffusion_schedule,
        inference_schedule=args.inference_schedule,
        inference_steps=args.inference_steps,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params:,}')

    # 优化器: AdamW (DIFUSCO 官方)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度: cosine-decay (DIFUSCO 官方)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_decay_scheduler(optimizer, total_steps, args.warmup_steps)

    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 1
    history = {
        'mode': args.mode,
        'encoder_type': args.encoder_type,
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }

    # 续训
    if args.resume:
        ckpt_path = args.resume
        print(f'Resuming from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        ema_model.load_state_dict(ckpt['ema_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        print(f'  Resumed from epoch {ckpt["epoch"]}, best_val_loss={best_val_loss:.6f}')
        hist_path = os.path.join(args.save_dir, 'history.json')
        if os.path.exists(hist_path):
            history = json.load(open(hist_path))

    # --------------- 训练循环 ---------------
    global_step = (start_epoch - 1) * len(train_loader)
    last_ckpt = None

    # 续训时恢复 scheduler
    if args.resume and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    elif args.resume and 'scheduler_state' not in ckpt:
        for _ in range(global_step):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for coords, adj_0, _ in train_loader:
            coords = coords.to(device)
            adj_0  = adj_0.to(device)

            loss = model.compute_loss(coords, adj_0)

            if not torch.isfinite(loss):
                global_step += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model, model, decay=args.ema_decay)

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]['lr']

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for coords, adj_0, _ in val_loader:
                coords = coords.to(device)
                adj_0  = adj_0.to(device)
                val_loss += model.compute_loss(coords, adj_0).item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | '
            f'LR: {current_lr:.2e} | {elapsed:.1f}s'
        )

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # 构建 checkpoint 数据 (每个 epoch 都构建，解决作用域问题)
        ckpt_data = {
            'epoch': epoch,
            'mode': args.mode,
            'encoder_type': args.encoder_type,
            'model_state': model.state_dict(),
            'ema_state': ema_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'args': vars(args),
        }
        last_ckpt = ckpt_data

        # 保存最佳
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_data, os.path.join(args.save_dir, 'best.pt'))

        # 每 10 epoch 保存 (使用当前 epoch 的 ckpt，不再有作用域问题)
        if epoch % 10 == 0:
            torch.save(ckpt_data, os.path.join(args.save_dir, f'epoch{epoch:03d}.pt'))

        # 保存最后一个 (方便续训)
        torch.save(ckpt_data, os.path.join(args.save_dir, 'last.pt'))

        # 历史
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f'\nDone. Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoint: {args.save_dir}/best.pt')


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train TSP Diffusion Model (FM / D3PM / DDPM)')

    # 核心参数
    p.add_argument('--mode', type=str, default='flow_matching',
                   choices=['flow_matching', 'discrete_ddpm', 'continuous_ddpm'],
                   help='生成框架')
    p.add_argument('--data_file', type=str, default='data/tsp20_train.txt')
    p.add_argument('--save_dir',  type=str, default=None)

    # 模型结构
    p.add_argument('--encoder_type', type=str, default='gated_gcn',
                   choices=['gated_gcn', 'gat', 'gcn'])
    p.add_argument('--n_layers',   type=int, default=12)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--T',          type=int, default=1000)
    p.add_argument('--inference_steps', type=int, default=None,
                   help='推理步数; FM 默认 20, D3PM/DDPM 默认 50')

    # 扩散 schedule
    p.add_argument('--diffusion_schedule', type=str, default='linear',
                   choices=['linear', 'cosine'],
                   help='Beta schedule (DIFUSCO 训练用 linear)')
    p.add_argument('--inference_schedule', type=str, default='cosine',
                   choices=['linear', 'cosine'],
                   help='推理时间步分布 (DIFUSCO 默认 cosine)')

    # 训练超参数 (对齐 DIFUSCO 官方)
    p.add_argument('--batch_size',   type=int,   default=64,
                   help='DIFUSCO: 64 (TSP50), 32 (TSP100)')
    p.add_argument('--lr',           type=float, default=2e-4,
                   help='DIFUSCO 官方: 0.0002')
    p.add_argument('--weight_decay', type=float, default=1e-4,
                   help='DIFUSCO 官方: 0.0001')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--warmup_steps', type=int,   default=0,
                   help='线性 warmup 步数 (DIFUSCO 无 warmup)')
    p.add_argument('--ema_decay',    type=float, default=0.999)
    p.add_argument('--resume', type=str, default=None,
                   help='从 checkpoint 续训')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子 (可复现)')

    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = f'checkpoints/{args.mode}_{args.encoder_type}'

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
