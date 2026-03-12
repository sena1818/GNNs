"""
训练入口脚本 — TSP Flow Matching 模型训练

用法:
    # TSP-20 快速调试（5 epoch，几分钟）
    python train.py --data_file data/tsp20_train.txt --epochs 5 --batch_size 32

    # TSP-50 正式训练（50 epoch，推荐在 GPU 上跑）
    python train.py --data_file data/tsp50_train.txt --epochs 50 --encoder_type gated_gcn

    # 消融实验：切换架构
    python train.py --data_file data/tsp50_train.txt --encoder_type gat   --save_dir checkpoints/fm_gat
    python train.py --data_file data/tsp50_train.txt --encoder_type gcn   --save_dir checkpoints/fm_gcn
"""

import argparse
import copy
import os
import sys
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from models.tsp_dataset import TSPDataset, collate_fn
from models.tsp_model import TSPFlowMatchingModel


# ---------------------------------------------------------------------------
# EMA 更新
# ---------------------------------------------------------------------------

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999):
    """指数移动平均：ema_params = decay * ema_params + (1-decay) * params"""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# 学习率预热 + Cosine 调度
# ---------------------------------------------------------------------------

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """线性预热 → Cosine 衰减"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(args):
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # 数据集（90% 训练，10% 验证）
    dataset = TSPDataset(args.data_file)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f'Train: {n_train} | Val: {n_val} | Batches/epoch: {len(train_loader)}')

    # 模型
    model = TSPFlowMatchingModel(
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder_type,
        inference_steps=args.inference_steps,
    ).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Encoder: {args.encoder_type} | Params: {n_params:,}')

    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    # 保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # --------------- 训练循环 ---------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for coords, adj_0, _ in train_loader:
            coords = coords.to(device)
            adj_0  = adj_0.to(device)

            loss = model.compute_loss(coords, adj_0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model, model, decay=0.999)

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for coords, adj_0, _ in val_loader:
                coords = coords.to(device)
                adj_0  = adj_0.to(device)
                val_loss += model.compute_loss(coords, adj_0).item()
        avg_val_loss = val_loss / len(val_loader)

        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | '
              f'LR: {current_lr:.2e} | {elapsed:.1f}s')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # 保存最佳 checkpoint（用 EMA 模型）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))

        # 每 10 epoch 额外保存一次
        if epoch % 10 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f'epoch{epoch:03d}.pt'))

    # 保存训练历史（用于画 loss 曲线）
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining done. Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoint saved to: {args.save_dir}/best.pt')


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train TSP Flow Matching model')
    p.add_argument('--data_file',       type=str,   default='data/tsp20_train.txt')
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--warmup_steps',    type=int,   default=1000)
    p.add_argument('--n_layers',        type=int,   default=4)
    p.add_argument('--hidden_dim',      type=int,   default=128)
    p.add_argument('--inference_steps', type=int,   default=20)
    p.add_argument('--encoder_type',    type=str,   default='gated_gcn',
                   choices=['gated_gcn', 'gat', 'gcn'])
    p.add_argument('--save_dir',        type=str,   default='checkpoints/fm_gatedgcn')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
