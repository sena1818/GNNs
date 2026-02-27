"""
训练入口脚本 — TSP 扩散模型训练

需要实现:
1. 参数解析 (argparse):
   --data_file: 训练数据路径 (默认 data/tsp20_train.txt)
   --batch_size: 批大小 (默认 64, GPU不足降到32)
   --lr: 学习率 (默认 2e-4)
   --weight_decay: 权重衰减 (默认 1e-4)
   --epochs: 训练轮数 (默认 50)
   --warmup_steps: 学习率预热步数 (默认 1000)
   --n_layers: GNN 层数 (默认 4)
   --hidden_dim: 隐藏维度 (默认 128)
   --diffusion_steps: 扩散步数 (默认 1000)
   --save_dir: checkpoint 保存路径
   --encoder_type: GNN 编码器类型 (gated_gcn / gat / gcn)

2. 训练循环:
   - Adam optimizer + CosineAnnealingLR
   - 梯度裁剪 (max_norm=1.0)
   - EMA 模型 (decay=0.999)
   - 每 epoch 打印: Epoch X | Loss: X.XXX | LR: X.XXe-4
   - 每 epoch 保存 checkpoint

3. 设备支持:
   - Apple MPS (本地调试)
   - CUDA (Colab T4 训练大数据集)
   - CPU (fallback)

用法:
    # TSP-20 调试
    python train.py --data_file data/tsp20_train.txt --epochs 5

    # TSP-50 正式训练
    python train.py --data_file data/tsp50_train.txt --epochs 50 --batch_size 32
"""

import argparse
import torch

# TODO: 实现训练逻辑

if __name__ == '__main__':
    pass
