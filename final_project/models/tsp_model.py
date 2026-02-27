"""
TSP 扩散主模型 — 简化版 DIFUSCO pl_tsp_model.py (纯 PyTorch)

参考: refs/DIFUSCO/difusco/pl_tsp_model.py

需要实现:
1. TSPDiffusionModel(nn.Module):
   - __init__: 组合 GNNEncoder + CategoricalDiffusion
   - forward(coords, adj_t, t): 预测去噪邻接矩阵
   - compute_loss(coords, adj_0): 完整训练 loss 计算
     * 采样 t ~ Uniform(1, T)
     * 加噪 adj_t = diffusion.sample(adj_0, t)
     * 预测 pred_adj_0 = forward(coords, adj_t, t)
     * BCE loss
   - denoise(coords, inference_steps=50): 从噪声逐步去噪
     * 返回概率热力图 (N, N)
   - get_intermediate_heatmap(coords, t): 获取中间步骤的热力图 (用于可视化)

关键超参数:
    n_layers = 4
    hidden_dim = 128
    diffusion_steps = 1000

验证: 随机初始化时 loss ≈ ln(2) ≈ 0.693
"""

import torch
import torch.nn as nn

# TODO: 基于 DIFUSCO 的 pl_tsp_model.py 简化实现
