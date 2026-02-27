"""
GNN 编码器模块 — 基于 DIFUSCO 的 Gated GCN 改写

参考: refs/DIFUSCO/difusco/models/gnn_encoder.py

需要实现:
1. GNNLayer: 门控图卷积层
   - gate_ij = sigmoid(A*h_i + B*h_j + C*e_ij)
   - h_i_new = ReLU(U*h_i + scatter_add(gate_ij * V*h_j))
2. GNNEncoder: 多层 GNN 编码器
   - 输入: coords (B, N, 2), adj_noisy (B, N, N), t (B,)
   - 输出: pred_adj (B, N, N) — 预测的去噪邻接矩阵
3. PositionEmbeddingSine: 城市坐标的正弦位置编码
4. 时间步嵌入 (sinusoidal timestep embedding)

单元测试:
    encoder = GNNEncoder(n_layers=4, hidden_dim=128)
    coords = torch.rand(2, 20, 2)
    adj_noisy = torch.rand(2, 20, 20)
    t = torch.tensor([500, 200])
    out = encoder(coords, adj_noisy, t)  # 期望 (2, 20, 20)
"""

import torch
import torch.nn as nn

# TODO: 从 DIFUSCO 的 gnn_encoder.py 改写实现
