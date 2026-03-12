"""
TSP Flow Matching 主模型 — 基于 Lipman et al. (ICLR 2023) + DIFUSCO 架构

核心思想: 用直线 ODE 替代 DDPM 的随机 SDE
  训练: 学习速度场 v_θ(A_t, t, coords)，使 A_t 从噪声沿直线走向 A_0
  推理: 20步欧拉积分 A_{t-dt} = A_t - dt * v_θ(A_t, t, coords)

关键超参数:
    n_layers = 4
    hidden_dim = 128
    inference_steps = 20   (替代 DDPM 的 diffusion_steps=1000)

验证: 随机初始化时 MSE loss ≈ 1.0 (速度预测随机时的 L2 误差)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import GNNEncoder
from .diffusion_schedulers import FlowMatchingScheduler, InferenceSchedule


class TSPFlowMatchingModel(nn.Module):
    """
    Flow Matching 模型，用于 TSP 邻接矩阵生成。

    接口:
        model = TSPFlowMatchingModel(n_layers=4, hidden_dim=128, encoder_type='gated_gcn')

        # 训练
        loss = model.compute_loss(coords, adj_0)
        loss.backward()

        # 推理
        heatmap = model.sample(coords, inference_steps=20)  # (B, N, N) ∈ [0,1]
    """

    def __init__(
        self,
        n_layers: int = 4,
        hidden_dim: int = 128,
        encoder_type: str = 'gated_gcn',
        inference_steps: int = 20,
    ):
        super().__init__()
        self.inference_steps = inference_steps
        self.encoder = GNNEncoder(n_layers=n_layers, hidden_dim=hidden_dim, encoder_type=encoder_type)
        self.scheduler = FlowMatchingScheduler()

    def forward(
        self,
        coords: torch.Tensor,   # (B, N, 2)
        adj_t: torch.Tensor,    # (B, N, N)  插值状态 A_t
        t: torch.Tensor,        # (B,)       时间步 ∈ [0,1]
    ) -> torch.Tensor:          # (B, N, N)  预测速度场 v_θ
        return self.encoder(coords, adj_t, t)

    def compute_loss(
        self,
        coords: torch.Tensor,   # (B, N, 2)
        adj_0: torch.Tensor,    # (B, N, N)  真实邻接矩阵（训练标签）
    ) -> torch.Tensor:          # scalar MSE loss
        """
        Flow Matching 训练损失:
            1. t ~ Uniform(0, 1)
            2. ε ~ N(0, I)
            3. A_t = (1-t)*A_0 + t*ε
            4. v_pred = GNN(coords, A_t, t)
            5. loss = MSE(v_pred, ε - A_0)
        """
        B = adj_0.shape[0]

        # 采样时间步和噪声
        t = torch.rand(B, device=adj_0.device)
        epsilon = torch.randn_like(adj_0)

        # 线性插值：构造中间状态
        adj_t = self.scheduler.interpolate(adj_0, epsilon, t)

        # GNN 预测速度场
        pred_v = self.forward(coords, adj_t, t)

        # MSE 损失（目标是恒定速度 ε - A_0）
        v_target = self.scheduler.get_velocity_target(adj_0, epsilon)
        return F.mse_loss(pred_v, v_target)

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,       # (B, N, 2)
        inference_steps: int = None,
    ) -> torch.Tensor:              # (B, N, N) ∈ [0,1]
        """
        推理：从 X_1 ~ N(0,I) 欧拉积分到 X_0，返回边概率热力图。

        每步: A_{t-dt} = A_t - dt * v_θ(A_t, t, coords)
        最终: heatmap = sigmoid(A_0)，对称化（TSP 无向图）
        """
        if inference_steps is None:
            inference_steps = self.inference_steps

        B, N, _ = coords.shape
        device = coords.device

        # 起点：纯高斯噪声
        x = torch.randn(B, N, N, device=device)

        # 欧拉积分：t 从 1.0 走到 0.0
        for t_val, dt in InferenceSchedule(inference_steps):
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.forward(coords, x, t_tensor)
            x = x - dt * v

        # sigmoid 映射到 [0,1]，对称化（无向图）
        heatmap = torch.sigmoid(x)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0

        return heatmap

    @torch.no_grad()
    def get_intermediate_heatmap(
        self,
        coords: torch.Tensor,   # (1, N, 2) or (B, N, 2)
        target_t: float,        # 停在哪个时刻截图（0.0~1.0）
        total_steps: int = 20,
    ) -> torch.Tensor:          # (B, N, N) ∈ [0,1]  中间状态热力图
        """
        用于可视化：推理中途在 target_t 时刻截图，生成扩散 GIF 的每一帧。
        target_t=1.0 → 纯噪声；target_t=0.0 → 最终结果。
        """
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)
        dt = 1.0 / total_steps

        for i in range(total_steps):
            t_val = 1.0 - i * dt
            if t_val < target_t:
                break
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.forward(coords, x, t_tensor)
            x = x - dt * v

        heatmap = torch.sigmoid(x)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap


# ---------------------------------------------------------------------------
# 快速单元测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, N = 4, 20
    device = 'cpu'

    coords = torch.rand(B, N, 2)
    adj_0 = torch.zeros(B, N, N)
    # 构造合法邻接矩阵（随机 tour）
    for b in range(B):
        perm = torch.randperm(N)
        for k in range(N):
            i, j = perm[k].item(), perm[(k+1) % N].item()
            adj_0[b, i, j] = 1.0
            adj_0[b, j, i] = 1.0

    for enc_type in ['gated_gcn', 'gat', 'gcn']:
        model = TSPFlowMatchingModel(n_layers=2, hidden_dim=64, encoder_type=enc_type)

        # 验证 loss（随机初始化时应 ≈ 1.0）
        loss = model.compute_loss(coords, adj_0)
        print(f'[{enc_type:12s}] loss={loss.item():.4f} (expect ~1.0)')

        # 验证推理
        heatmap = model.sample(coords, inference_steps=5)
        assert heatmap.shape == (B, N, N)
        assert 0.0 <= heatmap.min().item() and heatmap.max().item() <= 1.0
        assert (heatmap - heatmap.transpose(-1,-2)).abs().max() < 1e-5, "not symmetric"

    print('TSPFlowMatchingModel OK')
