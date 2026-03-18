"""
TSP 扩散模型 — 三种生成框架的统一入口

支持三种 mode，共享同一个 GNNEncoder：
  'flow_matching'   — 连续直线 ODE (FM)
  'discrete_ddpm'   — 离散伯努利扩散 (D3PM, DIFUSCO SOTA)
  'continuous_ddpm' — 连续高斯扩散 (DDPM, 对照组)

用法:
    model = TSPDiffusionModel(mode='flow_matching',   n_layers=4, hidden_dim=128)
    model = TSPDiffusionModel(mode='discrete_ddpm',   n_layers=4, hidden_dim=128)
    model = TSPDiffusionModel(mode='continuous_ddpm', n_layers=4, hidden_dim=128)

    # 训练（三种 mode 接口完全一致）
    loss = model.compute_loss(coords, adj_0)
    loss.backward()

    # 推理（三种 mode 接口完全一致）
    heatmap = model.sample(coords)  # (B, N, N) in [0, 1]

GNN 时间步归一化约定:
    FM:   t in [0,1]，直接传入
    D3PM: t in {1,...,T}，传入前除以 T → [0,1]
    DDPM: 同 D3PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import GNNEncoder
from .diffusion_schedulers import (
    FlowMatchingScheduler, InferenceSchedule,
    BernoulliDiffusion,
    GaussianDiffusion,
)


class TSPDiffusionModel(nn.Module):
    """
    三方对比统一模型。mode 参数决定生成框架，其余架构完全相同。

    Args:
        mode:          'flow_matching' | 'discrete_ddpm' | 'continuous_ddpm'
        n_layers:      GNN 层数，默认 4
        hidden_dim:    隐藏层维度，默认 128
        encoder_type:  'gated_gcn' | 'gat' | 'gcn'
        T:             扩散步数，仅 D3PM/DDPM 使用，默认 1000
        inference_steps: 推理步数，FM 默认 20，D3PM/DDPM 默认 50
    """

    MODES = ('flow_matching', 'discrete_ddpm', 'continuous_ddpm')

    def __init__(
        self,
        mode: str = 'flow_matching',
        n_layers: int = 12,
        hidden_dim: int = 256,
        encoder_type: str = 'gated_gcn',
        T: int = 1000,
        inference_steps: int = None,
    ):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}, got '{mode}'"

        self.mode = mode
        self.T = T
        self.inference_steps = inference_steps or (20 if mode == 'flow_matching' else 50)

        # 共享 GNN 编码器（三种框架完全相同的网络结构）
        self.encoder = GNNEncoder(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
        )

        # 各框架专属调度器（不是 nn.Module，不含可训练参数）
        if mode == 'flow_matching':
            self.scheduler = FlowMatchingScheduler()
        elif mode == 'discrete_ddpm':
            self.scheduler = BernoulliDiffusion(T=T)
        elif mode == 'continuous_ddpm':
            self.scheduler = GaussianDiffusion(T=T)

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def compute_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        计算训练损失。三种 mode 接口相同，内部逻辑不同。

        Args:
            coords: (B, N, 2) 城市坐标
            adj_0:  (B, N, N) 真实邻接矩阵 {0,1}（ground truth 标签）
        Returns:
            loss: scalar
        """
        if self.mode == 'flow_matching':
            return self._fm_loss(coords, adj_0)
        elif self.mode == 'discrete_ddpm':
            return self._d3pm_loss(coords, adj_0)
        elif self.mode == 'continuous_ddpm':
            return self._ddpm_loss(coords, adj_0)

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,       # (B, N, 2)
        inference_steps: int = None,
    ) -> torch.Tensor:              # (B, N, N) in [0, 1]
        """
        推理：生成 TSP 边概率热力图。

        Args:
            coords:          (B, N, 2)
            inference_steps: 推理步数，None 则使用初始化时设定的默认值
        Returns:
            heatmap: (B, N, N) in [0, 1]，对称矩阵，值越大表示该边存在概率越高
        """
        steps = inference_steps or self.inference_steps
        if self.mode == 'flow_matching':
            return self._fm_sample(coords, steps)
        elif self.mode == 'discrete_ddpm':
            return self._d3pm_sample(coords, steps)
        elif self.mode == 'continuous_ddpm':
            return self._ddpm_sample(coords, steps)

    # -------------------------------------------------------------------------
    # Flow Matching 内部实现
    # -------------------------------------------------------------------------

    def _fm_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        FM 训练损失：MSE on velocity field。

        步骤:
          1. t ~ Uniform(0, 1)
          2. ε ~ N(0, I)
          3. A_t = (1-t)*A_0 + t*ε
          4. v_pred = GNN(coords, A_t, t)
          5. loss = MSE(v_pred, ε - A_0)
        """
        B = adj_0.shape[0]
        t = torch.rand(B, device=adj_0.device)                    # (B,) in [0,1]
        epsilon = torch.randn_like(adj_0)

        adj_t = self.scheduler.interpolate(adj_0, epsilon, t)     # (B,N,N)
        pred_v = self.encoder(coords, adj_t, t)                   # (B,N,N)

        v_target = self.scheduler.get_velocity_target(adj_0, epsilon)
        return F.mse_loss(pred_v, v_target)

    def _fm_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        """
        FM 推理：从 X_1~N(0,I) 欧拉积分到 X_0。

        每步: A_{t-dt} = A_t - dt * v_θ(A_t, t, coords)
        最终: heatmap = sigmoid(A_0)，对称化
        """
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)                   # 起点

        for t_val, dt in InferenceSchedule(steps):
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.encoder(coords, x, t_tensor)
            x = x - dt * v                                         # 欧拉步

        heatmap = torch.sigmoid(x)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0     # 对称化
        return heatmap

    # -------------------------------------------------------------------------
    # Discrete DDPM (D3PM) 内部实现
    # -------------------------------------------------------------------------

    def _d3pm_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        D3PM 训练损失：BCE on x_0 prediction。

        步骤:
          1. t ~ randint(1, T)
          2. x_t = q_sample(x_0, t)   ← 比特翻转加噪，x_t 仍是 {0,1}
          3. pred_logits = GNN(coords, x_t, t/T)
          4. loss = BCE(pred_logits, x_0)   ← 分类损失，预测每条边是否存在
        """
        B = adj_0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=adj_0.device)   # (B,) int

        x_t = self.scheduler.q_sample(adj_0, t)                   # (B,N,N) {0,1}

        # GNN 接收归一化时间步 t/T ∈ (0,1]
        t_norm = t.float() / self.T
        pred_logits = self.encoder(coords, x_t, t_norm)            # (B,N,N)

        return F.binary_cross_entropy_with_logits(pred_logits, adj_0)

    def _d3pm_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        """
        D3PM 推理：从均匀 Bernoulli(0.5) 出发，逐步去噪。

        每步:
          pred_logits = GNN(coords, x_t, t/T)
          x_{t-1} = posterior_sample(x_t, pred_logits, t)
        最终热力图 = sigmoid(最后一步的 pred_logits)
        """
        B, N, _ = coords.shape
        device = coords.device

        # 起点：均匀随机 {0,1}^(N×N)
        x = torch.bernoulli(torch.full((B, N, N), 0.5, device=device))

        timestep_pairs = self.scheduler.get_inference_timesteps(num_steps=steps)
        assert len(timestep_pairs) > 0, "get_inference_timesteps returned empty list"
        # 初始化为全零 logits（对应 Bernoulli(0.5)）；会被循环中的真实预测覆盖
        pred_logits = torch.zeros(B, N, N, device=device)

        for t_val, t_prev_val in timestep_pairs:
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
            t_norm = torch.full((B,), t_val / self.T, device=device)

            pred_logits = self.encoder(coords, x, t_norm)          # (B,N,N)

            t_prev_tensor = torch.full((B,), t_prev_val, dtype=torch.long, device=device)

            if t_prev_val > 0:
                # 继续去噪：采样 x_{t-1}
                x = self.scheduler.posterior_sample(x, pred_logits, t_tensor)
            # t_prev=0 时不再采样，直接用最终 pred_logits 作为输出

        # 用最后一步预测的 logits 生成热力图（比采样的 {0,1} 矩阵信息更丰富）
        heatmap = torch.sigmoid(pred_logits)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap

    # -------------------------------------------------------------------------
    # Continuous DDPM (Gaussian) 内部实现
    # -------------------------------------------------------------------------

    def _ddpm_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        连续 DDPM 训练损失：MSE on noise prediction (ε-prediction)。

        步骤:
          1. adj_0_scaled = scale_x0(adj_0)   ← {0,1} → {-1,+1}
          2. t ~ randint(1, T)
          3. (x_t, ε) = q_sample(adj_0_scaled, t)
          4. pred_eps = GNN(coords, x_t, t/T)
          5. loss = MSE(pred_eps, ε)
        """
        B = adj_0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=adj_0.device)

        adj_0_scaled = GaussianDiffusion.scale_x0(adj_0)          # {-1,+1}
        x_t, epsilon = self.scheduler.q_sample(adj_0_scaled, t)   # (B,N,N), (B,N,N)

        t_norm = t.float() / self.T
        pred_eps = self.encoder(coords, x_t, t_norm)               # (B,N,N)

        return F.mse_loss(pred_eps, epsilon)

    def _ddpm_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        """
        连续 DDPM 推理：DDIM 确定性采样，从 N(0,I) 逐步去噪。

        每步:
          pred_eps = GNN(coords, x_t, t/T)
          x_{t_prev} = ddim_step(x_t, pred_eps, t, t_prev)
        最终热力图 = x0_hat * 0.5 + 0.5   （反转 {-1,+1} 缩放）
        """
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)                    # 起点

        timestep_pairs = self.scheduler.get_inference_timesteps(num_steps=steps)

        for t_val, t_prev_val in timestep_pairs:
            t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
            t_norm = torch.full((B,), t_val / self.T, device=device)
            t_prev_tensor = torch.full((B,), t_prev_val, dtype=torch.long, device=device)

            pred_eps = self.encoder(coords, x, t_norm)             # (B,N,N)
            x = self.scheduler.ddim_step(x, pred_eps, t_tensor, t_prev_tensor)

        # x 现在是 {-1,+1} 空间的 x̂_0 估计，转换到 [0,1]
        heatmap = GaussianDiffusion.to_heatmap(x)                  # x*0.5+0.5
        heatmap = heatmap.clamp(0.0, 1.0)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap

    # -------------------------------------------------------------------------
    # 可视化辅助（FM 专用）
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def get_intermediate_heatmap(
        self,
        coords: torch.Tensor,   # (B, N, 2)
        target_t: float,        # 在哪个时刻截图，1.0=纯噪声，0.0=最终结果
        total_steps: int = 20,
    ) -> torch.Tensor:          # (B, N, N) in [0,1]
        """
        FM 推理中途截图，用于生成扩散过程 GIF。
        仅在 mode='flow_matching' 下有意义。
        """
        assert self.mode == 'flow_matching', "get_intermediate_heatmap only supports flow_matching"
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)
        dt = 1.0 / total_steps

        for i in range(total_steps):
            t_val = 1.0 - i * dt
            if t_val < target_t:
                break
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.encoder(coords, x, t_tensor)
            x = x - dt * v

        heatmap = torch.sigmoid(x)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap


# 向后兼容别名
TSPFlowMatchingModel = lambda **kw: TSPDiffusionModel(mode='flow_matching', **kw)


# =============================================================================
# 快速单元测试
# =============================================================================

if __name__ == '__main__':
    import time

    B, N = 4, 20
    device = 'cpu'

    coords = torch.rand(B, N, 2)
    adj_0 = torch.zeros(B, N, N)
    for b in range(B):
        perm = torch.randperm(N)
        for k in range(N):
            i, j = perm[k].item(), perm[(k + 1) % N].item()
            adj_0[b, i, j] = 1.0
            adj_0[b, j, i] = 1.0

    modes = [
        ('flow_matching',   'gated_gcn'),
        ('discrete_ddpm',   'gated_gcn'),
        ('continuous_ddpm', 'gated_gcn'),
        ('flow_matching',   'gat'),
        ('flow_matching',   'gcn'),
    ]

    for mode, enc in modes:
        model = TSPDiffusionModel(
            mode=mode, n_layers=2, hidden_dim=64, encoder_type=enc,
            T=100,  # 小 T 加快测试
        )
        n_params = sum(p.numel() for p in model.parameters())

        # --- 训练损失 ---
        t0 = time.time()
        loss = model.compute_loss(coords, adj_0)
        assert loss.item() > 0, "loss should be positive"
        loss.backward()
        t_loss = time.time() - t0

        # --- 推理 ---
        t0 = time.time()
        # 用少步数加速测试
        inf_steps = 5
        heatmap = model.sample(coords, inference_steps=inf_steps)
        t_sample = time.time() - t0

        assert heatmap.shape == (B, N, N), f"heatmap shape mismatch: {heatmap.shape}"
        assert 0.0 <= heatmap.min().item(), f"heatmap min < 0: {heatmap.min()}"
        assert heatmap.max().item() <= 1.0, f"heatmap max > 1: {heatmap.max()}"
        sym_err = (heatmap - heatmap.transpose(-1, -2)).abs().max().item()
        assert sym_err < 1e-5, f"heatmap not symmetric: {sym_err}"

        print(
            f"[{mode:17s} | {enc:10s}] "
            f"params={n_params:,}  loss={loss.item():.4f}  "
            f"t_loss={t_loss*1000:.1f}ms  t_sample={t_sample*1000:.1f}ms  "
            f"heatmap=[{heatmap.min():.3f},{heatmap.max():.3f}]"
        )

    # --- FM get_intermediate_heatmap ---
    fm_model = TSPDiffusionModel(mode='flow_matching', n_layers=2, hidden_dim=64)
    mid = fm_model.get_intermediate_heatmap(coords, target_t=0.5)
    assert mid.shape == (B, N, N)
    print(f"\nget_intermediate_heatmap(target_t=0.5): shape={tuple(mid.shape)} OK")

    print("\nAll TSPDiffusionModel tests passed")
