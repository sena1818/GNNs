"""
TSP 扩散模型 — 三种生成框架的统一入口 (对齐 DIFUSCO + FM 扩展)

支持三种 mode，共享同一 GNNEncoder:
  'flow_matching'   — 连续直线 ODE [IMPROVEMENT: FM 扩展]
  'discrete_ddpm'   — D3PM 离散扩散 (与 DIFUSCO 对齐)
  'continuous_ddpm' — DDPM 连续扩散 (与 DIFUSCO 对齐)

与 DIFUSCO 官方的对齐项:
  ✅ categorical: one_hot → Q_bar 前向 + CrossEntropyLoss (2类输出)
  ✅ categorical: xt*2-1 + 5% jitter 作为 GNN 输入
  ✅ categorical: Q_bar 后验采样 (完整 Bayes 公式)
  ✅ gaussian: adj_0*2-1 + 5% jitter 预处理
  ✅ gaussian: ε-prediction + MSE loss
  ✅ gaussian: DDIM 确定性采样
  ✅ 推理时间步: cosine InferenceSchedule (DIFUSCO 默认)
  ✅ out_channels: 2 (categorical) / 1 (gaussian, FM)

参考:
  - DIFUSCO difusco/pl_tsp_model.py
  - DIFUSCO difusco/pl_meta_model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import GNNEncoder
from .diffusion_schedulers import (
    FlowMatchingScheduler, FMInferenceSchedule,
    CategoricalDiffusion, InferenceSchedule,
    GaussianDiffusion,
)


class TSPDiffusionModel(nn.Module):
    """
    三方对比统一模型。mode 决定生成框架，其余架构完全相同。

    Args:
        mode:             'flow_matching' | 'discrete_ddpm' | 'continuous_ddpm'
        n_layers:         GNN 层数 (DIFUSCO 默认 12)
        hidden_dim:       隐藏维度 (DIFUSCO 默认 256)
        encoder_type:     'gated_gcn' | 'gat' | 'gcn'
        T:                扩散步数 (D3PM/DDPM, 默认 1000)
        diffusion_schedule: beta schedule ('linear' | 'cosine')
        inference_schedule: 推理时间步分布 ('linear' | 'cosine')
        inference_steps:  推理步数 (FM 默认 20, D3PM/DDPM 默认 50)
    """

    MODES = ('flow_matching', 'discrete_ddpm', 'continuous_ddpm')

    def __init__(
        self,
        mode: str = 'flow_matching',
        n_layers: int = 12,
        hidden_dim: int = 256,
        encoder_type: str = 'gated_gcn',
        T: int = 1000,
        diffusion_schedule: str = 'linear',
        inference_schedule: str = 'cosine',
        inference_steps: int = None,
    ):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}"

        self.mode = mode
        self.T = T
        self.diffusion_schedule = diffusion_schedule
        self.inference_schedule_type = inference_schedule
        self.inference_steps = inference_steps or (20 if mode == 'flow_matching' else 50)

        # out_channels: 2 for categorical (CrossEntropyLoss), 1 for gaussian/FM
        if mode == 'discrete_ddpm':
            out_channels = 2
        else:
            out_channels = 1

        # 共享 GNN 编码器
        self.encoder = GNNEncoder(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            encoder_type=encoder_type,
        )

        # 调度器
        if mode == 'flow_matching':
            self.scheduler = FlowMatchingScheduler()
        elif mode == 'discrete_ddpm':
            self.scheduler = CategoricalDiffusion(T=T, schedule=diffusion_schedule)
        elif mode == 'continuous_ddpm':
            self.scheduler = GaussianDiffusion(T=T, schedule=diffusion_schedule)

    def to(self, *args, **kwargs):
        """Override to() 将非 nn.Module 的 scheduler tensor 也移至目标设备。"""
        result = super().to(*args, **kwargs)
        if hasattr(self.scheduler, 'to'):
            device = next(self.parameters()).device
            self.scheduler.to(device)
        return result

    # =========================================================================
    # 公共接口
    # =========================================================================

    def compute_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        计算训练损失。

        Args:
            coords: (B, N, 2) 城市坐标
            adj_0:  (B, N, N) 真实邻接矩阵 {0,1}
        Returns:
            loss: scalar
        """
        if self.mode == 'flow_matching':
            return self._fm_loss(coords, adj_0)
        elif self.mode == 'discrete_ddpm':
            return self._categorical_loss(coords, adj_0)
        elif self.mode == 'continuous_ddpm':
            return self._gaussian_loss(coords, adj_0)

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        inference_steps: int = None,
    ) -> torch.Tensor:
        """
        推理: 生成边概率热力图。

        Returns:
            heatmap: (B, N, N) in [0, 1]
        """
        steps = inference_steps or self.inference_steps
        if self.mode == 'flow_matching':
            return self._fm_sample(coords, steps)
        elif self.mode == 'discrete_ddpm':
            return self._categorical_sample(coords, steps)
        elif self.mode == 'continuous_ddpm':
            return self._gaussian_sample(coords, steps)

    # =========================================================================
    # Flow Matching [IMPROVEMENT: 本项目扩展]
    # =========================================================================

    def _fm_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """FM loss: MSE on velocity field."""
        B = adj_0.shape[0]
        t = torch.rand(B, device=adj_0.device)
        epsilon = torch.randn_like(adj_0)

        adj_t = self.scheduler.interpolate(adj_0, epsilon, t)

        # GNN 输出 (B, 1, N, N)，squeeze 到 (B, N, N)
        pred_v = self.encoder(coords, adj_t, t).squeeze(1)

        v_target = self.scheduler.get_velocity_target(adj_0, epsilon)
        return F.mse_loss(pred_v, v_target)

    def _fm_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        """FM 推理: Euler ODE 从 t=1 积分到 t=0。"""
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)

        for t_val, dt in FMInferenceSchedule(steps):
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.encoder(coords, x, t_tensor).squeeze(1)
            x = x - dt * v

        heatmap = torch.sigmoid(x)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap

    # =========================================================================
    # Categorical / D3PM (与 DIFUSCO 官方对齐)
    # =========================================================================

    def _categorical_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        D3PM 训练损失 — 与 DIFUSCO pl_tsp_model.py categorical_training_step 对齐。

        步骤:
          1. adj_0 → one_hot (B, N, N, 2)
          2. t ~ randint(1, T)
          3. x_t = Q_bar 前向采样 → {0, 1}
          4. x_t_input = x_t * 2 - 1 + 5% jitter (DIFUSCO 约定)
          5. pred = GNN(coords, x_t_input, t) → (B, 2, N, N)
          6. loss = CrossEntropyLoss(pred, adj_0.long())
        """
        B = adj_0.shape[0]

        # 时间步
        t = np.random.randint(1, self.T + 1, B).astype(int)

        # 前向加噪: one_hot → Q_bar 采样
        adj_matrix_onehot = F.one_hot(adj_0.long(), num_classes=2).float()  # (B,N,N,2)
        xt = self.scheduler.sample(adj_matrix_onehot, t)                     # (B,N,N) {0,1}

        # GNN 输入预处理 (DIFUSCO 约定): {0,1} → {-1,+1} → +5% jitter
        xt_input = xt * 2 - 1
        xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input))

        # 时间步转 float
        t_tensor = torch.from_numpy(t).float().to(adj_0.device)

        # GNN 预测: (B, 2, N, N) — 两类 logits
        x0_pred = self.encoder(coords.float(), xt_input.float(), t_tensor)

        # CrossEntropyLoss (DIFUSCO 官方用法)
        # x0_pred: (B, 2, N, N), target: (B, N, N) long
        loss = F.cross_entropy(x0_pred, adj_0.long())
        return loss

    def _categorical_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        """
        D3PM 推理 — 与 DIFUSCO pl_tsp_model.py test_step categorical 对齐。

        从随机 {0,1} 出发，用 Q_bar 后验逐步去噪。
        """
        B, N, _ = coords.shape
        device = coords.device

        # 初始化: 随机 {0, 1}
        xt = torch.randn(B, N, N, device=device)
        xt = (xt > 0).long()

        schedule = InferenceSchedule(
            inference_schedule=self.inference_schedule_type,
            T=self.T, inference_T=steps,
        )

        for i in range(steps):
            t1, t2 = schedule(i)
            t1_idx = int(t1)
            t2_idx = int(t2)

            # GNN 输入
            xt_input = xt.float() * 2 - 1
            xt_input = xt_input * (1.0 + 0.05 * torch.rand_like(xt_input))

            t_tensor = torch.tensor([t1_idx], dtype=torch.float, device=device)
            x0_pred = self.encoder(coords.float(), xt_input.float(), t_tensor)

            # softmax → x0 概率分布
            x0_pred_prob = x0_pred.permute(0, 2, 3, 1).contiguous().softmax(dim=-1)
            # (B, N, N, 2)

            # 后验采样
            xt = self._categorical_posterior(t2_idx, t1_idx, x0_pred_prob, xt)

        # 最终热力图: 用最后一步的 softmax 概率
        heatmap = x0_pred_prob[..., 1]  # P(edge=1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap.clamp(0, 1)

    def _categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """
        D3PM 后验采样 — 与 DIFUSCO pl_meta_model.py categorical_posterior 完全对齐。

        q(x_{t-1} | x_t, x_0_hat) 的精确 Bayes 公式。

        Args:
            target_t: int, 目标时间步索引
            t:        int, 当前时间步索引
            x0_pred_prob: (B, N, N, 2) softmax 概率
            xt:       (B, N, N) 当前状态 {0, 1}
        """
        diffusion = self.scheduler

        t_idx = int(t)
        tgt_idx = int(target_t) if target_t is not None else t_idx - 1

        device = x0_pred_prob.device

        Q_t = np.linalg.inv(diffusion.Q_bar[tgt_idx]) @ diffusion.Q_bar[t_idx]
        Q_t = torch.from_numpy(Q_t).float().to(device)                    # (2, 2)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t_idx]).float().to(device)   # (2, 2)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[tgt_idx]).float().to(device) # (2, 2)

        xt_onehot = F.one_hot(xt.long(), num_classes=2).float()
        xt_onehot = xt_onehot.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt_onehot, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt_onehot).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt_onehot).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (
            x_t_target_prob_part_1 * x_t_target_prob_part_2_new
        ) / x_t_target_prob_part_3_new

        sum_x_t_target_prob = sum_x_t_target_prob + x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if tgt_idx > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        return xt

    # =========================================================================
    # Gaussian / DDPM (与 DIFUSCO 官方对齐)
    # =========================================================================

    def _gaussian_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
        """
        Gaussian DDPM 训练损失 — 与 DIFUSCO pl_tsp_model.py gaussian_training_step 对齐。

        步骤:
          1. adj_0 → {-1,+1} + 5% jitter (DIFUSCO 约定)
          2. t ~ randint(1, T)
          3. (x_t, ε) = q_sample(adj_0_scaled, t)
          4. pred_eps = GNN(coords, x_t, t) → (B, 1, N, N)
          5. loss = MSE(pred_eps.squeeze(1), ε)
        """
        B = adj_0.shape[0]

        # 预处理: {0,1} → {-1,+1} + 5% jitter
        adj_scaled = adj_0 * 2 - 1
        adj_scaled = adj_scaled * (1.0 + 0.05 * torch.rand_like(adj_scaled))

        t = np.random.randint(1, self.T + 1, B).astype(int)
        xt, epsilon = self.scheduler.sample(adj_scaled, t)

        t_tensor = torch.from_numpy(t).float().to(adj_0.device)
        pred_eps = self.encoder(coords.float(), xt.float(), t_tensor)
        pred_eps = pred_eps.squeeze(1)  # (B, 1, N, N) → (B, N, N)

        return F.mse_loss(pred_eps, epsilon.float())

    def _gaussian_sample(self, coords: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Gaussian DDPM 推理 — 与 DIFUSCO 对齐，支持 DDIM。

        所有调度参数预存于 GPU tensor，循环内无 numpy 转换。
        """
        B, N, _ = coords.shape
        device = coords.device

        xt = torch.randn(B, N, N, device=device)

        schedule = InferenceSchedule(
            inference_schedule=self.inference_schedule_type,
            T=self.T, inference_T=steps,
        )

        # 获取预计算的 GPU tensor
        diffusion = self.scheduler
        sqrt_abar = diffusion.sqrt_alphabar_torch          # (T+1,) on device
        sqrt_one_minus_abar = diffusion.sqrt_one_minus_alphabar_torch  # (T+1,)
        abar = diffusion.alphabar_torch                     # (T+1,)

        for i in range(steps):
            t1, t2 = schedule(i)
            t1_idx = int(t1)
            t2_idx = int(t2)

            t_tensor = torch.tensor([t1_idx], dtype=torch.float, device=device)

            with torch.no_grad():
                pred = self.encoder(coords.float(), xt.float(), t_tensor)
                pred = pred.squeeze(1)  # (B, N, N)

            # DDIM 纯 tensor 后验 (无 numpy 转换)
            xt = self._gaussian_posterior_tensor(
                t2_idx, t1_idx, pred, xt,
                abar, sqrt_abar, sqrt_one_minus_abar,
            )

        # {-1, +1} → [0, 1]
        heatmap = xt.detach() * 0.5 + 0.5
        heatmap = heatmap.clamp(0, 1)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap

    @staticmethod
    def _gaussian_posterior_tensor(
        target_t_idx: int, t_idx: int,
        pred: torch.Tensor, xt: torch.Tensor,
        abar: torch.Tensor,
        sqrt_abar: torch.Tensor,
        sqrt_one_minus_abar: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM 确定性后验 — 纯 tensor 运算，与 DIFUSCO 公式完全等价。

        公式 (DDIM, σ=0):
          x_{t-1} = √(ᾱ_{t-1}/ᾱ_t) · (x_t - √(1-ᾱ_t) · ε_θ)
                  + √(1-ᾱ_{t-1}) · ε_θ

        所有系数直接从预计算 tensor 中用 int 索引取标量，
        无 numpy 转换，无 .item() 调用。
        """
        coeff_xt = (sqrt_abar[target_t_idx] / sqrt_abar[t_idx])          # scalar tensor
        coeff_eps_sub = sqrt_one_minus_abar[t_idx]                        # scalar tensor
        coeff_eps_add = sqrt_one_minus_abar[target_t_idx]                 # scalar tensor

        xt_target = coeff_xt * (xt - coeff_eps_sub * pred) + coeff_eps_add * pred
        return xt_target

    # =========================================================================
    # 可视化辅助 (FM 专用)
    # =========================================================================

    @torch.no_grad()
    def get_intermediate_heatmap(
        self,
        coords: torch.Tensor,
        target_t: float,
        total_steps: int = 20,
    ) -> torch.Tensor:
        """FM 推理中途截图，用于可视化。"""
        assert self.mode == 'flow_matching'
        B, N, _ = coords.shape
        device = coords.device

        x = torch.randn(B, N, N, device=device)
        dt = 1.0 / total_steps

        for i in range(total_steps):
            t_val = 1.0 - i * dt
            if t_val < target_t:
                break
            t_tensor = torch.full((B,), t_val, device=device)
            v = self.encoder(coords, x, t_tensor).squeeze(1)
            x = x - dt * v

        heatmap = torch.sigmoid(x)
        heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2.0
        return heatmap


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    import time as _time

    B, N = 4, 20
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
            T=100,
        )
        n_params = sum(p.numel() for p in model.parameters())

        t0 = _time.time()
        loss = model.compute_loss(coords, adj_0)
        loss.backward()
        t_loss = _time.time() - t0

        t0 = _time.time()
        heatmap = model.sample(coords, inference_steps=5)
        t_sample = _time.time() - t0

        assert heatmap.shape == (B, N, N)
        assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
        sym_err = (heatmap - heatmap.transpose(-1, -2)).abs().max().item()
        assert sym_err < 1e-5

        print(
            f"[{mode:17s} | {enc:10s}] "
            f"params={n_params:,}  loss={loss.item():.4f}  "
            f"t_loss={t_loss*1000:.0f}ms  t_sample={t_sample*1000:.0f}ms"
        )

    print("\nAll TSPDiffusionModel tests passed")
