"""
扩散调度器 — 三种生成框架的统一接口 (对齐 DIFUSCO + FM 扩展)

本文件实现三种调度器:
  1. FlowMatchingScheduler — 连续直线 ODE [IMPROVEMENT: 本项目扩展]
  2. CategoricalDiffusion  — D3PM 离散扩散 (与 DIFUSCO 官方对齐)
  3. GaussianDiffusion     — 连续高斯 DDPM  (与 DIFUSCO 官方对齐)

与 DIFUSCO 官方的对齐项:
  ✅ CategoricalDiffusion: Q_bar 2×2 转移矩阵 (不再是简化版 Bernoulli)
  ✅ 支持 cosine 和 linear beta schedule
  ✅ InferenceSchedule: 支持 cosine 和 linear spacing
  ✅ Jitter: 5% 随机扰动 (train_preprocess 中加入)

参考:
  - DIFUSCO difusco/utils/diffusion_schedulers.py
  - D3PM (Austin et al., NeurIPS 2021)
  - Flow Matching (Lipman et al., ICLR 2023)
"""

import math
import numpy as np
import torch


# =============================================================================
# 1. Flow Matching（连续直线 ODE）
# [IMPROVEMENT] 本项目扩展，DIFUSCO 官方没有此调度器
# =============================================================================

class FlowMatchingScheduler:
    """
    直线插值调度器 — 连续 Flow Matching。

    前向路径: X_t = (1-t)*X_0 + t*epsilon,  t ∈ [0, 1]
    速度目标: v* = epsilon - X_0             (恒定，与 t 无关)
    训练损失: MSE(v_theta(X_t, t), v*)
    推理:     从 X_1 ~ N(0,I) 欧拉积分到 X_0

    核心优势:
      - 单方向 ODE 路径，无需复杂后验
      - 推理只需 ~20 步 Euler vs DDPM/D3PM 的 50 步
      - 速度目标是常数场，训练更稳定
    """

    def interpolate(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """线性插值: X_t = (1-t)*x0 + t*epsilon"""
        t = t.view(-1, 1, 1)
        return (1.0 - t) * x0 + t * epsilon

    def get_velocity_target(self, x0: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """恒定速度场目标: v* = epsilon - x0"""
        return epsilon - x0


class FMInferenceSchedule:
    """
    FM 推理时间步: 从 t=1.0 均匀步进到 t≈0.0。

    用法:
        for t_val, dt in FMInferenceSchedule(steps=20):
            v = model(coords, x, t_tensor)
            x = x - dt * v
    """
    def __init__(self, inference_steps: int = 20):
        self.steps = inference_steps

    def __iter__(self):
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t_current = 1.0 - i * dt
            yield t_current, dt


# =============================================================================
# 2. Categorical Diffusion (D3PM) — 与 DIFUSCO 官方对齐
# =============================================================================

class CategoricalDiffusion:
    """
    D3PM 离散扩散 — 与 DIFUSCO 官方 CategoricalDiffusion 完全对齐。

    使用 2×2 转移矩阵 Q_bar 处理二值邻接矩阵:
      Q_t = (1-β_t)*I + (β_t/2)*ones   (均匀翻转)
      Q̄_t = Q_1 @ Q_2 @ ... @ Q_t      (累积转移)

    前向: q(x_t | x_0) = x_0_onehot @ Q̄_t，然后 Bernoulli 采样
    后验: q(x_{t-1} | x_t, x_0_hat) 用 Bayes + Q̄ 矩阵计算

    与我们之前简化版 BernoulliDiffusion 的区别:
      - 之前: 用标量 alpha_bar 公式简化后验 (数学等价但实现不同)
      - 现在: 用完整 Q_bar 矩阵 (与 DIFUSCO 官方代码行为完全一致)

    支持 beta schedule: 'linear' 和 'cosine'
    """

    def __init__(self, T: int = 1000, schedule: str = 'linear'):
        self.T = T

        # Beta schedule
        if schedule == 'linear':
            b0, bT = 1e-4, 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self._cos_noise(np.arange(0, T + 1, 1)) / self._cos_noise(0)
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

        # Q_bar: (T+1, 2, 2) 累积转移矩阵
        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))

        self.Qs = (1 - beta) * eye + (beta / 2) * ones    # (T, 2, 2) 单步转移

        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
        self.Q_bar = np.stack(Q_bar, axis=0)                # (T+1, 2, 2)

    def _cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def sample(self, x0_onehot: torch.Tensor, t: np.ndarray) -> torch.Tensor:
        """
        前向加噪: q(x_t | x_0)。
        与 DIFUSCO 官方 CategoricalDiffusion.sample() 完全一致。

        Args:
            x0_onehot: (B, N, N, 2) one-hot 编码的邻接矩阵
            t:         (B,) numpy int array, 范围 [1, T]
        Returns:
            x_t: (B, N, N) Bernoulli 采样结果 {0, 1}
        """
        Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
        # Q_bar: (B, 2, 2) → (B, 1, 2, 2) for broadcasting with (B, N, N, 2)
        xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
        return torch.bernoulli(xt[..., 1].clamp(0, 1))


class InferenceSchedule:
    """
    DDPM/D3PM 推理时间步调度 — 与 DIFUSCO 官方完全一致。

    支持 'linear' 和 'cosine' spacing:
      - linear: 均匀间隔从 T 到 1
      - cosine: sin 曲线间隔，初期慢，后期快
    """
    def __init__(self, inference_schedule="linear", T=1000, inference_T=50):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)
            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2

        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T
            )
            t1 = np.clip(t1, 1, self.T)
            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T
            )
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2

        else:
            raise ValueError(f"Unknown inference schedule: {self.inference_schedule}")


# =============================================================================
# 3. Gaussian Diffusion (DDPM) — 与 DIFUSCO 官方对齐
# =============================================================================

class GaussianDiffusion:
    """
    连续高斯扩散 — 与 DIFUSCO 官方 GaussianDiffusion 完全对齐。

    前向: q(x_t | x_0) = N(√ᾱ_t * x_0, (1-ᾱ_t) * I)
    训练: ε-prediction, MSE loss
    推理: DDIM 确定性采样

    支持 beta schedule: 'linear' 和 'cosine'

    注意: alphabar / sqrt_alphabar / sqrt_one_minus_alphabar 同时提供
          numpy 版本 (训练兼容) 和 torch 版本 (推理加速)。
          调用 to(device) 将预计算 tensor 移至 GPU，消除推理循环中的
          numpy↔torch 转换开销。
    """

    def __init__(self, T: int = 1000, schedule: str = 'linear'):
        self.T = T

        if schedule == 'linear':
            b0, bT = 1e-4, 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self._cos_noise(np.arange(0, T + 1, 1)) / self._cos_noise(0)
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")

        self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
        self.alphabar = np.cumprod(self.alpha)

        # 预计算 torch tensor 版本 (推理时用，避免循环内 numpy 转换)
        self.alphabar_torch = torch.from_numpy(self.alphabar).float()
        self.sqrt_alphabar_torch = torch.sqrt(self.alphabar_torch)
        self.sqrt_one_minus_alphabar_torch = torch.sqrt(1.0 - self.alphabar_torch)

        # DDPM 随机后验额外需要的 tensor (与 DIFUSCO 官方对齐)
        self.alpha_torch = torch.from_numpy(self.alpha).float()
        # beta 前补 0 使索引对齐: beta_torch[t] 对应第 t 步
        self.beta_torch = torch.from_numpy(
            np.concatenate([[0.0], self.beta])
        ).float()

    def _cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

    def to(self, device):
        """将预计算的推理 tensor 移至指定设备 (在 model.to(device) 后调用)。"""
        self.alphabar_torch = self.alphabar_torch.to(device)
        self.sqrt_alphabar_torch = self.sqrt_alphabar_torch.to(device)
        self.sqrt_one_minus_alphabar_torch = self.sqrt_one_minus_alphabar_torch.to(device)
        self.alpha_torch = self.alpha_torch.to(device)
        self.beta_torch = self.beta_torch.to(device)
        return self

    def sample(self, x0: torch.Tensor, t: np.ndarray):
        """
        前向加噪 — 与 DIFUSCO 官方完全一致。

        Args:
            x0: (B, N, N) 已经过预处理的邻接矩阵 (值域 {-1,+1} + jitter)
            t:  (B,) numpy int array, 范围 [1, T]
        Returns:
            x_t:     (B, N, N) 带噪声状态
            epsilon: (B, N, N) 采样的噪声
        """
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
        atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
        return xt, epsilon


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    B, N = 4, 20

    # FM
    print('=== FlowMatchingScheduler ===')
    fm = FlowMatchingScheduler()
    adj_0 = torch.zeros(B, N, N)
    t = torch.rand(B)
    eps = torch.randn_like(adj_0)
    x_t = fm.interpolate(adj_0, eps, t)
    v = fm.get_velocity_target(adj_0, eps)
    print(f'  x_t: {tuple(x_t.shape)}, v: {tuple(v.shape)} OK')

    # Categorical
    print('=== CategoricalDiffusion ===')
    for sched in ['linear', 'cosine']:
        cd = CategoricalDiffusion(T=1000, schedule=sched)
        import torch.nn.functional as F_
        adj_oh = F_.one_hot(adj_0.long(), num_classes=2).float()
        t_np = np.random.randint(1, 1001, B).astype(int)
        x_t = cd.sample(adj_oh, t_np)
        assert x_t.shape == (B, N, N)
        assert set(x_t.unique().tolist()).issubset({0.0, 1.0})
        print(f'  [{sched:6s}] sample OK, Q_bar shape: {cd.Q_bar.shape}')

    # InferenceSchedule
    print('=== InferenceSchedule ===')
    for sched in ['linear', 'cosine']:
        iss = InferenceSchedule(inference_schedule=sched, T=1000, inference_T=50)
        pairs = [iss(i) for i in range(50)]
        print(f'  [{sched:6s}] first={pairs[0]}, last={pairs[-1]}')

    # Gaussian
    print('=== GaussianDiffusion ===')
    for sched in ['linear', 'cosine']:
        gd = GaussianDiffusion(T=1000, schedule=sched)
        adj_scaled = adj_0 * 2 - 1
        t_np = np.random.randint(1, 1001, B).astype(int)
        x_t, eps = gd.sample(adj_scaled, t_np)
        print(f'  [{sched:6s}] sample OK, shape: {tuple(x_t.shape)}')

    print('\nAll schedulers OK')
