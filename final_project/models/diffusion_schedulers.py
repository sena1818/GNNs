"""
扩散调度器 — 三种生成框架的统一接口

本文件实现三种调度器，用于三方对比实验：
  1. FlowMatchingScheduler — 连续直线 ODE (Lipman et al., ICLR 2023)
  2. BernoulliDiffusion    — 离散伯努利扩散 / D3PM (DIFUSCO SOTA)
  3. GaussianDiffusion     — 连续高斯 DDPM (DIFUSCO 较弱变体，对照组)

核心研究问题：FM 的直线 ODE 能否弥合连续-离散扩散的性能差距？

三种调度器共享同一 GNN 编码器，仅改变:
  - 加噪方式 (前向过程)
  - 训练目标 (损失函数)
  - 推理方式 (逆向/积分过程)

时间步约定:
  - FM:   t ∈ [0,1]，连续浮点数
  - D3PM: t ∈ {1,...,T}，整数；传入 GNN 时除以 T 归一化到 [0,1]
  - DDPM: 同 D3PM
"""

import torch


# =============================================================================
# 1. Flow Matching（连续直线 ODE）
# =============================================================================

class FlowMatchingScheduler:
    """
    直线插值调度器 — 连续 Flow Matching。

    前向路径: X_t = (1-t)*X_0 + t*epsilon,  t ∈ [0, 1]
    速度目标: v* = epsilon - X_0             (恒定，与 t 无关)
    训练损失: MSE(v_theta(X_t, t), v*)
    推理:     从 X_1 ~ N(0,I) 欧拉积分到 X_0

    训练示例:
        scheduler = FlowMatchingScheduler()
        t = torch.rand(B)
        eps = torch.randn_like(x0)
        x_t = scheduler.interpolate(x0, eps, t)
        v_target = scheduler.get_velocity_target(x0, eps)
        loss = F.mse_loss(model(coords, x_t, t), v_target)

    推理示例:
        x = torch.randn(B, N, N)
        for t_val, dt in InferenceSchedule(steps=20):
            t_tensor = torch.full((B,), t_val, device=x.device)
            v = model(coords, x, t_tensor)
            x = x - dt * v
        heatmap = torch.sigmoid(x)
    """

    def interpolate(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        线性插值: X_t = (1-t)*x0 + t*epsilon

        Args:
            x0:      (B, N, N) 真实邻接矩阵，值域 {0,1}（连续松弛后当浮点处理）
            epsilon: (B, N, N) 标准高斯噪声
            t:       (B,)      时间步 ∈ [0, 1]
        Returns:
            x_t:     (B, N, N)
        """
        t = t.view(-1, 1, 1)
        return (1.0 - t) * x0 + t * epsilon

    def get_velocity_target(self, x0: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        恒定速度场目标: v* = epsilon - x0

        与 t 无关——这是 FM 相比 DDPM 的核心简化。

        Args:
            x0:      (B, N, N)
            epsilon: (B, N, N)
        Returns:
            v_target: (B, N, N)
        """
        return epsilon - x0


class InferenceSchedule:
    """
    FM 推理时间步序列：从 t=1.0 均匀步进到 t≈0.0，用于欧拉积分。

    用法:
        for t_val, dt in InferenceSchedule(inference_steps=20):
            t_tensor = torch.full((B,), t_val, device=device)
            v = model(coords, x, t_tensor)
            x = x - dt * v
    """

    def __init__(self, inference_steps: int = 20):
        self.steps = inference_steps

    def __iter__(self):
        """生成 (t_current, dt) 对，t 从 1.0 递减到约 0.05（最后一步止于 dt 处）。"""
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t_current = 1.0 - i * dt   # 1.0, 0.95, 0.90, ..., 0.05
            yield t_current, dt


# =============================================================================
# 2. 离散伯努利扩散（D3PM — DIFUSCO SOTA）
# =============================================================================

class BernoulliDiffusion:
    """
    离散伯努利扩散调度器，基于 DIFUSCO (NeurIPS 2023) 的 CategoricalDiffusion。

    核心思想：将邻接矩阵的每个 bit 以累积概率随机翻转（0↔1），
    中间状态 X_t 始终是合法的 {0,1} 二值矩阵，GNN 在每一步都能做有意义的消息传递。

    前向过程（比特翻转，"absorbing to uniform"公式）:
        q(x_t=1 | x_0) = alpha_bar_t * x_0 + (1 - alpha_bar_t) * 0.5
        其中 alpha_bar_t = Π_{s=1}^{t} (1 - beta_s)
        → t=0: x_t = x_0 (无噪声)
        → t=T: x_t ≈ Bernoulli(0.5) (完全随机)

    逆向过程（D3PM 后验，x_0-prediction 参数化）:
        模型预测 pred_logits（每个 bit 属于 1 的 logit）
        后验 q(x_{t-1} | x_t, x_0_hat) 用 Bayes 公式计算

    训练损失: BCE(pred_logits, x_0)    ← 分类问题，每条边是/否

    参考: DIFUSCO CategoricalDiffusion + D3PM (Austin et al., 2021)
    """

    def __init__(self, T: int = 1000, beta_schedule: str = 'linear'):
        self.T = T

        # Beta schedule（与 DIFUSCO 一致）
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, T)   # β_1, ..., β_T
        else:
            raise ValueError(f"Unsupported schedule: {beta_schedule}")

        alphas = 1.0 - betas                             # α_t = 1 - β_t
        alpha_bar = torch.cumprod(alphas, dim=0)         # ᾱ_t = Π_{s=1}^{t} α_s

        # 存为实例变量（非 nn.Module，不注册 buffer，需要手动搬到 device）
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar                       # shape: (T,)

    def _to(self, device: torch.device):
        """将预计算张量移动到指定设备（懒加载，首次使用时调用）。"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

    # ------------------------------------------------------------------
    # 前向过程：训练时用
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向加噪：从 q(x_t | x_0) 采样。

        公式: p(x_t=1 | x_0) = ᾱ_t * x_0 + (1-ᾱ_t) * 0.5

        Args:
            x0: (B, N, N) 真实邻接矩阵，浮点数 {0.0, 1.0}
            t:  (B,) 整数时间步，范围 [1, T]
        Returns:
            x_t: (B, N, N) 加噪后的二值矩阵，仍为 {0.0, 1.0}
        """
        if self.alpha_bar.device != x0.device:
            self._to(x0.device)

        abar = self.alpha_bar[t - 1].view(-1, 1, 1)     # ᾱ_t, shape (B,1,1)
        probs = x0 * abar + (1.0 - abar) * 0.5          # p(x_t=1 | x_0)
        return torch.bernoulli(probs)                    # {0,1} 采样

    # ------------------------------------------------------------------
    # 逆向过程：推理时用
    # ------------------------------------------------------------------

    def posterior_sample(
        self,
        x_t: torch.Tensor,           # (B, N, N) 当前二值状态 {0,1}
        pred_x0_logits: torch.Tensor, # (B, N, N) GNN 输出 logits（未经 sigmoid）
        t: torch.Tensor,             # (B,) 当前时间步，整数 [1, T]
    ) -> torch.Tensor:
        """
        从 p_theta(x_{t-1} | x_t) 采样（D3PM 后验，x_0-prediction 参数化）。

        推导：
            q(x_{t-1}=1 | x_t, x_0) ∝ q(x_t | x_{t-1}=1) * q(x_{t-1}=1 | x_0)

        其中（absorbing uniform 转移）：
            q(x_t | x_{t-1}=w) = α_t * I(x_t=w) + (1-α_t)/2
            → q(x_t | x_{t-1}=1) = (1 + α_t*(2x_t-1)) / 2
            → q(x_t | x_{t-1}=0) = (1 - α_t*(2x_t-1)) / 2

        模型给出软预测 x_0_hat = sigmoid(pred_x0_logits)，
        代入 q(x_{t-1}=1|x_0) = ᾱ_{t-1}*x_0_hat + (1-ᾱ_{t-1})*0.5

        Args:
            x_t:            (B, N, N) 当前状态 {0.0, 1.0}
            pred_x0_logits: (B, N, N) 模型预测的 x_0 logits
            t:              (B,) 整数 [1, T]
        Returns:
            x_{t-1}: (B, N, N) 采样的前一时刻状态 {0.0, 1.0}
        """
        if self.alpha_bar.device != x_t.device:
            self._to(x_t.device)

        B = x_t.shape[0]
        device = x_t.device

        alpha_t = self.alphas[t - 1].view(B, 1, 1)      # α_t

        # ᾱ_{t-1}：t=1 时前一步为 t=0，定义 ᾱ_0 = 1.0（无噪声状态）
        t_prev = (t - 1).clamp(min=0)
        abar_tm1 = torch.where(
            t_prev > 0,
            self.alpha_bar[t_prev - 1],
            torch.ones(B, device=device),
        ).view(B, 1, 1)                                  # ᾱ_{t-1}

        # 软 x_0 预测
        x0_hat = torch.sigmoid(pred_x0_logits)           # (B, N, N) ∈ [0, 1]

        # E[q(x_{t-1}=1 | x_0)] 和 E[q(x_{t-1}=0 | x_0)]
        q_tm1_1 = abar_tm1 * x0_hat + (1.0 - abar_tm1) * 0.5   # p(x_{t-1}=1)
        q_tm1_0 = 1.0 - q_tm1_1                                  # p(x_{t-1}=0)

        # q(x_t | x_{t-1}=1) 和 q(x_t | x_{t-1}=0)，利用 {0,1}→{-1,+1} 技巧
        xt_sign = 2.0 * x_t - 1.0                                # {0,1} → {-1,+1}
        step_from_1 = 0.5 * (1.0 + alpha_t * xt_sign)            # q(x_t | x_{t-1}=1)
        step_from_0 = 0.5 * (1.0 - alpha_t * xt_sign)            # q(x_t | x_{t-1}=0)

        # Bayes 分子
        num_1 = step_from_1 * q_tm1_1                  # ∝ p(x_{t-1}=1 | x_t)
        num_0 = step_from_0 * q_tm1_0                  # ∝ p(x_{t-1}=0 | x_t)

        # 归一化后验概率，采样
        post_prob = num_1 / (num_1 + num_0 + 1e-10)
        return torch.bernoulli(post_prob)               # {0.0, 1.0}

    def get_inference_timesteps(self, num_steps: int = 50) -> list:
        """
        生成推理时的时间步列表（从 T 到 1，均匀间隔），共 num_steps 步。
        DIFUSCO 默认用 50 步。

        Returns:
            [(t, t_prev), ...] 从大到小，如 [(1000,980), (980,960), ..., (20,0)]
        """
        stride = max(1, self.T // num_steps)
        timesteps = list(range(self.T, 0, -stride))    # [T, T-stride, ..., stride]
        pairs = []
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            pairs.append((t, t_prev))
        return pairs


# =============================================================================
# 3. 连续高斯扩散（DDPM/DDIM — 对照组）
# =============================================================================

class GaussianDiffusion:
    """
    连续高斯扩散调度器，基于 DIFUSCO (NeurIPS 2023) 的 Gaussian 变体。

    数据预处理（DIFUSCO 约定）：
        adj_0 先从 {0,1} 缩放到 {-1,+1}：adj_0_scaled = adj_0 * 2 - 1
        再加少量抖动：adj_0_scaled *= (1 + 0.05 * torch.rand_like(adj_0_scaled))
        （防止边界处模式坍塌）

    前向过程（标准 DDPM）:
        q(x_t | x_0) = N(sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)
        等价采样: x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε,  ε ~ N(0,I)

    训练目标（ε-prediction）:
        GNN 预测加入的噪声 ε̂
        损失: MSE(ε̂, ε)

    推理（DDIM，确定性，50 步）:
        x_{t_prev} = sqrt(ᾱ_{t_prev}) * x̂_0 + sqrt(1-ᾱ_{t_prev}) * ε̂
        其中 x̂_0 = (x_t - sqrt(1-ᾱ_t)*ε̂) / sqrt(ᾱ_t)

    热力图恢复: heatmap = x̂_0 * 0.5 + 0.5  （反转 {-1,+1} 缩放，不用 sigmoid）

    参考: DIFUSCO pl_tsp_model.py gaussian_training_step / gaussian_denoise_step
    """

    def __init__(self, T: int = 1000, beta_schedule: str = 'linear'):
        self.T = T

        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, T)
        else:
            raise ValueError(f"Unsupported schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)         # ᾱ_t

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()           # sqrt(ᾱ_t)
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()   # sqrt(1-ᾱ_t)

    def _to(self, device: torch.device):
        for attr in ['betas', 'alphas', 'alpha_bar',
                     'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar']:
            setattr(self, attr, getattr(self, attr).to(device))

    # ------------------------------------------------------------------
    # 数据预处理（训练前调用）
    # ------------------------------------------------------------------

    @staticmethod
    def scale_x0(x0: torch.Tensor) -> torch.Tensor:
        """
        {0,1} → {-1,+1} 缩放（与 DIFUSCO 一致）。

        DIFUSCO 原始代码在此处加入随机抖动（* (1 + 0.05*rand)），但这会导致
        验证集 loss 在相同 checkpoint 下不可复现，影响模型选择。此处去掉抖动，
        保持确定性。数值上等价于 DIFUSCO 无抖动模式。

        Args:
            x0: (B, N, N) 原始邻接矩阵 {0,1}
        Returns:
            x0_scaled: (B, N, N) 缩放后，值域 {-1, +1}
        """
        return x0 * 2.0 - 1.0

    # ------------------------------------------------------------------
    # 前向过程：训练时用
    # ------------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        """
        前向加噪：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε

        注意：调用前需先对 x0 做 scale_x0() 缩放！

        Args:
            x0: (B, N, N) 已缩放到 {-1,+1} 的邻接矩阵
            t:  (B,) 整数时间步 [1, T]
        Returns:
            x_t:     (B, N, N) 带噪声的连续状态
            epsilon: (B, N, N) 采样的噪声（训练目标）
        """
        if self.sqrt_alpha_bar.device != x0.device:
            self._to(x0.device)

        sqrt_abar = self.sqrt_alpha_bar[t - 1].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t - 1].view(-1, 1, 1)

        epsilon = torch.randn_like(x0)
        x_t = sqrt_abar * x0 + sqrt_one_minus * epsilon
        return x_t, epsilon

    # ------------------------------------------------------------------
    # 逆向过程：推理时用（DDIM 确定性采样）
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        x_t: torch.Tensor,      # (B, N, N) 当前状态
        pred_eps: torch.Tensor,  # (B, N, N) GNN 预测的噪声 ε̂
        t: torch.Tensor,         # (B,) 当前时间步，整数 [1, T]
        t_prev: torch.Tensor,    # (B,) 目标时间步，整数 [0, T-stride]
    ) -> torch.Tensor:
        """
        DDIM 确定性采样步（ε-parameterized）。

        公式:
            x̂_0 = (x_t - sqrt(1-ᾱ_t)*ε̂) / sqrt(ᾱ_t)    [重建 x_0]
            x_{t_prev} = sqrt(ᾱ_{t_prev})*x̂_0 + sqrt(1-ᾱ_{t_prev})*ε̂

        当 t_prev=0: ᾱ_0 = 1.0，退化为 x_prev = x̂_0（直接输出干净估计）。

        Args:
            x_t:      (B, N, N)
            pred_eps: (B, N, N)
            t:        (B,) 整数 [1, T]
            t_prev:   (B,) 整数 [0, T)
        Returns:
            x_{t_prev}: (B, N, N)
        """
        if self.sqrt_alpha_bar.device != x_t.device:
            self._to(x_t.device)

        B = x_t.shape[0]
        device = x_t.device

        sqrt_abar_t = self.sqrt_alpha_bar[t - 1].view(B, 1, 1)
        sqrt_one_minus_t = self.sqrt_one_minus_alpha_bar[t - 1].view(B, 1, 1)

        # x̂_0 重建，clamp 防止数值溢出
        x0_hat = (x_t - sqrt_one_minus_t * pred_eps) / sqrt_abar_t.clamp(min=1e-8)
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        # ᾱ_{t_prev}：当 t_prev=0 定义 ᾱ_0=1.0 → sqrt=1, sqrt(1-ᾱ)=0 → x_prev=x̂_0
        abar_tm1 = torch.where(
            t_prev > 0,
            self.alpha_bar[t_prev - 1],
            torch.ones(B, device=device),
        ).view(B, 1, 1)

        sqrt_abar_tm1 = abar_tm1.sqrt()
        sqrt_one_minus_tm1 = (1.0 - abar_tm1).sqrt()

        return sqrt_abar_tm1 * x0_hat + sqrt_one_minus_tm1 * pred_eps

    def get_inference_timesteps(self, num_steps: int = 50) -> list:
        """
        生成推理时间步对列表（DDIM 跳步采样），共 num_steps 步。

        Returns:
            [(t, t_prev), ...] 如 [(1000,980), (980,960), ..., (20,0)]
        """
        stride = max(1, self.T // num_steps)
        timesteps = list(range(self.T, 0, -stride))
        pairs = []
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            pairs.append((t, t_prev))
        return pairs

    @staticmethod
    def to_heatmap(x0_hat: torch.Tensor) -> torch.Tensor:
        """
        将 DDPM 输出的 {-1,+1} 空间反转到 [0,1] 热力图。
        heatmap = x̂_0 * 0.5 + 0.5
        （不用 sigmoid，直接线性映射，与 DIFUSCO 一致）
        """
        return x0_hat * 0.5 + 0.5


# =============================================================================
# 快速单元测试
# =============================================================================

if __name__ == '__main__':
    import torch

    B, N = 4, 20
    device = 'cpu'

    # 构造合法邻接矩阵
    adj_0 = torch.zeros(B, N, N)
    for b in range(B):
        perm = torch.randperm(N)
        for k in range(N):
            i, j = perm[k].item(), perm[(k + 1) % N].item()
            adj_0[b, i, j] = 1.0
            adj_0[b, j, i] = 1.0

    # --- 1. Flow Matching ---
    print('=== FlowMatchingScheduler ===')
    fm = FlowMatchingScheduler()
    t = torch.rand(B)
    eps = torch.randn_like(adj_0)
    x_t = fm.interpolate(adj_0, eps, t)
    v_target = fm.get_velocity_target(adj_0, eps)
    assert x_t.shape == (B, N, N)
    assert v_target.shape == (B, N, N)
    print(f'  x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]  (expected roughly between adj_0 and eps)')
    print(f'  v_target = eps - adj_0  check: {(v_target - (eps - adj_0)).abs().max():.2e}')

    print('=== InferenceSchedule ===')
    steps = list(InferenceSchedule(inference_steps=20))
    assert len(steps) == 20
    ts, dts = zip(*steps)
    assert abs(ts[0] - 1.0) < 1e-6, f'first t should be 1.0, got {ts[0]}'
    assert abs(dts[0] - 0.05) < 1e-6, f'dt should be 0.05, got {dts[0]}'
    print(f'  steps={len(steps)}, t[0]={ts[0]:.2f}, t[-1]={ts[-1]:.2f}, dt={dts[0]:.2f}')

    # --- 2. Bernoulli Diffusion ---
    print('=== BernoulliDiffusion ===')
    bd = BernoulliDiffusion(T=1000)

    # 前向：检查二值性
    t_int = torch.randint(1, 1001, (B,))
    x_t_b = bd.q_sample(adj_0, t_int)
    assert set(x_t_b.unique().tolist()).issubset({0.0, 1.0}), 'x_t should be binary'
    print(f'  q_sample: shape={tuple(x_t_b.shape)}, unique values={x_t_b.unique().tolist()} OK')

    # 早期时间步：应接近 x_0
    t_early = torch.ones(B, dtype=torch.long) * 10
    x_t_early = bd.q_sample(adj_0, t_early)
    agreement = (x_t_early == adj_0).float().mean()
    print(f'  t=10 agreement with x_0: {agreement:.3f} (expect > 0.95)')

    # 晚期时间步：应接近 Bernoulli(0.5)
    t_late = torch.ones(B, dtype=torch.long) * 1000
    x_t_late = bd.q_sample(adj_0, t_late)
    mean_late = x_t_late.mean()
    print(f'  t=1000 mean: {mean_late:.3f} (expect ~0.5)')

    # 逆向：检查输出仍是二值
    fake_logits = torch.randn(B, N, N)
    x_tm1 = bd.posterior_sample(x_t_b, fake_logits, t_int)
    assert set(x_tm1.unique().tolist()).issubset({0.0, 1.0}), 'x_{t-1} should be binary'
    print(f'  posterior_sample: shape={tuple(x_tm1.shape)}, binary OK')

    # 时间步对
    pairs = bd.get_inference_timesteps(num_steps=50)
    assert pairs[0] == (1000, 980), f'first pair: {pairs[0]}'
    assert pairs[-1][1] == 0, f'last t_prev should be 0'
    print(f'  inference pairs: {pairs[0]} ... {pairs[-1]}, total={len(pairs)}')

    # --- 3. Gaussian Diffusion ---
    print('=== GaussianDiffusion ===')
    gd = GaussianDiffusion(T=1000)

    # 数据预处理
    adj_scaled = GaussianDiffusion.scale_x0(adj_0)
    assert adj_scaled.min() >= -1.1 and adj_scaled.max() <= 1.1
    print(f'  scale_x0 range: [{adj_scaled.min():.3f}, {adj_scaled.max():.3f}] (expect ~[-1.05, 1.05])')

    # 前向
    x_t_g, eps_g = gd.q_sample(adj_scaled, t_int)
    assert x_t_g.shape == (B, N, N)
    print(f'  q_sample: shape={tuple(x_t_g.shape)}, range=[{x_t_g.min():.2f}, {x_t_g.max():.2f}]')

    # DDIM 步
    t_curr = torch.ones(B, dtype=torch.long) * 500
    t_prev_val = torch.ones(B, dtype=torch.long) * 480
    fake_eps = torch.randn(B, N, N)
    x_prev = gd.ddim_step(x_t_g, fake_eps, t_curr, t_prev_val)
    assert x_prev.shape == (B, N, N)
    print(f'  ddim_step: shape={tuple(x_prev.shape)} OK')

    # t_prev=0 边界
    t_zero = torch.zeros(B, dtype=torch.long)
    x_final = gd.ddim_step(x_t_g, fake_eps, t_curr, t_zero)
    print(f'  ddim_step(t_prev=0): shape={tuple(x_final.shape)} OK')

    # 热力图转换
    heatmap = GaussianDiffusion.to_heatmap(x_final)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.1  # ~[0,1]
    print(f'  to_heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]')

    print('\nAll schedulers OK')
