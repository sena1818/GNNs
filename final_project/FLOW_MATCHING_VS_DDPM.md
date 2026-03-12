# Flow Matching vs DDPM：理论深度融合与实验对比指南

> 针对"用生成模型求解 TSP"项目的技术决策文档
> 核心研究问题：**Flow Matching 的直线 ODE 能否弥合连续扩散与离散扩散之间的性能差距？**

---

## ⚠️ 重要事实修正（v3 更新）

**DIFUSCO 论文的实验结论**（[Sun et al., NeurIPS 2023](https://arxiv.org/abs/2302.08224)）：

> "discrete diffusion consistently outperforms the continuous diffusion models by a large margin"

| 扩散类型 | TSP-50 Optimality Gap | 状态 |
|---------|----------------------|------|
| **离散伯努利扩散 (D3PM)** | ~0.10-0.17% | ← DIFUSCO 最终采用（SOTA） |
| 连续高斯扩散 (DDPM) | ~0.25-0.30% | ← 明显更差 |
| 连续 Flow Matching (Ours) | ??? | ← 本项目的探索目标 |

**本文档的定位**：FM 只能替代连续高斯 DDPM（较弱版本），不能替代离散伯努利扩散（较强版本）。FM 的理论分析仍然正确——它确实比连续 DDPM 有结构性优势——但这是一个**有待验证的研究问题**，而非已确认的结论。

**项目采用三方对比实验设计**：同时实现 离散 DDPM + 连续 DDPM + 连续 FM，系统对比三者在 TSP 上的表现。

---

## 第一部分：三种方法的理论本质

### 1.0 离散伯努利扩散（DIFUSCO 的 SOTA 方法）

DIFUSCO 的最佳结果使用**离散扩散** (D3PM)，在 {0,1} 空间内做噪声注入：

**前向过程（比特翻转）：**

$$q(\mathbf{X}_t[i,j] \mid \mathbf{X}_0[i,j]) = \text{Bernoulli}\!\left((1-\beta_t)\,\mathbf{X}_0[i,j] + \beta_t\,(1-\mathbf{X}_0[i,j])\right)$$

即每个元素以概率 $\beta_t$ 被翻转（0→1 或 1→0），以概率 $(1-\beta_t)$ 保持不变。

**关键特性**：中间状态 $\mathbf{X}_t$ **始终是二值矩阵** {0,1}^(N×N)——它在任何时刻都是一张合法的图，GNN 的消息传递天然适配。

**逆向过程**：GNN 预测每个元素为 1 的概率（logits），然后用 D3PM 的离散后验公式计算 $p(\mathbf{X}_{t-1} \mid \mathbf{X}_t)$。

**损失函数**：BCE（二值交叉熵），预测每个位置的边概率。

**为什么离散版本更好**：TSP 邻接矩阵本身就是 {0,1}，离散扩散的中间状态始终保持二值结构，GNN 在每一步都能有效地进行图上的消息传递。连续版本的中间状态充满 0.37、-0.82 这类值，不构成自然的图结构。

---

### 1.1 连续高斯 DDPM（DIFUSCO 的较弱变体）

DDPM 定义了一个**随机**前向过程，将数据 $\mathbf{X}_0$ 逐步破坏为高斯噪声：

**前向过程（加噪）：**

$$q(\mathbf{X}_t \mid \mathbf{X}_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,\mathbf{X}_0,\; (1-\bar\alpha_t)\,\mathbf{I}\right)$$

其中 $\bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$，$\{\beta_t\}$ 是预先设定的噪声调度（linear/cosine）。

这意味着可以直接采样任意时刻：

$$\mathbf{X}_t = \sqrt{\bar\alpha_t}\,\mathbf{X}_0 + \sqrt{1-\bar\alpha_t}\;\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**逆向过程（去噪）：**

神经网络学习预测噪声 $\boldsymbol{\epsilon}_\theta$（DIFUSCO 连续高斯变体使用 ε-prediction）或预测 $\mathbf{X}_0$（DIFUSCO 离散变体使用 x₀-prediction）：

$$\mathcal{L}_\text{DDPM} = \mathbb{E}_{t,\mathbf{X}_0,\boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t) - \boldsymbol{\epsilon}\right\|^2\right]$$

推理时沿**随机** SDE 逐步去噪，需要 $T = 50 \sim 1000$ 步。

---

### 1.2 Flow Matching 的核心机制

Flow Matching 定义了一个**确定性** ODE 流，在数据分布 $p_0$ 和噪声分布 $p_1 = \mathcal{N}(0,I)$ 之间建立直线路径：

**前向路径（插值）：**

$$\mathbf{X}_t = (1-t)\,\mathbf{X}_0 + t\,\boldsymbol{\epsilon}, \quad t \in [0,1], \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

> **直觉**：矩阵中每一个元素都沿各自维度的一条直线，从数据值（0 或 1）匀速滑向对应的高斯随机数。没有随机扰动，完全确定。

**目标速度场（恒定）：**

$$\mathbf{v}^*(\mathbf{X}_t, t) = \frac{d\mathbf{X}_t}{dt} = \boldsymbol{\epsilon} - \mathbf{X}_0$$

神经网络学习这个速度场：

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\;\mathbf{X}_0,\;\boldsymbol{\epsilon}}\left[\left\|v_\theta(\mathbf{X}_t, t) - (\boldsymbol{\epsilon} - \mathbf{X}_0)\right\|^2\right]$$

推理时积分 ODE（从 $t=1$ 到 $t=0$），仅需 $K = 10 \sim 20$ 步欧拉积分。

---

### 1.3 两者的数学深层联系（核心融合洞见）

这是最关键的部分：**DDPM 和 FM 描述的是同一件事的两种语言**。

#### 联系 1：概率流 ODE（Probability Flow ODE）

Song et al. (2021) 证明：DDPM 的**随机** SDE 对应一个等价的**确定性** ODE，称为概率流 ODE：

$$\frac{d\mathbf{X}}{dt} = \mathbf{f}(\mathbf{X}, t) - \frac{1}{2}g(t)^2\,\nabla_\mathbf{X}\log p_t(\mathbf{X})$$

其中 $\nabla \log p_t$ 是 **得分函数（score function）**，正是 DDPM 学习的核心。

FM 的速度场也是一个确定性 ODE：

$$\frac{d\mathbf{X}}{dt} = v_\theta(\mathbf{X}_t, t)$$

**结论**：FM 的速度场 $v_\theta$ 和 DDPM 的得分函数 $\nabla \log p_t$ 本质上等价——它们都定义了边际分布 $p_t$ 上的一个合法向量场。DDIM（DDPM 的确定性采样器）其实就是 DDPM 在走它自己的概率流 ODE，数学上与 FM 的 ODE 对应。

#### 联系 2：速度场与噪声预测的转换关系

在 FM 框架下，对于条件流（conditioned on a single pair $(X_0, \epsilon)$）：

$$v^*(\mathbf{X}_t, t) = \boldsymbol{\epsilon} - \mathbf{X}_0$$

而已知 $\mathbf{X}_t = (1-t)\mathbf{X}_0 + t\boldsymbol{\epsilon}$，我们可以解出：

$$\mathbf{X}_0 = \frac{\mathbf{X}_t - t\,v^*}{1} = \mathbf{X}_t - t\,v^*$$

$$\boldsymbol{\epsilon} = \mathbf{X}_t + (1-t)\,v^*$$

因此，如果 DDPM 模型预测 $\hat{\mathbf{X}}_0$，可以直接转换为 FM 的速度预测：

$$\hat{v}_\theta = \boldsymbol{\epsilon} - \hat{\mathbf{X}}_0 = \frac{\mathbf{X}_t - \hat{\mathbf{X}}_0}{1-t} \cdot t + \frac{\mathbf{X}_t - \hat{\mathbf{X}}_0}{1} \cdot (1-t-t) \quad \text{(简化：} \hat{v} = \frac{\mathbf{X}_t - \hat{\mathbf{X}}_0}{1-t}\text{)}$$

> **实践意义**：一个训练好的"预测 $\mathbf{X}_0$"的 DDPM 模型，可以零成本地转化为 FM 采样器使用，只需在推理时改变时间步和 ODE 积分方式。

#### 联系 3：参数化选择的等价性

| 预测目标 | DDPM 形式 | FM 形式 | 数学等价 |
|---------|----------|---------|---------|
| 预测噪声 | $\boldsymbol{\epsilon}_\theta$ | $v_\theta = \boldsymbol{\epsilon}_\theta - \mathbf{X}_0$ | ✅ 等价（需知道 $X_0$） |
| 预测数据 | $\hat{\mathbf{X}}_0$ | $v_\theta = \frac{\mathbf{X}_t - \hat{\mathbf{X}}_0}{1-t} \cdot \text{sgn}$ | ✅ 可互转 |
| 预测速度 | 无直接对应 | $v_\theta$ | FM 独有（最简洁） |

---

## 第二部分：FM 相比连续 DDPM 的结构性优势（注意：离散 DDPM 仍是 SOTA）

### 2.1 连续高斯 DDPM 的痛点（对比离散版本更差的原因）

1. **布朗运动破坏稀疏性**：DDPM 的加噪过程通过 SDE 引入随机游走，中间状态 $\mathbf{X}_t$ 的元素值在 $\mathbb{R}$ 上剧烈震荡。对于 TSP 邻接矩阵这种极度稀疏（N 个非零元素 vs $N^2$ 总元素）的结构，随机噪声会彻底淹没稀疏信号。

2. **非线性噪声调度引入误差**：$\bar\alpha_t$ 的 $\sqrt{\cdot}$ 缩放导致 $\mathbf{X}_t$ 的均值在 $t$ 趋近 0 时衰减，造成训练损失在不同时间步之间幅度差异大，需要权重调整（loss weighting）。

3. **离散时间步的 $T=1000$ 问题**：推理需要 50~1000 步，每步调用完整的 GNN 前向传播，对于 $N=100$ 节点的图（GNN 复杂度 $O(N^2)$）代价极高。

### 2.2 FM 相比连续 DDPM 的优化点

> 以下对比的是"连续 DDPM"vs"FM"，**不是**"离散 DDPM"vs"FM"。FM 能否追平离散 DDPM 是本项目的核心研究问题。

| 维度 | 连续 DDPM | Flow Matching | 优化幅度 |
|------|------|--------------|---------|
| **训练损失幅度一致性** | 不同 $t$ 的 loss 幅度差异大（需加权） | 恒定速度场，MSE 幅度一致 | 训练更稳定 |
| **推理步数** | 50~1000 步 | 10~20 步（欧拉） | 5~50× 加速 |
| **中间状态可解释性** | $\mathbf{X}_t$ 是加权高斯混合 | $\mathbf{X}_t$ 是 $\mathbf{X}_0$ 和噪声的线性插值 | 更直观 |
| **超参数复杂度** | 需要 beta schedule（linear/cosine/quadratic） | 无需 schedule | 更简单 |
| **GNN 特征稳定性** | 图结构在中间时刻被随机扰动破坏 | 线性插值保持相对结构稳定 | 更适合图结构 |
| **ODE 积分精度** | 随机，依赖 ancestral sampling | 确定性，可用高阶积分（RK4） | 精度更高 |

### 2.3 深度技术原因：为什么 GNN + FM 是天然组合

在 DDPM 框架中，GNN 在时刻 $t$ 接收的输入 $\mathbf{X}_t$ 满足：

$$\mathbb{E}[\mathbf{X}_t[i,j]] = \sqrt{\bar\alpha_t} \cdot \mathbf{X}_0[i,j]$$

当 $t$ 较大（接近 $T$）时，$\sqrt{\bar\alpha_t} \to 0$，即整个邻接矩阵的均值趋向零，图结构彻底消失。GNN 的消息传递在此时几乎是在全零图上操作，**过平滑问题**（over-smoothing）会被严重放大。

而在 FM 框架中：

$$\mathbb{E}[\mathbf{X}_t[i,j]] = (1-t) \cdot \mathbf{X}_0[i,j]$$

图结构以 $(1-t)$ 的比例线性保留，即使在 $t=0.8$ 时，仍有 20% 的原始图结构信息存在。**GNN 在整个训练过程中都能接触到有意义的图拓扑**，消息传递更有效。

---

## 第三部分：三方对比实验设计

### 3.1 核心对比实验（Three-Way Comparison）

**目标**：在完全相同的 GNN 架构下，对比三种生成框架，回答核心研究问题：**FM 的直线 ODE 能否弥合连续扩散与离散扩散之间的性能差距？**

**实验配置：**

```
共享配置（三组完全相同）：
  模型结构：     GNNEncoder (4层, hidden=128)
  数据集：       TSP-50, 5000 训练 / 1000 测试
  训练 epoch：  50
  批大小：       64
  优化器：       Adam (lr=2e-4)

A 组 — 离散伯努利 DDPM（DIFUSCO SOTA）：
  - 前向：以概率 β_t 翻转 {0,1} 矩阵元素
  - 中间状态：始终是二值矩阵 {0,1}^(N×N)
  - 损失：BCE(pred_logits, X0)
  - 推理：50 步 D3PM 逆向采样

B 组 — 连续高斯 DDPM：
  - 前向：先 {0,1}→{-1,1} 缩放，再 q(X_t|X_0') = N(√ᾱ_t · X_0', (1-ᾱ_t)·I)，T=1000
  - 中间状态：实数矩阵 ℝ^(N×N)
  - GNN 预测：噪声 ε̂（ε-prediction，1 通道输出）
  - 损失：MSE(pred_ε, ε)  [DIFUSCO 原始设置]
  - 推理：50 步 DDIM 采样，最终 heatmap = 0.5*(X̂₀+1)

C 组 — 连续 Flow Matching（本项目核心探索）：
  - 前向：X_t = (1-t)X_0 + t·ε，t ~ U(0,1)
  - 中间状态：实数矩阵 ℝ^(N×N)
  - 损失：MSE(v_θ(X_t,t), ε - X_0)
  - 推理：20 步欧拉积分
```

**记录指标：**

| 指标 | A: 离散 DDPM | B: 连续 DDPM | C: FM | B→C 改进 |
|------|-------------|-------------|-------|---------|
| Best Optimality Gap (TSP-50) | — | — | — | — |
| 训练至 loss 收敛所需 epoch | — | — | — | — |
| 单实例推理时间 (ms) | — | — | — | — |
| 推理步数 | 50 | 50 | 20 | -60% |
| 训练 loss 曲线平滑度 | — | — | — | — |

**预期结论**：
- A（离散）> C（FM）≥ B（连续 DDPM）：FM 优于连续 DDPM，但仍不及离散版本
- 或 C ≈ A：FM 的直线 ODE 成功弥合差距（最理想结果）
- 核心洞见：无论哪种结果都有学术价值——前者解释了"为什么离散扩散更适合图结构"，后者证明了"确定性 ODE 可以替代随机 SDE"

---

### 3.2 FM 推理步数消融实验

**目标**：找到 FM 在"推理质量 vs 推理速度"上的最优 Pareto 点。

```bash
# 同一个训练好的 FM 模型，用不同步数推理
python evaluate.py --model fm --inference_steps 5
python evaluate.py --model fm --inference_steps 10
python evaluate.py --model fm --inference_steps 20
python evaluate.py --model fm --inference_steps 50
python evaluate.py --model ddpm --inference_steps 50   # 对照
```

**预期结果曲线：**

```
Optimality Gap (%)
  2.5|  * (FM-5步)
  2.0|     * (FM-10步)
  1.5|        * (FM-20步)   ≈ * (DDPM-50步)
  1.2|           * (FM-50步)
     +---+---+---+---+--- 推理时间(ms)
        5  10  15  20  25
```

FM-20步 应能达到与 DDPM-50步 相当甚至更好的质量，而推理时间减少 60%。

---

### 3.3 训练收敛速度对比

**目标**：可视化 FM 比 DDPM 收敛更快的现象。

```python
# 每个 epoch 记录 loss，画在同一张图上
epochs = range(1, 51)
plt.plot(epochs, ddpm_losses, label='DDPM (BCE loss, normalized)', color='blue')
plt.plot(epochs, fm_losses,   label='Flow Matching (MSE loss, normalized)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Normalized Training Loss')
plt.title('Convergence Comparison: DDPM vs Flow Matching')
plt.legend()
```

> **注意**：DDPM 和 FM 的损失量纲不同（BCE vs MSE），需要分别对初始 loss 做归一化再比较收敛速度。

---

### 3.4 实验结果汇报格式（报告用三方对比表格）

```
Table 2: Three-Way Comparison on TSP-50

| Method              | Type       | Inference Steps | TSP-50 Gap (%) | Inference Time |
|---------------------|------------|----------------|----------------|----------------|
| Discrete DDPM       | {0,1}      | 50             | X.X ± 0.X      | XXX ms         |
| Continuous DDPM     | ℝ          | 50             | X.X ± 0.X      | XXX ms         |
| Flow Matching       | ℝ (ODE)   | 20             | X.X ± 0.X      | XXX ms         |
| Flow Matching       | ℝ (ODE)   | 50             | X.X ± 0.X      | XXX ms         |

Table 3: FM Inference Steps Ablation

| Method              | Steps | TSP-50 Gap (%) | Inference Time |
|---------------------|-------|----------------|----------------|
| FM (Ours)           | 5     | X.X ± 0.X      | XXX ms         |
| FM (Ours)           | 10    | X.X ± 0.X      | XXX ms         |
| FM (Ours)           | 20    | X.X ± 0.X      | XXX ms         |
| FM (Ours)           | 50    | X.X ± 0.X      | XXX ms         |
| Continuous DDPM     | 50    | X.X ± 0.X      | XXX ms         |  ← 对照
```

---

## 第四部分：具体实现路径

### 4.1 代码改动总览（最小化改动原则）

FM 替换 DDPM 的改动集中在 **3 个函数**，GNN 编码器结构完全不变：

```
models/
  gnn_encoder.py          ← 完全不变（预测速度场 v_θ，而非预测 ε 或 X_0）
  diffusion_schedulers.py ← 核心改动（新增 FlowMatchingScheduler）
  tsp_model.py            ← 改 compute_loss 和 sample 两个函数
train.py                  ← 超参数微调（去掉 T=1000，加 t~U(0,1) 采样）
```

### 4.2 FlowMatchingScheduler 实现

```python
class FlowMatchingScheduler:
    def interpolate(self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x0:      (B, N, N) — 真实邻接矩阵
        epsilon: (B, N, N) — 高斯噪声
        t:       (B,)      — 时间步，在 [0, 1] 之间
        返回:    X_t = (1-t)*x0 + t*epsilon
        """
        # t 需要 reshape 为 (B, 1, 1) 以支持广播
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * epsilon

    def get_velocity_target(self, x0: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        返回恒定速度场目标 v* = epsilon - x0
        注意：与 t 无关，这是 FM 相比 DDPM 的一大简化
        """
        return epsilon - x0


class InferenceSchedule:
    def __init__(self, inference_steps: int = 20):
        self.steps = inference_steps

    def __iter__(self):
        """
        生成 (t_current, dt) 对，从 t=1.0 积分到 t=0.0
        每步: X_{t-dt} = X_t - dt * v_θ(X_t, t)
        """
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t_current = 1.0 - i * dt
            yield t_current, dt
```

### 4.3 TSPFlowMatchingModel 的 compute_loss

```python
def compute_loss(self, coords: torch.Tensor, adj_0: torch.Tensor) -> torch.Tensor:
    """
    coords: (B, N, 2) — 城市坐标
    adj_0:  (B, N, N) — 真实邻接矩阵（0/1，连续松弛后当作实数处理）
    """
    B = adj_0.shape[0]

    # 1. 采样时间步 t ~ Uniform(0, 1)
    t = torch.rand(B, device=adj_0.device)

    # 2. 采样噪声
    epsilon = torch.randn_like(adj_0)

    # 3. 线性插值得到中间状态
    adj_t = self.scheduler.interpolate(adj_0, epsilon, t)

    # 4. GNN 前向预测速度场
    pred_velocity = self.encoder(coords, adj_t, t)   # (B, N, N)

    # 5. 计算 MSE 损失（目标是恒定速度 epsilon - x0）
    velocity_target = self.scheduler.get_velocity_target(adj_0, epsilon)
    loss = F.mse_loss(pred_velocity, velocity_target)

    return loss
```

### 4.4 TSPFlowMatchingModel 的 sample（推理）

```python
@torch.no_grad()
def sample(self, coords: torch.Tensor, inference_steps: int = 20) -> torch.Tensor:
    """
    从纯噪声 X_1 ~ N(0,I) 出发，用欧拉积分还原邻接矩阵热力图。
    coords: (B, N, 2)
    返回:   heatmap (B, N, N) ∈ [0, 1]
    """
    B, N, _ = coords.shape

    # 起点：纯高斯噪声
    x = torch.randn(B, N, N, device=coords.device)

    # 欧拉积分：从 t=1 走到 t=0
    schedule = InferenceSchedule(inference_steps)
    for t_val, dt in schedule:
        t_tensor = torch.full((B,), t_val, device=coords.device)

        # GNN 预测速度场
        v = self.encoder(coords, x, t_tensor)   # (B, N, N)

        # 欧拉步：向 t=0 方向走一步
        x = x - dt * v

    # 映射到 [0, 1] 得到"边存在概率"热力图
    heatmap = torch.sigmoid(x)

    # TSP 是无向图，邻接矩阵对称化
    heatmap = (heatmap + heatmap.transpose(-1, -2)) / 2

    return heatmap
```

### 4.5 GNN Encoder 的适配（几乎不变）

GNN 编码器的输入/输出接口完全相同：

- **输入**：`(coords, adj_t, t)` — 坐标、当前状态矩阵、时间步
- **输出**：`(B, N, N)` — 预测量（DDPM 时预测 $\hat{\mathbf{X}}_0$，FM 时预测速度 $v_\theta$）

**唯一需要注意的**：时间步嵌入的归一化方式。

DDPM：时间步 $t \in \{1, 2, ..., 1000\}$（整数），嵌入时除以 $T$：`t_emb = t / 1000`

FM：时间步 $t \in [0, 1]$（连续浮点数），直接使用：`t_emb = t`

如果复用 DIFUSCO 的 `GNNEncoder`，需要检查时间步嵌入代码，确保 FM 的 $t \in [0,1]$ 不会导致嵌入维度或尺度问题（通常在 `PositionEmbeddingSine` 或 `sinusoidal embedding` 中处理）。

---

## 第五部分：两种方法的深度融合点

### 5.1 融合策略 1：混合训练目标

可以在同一模型中结合两种损失，利用 FM 损失的稳定性和 DDPM 的对角化预测精度：

```python
# FM 损失（主损失）
loss_fm = F.mse_loss(pred_velocity, velocity_target)

# 附加的 X0 预测监督（辅助损失）
# 从预测速度反推 X0：x̂_0 = x_t - (1-t) * pred_v  ← 由插值公式推导
x0_pred = adj_t - (1 - t).view(-1,1,1) * pred_velocity
loss_x0 = F.binary_cross_entropy_with_logits(x0_pred, adj_0)

# 组合
loss = loss_fm + 0.1 * loss_x0
```

这种组合利用了 FM 的稳定训练信号 + BCE 对二值目标的监督精度，是一种有意义的融合实验创新点。

### 5.2 融合策略 2：DDIM 采样作为 FM 的特殊情形

DDIM（Denoising Diffusion Implicit Models）使用确定性采样，其更新公式为：

$$\mathbf{X}_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat{\mathbf{X}}_0(\mathbf{X}_t) + \sqrt{1-\bar\alpha_{t-1}}\,\boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t)$$

当 $\bar\alpha_t = 1 - t^2$（特殊调度）时，此公式退化为 FM 的欧拉步。这说明：**使用线性 FM 训练的模型，在推理时也可以用 DDIM 格式的采样器，两者是对偶的**。

在实验中可以验证：

```python
# 实验验证：同一个 FM 模型，用 Euler ODE vs 用等效 DDIM 格式
gap_euler = evaluate(model, sampler='euler', steps=20)
gap_ddim  = evaluate(model, sampler='ddim_equivalent', steps=20)
# 两者应该非常接近
```

### 5.3 融合策略 3：Optimal Transport Flow Matching（进阶）

标准 FM 随机配对 $(X_0, \epsilon)$，导致路径有"交叉"（不同数据点到同一噪声）。Mini-batch OT（最优传输）在每个 batch 内寻找最优配对，进一步拉直路径：

```
标准 FM 路径：          OT-FM 路径：
X_0 → ε（随机配对）     X_0 → ε（最优传输配对）
路径可能交叉            路径不交叉，更短
推理更容易              推理步数可进一步减少至 5 步
```

对于 TSP 项目，如果效果好可以作为额外创新点，但实现需要 `torch_linear_assignment` 或 `ot` 库。

---

## 第六部分：实验章节写作模板

在报告的 `Section 4: Experiments` 中，对比实验建议这样组织：

```
4.1 Training Convergence (必做)
    图：离散DDPM vs 连续DDPM vs FM 的 training loss 曲线（50 epochs）
    注意：三种方法的损失量纲不同，需分别做归一化再比较收敛速度
    结论：分析三者收敛速度差异

4.2 Three-Way Comparison: Core Results (必做, 本项目核心实验)
    表：离散DDPM / 连续DDPM / FM 的 Gap/时间/成功率
    核心研究问题：FM 能否弥合连续-离散差距？
    结论：定量回答研究问题

4.3 Flow Matching Inference Steps Ablation (必做)
    表：FM 在 K=5/10/20/50 步下的 Gap 和推理时间
    图：Pareto 曲线（横轴=时间，纵轴=Gap）
    结论：K=20 是最优 Pareto 点

4.4 GNN Architecture Ablation（原有实验，不受影响）
    表：Gated GCN / GAT / GCN 在 FM 框架下的对比

4.5 Cross-Scale Generalization（原有实验，不受影响）
    曲线：TSP-20/50/100 的 Gap 趋势
```

---

## 附录：关键参考文献

| 论文 | 核心贡献 | 与本项目关联 |
|------|---------|------------|
| **DDPM** Ho et al. (NeurIPS 2020) | 扩散模型基础 | 对比基线 |
| **Score SDE** Song et al. (ICLR 2021) | 概率流 ODE 理论 | FM/DDPM 统一框架 |
| **DDIM** Song et al. (ICLR 2021) | DDPM 确定性采样 | FM 的等价形式 |
| **Flow Matching** Lipman et al. (ICLR 2023) | 直线 ODE 框架 | 本项目核心替换 |
| **DIFUSCO** Sun et al. (NeurIPS 2023) | TSP + 扩散 | 本项目基础架构 |
| **T2T-CO** Chen et al. (NeurIPS 2024) | Anisotropic GNN | GNN 编码器参考 |

**待读**（推理时如需更多优化）：
- **Rectified Flow** Liu et al. (2022)：进一步直化 FM 路径
- **Consistency Models** Song et al. (2023)：极少步数（1-2步）高质量采样

---

## 第七部分：矩阵层面的逐步拆解——DIFUSCO 的内部机制

> 以 TSP-4（4座城市）为例，把每个矩阵元素在每个时刻的值写出来，彻底弄清楚数字在哪里走。

### 7.1 基础：什么是邻接矩阵 adj_0

4座城市，最优路径为 **0→1→2→3→0**（一个 Hamiltonian 回路），对应的邻接矩阵：

```
adj_0 ∈ {0,1}^(4×4):

         城市0  城市1  城市2  城市3
城市0  [  0     1     0     1  ]   ← 城市0与城市1、城市3相连
城市1  [  1     0     1     0  ]
城市2  [  0     1     0     1  ]
城市3  [  1     0     1     0  ]
```

矩阵特性：
- **对角线全 0**（城市不自连）
- **对称**（无向图）
- **每行恰好 2 个 1**（每个城市恰好连接 2 条边）
- **N 个非零元素，N²个总元素** — 极度稀疏（本例稀疏度 50%，真实 TSP-50 约 4%）

---

### 7.2 DIFUSCO 的前向加噪：每个矩阵元素的轨迹

DIFUSCO 连续变体先将 {0,1} 缩放到 {-1,1}（`adj_0 = adj_0 * 2 - 1`），然后用高斯扩散公式（直接采样任意时刻）：

$$\text{adj}_t = \sqrt{\bar\alpha_t} \cdot \text{adj}_0' + \sqrt{1-\bar\alpha_t} \cdot \boldsymbol{\epsilon}$$

其中 $\text{adj}_0' \in \{-1, +1\}^{N \times N}$ 是缩放后的邻接矩阵，$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ 是高斯噪声。

设线性 schedule 下的 $\sqrt{\bar\alpha_t}$ 在不同时刻的值：

| 时刻 t（DIFUSCO离散整数） | $\sqrt{\bar\alpha_t}$（信号保留比例） | $\sqrt{1-\bar\alpha_t}$（噪声强度） |
|--------------------------|--------------------------------------|-------------------------------------|
| t = 0（干净数据）          | 1.000                                | 0.000                               |
| t = 100                   | 0.866                                | 0.500                               |
| t = 250                   | 0.707                                | 0.707                               |
| t = 500                   | 0.500                                | 0.866                               |
| t = 750                   | 0.250                                | 0.968                               |
| t = 1000（纯噪声）         | 0.000                                | 1.000                               |

**具体数字示例（t = 500，ε 是一个具体采样值）：**

假设噪声矩阵 ε 的前两行如下：
```
ε ≈ [[-0.3,  0.8, -1.2,  0.5],
     [ 0.6, -0.4,  0.9, -0.7], ...]
```

注意：adj_0 已缩放到 {-1,1}，原来的 0→-1，1→+1。

则在 t = 500 时（√ᾱ_500 ≈ 0.5，√(1-ᾱ_500) ≈ 0.866），adj_500 的前两行：
```
adj_0' = [-1, +1, -1, +1]   ← 缩放后：非边=-1, 边=+1
adj_500[0,:] = 0.5 * [-1, +1, -1, +1] + 0.866 * [-0.3, 0.8, -1.2, 0.5]
             = [-0.5, 0.5, -0.5, 0.5] + [-0.26, 0.69, -1.04, 0.43]
             = [-0.76, 1.19, -1.54, 0.93]

adj_0' = [+1, -1, +1, -1]
adj_500[1,:] = 0.5 * [+1, -1, +1, -1] + 0.866 * [0.6, -0.4, 0.9, -0.7]
             = [0.5, -0.5, 0.5, -0.5] + [0.52, -0.35, 0.78, -0.61]
             = [1.02, -0.85, 1.28, -1.11]
```

**关键观察**：
- `adj_500[0,1] = 1.19` → 真实边（adj_0'=+1），值偏正（弱信号保留）
- `adj_500[0,2] = -1.54` → 真实非边（adj_0'=-1），值偏负
- 由于初始值以 0 为中心（{-1,+1}），边和非边的均值差 = 2×√ᾱ_t = 1.0（比 {0,1} 编码的 0.5 差更大）
- 但噪声标准差已达 0.866，**信噪比仍然较低**

**连续版本比离散版本差的核心原因**：即使有 {-1,1} 编码优势，中间状态仍是任意实数，不构成自然的图结构。GNN 的消息传递在实数加权图上的效果不如在二值图上清晰。

---

### 7.3 DIFUSCO 的逆向去噪：DDPM 后验公式

已知 GNN 预测噪声 $\hat{\boldsymbol{\epsilon}}$（ε-prediction），DIFUSCO 用 DDPM 后验计算 $\text{adj}_{t-1}$：

**Step 1**：从预测的噪声 $\hat{\boldsymbol{\epsilon}}$ 推导干净数据估计：
$$\widehat{\text{adj}}_0 = \frac{\text{adj}_t - \sqrt{1-\bar\alpha_t}\,\hat{\boldsymbol{\epsilon}}}{\sqrt{\bar\alpha_t}}$$

**Step 2**：使用 ε-parameterized DDPM 后验直接计算：
$$\text{adj}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(\text{adj}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\hat{\boldsymbol{\epsilon}}\right) + \sigma_t \cdot \boldsymbol{\eta}$$

或用 DDIM 确定性公式（σ=0）：
$$\text{adj}_{t-1} = \sqrt{\frac{\bar\alpha_{t-1}}{\bar\alpha_t}} \left(\text{adj}_t - \sqrt{1-\bar\alpha_t}\,\hat{\boldsymbol{\epsilon}}\right) + \sqrt{1-\bar\alpha_{t-1}}\,\hat{\boldsymbol{\epsilon}}$$

**矩阵层面的含义**：每走一步，矩阵中每个元素都要经过"先用 GNN 预测噪声 ε̂，再用后验公式去噪"的操作。需要 50~1000 次循环。

---

## 第八部分：矩阵层面的逐步拆解——Flow Matching 的具体实现

### 8.1 FM 的前向路径：每个元素的直线运动

同一个 TSP-4 例子，同一组噪声 ε，FM 的插值：

$$\text{adj}_t[i,j] = (1-t) \cdot \text{adj}_0[i,j] + t \cdot \boldsymbol{\epsilon}[i,j]$$

用同一噪声矩阵在不同时刻 t ∈ [0,1] 的元素值：

**元素 (0,1)：真实边，adj_0[0,1] = 1，ε[0,1] = 0.8**

| t | adj_t[0,1] = (1-t)·1 + t·0.8 | 说明 |
|---|-------------------------------|------|
| 0.0 | 1.000 | 纯数据 |
| 0.2 | 0.960 | 96% 信号 |
| 0.5 | 0.900 | 90% 信号 |
| 0.8 | 0.840 | 84% 信号 |
| 1.0 | 0.800 | 纯噪声 |

**元素 (0,2)：真实非边，adj_0[0,2] = 0，ε[0,2] = -1.2**

| t | adj_t[0,2] = (1-t)·0 + t·(-1.2) | 说明 |
|---|----------------------------------|------|
| 0.0 | 0.000 | 纯数据 |
| 0.2 | -0.240 | 20% 噪声 |
| 0.5 | -0.600 | 50% 噪声 |
| 0.8 | -0.960 | 80% 噪声 |
| 1.0 | -1.200 | 纯噪声 |

**关键对比**：在 t=0.5 时，
- 真实边元素：adj_t[0,1] = 0.9（明显正）
- 真实非边元素：adj_t[0,2] = -0.6（明显负）
- **两类元素在 t=0.5 时仍然可分**，而 DIFUSCO 在 t=500 时已接近不可分

这正是 FM 对 GNN 更友好的根本原因：**在整个训练过程中，GNN 接收到的输入矩阵始终保留着可区分的边/非边信号**。

---

### 8.2 FM 的速度场：矩阵中每个元素的"方向"

速度目标 $v^*[i,j] = \boldsymbol{\epsilon}[i,j] - \text{adj}_0[i,j]$（与 t 完全无关）：

```
速度目标矩阵 v*：

         城市0  城市1  城市2  城市3
城市0  [  ε₀₀-0   ε₀₁-1   ε₀₂-0   ε₀₃-1  ]
城市1  [  ε₁₀-1   ε₁₁-0   ε₁₂-1   ε₁₃-0  ]
城市2  [  ε₂₀-0   ε₂₁-1   ε₂₂-0   ε₂₃-1  ]
城市3  [  ε₃₀-1   ε₃₁-0   ε₃₂-1   ε₃₃-0  ]
```

用具体数字（同一组 ε）：

```
v* = ε - adj_0：

  对角元素（adj_0=0）: v* = ε（纯噪声方向，数值随机）
  真实边（adj_0=1）:   v* = ε - 1（指向"比噪声还小1"的方向）
  真实非边（adj_0=0）: v* = ε（指向纯噪声方向）
```

**GNN 需要学到什么**：
- 看到城市 i 和城市 j 的坐标距离近 + 当前 adj_t[i,j] 偏大 → 这可能是真实边 → 预测 v ≈ ε - 1（负的，推着矩阵元素在"去噪"时往1走）
- 看到城市 i 和城市 j 距离远 + adj_t[i,j] 偏小 → 这可能是非边 → 预测 v ≈ ε（中性，推着元素往0走）

---

### 8.3 FM 推理的矩阵演化：逐步展示 20 步欧拉积分

初始状态 $\text{adj}_1 \sim \mathcal{N}(0, I)$，设具体值：

```
adj_1（纯噪声，t=1）：
[[ 0.3, -0.8,  1.2, -0.5],
 [ 0.6,  0.4, -0.9,  0.7],
 [-0.3,  0.8,  0.2, -1.1],
 [ 0.5, -0.7,  0.9,  0.3]]
```

**欧拉步（dt = 1/20 = 0.05）**：

$$\text{adj}_{t-0.05} = \text{adj}_t - 0.05 \times v_\theta(\text{adj}_t,\, t,\, \text{coords})$$

假设 GNN 在 t=1.0 预测出理想速度场（接近 v*），推演前两步：

```
Step 1（t=1.0 → t=0.95）:
  v_θ ≈ v* = ε - adj_0
  adj_0.95 = adj_1 - 0.05 * v*
           ≈ adj_1 - 0.05 * (adj_1 - adj_0)     ← 因为 adj_1 ≈ ε 在 t=1 时
           = 0.95 * adj_1 + 0.05 * adj_0

  adj_0.95[0,1] = 0.95*(-0.8) + 0.05*1 = -0.71
  adj_0.95[0,2] = 0.95*(1.2)  + 0.05*0 =  1.14

Step 2（t=0.95 → t=0.90）:
  adj_0.90[0,1] = 0.95*(-0.71) + 0.05*1 ≈ -0.62
  adj_0.90[0,2] = 0.95*(1.14)  + 0.05*0 ≈  1.08

...（每步，真实边元素朝+1方向收敛，非边元素朝0方向收敛）...

Step 20（t=0.05 → t=0）:
  adj_0[0,1] ≈ 0.95          ← 趋向 1（真实边）
  adj_0[0,2] ≈ 0.06          ← 趋向 0（真实非边）

最终 heatmap = sigmoid(adj_0):
  heatmap[0,1] = sigmoid(0.95) ≈ 0.72   ← 高概率，是边
  heatmap[0,2] = sigmoid(0.06) ≈ 0.51   ← 接近 0.5，不确定（需解码决策）
```

**这就是"从噪声画出路径"的完整矩阵级过程**：20 步欧拉积分，每步 GNN 用城市坐标信息将随机噪声矩阵中的每个元素推向它应该属于的值（接近1的边，接近0的非边）。

---

## 第九部分：三种方法的操作对照表

### 9.1 核心操作三方对比

| 操作 | 离散伯努利 DDPM（SOTA） | 连续高斯 DDPM | Flow Matching |
|------|----------------------|--------------|--------------|
| **数据空间** | {0,1}^(N×N) | ℝ^(N×N) | ℝ^(N×N) |
| **加噪方式** | 以 β_t 概率翻转 0↔1 | `√ᾱ_t · x0 + √(1-ᾱ_t) · ε` | `(1-t) · x0 + t · ε` |
| **中间状态** | 二值矩阵（合法图） | 实数矩阵（非自然图） | 实数矩阵（线性插值） |
| **t=0.5时信号** | ~50% 元素被翻转 | `√ᾱ_500 ≈ 0.16`（16%） | `1-0.5 = 0.5`（50%） |
| **数据预处理** | — | {0,1}→{-1,1} 缩放 | — |
| **GNN 预测目标** | 每元素 P(=1) 的 logit (2ch) | `epsilon_hat`（噪声，1ch） | `v_θ = ε - adj_0`（速度） |
| **训练损失** | CrossEntropyLoss | MSE(ε̂, ε) | MSE（速度回归） |
| **推理出发点** | 均匀随机 {0,1}^(N×N) | `N(0, I)` | `N(0, I)` |
| **推理更新** | D3PM 离散后验公式 | DDPM 三项加权公式 | `adj -= dt · v_θ`（一行） |
| **推理步数** | 50 步 | 50~1000 步 | 10~20 步 |
| **最终输出** | `softmax(logits)[:,1]` | `0.5*(X̂₀+1)` | `sigmoid(adj_0)` |
| **超参数复杂度** | β schedule + T | β schedule + T | 仅推理步数 K |

### 9.2 参数化等价转换（可互相推导）

DIFUSCO 连续变体的 GNN 实际预测噪声 `epsilon_hat`（ε-prediction），可以在推理时等价地转换为 FM 方式。

**Step 1**：从 ε̂ 推导出 x̂₀：

```python
# DIFUSCO 方式：GNN 输出 epsilon_hat
epsilon_hat = gnn(coords, adj_t, t)   # 预测噪声

# 推导 adj_0_hat
adj_0_hat = (adj_t - sqrt(1-abar_t) * epsilon_hat) / sqrt(abar_t)
```

**Step 2**：转换为 FM 速度进行积分：

```python
# 等价转换为 FM 速度（在推理时使用，无需重新训练）：
v_equivalent = (adj_t - adj_0_hat) / t      # 当 t > 0 时有效

# FM 欧拉步
adj_next = adj_t - dt * v_equivalent
```

这意味着：**一个训练好的 DIFUSCO 模型可以"零成本"转化为用 FM-Euler 方式推理**，直接比较两种推理方式的质量，而不需要重新训练——这本身就是一个有说服力的消融实验。

### 9.3 训练目标的等价性证明

DIFUSCO 连续变体采用 ε-prediction 参数化：`epsilon_hat = gnn(adj_t, t, coords)`

DIFUSCO 的 MSE 损失：

$$\mathcal{L}_\text{DIFUSCO} = \mathbb{E}\left[\left\|\hat{\boldsymbol{\epsilon}} - \boldsymbol{\epsilon}\right\|^2\right]$$

而 FM 的 MSE 损失，其中 $v_\theta = (\text{adj}_t - \widehat{\text{adj}}_0) / t$（等价参数化），展开得：

$$\mathcal{L}_\text{FM} = \mathbb{E}\left[\left\|\frac{\text{adj}_t - \widehat{\text{adj}}_0}{t} - (\boldsymbol{\epsilon} - \text{adj}_0)\right\|^2\right]$$

两个损失的**最小值点相同**（都在 `adj_0_hat = adj_0` 时取到），但**训练动态不同**：
- BCE：边/非边的梯度方向明确（推向0或1），但中间时刻 t≈0.5 的样本梯度被标签二值性"拉扯"
- MSE on velocity：梯度均匀（L2 损失），中间时刻的样本梯度幅度稳定，训练曲线更平滑

---

## 第十部分：三种方法的代码结构精确映射

### 10.1 逐函数对照清单

```
离散伯努利 DDPM (A)                    连续高斯 DDPM (B)                     Flow Matching (C)
───────────────────────────────────    ───────────────────────────────────    ──────────────────────────────
BernoulliDiffusion.__init__()          CategoricalDiffusion.__init__()        FlowMatchingScheduler.__init__()
  设置 β_schedule (T步)                 设置 β_schedule, α_bar_t (T步)          无需任何预计算

BernoulliDiffusion.sample(x0, t)       CategoricalDiffusion.sample(x0, t)     FMS.interpolate(x0, ε, t)
  以 β_t 概率翻转 x0 的每个 bit          adj_t = √ᾱ_t*x0 + √(1-ᾱ_t)*ε         adj_t = (1-t)*x0 + t*ε
  结果仍为 {0,1} 矩阵                   结果为 ℝ 矩阵                          结果为 ℝ 矩阵

training_step():                       training_step():                       compute_loss():
  t = randint(1, T)                     t = randint(1, T)                      t = rand(0, 1)
  adj_t = bernoulli_flip(adj_0, t)      adj_0' = adj_0*2-1  # {-1,1}缩放      adj_t = interpolate(adj_0, ε, t)
  pred = gnn(coords, adj_t, t/T)        adj_t = gauss_noise(adj_0', t)         pred_v = gnn(coords, adj_t, t)
  loss = CE(pred, adj_0)                pred_eps = gnn(coords, adj_t, t/T)     loss = MSE(pred_v, ε - adj_0)
                                        loss = MSE(pred_eps, epsilon)

test_step() 推理:                       test_step() 推理:                       sample() 推理:
  adj ~ Bernoulli(0.5)^(N×N)            adj ~ N(0, I)                          adj ~ N(0, I)
  for t in T,...,1:                      for t in T,...,1:                      for t,dt in Schedule(20):
    logits = gnn(adj, t)                  pred_eps = gnn(adj, t)                 v = gnn(adj, t)
    adj = d3pm_posterior(logits)           adj = ddpm_posterior(pred_eps)          adj = adj - dt * v
  heatmap = softmax(logits)[:,1]         heatmap = adj * 0.5 + 0.5            heatmap = sigmoid(adj)
```

### 10.2 完整的 DIFUSCO → FM 改造 diff（核心 3 处修改）

```python
# ========================= 改动 1：FlowMatchingScheduler 替代 CategoricalDiffusion =========================
# BEFORE（DIFUSCO）:
class CategoricalDiffusion:
    def __init__(self, T=1000, beta_schedule='linear'):
        betas = torch.linspace(1e-4, 2e-2, T)         # 1000个 β 值
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t，1000个值
    def sample(self, x0, t):
        abar = self.alpha_bar[t].view(-1,1,1)
        epsilon = torch.randn_like(x0)
        return abar.sqrt() * x0 + (1-abar).sqrt() * epsilon, epsilon

# AFTER（FM）:
class FlowMatchingScheduler:
    def interpolate(self, x0, epsilon, t):
        t = t.view(-1,1,1)
        return (1 - t) * x0 + t * epsilon              # 直线，无需任何预计算

    def get_velocity_target(self, x0, epsilon):
        return epsilon - x0                            # 恒定，与 t 无关


# ========================= 改动 2：compute_loss 中的采样和损失 =========================
# BEFORE（DIFUSCO 连续高斯变体）:
def compute_loss(self, coords, adj_0):
    adj_0_scaled = adj_0 * 2 - 1                       # {0,1}→{-1,1}
    t = torch.randint(1, self.T+1, (B,))               # 整数时间步
    adj_t, epsilon = self.diffusion.sample(adj_0_scaled, t)  # 高斯加噪
    pred_eps = self.gnn(coords, adj_t, t.float()/self.T)    # ε-prediction
    return F.mse_loss(pred_eps, epsilon)                 # MSE on noise

# AFTER（FM）:
def compute_loss(self, coords, adj_0):
    t = torch.rand(B)                                 # 连续均匀时间步
    epsilon = torch.randn_like(adj_0)
    adj_t = self.scheduler.interpolate(adj_0, epsilon, t)
    pred_velocity = self.gnn(coords, adj_t, t)        # 同一个 GNN
    v_target = self.scheduler.get_velocity_target(adj_0, epsilon)
    return F.mse_loss(pred_velocity, v_target)         # MSE（更稳定）


# ========================= 改动 3：推理循环 =========================
# BEFORE（DIFUSCO 连续高斯变体，50步 DDIM）:
@torch.no_grad()
def denoise(self, coords, steps=50):
    adj = torch.randn(B, N, N)
    for t in reversed(range(0, 1000, 1000//steps)):
        pred_eps = self.gnn(coords, adj, torch.full((B,), t/1000.))
        # DDIM 确定性后验（ε-parameterized）
        t_prev = t - 1000//steps
        adj = (abar[t_prev]/abar[t]).sqrt() * (adj - (1-abar[t]).sqrt() * pred_eps) \
              + (1-abar[t_prev]).sqrt() * pred_eps
    return adj * 0.5 + 0.5  # 反转 {-1,1} 缩放到 [0,1]

# AFTER（FM，20步欧拉）:
@torch.no_grad()
def sample(self, coords, steps=20):
    adj = torch.randn(B, N, N)
    dt = 1.0 / steps
    for i in range(steps):
        t = 1.0 - i * dt
        v = self.gnn(coords, adj, torch.full((B,), t))
        adj = adj - dt * v                             # 一行搞定
    return torch.sigmoid(adj)
```

### 10.3 实现复杂度对比

| 维度 | 离散 DDPM | 连续 DDPM | Flow Matching |
|------|----------|----------|--------------|
| Scheduler 代码行数 | ~100行（D3PM 后验） | ~80行（β schedule） | ~15行 |
| 训练时采样代码 | 5行（bit flip） | 5行（高斯噪声） | 3行（线性插值） |
| 推理循环复杂度 | 8行（离散后验公式） | 7行（三项加权公式） | 2行（一步欧拉） |
| 超参数数量 | 5个 | 5个 | 1个（推理步数K） |
| GNN 结构改动 | 基准 | 无需改动 | 无需改动 |
| 实现难度 | 中等（需理解 D3PM） | 中等 | 最简单 |
