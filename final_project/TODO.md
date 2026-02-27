# 最终大作业：落地级执行 TODO List
## 项目：Diffusion Models for TSP — 基于 DIFUSCO/T2T-CO 的复现与扩展

---

## Context

**课程**：Generative Neural Networks for the Sciences（海德堡大学 2025/26 冬季学期）
**截止日期**：2026年3月30日 22:00，上传至 MaMPF（文件名：`final-project-report.zip`）
**小组**：2人，每人约 90h，共 ~180h
**报告要求**：~8000字（2×4000），LaTeX PDF，每节标注主要作者，含代码仓库链接

**策略选择**：以 **DIFUSCO**（NeurIPS 2023）为主要复现基础（代码更清晰、文档更完整），以 **T2T-CO** 的 Anisotropic GNN 为改进参考。

**核心文件位置**：
- 现有环境：`T0/env/environment.yml`（Python 3.11 + PyTorch 2.5.1 + MPS）
- 参考代码：DIFUSCO → https://github.com/Edward-Sun/DIFUSCO
- 参考代码：T2T-CO → https://github.com/Thinklab-SJTU/T2TCO

---

## 核心论文列表（加入 Zotero）

| 论文 | 作者 | 会议 | 链接 |
|------|------|------|------|
| **Denoising Diffusion Probabilistic Models (DDPM)** | Ho, Jain, Abbeel | NeurIPS 2020 | https://arxiv.org/abs/2006.11239 |
| **DIFUSCO** | Sun et al. | NeurIPS 2023 | https://arxiv.org/abs/2303.18138 |
| **T2T-CO (Fast T2T)** | Chen et al. | NeurIPS 2024 | https://github.com/Thinklab-SJTU/T2TCO |
| **Graph Attention Networks (GAT)** | Veličković et al. | ICLR 2018 | https://arxiv.org/abs/1710.10903 |
| **Attention Model** | Kool et al. | ICLR 2019 | https://arxiv.org/abs/1803.08475 |

---

## 总体目录结构（目标）

```
sheet/final_project/
├── TODO.md                            # 本文件
├── data/
│   ├── generate_tsp_data.py           # 数据生成（改自 DIFUSCO 的同名脚本）
│   ├── tsp20_train.txt                # 生成后存放
│   ├── tsp50_train.txt
│   └── tsp100_test.txt
├── models/
│   ├── gnn_encoder.py                 # GNN 编码器（改自 DIFUSCO）
│   ├── diffusion_schedulers.py        # 扩散调度器（直接复用 DIFUSCO）
│   ├── tsp_model.py                   # 主模型（简化版 pl_tsp_model.py）
│   └── tsp_dataset.py                 # Dataset 类
├── utils/
│   ├── tsp_utils.py                   # 2-opt、merge_tours（直接复用 DIFUSCO）
│   ├── decode.py                      # Greedy + Beam Search 解码
│   └── visualize.py                   # 可视化 + GIF 生成（新写）
├── train.py                           # 训练入口
├── evaluate.py                        # 评估 + Optimality Gap
├── experiments/
│   ├── run_ablation_arch.sh           # 消融：GNN 架构
│   ├── run_generalization.sh          # 泛化测试
│   └── run_decoding.sh                # 解码策略对比
├── report/
│   ├── main.tex                       # LaTeX 主文件
│   └── figs/                          # 生成的图表
└── README.md
```

---

## PHASE 0 — 文献阅读（Week 1 前 2 天，每人约 15h）

> 目标：读懂再动手，避免黑箱复现

### P0-A：DDPM 基础理论（2h，两人都读）
- [ ] 读 **DDPM**（Ho, Jain, Abbeel — NeurIPS 2020）：https://arxiv.org/abs/2006.11239
  - Section 2-3：前向加噪 `q(x_t | x_{t-1}) = N(√(1-β_t)·x_{t-1}, β_t·I)`
  - 逆向去噪：学习 `p_θ(x_{t-1} | x_t)`
  - 理解为什么能直接预测 `x_0`（DIFUSCO 用的是直接预测 x_0 而非 ε）
  - 重点：搞懂"每步去噪是一个分类/回归问题"

### P0-B：DIFUSCO 论文（5h，重点是 Section 3）
- [ ] 读 DIFUSCO（NeurIPS 2023）：https://arxiv.org/abs/2303.18138
  - Section 3.1：如何把 TSP 邻接矩阵 A ∈ {0,1}^(N×N) 当作扩散对象
  - Section 3.2：Categorical Diffusion（离散扩散）vs Gaussian Diffusion
  - Section 3.3：GNN 编码器如何注入城市坐标和时间步信息
  - Section 3.4：从概率热力图到合法 TSP 路径的 `merge_tours` 算法
  - 记录：超参数表（T=1000，lr=2e-4，batch=64，epochs=50）

### P0-C：T2T-CO 论文（3h，只看 Anisotropic GNN 部分）
- [ ] 读 T2T-CO（NeurIPS 2024）的模型架构部分：
  - Anisotropic GCN：为什么用独立的出边/入边权重？
  - 相比标准 GCN 的优势：更好区分方向性特征
  - 记录：`GNNLayer` 的更新公式 `h_i = ReLU(U*h_i + Σ sigmoid(gate_ij) * V*h_j)`

### P0-D：通读代码（5h，各自分工）
- [ ] **人员A**：通读 DIFUSCO 的以下文件（按顺序）：
  1. `difusco/models/gnn_encoder.py`（GNNLayer、GNNEncoder 类）
  2. `difusco/utils/diffusion_schedulers.py`（CategoricalDiffusion、InferenceSchedule）
  3. `difusco/utils/tsp_utils.py`（merge_tours、batched_two_opt_torch）
- [ ] **人员B**：通读 DIFUSCO 的以下文件：
  1. `difusco/co_datasets/tsp_graph_dataset.py`（数据格式、邻接矩阵构建）
  2. `difusco/pl_meta_model.py`（训练循环、loss 计算、推理逻辑）
  3. `difusco/pl_tsp_model.py`（TSP 特定逻辑、test_step 推理流程）
  4. `data/generate_tsp_data.py`（数据格式 `x1 y1 x2 y2 ... output tour`）

---

## PHASE 1 — 环境配置（Week 1，~10h）

> 目标：能 `import` 所有必要库，跑通 DIFUSCO 的 mini demo

### P1-1：扩展现有 GNNS conda 环境
```bash
conda activate GNNS

# 安装 PyTorch Geometric
pip install torch_geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# 其他依赖
pip install pytorch-lightning einops imageio[ffmpeg] networkx
pip install wandb  # 可选，也可用 tensorboard
```
- [ ] 验证：`python -c "import torch_geometric; print(torch_geometric.__version__)"`
- [ ] 注意：DIFUSCO 原来用 PyTorch 1.11，我们用 2.5.1，需检查 API 差异

### P1-2：安装 TSP 求解器（关键！数据生成依赖它）

**方案A：LKH（推荐，Mac M 系列更稳定）**
```bash
# 下载 LKH 二进制：http://webhotel4.ruc.dk/~keld/research/LKH-3/
pip install lkh  # Python wrapper
python -c "import lkh; print('LKH OK')"  # 验证
```

**方案B：Concorde（备选，可能需要编译）**
```bash
pip install pyconcorde  # 如果失败则用 LKH
```

- [ ] 至少保证一个求解器可用
- [ ] 写一个测试脚本验证：生成 5 个随机城市 → 求解 → 打印最优路径

### P1-3：克隆参考代码仓库（只读参考，不在项目里直接用）
```bash
git clone https://github.com/Edward-Sun/DIFUSCO /tmp/DIFUSCO
git clone https://github.com/Thinklab-SJTU/T2TCO /tmp/T2TCO
```
- [ ] 不要把这两个仓库复制进自己的项目（知识产权，报告里注明"参考实现"）

### P1-4：初始化自己的项目仓库
```bash
cd /Users/sena/Desktop/Heidelberg_SciComp/GNNs/sheet/final_project
git init  # 或用 GitHub 建 repo
mkdir -p data models utils experiments report/figs
```
- [ ] 建好 GitHub/GitLab repo（报告封面需要仓库链接）
- [ ] 写 `.gitignore`（排除 `*.txt` 大数据文件、`__pycache__`、`*.ckpt`）

---

## PHASE 2 — 数据生成（Week 1 后半，~15h）

> 目标：生成标准格式的 TSP 数据集，格式与 DIFUSCO 完全兼容

### P2-1：理解数据格式
DIFUSCO 的数据格式（每行一个实例）：
```
x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1
```
- 坐标是 [0,1]² 范围的浮点数，空格分隔
- `output` 是分隔符
- 后面是 1-indexed 的城市访问顺序（包含回到起点）

### P2-2：编写数据生成脚本（`data/generate_tsp_data.py`）
- [ ] 参考 DIFUSCO 的 `data/generate_tsp_data.py` 进行改写
- [ ] 核心函数：
  ```python
  def generate_instance(n_cities, solver='lkh'):
      coords = np.random.uniform(0, 1, (n_cities, 2))
      tour = solve_tsp(coords, solver)   # 调用 LKH
      return coords, tour

  def coords_tour_to_line(coords, tour):
      # 格式化为 DIFUSCO 标准格式的一行字符串
  ```
- [ ] 参数：`--num_nodes`、`--num_samples`、`--solver`、`--output_file`
- [ ] 支持多进程并行（`multiprocessing.Pool`）

### P2-3：生成数据集（跑脚本）
```bash
# TSP-20（快，用于调试，约 10 分钟）
python data/generate_tsp_data.py --num_nodes 20 --num_samples 1000 --output_file data/tsp20_train.txt

# TSP-50（慢，约 1-2 小时）
python data/generate_tsp_data.py --num_nodes 50 --num_samples 5000 --output_file data/tsp50_train.txt

# TSP-100（测试用，约 30 分钟）
python data/generate_tsp_data.py --num_nodes 100 --num_samples 1000 --output_file data/tsp100_test.txt
```
- [ ] 生成后验证：读取几条数据，检查 tour 是否合法（每城市恰好出现一次）

### P2-4：编写 Dataset 类（`models/tsp_dataset.py`）
- [ ] 参考 DIFUSCO 的 `co_datasets/tsp_graph_dataset.py` 改写（稠密模式）：
  ```python
  class TSPDataset(Dataset):
      def __init__(self, data_file):
          # 读取 .txt 文件，解析坐标和 tour

      def __getitem__(self, idx):
          # 返回：coords (N,2), adj_matrix (N,N), tour (N,)
          # adj_matrix[i,j] = 1 当且仅当 (i,j) 在 tour 中
  ```
- [ ] 测试：`DataLoader(dataset, batch_size=4)` 输出形状正确

---

## PHASE 3 — 核心模型实现（Week 2 前半，~30h）

> 目标：一个能训练的端到端系统（不用担心性能，先跑起来）

### P3-1：GNN 编码器（`models/gnn_encoder.py`）
- [ ] 从 DIFUSCO 的 `difusco/models/gnn_encoder.py` 复制并适配：
  - 删除 MIS 相关代码，只保留 TSP 需要的部分
  - 确认 `GNNLayer` 的门控卷积正确：
    ```python
    gate_ij = torch.sigmoid(A*h_i + B*h_j + C*e_ij)
    h_i_new = ReLU(U*h_i + scatter_add(gate_ij * V*h_j))
    ```
  - 确认 `PositionEmbeddingSine`（城市坐标编码）和时间步嵌入
- [ ] 单元测试：
  ```python
  encoder = GNNEncoder(n_layers=4, hidden_dim=128)
  coords = torch.rand(2, 20, 2)      # batch=2, N=20, (x,y)
  adj_noisy = torch.rand(2, 20, 20)
  t = torch.tensor([500, 200])
  out = encoder(coords, adj_noisy, t)  # 期望输出 (2, 20, 20)
  ```

### P3-2：扩散调度器（`models/diffusion_schedulers.py`）
- [ ] 直接复制 DIFUSCO 的 `difusco/utils/diffusion_schedulers.py`（几乎不需要修改）
- [ ] 理解并注释以下关键函数：
  - `CategoricalDiffusion.sample(x0, t)`：x0 加噪 → x_t
  - `InferenceSchedule.__iter__()`：生成推理时间步序列（T → 0）

### P3-3：主模型（`models/tsp_model.py`）
- [ ] 基于 DIFUSCO 的 `pl_tsp_model.py` 大幅简化（去掉 Lightning，用纯 PyTorch）：
  ```python
  class TSPDiffusionModel(nn.Module):
      def __init__(self, n_layers=4, hidden_dim=128, diffusion_steps=1000):
          self.encoder = GNNEncoder(n_layers, hidden_dim)
          self.diffusion = CategoricalDiffusion(diffusion_steps)

      def forward(self, coords, adj_t, t):
          # 给定带噪邻接矩阵和时间步，预测原始邻接矩阵 adj_0
          return self.encoder(coords, adj_t, t)

      def compute_loss(self, coords, adj_0):
          # 1. 采样时间步 t ~ Uniform(1, T)
          # 2. 加噪：adj_t = diffusion.sample(adj_0, t)
          # 3. 前向：pred_adj_0 = self.forward(coords, adj_t, t)
          # 4. 损失：BCE(pred_adj_0, adj_0)
          return loss

      @torch.no_grad()
      def denoise(self, coords, inference_steps=50):
          # 从纯噪声开始，逐步去噪，返回概率热力图
          # 使用 InferenceSchedule 生成时间步序列
          return heatmap  # shape: (N, N)
  ```
- [ ] 验证：随机初始化时 loss ≈ ln(2) ≈ 0.693

### P3-4：训练脚本（`train.py`）
- [ ] 完整训练循环：
  ```python
  # 超参数（参考 DIFUSCO 论文）
  BATCH_SIZE = 64      # GPU 显存不足则降到 32
  LEARNING_RATE = 2e-4
  WEIGHT_DECAY = 1e-4
  EPOCHS = 50
  WARMUP_STEPS = 1000

  optimizer = Adam(model.parameters(), lr=LR, weight_decay=WD)
  scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
  ema_model = copy.deepcopy(model)

  for epoch in range(EPOCHS):
      for coords, adj_0, _ in dataloader:
          loss = model.compute_loss(coords, adj_0)
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()
          update_ema(ema_model, model, decay=0.999)
      # 每 epoch 保存 checkpoint
  ```
- [ ] 支持 Apple MPS：`device = "mps" if torch.backends.mps.is_available() else "cpu"`
- [ ] 支持 Google Colab T4（训练大数据集用）
- [ ] 每 epoch 打印：`Epoch X | Loss: X.XXX | LR: X.XXe-4`

---

## PHASE 4 — 解码器（Week 2 中，~15h）

> 目标：从概率热力图解码出合法 TSP 路径，并计算 Optimality Gap

### P4-1：复用 DIFUSCO 的 merge_tours（`utils/tsp_utils.py`）
- [ ] 从 DIFUSCO 复制以下函数（理解后复用）：
  - `merge_tours(heatmap, coords)`：按 `heatmap[i,j] / dist(i,j)` 降序贪心构建路径
  - `batched_two_opt_torch(tours, coords, max_iter=100)`：2-opt 局部优化
  - `TSPEvaluator.tour_cost(tour, coords)`：计算路径总长度
- [ ] 注意：Cython 版本在 M 系列芯片编译可能有问题，用纯 NumPy 版作备选

### P4-2：Greedy 解码（`utils/decode.py`）
- [ ] 实现简单贪心解码：
  ```python
  def greedy_decode(heatmap):
      # 从节点0出发，每次选择概率最高且未访问的下一节点
      # O(N^2)，适合 N ≤ 100
  ```
- [ ] 验证：输出路径是否是合法哈密顿回路

### P4-3：Beam Search 解码（`utils/decode.py`）
- [ ] 实现 k=5 的 Beam Search：
  ```python
  def beam_search_decode(heatmap, k=5):
      # 维护 k 条候选路径，每步扩展保留概率最高的 k 条
      # 最终选择总距离最短的路径
  ```
- [ ] 对比测试：Greedy vs Beam Search (k=5) 的 Optimality Gap

### P4-4：Optimality Gap 计算（`evaluate.py`）
- [ ] 对每个测试实例计算：`gap = (model_cost - opt_cost) / opt_cost * 100%`
- [ ] 输出：
  ```
  TSP-50 Test Results (1000 instances):
  Avg Optimality Gap: X.XX%
  Avg Inference Time: X.XXs
  Valid Tour Rate: XX.X%
  ```

---

## PHASE 5 — 里程碑验证（Week 2 中期，~1天）

> 在 TSP-20 上跑通完整 pipeline，确认正常后再上 TSP-50

- [ ] 在 TSP-20 上训练 5 个 epoch（目标：10 分钟内完成）
  - Loss 应从约 0.69 下降到 < 0.3
- [ ] 对 10 个测试实例运行推理，用 matplotlib 可视化路径
- [ ] 计算 Optimality Gap（目标：< 10%，说明模型在学习）
- [ ] 如果 Loss 不下降或路径随机，在这里 debug，不要继续往后走

---

## PHASE 5+ — 扩展实验（Week 2 后半 + Week 3 前半，~70h）

> 推荐做实验1 + 实验2

### 实验1：轻量化架构消融（~25h）

- [ ] **变体A（Baseline）**：原始 GNNEncoder（DIFUSCO 的 Gated GCN，4层，hidden=128）
- [ ] **变体B**：GAT（Graph Attention Network）
  ```python
  from torch_geometric.nn import GATConv
  class GATEncoder(nn.Module):
      def __init__(self, n_layers=4, hidden_dim=128, heads=4):
          self.convs = nn.ModuleList([
              GATConv(hidden_dim, hidden_dim//heads, heads=heads)
              for _ in range(n_layers)
          ])
  ```
  注意：需要把稠密图转为 `edge_index` 格式（torch_geometric 要求）
- [ ] **变体C**：简单 GCNConv（`torch_geometric.nn.GCNConv`，最轻量）
- [ ] 三种变体在 TSP-50 上各训练 50 epoch，记录结果填表：

| 模型 | 参数量 | 训练时间/epoch | 推理时间/实例 | Gap (TSP-50) |
|------|--------|--------------|-------------|-------------|
| Gated GCN | — | — | — | — |
| GAT | — | — | — | — |
| Simple GCN | — | — | — | — |

### 实验2：跨规模泛化测试（~25h）

- [ ] 用 TSP-50 训练的最佳模型直接在以下规模测试（不重新训练）：
  - TSP-20：期望 Gap < 2%
  - TSP-50：期望 Gap < 2%（训练集内）
  - TSP-100：期望 Gap < 5%（可能退化）
- [ ] 分析 TSP-100 失败案例：画出模型解 vs 最优解，解释原因
- [ ] 尝试混合规模训练改进泛化：
  ```bash
  cat data/tsp20_train.txt data/tsp50_train.txt > data/mixed_train.txt
  python train.py --data_file data/mixed_train.txt
  ```
- [ ] 画曲线：横轴 = TSP 规模，纵轴 = Optimality Gap

### 实验3（可选）：解码策略对比（~20h）

- [ ] 同一模型，对比：Greedy / Beam Search k=3,10 / Greedy+2-opt
- [ ] 画 Pareto 图（横轴=解码时间，纵轴=Gap），找最优权衡点

---

## PHASE 6 — 可视化（Week 3 前期，~15h）

### P6-1：扩散过程 GIF（必做！）
- [ ] 编写 `utils/visualize.py` 中的 `save_diffusion_gif` 函数：
  ```python
  def save_diffusion_gif(model, coords, output_path, n_frames=20):
      frames = []
      for t in reversed(range(0, 1000, 50)):   # 每50步截一帧
          with torch.no_grad():
              heatmap_t = model.get_intermediate_heatmap(coords, t)
          fig = plot_heatmap(coords, heatmap_t, title=f"t={t}")
          frames.append(fig_to_array(fig))
      imageio.mimsave(output_path, frames, fps=5)
  ```
- [ ] 目标效果：t=1000 全灰噪声 → t=0 清晰的城市连线
- [ ] 生成 3-5 个不同实例的 GIF

### P6-2：路径对比图
- [ ] 左图：LKH 最优解（蓝色）；右图：模型解（橙色）
- [ ] 下方标注：`LKH Cost: X.XXX | Model Cost: X.XXX | Gap: X.X%`
- [ ] 选 3 个实例（Gap 小/中/大各一个）

### P6-3：其他图表
- [ ] 训练 Loss 曲线（横轴=epoch，纵轴=BCE Loss）
- [ ] 消融实验 Bar Chart（不同架构的 Optimality Gap）
- [ ] 泛化曲线（横轴=TSP 规模，纵轴=Gap%）
- [ ] 边概率热力图（N×N 矩阵，imshow，t=0 时刻）

---

## PHASE 7 — LaTeX 报告撰写（Week 3，~10h）

### 封面要求（禁止使用大学/机构 Logo！）
```latex
\title{Diffusion Models for Combinatorial Optimization: \\
       Solving TSP via Generative Modeling}
\author{姓名1（学号1）\and 姓名2（学号2）}
\date{March 2026}
% 封面加上：Code: \url{https://github.com/xxx/tsp-diffusion}
```

### 各章节分工（每节标题后需注明主要作者）

| 章节 | 字数目标 | 负责人 | 关键内容 |
|------|---------|--------|---------|
| 摘要 | ~250字 | 共同 | 问题 + 方法 + 结果 + 贡献 |
| 1. 引言 | ~800字 | 人员A | TSP难度 + LKH局限 + 扩散模型动机 + 贡献列表 |
| 2. 背景 | ~1200字 | 人员B | TSP形式化 + DDPM公式 + 相关工作综述 |
| 3. 方法 | ~1500字 | 共同 | 架构图 + GNN公式 + 扩散公式 + 解码算法 + 超参数表 + **规划vs实际偏差**（课程要求！） |
| 4. 实验 | ~2000字 | 共同 | 收敛曲线 + 消融表格 + 泛化曲线 + GIF截图 |
| 5. 结论 | ~500字 | 共同 | 回答研究问题 + 局限性 + 未来工作 |
| 参考文献 | — | — | DDPM, DIFUSCO, T2T-CO, GAT, LKH, Attention Model |

**重要**：LaTeX 中每节标题后加 `\textit{[Author: 姓名]}` 标注主要作者

---

## PHASE 8 — 提交准备（3月29-30日）

- [ ] 整理 `README.md`：安装方法 + 数据生成命令 + 训练命令 + 评估命令
- [ ] 所有脚本加 `if __name__ == '__main__':` 保护
- [ ] LaTeX 编译无报错，生成 PDF
- [ ] 压缩（**不要打包大数据文件**）：
  ```bash
  zip -r final-project-report.zip report/main.pdf models/ utils/ train.py evaluate.py README.md
  ```
- [ ] 确认 MaMPF 账号（真实姓名，已加入小组）
- [ ] **截止：2026年3月30日 22:00 上传 MaMPF**

---

## 关键风险与解决方案

| 风险 | 概率 | 解决方案 |
|------|------|---------|
| Concorde 安装失败（Mac M 系列） | 高 | 换 LKH：`pip install lkh`，Python wrapper 更稳定 |
| DIFUSCO API 与 PyTorch 2.5 不兼容 | 中 | 逐行 debug，主要是 `torch_sparse` API 变化 |
| TSP-50 本地训练太慢（MPS） | 中 | 用 Google Colab T4（免费）训练，本地只跑 TSP-20 调试 |
| Optimality Gap 过高（>10%） | 中 | 先确保 Loss 下降，再增加 epochs，调整 lr |
| 路径解码出现子回路（非法路径） | 中 | 用 `merge_tours` 替代 Greedy，或加合法性修复后处理 |
| Week 3 时间不够写报告 | 高 | Week 2 就开始写引言和背景，边做实验边写结果节 |

---

## 3周执行时间线

```
Week 1（3/9-3/15）：
  Day 1-2:  P0 文献阅读（两人并行，分工读不同文件）
  Day 3-4:  P1 配环境（目标：torch_geometric + LKH 装好）
  Day 5-7:  P2 数据生成（先生成 TSP-20 验证，再生成 TSP-50）

Week 2（3/16-3/22）：
  Day 1-2:  P3-1/P3-2（GNN 编码器 + 扩散调度器）
  Day 3-4:  P3-3/P3-4（主模型 + 训练脚本）
  Day 5:    P4 解码器 + 里程碑验证（TSP-20 跑通）
  Day 6-7:  P5 实验1（消融实验，3个变体并行训练）
  ↑ 同期开始写报告 P7 引言+背景部分

Week 3（3/23-3/29）：
  Day 1-2:  P5 实验2（泛化测试）
  Day 3:    P6 可视化（重点做扩散 GIF）
  Day 4-5:  P7 完成报告（重点写实验结果章节）
  Day 6:    P8 提交准备
  Day 7:    留余量 buffer
  3月30日 22:00：上传 MaMPF ✅
```
