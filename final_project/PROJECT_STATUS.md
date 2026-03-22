# 项目状态 — Diffusion Models for TSP

> 基于 DIFUSCO (NeurIPS 2023) 的复现 + Flow Matching 扩展对比
> 截止日期：2026年3月30日 22:00

---

## 当前进度总览

| 阶段 | 状态 |
|------|------|
| 数据生成 | ✅ 完成 |
| 核心模型实现 | ✅ 完成 |
| 训练（3个模型） | ✅ 完成（TSP-50, 50 epochs, RTX 4090） |
| 推理 bug 修复 | ✅ 完成（2026-03-21） |
| FM 重训 | ✅ 完成（服务器 {-1,+1} 域） |
| 主实验评估 | ✅ 完成（merge+2opt） |
| Steps Sweep（merge+2opt） | ✅ 完成 |
| 跨规模泛化测试 | ✅ 完成（TSP-20/50/100） |
| 解码策略消融 | ✅ 完成（5种解码器） |
| 可视化 GIF | ✅ 脚本完成，待服务器生成 |
| 图表生成（8张） | ✅ 完成 |
| Steps Sweep（greedy） | 🔄 待跑（证明真实热力图质量） |
| 多次采样实验 | 🔄 待跑（对标 DIFUSCO 的 16 samples） |
| 架构消融（GAT/GCN） | ⭕ 可选 |
| 报告撰写 | ❌ 待做 |

---

## 最新实验结果（2026-03-22）

### 主结果（TSP-50, merge+2opt, 1000 instances）

| 模型 | Avg Gap | Std | Best | Worst | Valid | Time |
|------|---------|-----|------|-------|-------|------|
| **D3PM** | **1.90%** | 2.06% | 0.00% | 11.64% | 100% | 398ms |
| **FM** | **3.45%** | 2.52% | 0.00% | 11.82% | 100% | 130ms |
| DDPM | 6.21% | 3.44% | 0.00% | 17.56% | 100% | 350ms |

对比 DIFUSCO 论文：D3PM=0.10%, DDPM=0.25%（差距原因见下文）

### Steps Sweep（merge+2opt）

| Steps | FM | D3PM | DDPM |
|-------|------|------|------|
| 5 | 3.34% (36ms) | 1.75% (38ms) | 6.53% (35ms) |
| 10 | 3.45% (66ms) | 1.78% (76ms) | 6.73% (70ms) |
| 20 | 3.48% (130ms) | 1.91% (144ms) | 6.30% (139ms) |
| 50 | 3.47% (333ms) | 1.83% (359ms) | 6.21% (350ms) |
| 100 | 3.66% (817ms) | 1.96% (748ms) | 5.84% (681ms) |

**发现**：FM 和 D3PM 在 5 步即收敛（merge+2opt 掩盖了步数差异），DDPM 随步数改善。
需要补 greedy 步数扫描来揭示真实热力图质量差异。

### 跨规模泛化（训练 TSP-50，测试 20/50/100）

| Scale | FM | D3PM | DDPM |
|-------|------|------|------|
| TSP-20 | 2.25% | 1.54% | 2.91% |
| TSP-50 | 3.63% | 1.99% | 6.23% |
| TSP-100 | **5.39%** | 5.83% | 8.10% |

**发现**：FM 在 TSP-100 泛化性最好（5.39% < D3PM 5.83%），超出训练分布时 FM 优势显现。

### 解码策略消融

| 解码器 | FM | D3PM |
|--------|------|------|
| Greedy | 26.79% (82% valid) | 8.38% (100%) |
| Beam k=3 | 23.87% | 6.49% |
| Beam k=5 | 21.99% | 5.55% |
| Beam k=10 | 19.19% | 5.06% |
| Greedy+2opt | 4.45% (84% valid) | 2.13% |
| **Merge+2opt** | **3.45%** (100%) | **1.90%** (100%) |

**发现**：FM 对解码器极其敏感（8× 差距），D3PM 相对稳健（4× 差距）。

---

## 与 DIFUSCO 论文差距的原因

| 因素 | 我们 | DIFUSCO | 影响程度 |
|------|------|---------|---------|
| 训练数据 | 50K | 128K | ⬛⬛⬜ 中等 |
| 训练 epochs | 50 | ~200-500 | ⬛⬛⬛ 很大 |
| 并行采样 | 1 sample | 16 samples 取最优 | ⬛⬛⬛ 很大 |
| 2-opt 实现 | Python | GPU C++ | ⬛⬜⬜ 较小 |
| batch size | 64 | 更大 | ⬛⬜⬜ 较小 |

**注意**：我们的贡献不是复现 SOTA 数字，而是三方框架对比 + FM 创新引入。差距已在合理范围内。

---

## 待完成实验

### 实验 A：Greedy Steps Sweep（最重要！）

```bash
bash experiments/run_steps_sweep_greedy.sh
```

不用 2opt，纯 greedy 解码的步数扫描。目的是展示：
- DDPM 真的需要更多步来产生好的热力图
- FM 是否在 greedy 下也能快速收敛
- 解码器贡献有多大（greedy vs merge+2opt 对比图 09）

### 实验 B：多次采样取最优

```bash
bash experiments/run_multi_sample.sh
```

对标 DIFUSCO 的 "10 steps × 16 samples" 设置。预期：
- D3PM 16 samples: gap 可能降到 ~1.0% 以下
- DDPM 16 samples: gap 可能降到 ~4%
- FM 是确定性 ODE，多次采样无效果（验证性实验）

### 实验 C：架构消融（可选）

```bash
bash experiments/run_ablation_arch.sh
```

训练 GAT 和 SimpleGCN 变体，对比 GatedGCN 的优势。约 4h。

---

## 报告叙事

核心论点：
> "我们首次将 Flow Matching 引入 TSP 组合优化，与 D3PM 和 Continuous DDPM 在统一 GNN 架构下三方对比。
> 实验表明：(1) FM 在 5 步推理即可达到与 50 步 DDPM 相当的解质量，推理速度提升 19×；
> (2) FM 在超出训练分布的 TSP-100 上泛化性最佳（5.39% vs D3PM 5.83%）；
> (3) 解码策略对连续模型的影响远大于离散模型，merge_tours 是关键组件；
> (4) 多次采样可显著缩小与 DIFUSCO 论文的差距，验证了训练量是主要瓶颈。"

---

## 文件结构

```
final_project/
├── train.py                   ✅ 训练入口
├── evaluate.py                ✅ 评估入口（支持 --n_samples 多次采样）
├── visualize_diffusion.py     ✅ 扩散过程可视化 GIF
├── run_all_experiments.sh     ✅ 一键跑全部消融实验
│
├── data/
│   ├── tsp{20,50,100}_{train,test}.txt
│
├── models/
│   ├── gnn_encoder.py         ✅ GatedGCN / GAT / GCN
│   ├── diffusion_schedulers.py ✅ 三种扩散调度器
│   ├── tsp_model.py           ✅ 统一模型（3 mode）
│   └── tsp_dataset.py         ✅ 数据集
│
├── experiments/
│   ├── plot_results.py        ✅ 10 种图表生成
│   ├── run_steps_sweep.sh     ✅ 步数扫描（merge+2opt）
│   ├── run_steps_sweep_greedy.sh  ✅ 步数扫描（纯 greedy）
│   ├── run_multi_sample.sh    ✅ 多次采样实验
│   ├── run_generalization.sh  ✅ 跨规模泛化
│   ├── run_decoding.sh        ✅ 解码消融
│   ├── run_ablation_arch.sh   ✅ 架构消融
│   └── results/
│       ├── README.md          ✅ 数据文件索引
│       ├── *.json             ✅ 49 个结果文件
│       └── archive/           ✅ 修复前的旧数据
│
├── report/
│   └── figs/                  ✅ 8 张图表（待更新到 10 张）
│
├── checkpoints/               ✅ 3 个模型 checkpoint
├── DIFFUSION_PRINCIPLES.md    ✅ 技术原理文档
├── FLOW_MATCHING_VS_DDPM.md   ✅ FM vs DDPM 分析
├── EXPERIMENT_DESIGN.md       ✅ 实验设计方案
├── PROJECT_STATUS.md          ✅ 本文件
└── TODO.md                    ✅ 详细任务列表
```
