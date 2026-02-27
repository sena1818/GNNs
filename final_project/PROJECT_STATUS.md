# 项目文件清单与实现状态

> Diffusion Models for TSP — 基于 DIFUSCO/T2T-CO 的复现与扩展
> 截止日期：2026年3月30日 22:00

---

## 文件总览

```
sheet/final_project/
├── PROJECT_STATUS.md          ← 本文件
├── TODO.md                    ✅ 执行计划 (已完成)
├── idea.md                    ✅ 项目提案 (已完成)
├── requirements.txt           ✅ 依赖列表 (已完成)
├── .gitignore                 ✅ (已完成)
│
├── train.py                   ❌ 训练入口
├── evaluate.py                ❌ 评估入口
│
├── data/
│   ├── generate_tsp_data.py   ✅ 数据生成脚本 (已完成)
│   ├── tsp20_train.txt        ✅ TSP-20 训练数据 (1000条, 已生成)
│   ├── tsp50_train.txt        ❌ 需生成 (5000条)
│   └── tsp100_test.txt        ❌ 需生成 (1000条)
│
├── models/
│   ├── __init__.py            ✅ 包导入
│   ├── gnn_encoder.py         ❌ GNN 编码器
│   ├── diffusion_schedulers.py ❌ 扩散调度器
│   ├── tsp_model.py           ❌ 主模型
│   └── tsp_dataset.py         ❌ 数据集类
│
├── utils/
│   ├── __init__.py            ✅ 包导入
│   ├── tsp_utils.py           ❌ TSP 工具函数
│   ├── decode.py              ❌ 解码器
│   └── visualize.py           ❌ 可视化工具
│
├── experiments/
│   ├── run_ablation_arch.sh   ❌ 架构消融实验
│   ├── run_generalization.sh  ❌ 泛化测试实验
│   └── run_decoding.sh        ❌ 解码策略对比 (可选)
│
├── report/
│   ├── main.tex               ❌ LaTeX 报告
│   └── figs/                  ❌ 图表目录 (空)
│
└── refs/                      ✅ 参考代码 (只读)
    ├── DIFUSCO/               ✅ NeurIPS 2023 官方实现
    └── T2TCO/                 ✅ NeurIPS 2024 官方实现
```

---

## 各文件详细说明

### 📦 models/ — 核心模型 (PHASE 3)

| 文件 | 需要实现的内容 | 参考来源 | 优先级 |
|------|--------------|---------|--------|
| **gnn_encoder.py** | `GNNLayer` 门控图卷积层 + `GNNEncoder` 多层编码器 + 位置编码 + 时间步嵌入 | `refs/DIFUSCO/difusco/models/gnn_encoder.py` | ⭐⭐⭐ |
| **diffusion_schedulers.py** | `CategoricalDiffusion` 离散扩散 (前向加噪/逆向去噪) + `InferenceSchedule` 推理时间步 | `refs/DIFUSCO/difusco/utils/diffusion_schedulers.py` | ⭐⭐⭐ |
| **tsp_model.py** | `TSPDiffusionModel` 组合编码器+扩散 + `compute_loss` + `denoise` 推理 | `refs/DIFUSCO/difusco/pl_tsp_model.py` (去掉Lightning) | ⭐⭐⭐ |
| **tsp_dataset.py** | `TSPDataset(Dataset)` 读取txt数据 → 返回 coords/adj_matrix/tour | `refs/DIFUSCO/difusco/co_datasets/tsp_graph_dataset.py` | ⭐⭐⭐ |

### 🔧 utils/ — 工具函数 (PHASE 4 + 6)

| 文件 | 需要实现的内容 | 参考来源 | 优先级 |
|------|--------------|---------|--------|
| **tsp_utils.py** | `merge_tours` 贪心路径构建 + `batched_two_opt_torch` 2-opt优化 + `TSPEvaluator` 路径评估 | `refs/DIFUSCO/difusco/utils/tsp_utils.py` | ⭐⭐⭐ |
| **decode.py** | `greedy_decode` 贪心解码 + `beam_search_decode` Beam Search + `decode_with_2opt` 混合解码 | 自行实现 | ⭐⭐ |
| **visualize.py** | `save_diffusion_gif` 扩散GIF + `plot_tour_comparison` 路径对比 + `plot_heatmap` 热力图 + 训练曲线/消融图/泛化曲线 | 自行实现 (matplotlib + imageio) | ⭐⭐ |

### 🚀 根目录脚本 (PHASE 3-4)

| 文件 | 需要实现的内容 | 优先级 |
|------|--------------|--------|
| **train.py** | argparse 参数 + DataLoader + 训练循环 (Adam + CosineAnnealingLR + EMA + 梯度裁剪) + checkpoint 保存 + MPS/CUDA 支持 | ⭐⭐⭐ |
| **evaluate.py** | 加载checkpoint + 扩散推理 + 解码 + Optimality Gap 计算 + 结果打印 | ⭐⭐⭐ |

### 🧪 experiments/ — 实验脚本 (PHASE 5+)

| 文件 | 实验内容 | 优先级 |
|------|---------|--------|
| **run_ablation_arch.sh** | 3种GNN架构 (Gated GCN / GAT / GCN) 在TSP-50上对比 | ⭐⭐ |
| **run_generalization.sh** | TSP-50训练模型在TSP-20/50/100上的泛化测试 | ⭐⭐ |
| **run_decoding.sh** | Greedy / Beam Search / 2-opt 解码策略 Pareto 分析 | ⭐ (可选) |

### 📝 report/ — LaTeX 报告 (PHASE 7)

| 文件 | 需要完成的内容 | 优先级 |
|------|--------------|--------|
| **main.tex** | 8000字报告: 摘要(250) + 引言(800) + 背景(1200) + 方法(1500) + 实验(2000) + 结论(500) | ⭐⭐ |
| **figs/** | 训练Loss曲线、消融柱状图、泛化曲线、扩散GIF截图、路径对比图、热力图 | ⭐⭐ |

### 📊 data/ — 数据集 (PHASE 2)

| 文件 | 状态 | 生成命令 |
|------|------|---------|
| tsp20_train.txt | ✅ 已生成 (1000条) | — |
| **tsp50_train.txt** | ❌ 需生成 | `python data/generate_tsp_data.py --num_nodes 50 --num_samples 5000 --output_file data/tsp50_train.txt` |
| **tsp100_test.txt** | ❌ 需生成 | `python data/generate_tsp_data.py --num_nodes 100 --num_samples 1000 --output_file data/tsp100_test.txt` |

---

## 建议实现顺序

```
第1步 (PHASE 2): 生成 TSP-50/100 数据
       ↓
第2步 (PHASE 3): models/tsp_dataset.py → models/gnn_encoder.py
       ↓         → models/diffusion_schedulers.py → models/tsp_model.py
       ↓
第3步 (PHASE 3): train.py (训练循环)
       ↓
第4步 (PHASE 4): utils/tsp_utils.py → utils/decode.py → evaluate.py
       ↓
第5步 (PHASE 5): 在 TSP-20 上验证完整 pipeline ← 里程碑!
       ↓
第6步 (PHASE 5+): 运行 experiments/ 下的实验脚本
       ↓
第7步 (PHASE 6): utils/visualize.py → 生成所有图表到 report/figs/
       ↓
第8步 (PHASE 7): 撰写 report/main.tex
```

---

## 关键超参数 (参考 DIFUSCO)

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 64 (GPU不足→32) | — |
| learning_rate | 2e-4 | — |
| weight_decay | 1e-4 | — |
| epochs | 50 | — |
| warmup_steps | 1000 | — |
| diffusion_steps | 1000 | 训练时 |
| inference_steps | 50 | 推理时加速 |
| n_layers | 4 | GNN 层数 |
| hidden_dim | 128 | — |
| ema_decay | 0.999 | — |
| grad_clip | 1.0 | max_norm |
