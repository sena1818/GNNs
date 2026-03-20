# Experiment Design: Flow Matching vs Diffusion for TSP

## 项目信息

- **课程**: Heidelberg University - Scientific Computing / GNN
- **Deadline**: 2026-03-30
- **核心研究问题**: Flow Matching (ODE-based) 能否以更少推理步数达到与扩散模型相当的 TSP 解质量？

## 项目结构

```
final_project/
├── train.py                 # 统一训练入口 (三种模式)
├── evaluate.py              # 评估: 推理 → 解码 → gap 计算
├── models/
│   ├── tsp_model.py         # TSPDiffusionModel (FM / D3PM / DDPM)
│   ├── gnn_encoder.py       # GatedGCN + GAT + GCN 编码器
│   ├── diffusion_schedulers.py  # 三种调度器
│   ├── nn_utils.py          # GroupNorm32, zero_module 等
│   ├── tsp_dataset.py       # 数据加载
│   └── __init__.py
├── utils/
│   ├── decode.py            # greedy / beam search / 2-opt 解码
│   └── tsp_utils.py         # TSPEvaluator, merge_tours
├── experiments/
│   ├── run_main_comparison.sh    # 实验一: 三方主对比
│   ├── run_steps_sweep.sh        # 实验二: 推理步数扫描
│   ├── run_decoding.sh           # 实验三: 解码消融
│   ├── run_ablation_arch.sh      # 实验四: 架构消融
│   ├── run_generalization.sh     # 实验五: 泛化性
│   ├── plot_results.py           # 绘图脚本
│   └── results/                  # 评估结果 JSON
├── data/
│   ├── tsp20_train.txt      # 1,000 instances
│   ├── tsp20_test.txt        # 500 instances
│   ├── tsp50_train.txt      # 50,000 instances
│   └── tsp50_test.txt        # 1,000 instances
├── checkpoints/             # 训练保存的模型
└── report/
    └── main.tex             # LaTeX 报告
```

## 三方对比的统一设计

| 维度 | Flow Matching | Discrete DDPM (D3PM) | Continuous DDPM |
|------|--------------|---------------------|-----------------|
| **生成框架** | ODE (直线路径) | 离散马尔可夫 (Q_bar) | 连续高斯 (DDIM) |
| **GNN 骨干** | GatedGCN 12层 256维 | 同左 | 同左 |
| **训练目标** | MSE on velocity | CrossEntropy on x0 | MSE on noise ε |
| **输出通道** | 1 | 2 (二分类) | 1 |
| **推理方式** | Euler ODE | Q_bar Bayes 后验 | DDIM 确定性 |
| **默认推理步** | 20 | 50 | 50 |
| **Jitter** | 无 | 5% | 5% |

## 实验矩阵

### 实验一: 三方法主结果对比 (必做 ⭐⭐⭐)

**目标**: 在相同条件下对比三种方法的 optimality gap

| 配置 | 值 |
|------|---|
| 数据集 | TSP-50 (train 50k, test 1k) |
| Epochs | 50 |
| Batch size | 64 |
| 解码方式 | greedy, greedy+2-opt |
| 推理步数 | FM:20, D3PM:50, DDPM:50 |

**评估指标**:
- Avg Optimality Gap (%)
- Std Gap (%)
- Valid Tour Rate (%)
- Avg Inference Time (ms/instance)

**脚本**: `bash experiments/run_main_comparison.sh 50 50`

### 实验二: 推理步数 vs Gap 曲线 (必做 ⭐⭐⭐)

**目标**: 证明 FM 在少步数下效率优势（核心创新点的实验证据）

固定训练好的三个模型，分别用不同推理步数评估:

| Steps | FM | D3PM | DDPM |
|-------|-----|------|------|
| 5 | ✅ | ✅ | ✅ |
| 10 | ✅ | ✅ | ✅ |
| 20 | ✅ | ✅ | ✅ |
| 50 | ✅ | ✅ | ✅ |
| 100 | ✅ | ✅ | ✅ |

**预期结果**: FM 在 10-20 步即收敛，DDPM/D3PM 需 50 步

**产出**: steps-vs-gap 折线图（论文核心 Figure）

**脚本**: `bash experiments/run_steps_sweep.sh`

### 实验三: 解码方法消融 (建议做 ⭐⭐)

**目标**: 对比不同解码策略对最终 gap 的影响

| 解码方法 | FM Gap | D3PM Gap | DDPM Gap |
|---------|--------|---------|---------|
| Greedy | ? | ? | ? |
| Greedy + 2-opt | ? | ? | ? |
| Beam Search (k=5) | ? | ? | ? |
| Beam Search + 2-opt | ? | ? | ? |

**脚本**: `bash experiments/run_decoding.sh`

### 实验四: 训练收敛曲线 (必做 ⭐⭐⭐)

**目标**: 对比三种方法的训练稳定性和收敛速度

**数据源**: 训练过程中自动保存的 `checkpoints/*/history.json`

**产出**:
- epoch vs train_loss 三线图
- epoch vs val_loss 三线图
- epoch vs learning_rate (验证 cosine decay)

**脚本**: `python experiments/plot_results.py --plot convergence`

### 实验五: TSP-20 快速验证 (建议做 ⭐⭐)

**目标**: 小规模快速跑通全流程，确认代码正确

| 配置 | 值 |
|------|---|
| 数据集 | TSP-20 (train 1k, test 500) |
| Epochs | 20 |
| Batch size | 64 |

**脚本**: `bash experiments/run_main_comparison.sh 20 20`

### 实验六: 架构消融 (可选 ⭐)

**目标**: 验证 GatedGCN 是否优于 GAT/GCN

仅对 FM 模型做三种 encoder 对比:
- GatedGCN (DIFUSCO 默认)
- GAT
- SimpleGCN

**脚本**: `bash experiments/run_ablation_arch.sh`

## 执行计划 (时间线)

```
Day 1-2:  TSP-20 快速验证 (实验五) → 确认代码无 bug
Day 2-5:  TSP-50 三方训练 (实验一) → 主结果
Day 5-6:  Steps Sweep (实验二) → 核心 Figure
Day 6-7:  解码消融 + 收敛曲线 (实验三四) → 补充实验
Day 7-8:  写报告
Day 8-10: 完善和 buffer
```

## 训练命令速查

```bash
# TSP-20 快速调试 (每个模型 ~5 分钟)
python train.py --mode flow_matching   --data_file data/tsp20_train.txt --epochs 20 --batch_size 64
python train.py --mode discrete_ddpm   --data_file data/tsp20_train.txt --epochs 20 --batch_size 64
python train.py --mode continuous_ddpm --data_file data/tsp20_train.txt --epochs 20 --batch_size 64

# TSP-50 正式训练 (每个模型 ~2-4 小时 on V100)
python train.py --mode flow_matching   --data_file data/tsp50_train.txt --epochs 50 --batch_size 64
python train.py --mode discrete_ddpm   --data_file data/tsp50_train.txt --epochs 50 --batch_size 64
python train.py --mode continuous_ddpm --data_file data/tsp50_train.txt --epochs 50 --batch_size 64

# 评估
python evaluate.py --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
                   --data_file data/tsp50_test.txt --decode greedy --use_2opt

# 续训
python train.py --mode flow_matching --data_file data/tsp50_train.txt \
                --resume checkpoints/flow_matching_gated_gcn/last.pt
```

## 预期论文叙事

> **"我们将 Flow Matching 引入组合优化 (TSP)，与 D3PM 和 Gaussian DDPM 做三方对比。
> 实验表明 FM 以 20 步推理即可达到与 50 步扩散模型相当的解质量，
> 推理速度提升约 2.5 倍，验证了 ODE-based 生成模型在组合优化场景的效率优势。"**
