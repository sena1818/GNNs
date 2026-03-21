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
| FM 重训 | 🔄 待做（旧 checkpoint 与当前代码不匹配） |
| 重新评估（修复后） | 🔄 待做 |
| 可视化 | ❌ 待做 |
| 报告撰写 | ❌ 待做 |

---

## 文件状态

```
final_project/
├── train.py                   ✅ 训练入口
├── evaluate.py                ✅ 评估入口（默认解码器已改为 merge_tours）
│
├── data/
│   ├── generate_tsp_data.py   ✅ 数据生成脚本
│   ├── tsp20_train.txt        ✅ TSP-20 (1000条)
│   ├── tsp20_test.txt         ✅ TSP-20 测试集
│   ├── tsp50_train.txt        ✅ TSP-50 训练集 (50000条)
│   └── tsp50_test.txt         ✅ TSP-50 测试集 (1000条)
│
├── models/
│   ├── gnn_encoder.py         ✅ GNN 编码器（Gated GCN，对齐 DIFUSCO）
│   ├── diffusion_schedulers.py ✅ 扩散调度器（已添加 DDPM 所需 alpha/beta tensor）
│   ├── tsp_model.py           ✅ 主模型（已添加 DDPM 随机后验方法）
│   └── tsp_dataset.py         ✅ 数据集类
│
├── utils/
│   ├── tsp_utils.py           ✅ merge_tours + TSPEvaluator
│   ├── decode.py              ✅ greedy + beam_search + 2-opt
│   └── visualize.py           ❌ 待完成
│
├── checkpoints/               ✅ 已下载到本地备份
│   ├── flow_matching_gated_gcn/      best.pt + history.json
│   ├── discrete_ddpm_gated_gcn/      best.pt + history.json
│   └── continuous_ddpm_gated_gcn/    best.pt + history.json
│
└── experiments/results/
    ├── main_flow_matching_greedy.json          ✅ (旧结果, 修复前)
    ├── main_flow_matching_greedy2opt.json      ✅ (旧结果, 修复前)
    ├── main_discrete_ddpm_greedy.json          ✅ (旧结果, 修复前)
    ├── main_discrete_ddpm_greedy2opt.json      ✅ (旧结果, 修复前)
    ├── main_continuous_ddpm_greedy.json        ⚠️  (bug版本, 4% valid)
    └── main_continuous_ddpm_greedy2opt.json    ⚠️  (bug版本, 4% valid)
```

---

## 训练结果（服务器 RTX 4090，TSP-50，50 epochs）

| 模型 | Train Loss (ep1→50) | Val Loss (ep1→50) | 训练用时 |
|------|--------------------|--------------------|---------|
| discrete_ddpm | 0.0624 → 0.0122 | 0.0445 → 0.0139 | ~2h |
| continuous_ddpm | 0.0513 → 0.0052 | 0.0198 → 0.0052 | ~2h |
| flow_matching | 0.1265 → 0.0081 | 0.0679 → 0.0081 | ~2h |

超参数：batch_size=64, lr=2e-4, CosineAnnealingLR, EMA decay=0.999, grad_clip=1.0, n_layers=12, hidden_dim=256

---

## 评估结果

### 修复前（greedy 解码，旧代码）

| 模型 | 解码方式 | avg_gap | std_gap | valid_rate | 推理时间 |
|------|---------|---------|---------|-----------|---------|
| discrete_ddpm | greedy | 8.66% | 8.79% | 100% | 382ms |
| discrete_ddpm | greedy+2opt | **2.09%** | 2.40% | 100% | 361ms |
| continuous_ddpm | greedy | 5.92% | 4.31% | **4%** ❌ | 351ms |
| continuous_ddpm | greedy+2opt | 2.85% | 2.71% | **4%** ❌ | 353ms |
| flow_matching | greedy | 28.09% | 15.67% | 100% | 140ms |
| flow_matching | greedy+2opt | 4.45% | 3.13% | 100% | 154ms |

### 修复后（待跑——merge_tours + DDPM 随机后验 + FM 重训）

| 模型 | 解码方式 | avg_gap | valid_rate |
|------|---------|---------|-----------|
| discrete_ddpm | merge+2opt | 待测 | 待测 |
| continuous_ddpm | merge+2opt (DDPM 随机) | 待测 | 预期 ~100% |
| flow_matching (重训) | merge+2opt | 待测 | 预期 ~100% |

---

## Bug 记录与修复

### Bug 1：Continuous DDPM valid_rate = 4%（✅ 已修复）

**根因**：我们只实现了 DDIM 确定性推理，但 DIFUSCO 官方默认用 DDPM 随机推理。
- DDIM 50步推理中，每步预测误差不断叠加 → xt 变成极端大负数
- 还原公式 `heatmap = xt * 0.5 + 0.5` 导致输出全部接近 0
- greedy 解码无法找到有效路径

**DDIM vs DDPM 随机的区别**：
- DDIM：每步 `xt_next = 系数A × xt + 系数B × pred`，完全确定性，误差可叠加
- DDPM 随机：每步 `xt_next = 均值 + √β̃ × 随机噪声z`，随机噪声把累积误差打散

**修复内容**：
- `models/diffusion_schedulers.py`：新增 `alpha_torch`, `beta_torch` GPU tensor
- `models/tsp_model.py`：新增 `_gaussian_posterior_ddpm_tensor()` 随机后验
- `models/tsp_model.py`：`sample()` 新增 `inference_trick` 参数，默认 DDPM 随机，`'ddim'` 切换确定性

### Bug 2：默认解码器 greedy 脆弱（✅ 已修复）

**根因**：greedy 从节点0顺序选最大邻居，heatmap 稍有噪声就走进死路。

**merge_tours 工作原理**（DIFUSCO 官方解码器）：
1. 给所有边打分：`score = heatmap[i][j] / dist(i,j)`
2. 按分数降序排列所有边
3. 贪心加边：跳过会让节点超过 2 条边或提前成环的边（union-find 检测）
4. 保证输出合法哈密顿回路，对噪声 heatmap 完全鲁棒

**修复内容**：
- `evaluate.py`：默认解码器 `greedy` → `merge`
- `evaluate.py`：新增 `--inference_trick` 命令行参数

### Bug 3：FM checkpoint 与代码不匹配（🔄 待重训）

**根因**：FM checkpoint 用旧代码（`{0,1}` 目标空间）训练，当前推理代码用 `{-1,+1}` 空间（`x*0.5+0.5` 还原），训练与推理的值域定义完全不同，导致 28% 随机级别 gap。

**修复**：重新训练 FM，训练代码已正确，只需重跑。

---

## 待完成事项

### 服务器操作

```bash
tmux new -s work
cd /root/final_project

# 重新评估 D3PM 和 Continuous DDPM（不需重训）
python evaluate.py --checkpoint checkpoints/discrete_ddpm_gated_gcn/best.pt \
    --data_file data/tsp50_test.txt --use_2opt \
    --save_result experiments/results/fixed_d3pm_merge2opt.json

python evaluate.py --checkpoint checkpoints/continuous_ddpm_gated_gcn/best.pt \
    --data_file data/tsp50_test.txt --use_2opt \
    --save_result experiments/results/fixed_gaussian_ddpm_merge2opt.json

# 重训 FM（约 2 小时）
python train.py --mode flow_matching \
    --data_file data/tsp50_train.txt \
    --epochs 50 --batch_size 64 \
    --save_dir checkpoints/flow_matching_gated_gcn

# FM 重训完成后评估
python evaluate.py --checkpoint checkpoints/flow_matching_gated_gcn/best.pt \
    --data_file data/tsp50_test.txt --use_2opt \
    --save_result experiments/results/fixed_fm_merge2opt.json
```

### 报告（截止 3 月 30 日）

- [ ] 训练 loss 曲线图（三个模型对比）
- [ ] 最终结果对比表格
- [ ] 扩散过程热力图（去噪中间步骤截图）
- [ ] 路径对比图（模型解 vs 最优解）
- [ ] LaTeX 报告正文

---

## 我们的实现 vs DIFUSCO 官方

| 方面 | DIFUSCO 官方 | 我们的实现 | 状态 |
|------|-------------|-----------|------|
| 框架 | PyTorch Lightning | 纯 PyTorch | 等价 |
| GNN 架构 | Gated GCN | Gated GCN（对齐） | ✅ |
| D3PM 前向/后验 | Q_bar 转移矩阵 | Q_bar 转移矩阵（对齐） | ✅ |
| Gaussian 推理默认 | DDPM 随机 | DDPM 随机（已修复） | ✅ |
| 解码器 | merge_tours | merge_tours（已修复） | ✅ |
| 2-opt | GPU 加速 | Python 实现（较慢） | 近似等价 |
| 并行采样 | 支持 | 不支持 | 影响不大 |
| **扩展：Flow Matching** | ❌ 无 | ✅ 本项目新增 | 创新点 |
