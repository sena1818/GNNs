# Experiment Results Index

All experiments use **TSP-50** (train) unless noted. Decoder is `merge_tours + 2-opt` unless noted.
Models: `flow_matching` (FM), `discrete_ddpm` (D3PM), `continuous_ddpm` (DDPM).
Encoder: `gated_gcn` (12 layers, hidden=256) for all.

---

## 1. Main Results  ← **核心对比数据**

| File | Model | Decoder | Gap | Notes |
|------|-------|---------|-----|-------|
| `fm_merge2opt.json` | Flow Matching | merge+2opt | **3.45%** | 20 steps, 130ms |
| `d3pm_merge2opt.json` | Discrete DDPM | merge+2opt | **1.90%** | 50 steps, 398ms |
| `ddpm_merge2opt.json` | Continuous DDPM | merge+2opt | **6.21%** | 50 steps, 349ms |

> 生成图: `02_main_results_bar.png`, `03_inference_time.png`, `05_gap_distribution.png`

---

## 2. Inference Steps Sweep  ← **核心消融：步数 vs 质量**

测试不同推理步数（5/10/20/50/100）对解质量和速度的影响。解码器统一用 merge+2opt。

### Flow Matching
| File | Steps | Gap | Time |
|------|-------|-----|------|
| `steps_flow_matching_s5.json` | 5 | 3.34% | 36ms |
| `steps_flow_matching_s10.json` | 10 | 3.45% | 66ms |
| `steps_flow_matching_s20.json` | 20 | 3.48% | 130ms |
| `steps_flow_matching_s50.json` | 50 | 3.47% | 333ms |
| `steps_flow_matching_s100.json` | 100 | 3.66% | 817ms |

### Discrete DDPM (D3PM)
| File | Steps | Gap | Time |
|------|-------|-----|------|
| `steps_discrete_ddpm_s5.json` | 5 | 1.75% | 38ms |
| `steps_discrete_ddpm_s10.json` | 10 | 1.78% | 76ms |
| `steps_discrete_ddpm_s20.json` | 20 | 1.91% | 144ms |
| `steps_discrete_ddpm_s50.json` | 50 | 1.83% | 359ms |
| `steps_discrete_ddpm_s100.json` | 100 | 1.96% | 748ms |

### Continuous DDPM
| File | Steps | Gap | Time |
|------|-------|-----|------|
| `steps_continuous_ddpm_s5.json` | 5 | 6.53% | 35ms |
| `steps_continuous_ddpm_s10.json` | 10 | 6.73% | 70ms |
| `steps_continuous_ddpm_s20.json` | 20 | 6.30% | 139ms |
| `steps_continuous_ddpm_s50.json` | 50 | 6.21% | 350ms |
| `steps_continuous_ddpm_s100.json` | 100 | 5.84% | 681ms |

**关键发现**: FM 和 D3PM 在 5 步时已基本收敛；DDPM 随步数单调改善，说明其去噪路径需要更多步骤。

> 生成图: `06_steps_sweep.png`

---

## 3. Cross-Scale Generalization  ← **泛化能力测试**

模型在 TSP-50 上训练，分别在 TSP-20 / TSP-50 / TSP-100 上测试（20 steps, merge+2opt）。

| File | Model | Scale | Gap |
|------|-------|-------|-----|
| `gen_flow_matching_gated_gcn_tsp20.json` | FM | TSP-20 | 2.25% |
| `gen_flow_matching_gated_gcn_tsp50.json` | FM | TSP-50 | 3.63% |
| `gen_flow_matching_gated_gcn_tsp100.json` | FM | TSP-100 | 5.39% |
| `gen_discrete_ddpm_gated_gcn_tsp20.json` | D3PM | TSP-20 | 1.54% |
| `gen_discrete_ddpm_gated_gcn_tsp50.json` | D3PM | TSP-50 | 1.99% |
| `gen_discrete_ddpm_gated_gcn_tsp100.json` | D3PM | TSP-100 | 5.83% |
| `gen_continuous_ddpm_gated_gcn_tsp20.json` | DDPM | TSP-20 | 2.91% |
| `gen_continuous_ddpm_gated_gcn_tsp50.json` | DDPM | TSP-50 | 6.23% |
| `gen_continuous_ddpm_gated_gcn_tsp100.json` | DDPM | TSP-100 | 8.10% |

**关键发现**: D3PM 在小规模（TSP-20）表现最佳；所有模型在 TSP-100（超出训练分布）时 gap 均显著上升，FM 泛化性最好。

> 生成图: `07_generalization.png`

---

## 4. Decoding Strategy Ablation  ← **解码器消融**

固定模型权重，对比 greedy / greedy+2opt / beam(k=3,5,10) / merge+2opt 五种解码策略（20 steps for FM, 50 for D3PM）。

### Flow Matching
| File | Decoder | Gap | Valid Rate |
|------|---------|-----|-----------|
| `flow_matching_gated_gcn_decode_greedy.json` | Greedy | 26.79% | 82% |
| `flow_matching_gated_gcn_decode_greedy2opt.json` | Greedy+2opt | 4.45% | 84% |
| `flow_matching_gated_gcn_decode_beam3.json` | Beam k=3 | 23.87% | 100% |
| `flow_matching_gated_gcn_decode_beam5.json` | Beam k=5 | 21.99% | 100% |
| `flow_matching_gated_gcn_decode_beam10.json` | Beam k=10 | 19.19% | 100% |
| `fm_merge2opt.json` | **Merge+2opt** ★ | **3.45%** | 100% |

### Discrete DDPM
| File | Decoder | Gap | Valid Rate |
|------|---------|-----|-----------|
| `discrete_ddpm_gated_gcn_decode_greedy.json` | Greedy | 8.38% | 100% |
| `discrete_ddpm_gated_gcn_decode_greedy2opt.json` | Greedy+2opt | 2.13% | 100% |
| `discrete_ddpm_gated_gcn_decode_beam3.json` | Beam k=3 | 6.49% | 100% |
| `discrete_ddpm_gated_gcn_decode_beam5.json` | Beam k=5 | 5.55% | 100% |
| `discrete_ddpm_gated_gcn_decode_beam10.json` | Beam k=10 | 5.06% | 100% |
| `d3pm_merge2opt.json` | **Merge+2opt** ★ | **1.90%** | 100% |

**关键发现**: FM 对解码器极度敏感（greedy 26.8% vs merge+2opt 3.45%，8× 差距），D3PM 相对稳健（4× 差距）。merge+2opt 是两者最优解码策略。

> 生成图: `08_decoding_ablation.png`

---

## 5. Bug Fix Comparison  ← **修复记录**

存档于 `archive/`，记录两个关键 bug 修复前的数据。

| File | 说明 |
|------|------|
| `archive/pre_fix_fm_greedy.json` | FM bug修复前（{0,1}值域），greedy，gap=28.09% |
| `archive/pre_fix_fm_greedy2opt.json` | FM bug修复前，greedy+2opt，gap=19.xx% |
| `archive/pre_fix_d3pm_greedy.json` | D3PM bug修复前，greedy |
| `archive/pre_fix_d3pm_greedy2opt.json` | D3PM bug修复前，greedy+2opt |
| `archive/pre_fix_ddpm_greedy.json` | DDPM bug修复前（DDIM+greedy），valid_rate=4% |
| `archive/pre_fix_ddpm_greedy2opt.json` | DDPM bug修复前，greedy+2opt |
| `archive/fm_clamp_greedy.json` | FM 中间版本（仅加clamp，未重训） |
| `archive/fm_clamp_greedy2opt.json` | FM 中间版本，greedy+2opt |

> 生成图: `04_bug_fix_comparison.png`

---

## Report Figures Summary

| 图表文件 | 内容 | 数据来源 |
|---------|------|---------|
| `report/figs/01_training_curves.png` | 三模型训练/验证 loss 曲线（log scale） | checkpoints/*/history.json |
| `report/figs/02_main_results_bar.png` | 主结果柱状图 + DIFUSCO 论文参考线 | main results (§1) |
| `report/figs/03_inference_time.png` | 推理速度三合一（总时间/每步/Pareto） | main results (§1) |
| `report/figs/04_bug_fix_comparison.png` | Bug 修复前后对比 | archive (§5) |
| `report/figs/05_gap_distribution.png` | Gap 完整分布箱线图（1000 instances） | main results (§1) |
| `report/figs/06_steps_sweep.png` | 推理步数 vs 质量+速度折线图 | steps sweep (§2) |
| `report/figs/07_generalization.png` | 跨规模泛化折线图（TSP-20/50/100） | generalization (§3) |
| `report/figs/08_decoding_ablation.png` | 解码策略 Pareto 散点图 | decoding ablation (§4) |

重新生成所有图表:
```bash
cd final_project
python experiments/plot_results.py
```
