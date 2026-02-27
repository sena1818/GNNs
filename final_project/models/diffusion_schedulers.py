"""
扩散调度器 — 基于 DIFUSCO 的 CategoricalDiffusion

参考: refs/DIFUSCO/difusco/utils/diffusion_schedulers.py

需要实现:
1. CategoricalDiffusion: 离散扩散过程
   - __init__: 设置 beta schedule, 计算 alpha_bar 等
   - sample(x0, t): 前向加噪 x0 → x_t
   - 支持 Bernoulli 扩散 (TSP 邻接矩阵是 0/1)
2. InferenceSchedule: 推理时间步生成器
   - __iter__: 生成从 T → 0 的时间步序列
   - 支持 linear / cosine / quadratic spacing

关键超参数:
    diffusion_steps = 1000 (训练)
    inference_steps = 50 (推理，加速)
    beta_schedule = 'linear'
"""

import torch
import numpy as np

# TODO: 从 DIFUSCO 的 diffusion_schedulers.py 复制并适配
