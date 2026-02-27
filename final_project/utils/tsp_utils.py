"""
TSP 工具函数 — 复用 DIFUSCO 的核心解码/优化函数

参考: refs/DIFUSCO/difusco/utils/tsp_utils.py

需要实现:
1. merge_tours(heatmap, coords):
   - 输入: heatmap (N, N) 概率矩阵, coords (N, 2) 城市坐标
   - 按 heatmap[i,j] / dist(i,j) 降序贪心构建合法哈密顿路径
   - 输出: tour (N+1,) 合法路径

2. batched_two_opt_torch(tours, coords, max_iter=100):
   - 2-opt 局部优化: 反转路径子段以减少总距离
   - 支持批量处理

3. TSPEvaluator:
   - tour_cost(tour, coords): 计算路径总长度
   - is_valid_tour(tour, n_cities): 检查路径是否合法

注意: DIFUSCO 有 Cython 加速版本，但在 M 系列芯片可能编译失败
      优先使用纯 NumPy/PyTorch 实现作为备选
"""

import torch
import numpy as np

# TODO: 从 DIFUSCO 的 tsp_utils.py 复制并适配
