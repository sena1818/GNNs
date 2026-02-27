"""
TSP 解码器 — 从概率热力图解码出合法路径

需要实现:
1. greedy_decode(heatmap):
   - 从节点 0 出发，每次选概率最高且未访问的下一节点
   - O(N^2)，适合 N ≤ 100
   - 输出: tour (N+1,) 包含回到起点

2. beam_search_decode(heatmap, k=5):
   - 维护 k 条候选路径
   - 每步扩展所有候选，保留概率最高的 k 条
   - 最终选择总距离最短的路径
   - 输出: tour (N+1,)

3. decode_with_2opt(heatmap, coords, method='greedy'):
   - 先用 greedy/beam search 解码
   - 再用 2-opt 局部优化
   - 混合求解器 (扩散初始化 + 2-opt 精调)

对比测试: Greedy vs Beam Search (k=5) 的 Optimality Gap
"""

import torch
import numpy as np

# TODO: 实现解码策略
