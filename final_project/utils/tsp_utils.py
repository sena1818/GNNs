"""
TSP 工具函数

提供三类功能:
1. TSPEvaluator — 路径评估 (cost + validity)
2. merge_tours  — 按 heatmap/dist 比值贪心构建路径（DIFUSCO 风格）
3. compute_optimality_gap — 批量 gap 计算，供 evaluate.py 调用

注意：decode.py 已有 greedy_decode / beam_search_decode / two_opt_improve，
      本文件专注于评估指标和 DIFUSCO-style merge 逻辑，避免重复实现。
"""

import torch
import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# TSPEvaluator — 路径评估
# ---------------------------------------------------------------------------

class TSPEvaluator:
    """
    TSP 路径评估器。

    用法:
        evaluator = TSPEvaluator(coords)     # coords: (N, 2) tensor 或 ndarray
        cost = evaluator.tour_cost(tour)     # tour: list[int] 0-indexed
        valid = evaluator.is_valid(tour)
        gap = evaluator.optimality_gap(pred_tour, opt_tour)
    """

    def __init__(self, coords):
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        self.coords = coords  # (N, 2)
        self.N = coords.shape[0]

    def tour_cost(self, tour: List[int]) -> float:
        """计算 tour 的欧氏总路径长度。"""
        total = 0.0
        N = len(tour)
        for k in range(N):
            i, j = tour[k], tour[(k + 1) % N]
            diff = self.coords[i] - self.coords[j]
            total += diff.norm().item()
        return total

    def is_valid(self, tour: List[int]) -> bool:
        """检查是否为合法 Hamilton 回路（访问所有城市恰好一次）。"""
        return len(tour) == self.N and set(tour) == set(range(self.N))

    def optimality_gap(self, pred_tour: List[int], opt_tour: List[int]) -> float:
        """
        gap = (pred_cost - opt_cost) / opt_cost * 100%
        返回百分比 (%)，若 opt_cost ≈ 0 返回 0.0。
        """
        pred_cost = self.tour_cost(pred_tour)
        opt_cost  = self.tour_cost(opt_tour)
        if opt_cost < 1e-10:
            return 0.0
        return (pred_cost - opt_cost) / opt_cost * 100.0


# ---------------------------------------------------------------------------
# merge_tours — DIFUSCO 风格的贪心路径构建
# ---------------------------------------------------------------------------

def merge_tours(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
) -> List[int]:
    """
    DIFUSCO 风格路径构建：按 heatmap[i,j] / dist(i,j) 降序贪心选边。

    与 greedy_decode（按 heatmap 最大邻居逐步前进）不同，
    merge_tours 先对所有边排序，再贪心加边（类似 Christofides 的边集合方式），
    最终构建出连通的 Hamilton 回路。

    Args:
        heatmap: (N, N) 边概率矩阵，值域 [0, 1]
        coords:  (N, 2) 城市坐标
    Returns:
        tour: list[int]，0-indexed，长度 N，合法 Hamilton 回路
    """
    N = heatmap.shape[0]
    device = heatmap.device

    # 计算欧氏距离矩阵
    c = coords.float()
    diff = c.unsqueeze(1) - c.unsqueeze(0)          # (N, N, 2)
    dist = diff.norm(dim=-1)                          # (N, N)
    dist = dist + torch.eye(N, device=device) * 1e9  # 对角线置大数

    # 按 heatmap/dist 降序对所有边排序（取上三角避免重复）
    score = heatmap / (dist + 1e-8)
    score = score.cpu().numpy()
    dist_np = dist.cpu().numpy()

    # 提取上三角边
    rows, cols = np.triu_indices(N, k=1)
    edge_scores = score[rows, cols]
    order = np.argsort(-edge_scores)  # 降序
    sorted_edges = list(zip(rows[order].tolist(), cols[order].tolist()))

    # 贪心加边：满足 degree≤2 且不形成提前闭合的环
    degree = [0] * N
    adj = [[] for _ in range(N)]

    def find_root(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    parent = list(range(N))

    def union(parent, x, y):
        rx, ry = find_root(parent, x), find_root(parent, y)
        if rx != ry:
            parent[rx] = ry
            return True
        return False

    edges_added = 0
    for u, v in sorted_edges:
        if edges_added == N:
            break
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        # 检查：加入 (u,v) 是否形成提前闭合（除非这是最后一条边）
        if edges_added < N - 1 and find_root(parent, u) == find_root(parent, v):
            continue
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
        union(parent, u, v)
        edges_added += 1

    # 如果贪心边不够 N 条（极少情况），补全
    if edges_added < N:
        unvisited = [i for i in range(N) if degree[i] < 2]
        for i in range(0, len(unvisited) - 1, 2):
            u, v = unvisited[i], unvisited[i + 1]
            if degree[u] < 2 and degree[v] < 2:
                adj[u].append(v)
                adj[v].append(u)
                degree[u] += 1
                degree[v] += 1
                edges_added += 1

    # 将邻接表转为 tour（从节点 0 出发遍历）
    tour = [0]
    prev = -1
    current = 0
    for _ in range(N - 1):
        neighbors = [nb for nb in adj[current] if nb != prev]
        if not neighbors:
            # 断链，补全剩余节点
            remaining = [i for i in range(N) if i not in set(tour)]
            tour.extend(remaining)
            break
        next_node = neighbors[0]
        tour.append(next_node)
        prev = current
        current = next_node

    # 确保长度 N（防止极端情况）
    if len(tour) < N:
        missing = [i for i in range(N) if i not in set(tour)]
        tour.extend(missing)

    return tour[:N]


# ---------------------------------------------------------------------------
# 批量 gap 统计（evaluate.py 的辅助函数）
# ---------------------------------------------------------------------------

def compute_batch_gaps(
    pred_tours: List[List[int]],
    opt_tours,          # list[list[int]] 或 Tensor (B, N)
    coords_batch,       # list[(N,2) tensor] 或 Tensor (B, N, 2)
) -> Tuple[List[float], List[bool]]:
    """
    对一个 batch 计算每个实例的 optimality gap 和 validity。

    Returns:
        gaps:   list[float]，仅对合法 tour 有值
        valids: list[bool]
    """
    B = len(pred_tours)
    gaps   = []
    valids = []

    for i in range(B):
        pred = pred_tours[i]
        opt  = opt_tours[i].tolist() if hasattr(opt_tours[i], 'tolist') else list(opt_tours[i])
        c    = coords_batch[i]
        if hasattr(c, 'cpu'):
            c = c.cpu()

        N = c.shape[0]
        evaluator = TSPEvaluator(c)

        valid = evaluator.is_valid(pred)
        valids.append(valid)

        if valid:
            gap = evaluator.optimality_gap(pred, opt)
            gaps.append(gap)

    return gaps, valids


# ---------------------------------------------------------------------------
# 快速单元测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    N = 10
    coords = torch.rand(N, 2)

    # TSPEvaluator
    ev = TSPEvaluator(coords)
    tour = list(range(N))
    print(f'Tour cost:    {ev.tour_cost(tour):.4f}')
    print(f'Tour valid:   {ev.is_valid(tour)}')
    print(f'Gap (self):   {ev.optimality_gap(tour, tour):.2f}%')

    # merge_tours
    heatmap = torch.rand(N, N)
    heatmap = (heatmap + heatmap.T) / 2  # 对称
    merged = merge_tours(heatmap, coords)
    print(f'Merge tour:   {merged}')
    print(f'Merge valid:  {ev.is_valid(merged)}')
    print(f'Merge cost:   {ev.tour_cost(merged):.4f}')

    print('tsp_utils.py OK')
