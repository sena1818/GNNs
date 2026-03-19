"""
TSP 解码器 — 从概率热力图解码出合法 Hamilton 回路


三种解码策略（速度 vs 质量的 Pareto 权衡）:

  greedy_decode(heatmap, coords)
      O(N²)，最快，质量中等
      每步从当前节点选概率最高且未访问的邻居

  beam_search_decode(heatmap, coords, k=5)
      O(k·N²)，较慢，质量更好
      维护 k 条候选路径，最终选路径长度最短的

  decode_with_2opt(heatmap, coords, method, max_iter)
      在 greedy/beam 结果上做 2-opt 局部优化
      质量最好，适合最终评估

所有函数输入/输出:
    heatmap: (N, N) torch.Tensor，值域 [0,1]（单个实例，非 batch）
    coords:  (N, 2) torch.Tensor
    返回:    tour  list[int]，长度 N，0-indexed，合法 Hamilton 回路
"""

import math
import torch
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# 工具：路径长度计算
# ---------------------------------------------------------------------------

def tour_length(tour: List[int], coords: torch.Tensor) -> float:
    """
    计算 tour 的总路径长度（欧氏距离之和）。
    tour: 0-indexed，长度 N
    coords: (N, 2)
    """
    total = 0.0
    N = len(tour)
    for k in range(N):
        i, j = tour[k], tour[(k + 1) % N]
        diff = coords[i] - coords[j]
        total += diff.norm().item()
    return total


def is_valid_tour(tour: List[int], N: int) -> bool:
    """检查 tour 是否是合法 Hamilton 回路（访问所有城市恰好一次）。"""
    return len(tour) == N and set(tour) == set(range(N))


# ---------------------------------------------------------------------------
# 策略 1：贪心解码
# ---------------------------------------------------------------------------

def greedy_decode(heatmap: torch.Tensor, coords: torch.Tensor) -> List[int]:
    """
    贪心解码：从节点 0 出发，每步选 heatmap 概率最高且未访问的邻居。

    Args:
        heatmap: (N, N) 边概率矩阵
        coords:  (N, 2)
    Returns:
        tour: list[int]，长度 N，0-indexed
    """
    N = heatmap.shape[0]
    h = heatmap.clone()

    # 对角线置0（不能自连）
    h.fill_diagonal_(0.0)

    visited = [False] * N
    tour = [0]
    visited[0] = True

    for _ in range(N - 1):
        current = tour[-1]
        # 将已访问节点的概率置 0
        row = h[current].clone()
        for v in tour:
            row[v] = 0.0

        next_node = row.argmax().item()
        tour.append(next_node)
        visited[next_node] = True

    return tour


# ---------------------------------------------------------------------------
# 策略 2：Beam Search 解码
# ---------------------------------------------------------------------------

def beam_search_decode(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    k: int = 5,
) -> List[int]:
    """
    Beam Search 解码：维护 k 条候选路径，最终返回路径长度最短的。

    Args:
        heatmap: (N, N)
        coords:  (N, 2)
        k:       beam width
    Returns:
        tour: list[int]，长度 N
    """
    N = heatmap.shape[0]
    h = heatmap.clone()
    h.fill_diagonal_(0.0)

    # 每条 beam: (log_prob, path)
    # 从节点 0 出发
    beams = [(0.0, [0])]

    for step in range(N - 1):
        new_beams = []
        for log_prob, path in beams:
            current = path[-1]
            visited_set = set(path)

            # 候选下一节点（未访问）
            row = h[current].clone()
            for v in visited_set:
                row[v] = 0.0

            # 取 top-k 个候选（或剩余未访问节点数，取小者）
            remaining = N - len(path)
            top_k = min(k, remaining)
            if top_k == 0:
                continue

            probs, indices = row.topk(top_k)
            for prob, idx in zip(probs.tolist(), indices.tolist()):
                if prob <= 0:
                    continue
                new_log_prob = log_prob + math.log(prob + 1e-10)
                new_beams.append((new_log_prob, path + [idx]))

        if not new_beams:
            # fallback：用贪心完成剩余路径
            best_path = max(beams, key=lambda x: x[0])[1]
            remaining_nodes = [v for v in range(N) if v not in set(best_path)]
            beams = [(beams[0][0], best_path + remaining_nodes)]
            break

        # 保留 top-k beams（按 log_prob 降序）
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:k]

    # 从所有 beam 中选路径长度最短的
    best_tour = min(
        [b[1] for b in beams if len(b[1]) == N],
        key=lambda t: tour_length(t, coords),
        default=beams[0][1],
    )

    # 如果某个 beam 不完整（极少情况），补全
    if len(best_tour) < N:
        missing = [v for v in range(N) if v not in set(best_tour)]
        best_tour = best_tour + missing

    return best_tour


# ---------------------------------------------------------------------------
# 策略 3：2-opt 局部优化
# ---------------------------------------------------------------------------

def two_opt_improve(tour: List[int], coords: torch.Tensor, max_iter: int = 100) -> List[int]:
    """
    2-opt 局部搜索：重复交换两条边，直到无法改善。
    O(N² × max_iter)

    Args:
        tour:     初始 tour（0-indexed，长度 N）
        coords:   (N, 2)
        max_iter: 最大迭代轮数
    Returns:
        improved tour
    """
    N = len(tour)
    best = list(tour)
    best_len = tour_length(best, coords)

    for _ in range(max_iter):
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                # 当前路段：best[i]→best[i+1] 和 best[j]→best[(j+1)%N]
                a, b = best[i], best[(i + 1) % N]
                c, d = best[j], best[(j + 1) % N]

                # 计算交换前后的长度变化
                d_old = (coords[a] - coords[b]).norm() + (coords[c] - coords[d]).norm()
                d_new = (coords[a] - coords[c]).norm() + (coords[b] - coords[d]).norm()

                if d_new < d_old - 1e-10:
                    # 反转 best[i+1 : j+1]
                    best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                    best_len = best_len - d_old.item() + d_new.item()
                    improved = True

        if not improved:
            break

    return best


def decode_with_2opt(
    heatmap: torch.Tensor,
    coords: torch.Tensor,
    method: str = 'greedy',
    beam_k: int = 5,
    max_iter: int = 100,
) -> List[int]:
    """
    先用 greedy 或 beam search 解码，再用 2-opt 优化。

    Args:
        heatmap: (N, N)
        coords:  (N, 2)
        method:  'greedy' 或 'beam_search'
        beam_k:  beam search 宽度
        max_iter: 2-opt 最大迭代次数
    Returns:
        tour: list[int]
    """
    if method == 'greedy':
        tour = greedy_decode(heatmap, coords)
    elif method == 'beam_search':
        tour = beam_search_decode(heatmap, coords, k=beam_k)
    else:
        raise ValueError(f"Unknown method: {method}")

    return two_opt_improve(tour, coords, max_iter=max_iter)


# ---------------------------------------------------------------------------
# 批量解码（evaluate.py 使用）
# ---------------------------------------------------------------------------

def batch_decode(
    heatmaps: torch.Tensor,     # (B, N, N)
    coords: torch.Tensor,       # (B, N, 2)
    method: str = 'greedy',
    beam_k: int = 5,
    use_2opt: bool = False,
) -> List[List[int]]:
    """对一个 batch 的热力图批量解码，返回 tour 列表。"""
    B = heatmaps.shape[0]
    tours = []
    for i in range(B):
        h = heatmaps[i].cpu()
        c = coords[i].cpu()
        if use_2opt:
            tour = decode_with_2opt(h, c, method=method, beam_k=beam_k)
        elif method == 'beam_search':
            tour = beam_search_decode(h, c, k=beam_k)
        else:
            tour = greedy_decode(h, c)
        tours.append(tour)
    return tours


# ---------------------------------------------------------------------------
# 快速单元测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch

    N = 20
    torch.manual_seed(42)
    coords = torch.rand(N, 2)

    # 构造一个接近真实的热力图（沿 tour [0,1,2,...,N-1] 有高概率）
    tour_gt = list(range(N))
    heatmap = torch.rand(N, N) * 0.1
    for k in range(N):
        i, j = tour_gt[k], tour_gt[(k + 1) % N]
        heatmap[i, j] = 0.9
        heatmap[j, i] = 0.9

    # 贪心解码
    t1 = greedy_decode(heatmap, coords)
    assert is_valid_tour(t1, N), f"Greedy invalid: {t1}"
    print(f'Greedy:      len={tour_length(t1, coords):.4f}  valid={is_valid_tour(t1, N)}')

    # Beam Search
    t2 = beam_search_decode(heatmap, coords, k=5)
    assert is_valid_tour(t2, N), f"Beam invalid: {t2}"
    print(f'Beam(k=5):   len={tour_length(t2, coords):.4f}  valid={is_valid_tour(t2, N)}')

    # 2-opt
    t3 = decode_with_2opt(heatmap, coords, method='greedy')
    assert is_valid_tour(t3, N), f"2opt invalid: {t3}"
    print(f'Greedy+2opt: len={tour_length(t3, coords):.4f}  valid={is_valid_tour(t3, N)}')

    print('decode.py OK')
