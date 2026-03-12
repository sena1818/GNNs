"""
TSP 数据集类 — 读取 DIFUSCO 兼容格式的 .txt 文件

文件格式（每行一个实例）:
    x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1
    - 坐标是 [0,1]² 范围的浮点数
    - output 是分隔符
    - tour 是 1-indexed 城市访问顺序（含回到起点）

__getitem__ 返回:
    coords:     (N, 2)   城市坐标，float32
    adj_matrix: (N, N)   最优路径的邻接矩阵，float32（{0,1}，训练标签）
    tour:       (N,)     最优路径（0-indexed），int64
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class TSPDataset(Dataset):
    def __init__(self, data_file: str):
        """
        Args:
            data_file: .txt 文件路径，每行一个 TSP 实例
        """
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]

        # 按 "output" 分割坐标和 tour
        coord_part, tour_part = line.split(' output ')

        # 解析坐标
        vals = list(map(float, coord_part.split()))
        N = len(vals) // 2
        coords = np.array(vals, dtype=np.float32).reshape(N, 2)  # (N, 2)

        # 解析 tour（1-indexed → 0-indexed，去掉末尾回到起点）
        tour_1indexed = list(map(int, tour_part.split()))
        tour = np.array(tour_1indexed[:-1], dtype=np.int64) - 1  # (N,), 0-indexed

        # 构建邻接矩阵：tour 中相邻城市对 (i,j) 设为 1，无向图所以对称
        adj = np.zeros((N, N), dtype=np.float32)
        for k in range(N):
            i = tour[k]
            j = tour[(k + 1) % N]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        return (
            torch.from_numpy(coords),   # (N, 2)
            torch.from_numpy(adj),      # (N, N)
            torch.from_numpy(tour),     # (N,)
        )


def collate_fn(batch):
    """
    标准 collate：要求同一 batch 内节点数相同（正常情况均满足）。
    混合规模训练时需要在 DataLoader 外按规模分 batch。
    """
    coords_list, adj_list, tour_list = zip(*batch)
    return (
        torch.stack(coords_list),   # (B, N, 2)
        torch.stack(adj_list),      # (B, N, N)
        torch.stack(tour_list),     # (B, N)
    )
