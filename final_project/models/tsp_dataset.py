"""
TSP 数据集类 — 读取 DIFUSCO 格式的 .txt 数据文件

参考: refs/DIFUSCO/difusco/co_datasets/tsp_graph_dataset.py

需要实现:
1. TSPDataset(Dataset):
   - __init__(data_file): 读取 .txt 文件，解析坐标和 tour
   - __len__(): 返回实例数量
   - __getitem__(idx): 返回单个实例
     * coords: (N, 2) — 城市坐标
     * adj_matrix: (N, N) — 邻接矩阵 (tour 对应边为 1)
     * tour: (N+1,) — 访问顺序 (含回到起点)

数据格式 (每行一个实例):
    x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1
    - 坐标: [0,1]² 浮点数
    - output: 分隔符
    - tour: 1-indexed 城市序列

测试:
    dataset = TSPDataset('data/tsp20_train.txt')
    coords, adj, tour = dataset[0]
    assert coords.shape == (20, 2)
    assert adj.shape == (20, 20)
    loader = DataLoader(dataset, batch_size=4)
"""

import torch
from torch.utils.data import Dataset

# TODO: 参考 DIFUSCO 的 tsp_graph_dataset.py 实现
