"""
GNN 编码器模块 — 门控图卷积 (Gated GCN) + GAT + SimpleGCN

基于 DIFUSCO (NeurIPS 2023) 的 gnn_encoder.py 改写，适配 Flow Matching：
- 时间步 t ∈ [0,1] 的连续浮点数（而非 DDPM 的整数 t/1000）
- 输出速度场 v_θ(B, N, N)（而非预测 X_0）

三种编码器可通过 encoder_type 切换，用于消融实验：
  'gated_gcn'  —— 门控图卷积，最强，参数最多（Baseline）
  'gat'        —— 图注意力网络，中等
  'gcn'        —— 简单图卷积，最轻量

输入/输出接口统一：
    encoder = GNNEncoder(n_layers=4, hidden_dim=128, encoder_type='gated_gcn')
    coords    = torch.rand(B, N, 2)
    adj_noisy = torch.rand(B, N, N)   # 插值状态 A_t
    t         = torch.rand(B)          # 时间步 ∈ [0,1]
    out       = encoder(coords, adj_noisy, t)  # (B, N, N)  速度场预测
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 工具：正弦时间步嵌入
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    将标量时间步 t ∈ [0,1] 编码为 dim 维向量（sinusoidal）。

    Args:
        t:   (B,) 时间步，连续浮点数
        dim: 输出维度，必须为偶数
    Returns:
        emb: (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    # 频率：exp(-log(10000) * i / half) for i in range(half)
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    # t: (B,) → (B, 1); freqs: (half,) → (1, half)
    args = t[:, None] * freqs[None, :]   # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return emb


# ---------------------------------------------------------------------------
# 工具：坐标 → 节点初始特征
# ---------------------------------------------------------------------------

class CoordEmbedding(nn.Module):
    """把 (x,y) 坐标映射到 hidden_dim 维节点特征。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(2, hidden_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2)
        Returns:
            h:      (B, N, hidden_dim)
        """
        return F.relu(self.proj(coords))


# ---------------------------------------------------------------------------
# 工具：边初始特征（距离 + 邻接矩阵值）
# ---------------------------------------------------------------------------

class EdgeEmbedding(nn.Module):
    """把 (距离, 当前邻接矩阵值) 映射到 hidden_dim 维边特征。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # 输入：[dist(i,j), adj_t(i,j)]，2 维
        self.proj = nn.Linear(2, hidden_dim)

    def forward(self, coords: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2)
            adj_t:  (B, N, N)  当前插值状态
        Returns:
            e:      (B, N, N, hidden_dim)
        """
        # 计算欧氏距离矩阵
        diff = coords[:, :, None, :] - coords[:, None, :, :]   # (B, N, N, 2)
        dist = diff.norm(dim=-1, keepdim=True)                  # (B, N, N, 1)

        # 拼接：[dist, adj_t]
        adj_t_expand = adj_t.unsqueeze(-1)                      # (B, N, N, 1)
        edge_feat = torch.cat([dist, adj_t_expand], dim=-1)     # (B, N, N, 2)

        return F.relu(self.proj(edge_feat))                     # (B, N, N, hidden_dim)


# ---------------------------------------------------------------------------
# Gated GCN Layer（DIFUSCO 核心层）
# ---------------------------------------------------------------------------

class GatedGCNLayer(nn.Module):
    """
    门控图卷积层，来自 Bresson & Laurent (2017) / T2T-CO。

    更新公式:
        gate_ij   = sigmoid(A * h_i + B * h_j + C * e_ij)
        h_i_new   = ReLU(U * h_i + sum_j(gate_ij * V * h_j))
        e_ij_new  = ReLU(A * h_i + B * h_j + C * e_ij)   (边特征同步更新)

    其中 h_i 是节点特征，e_ij 是边特征。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        d = hidden_dim
        # 节点更新
        self.A = nn.Linear(d, d, bias=False)
        self.B = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, d, bias=False)
        # 边更新（与 DIFUSCO 一致：共享 Ah_i + Bh_j + Ce_ij，无额外线性层）
        self.C = nn.Linear(d, d, bias=False)

        self.bn_node = nn.BatchNorm1d(d)
        self.bn_edge = nn.BatchNorm1d(d)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, d)       节点特征
            e: (B, N, N, d)    边特征
        Returns:
            h_new: (B, N, d)
            e_new: (B, N, N, d)
        """
        B, N, d = h.shape

        # 边更新前半部分：计算 gate 的原材料
        Ah_i = self.A(h)                          # (B, N, d)
        Bh_j = self.B(h)                          # (B, N, d)
        Ce_ij = self.C(e)                         # (B, N, N, d)

        # gate_ij = sigmoid(Ah_i + Bh_j + Ce_ij)
        # Ah_i: (B, N, d) → broadcast 到 (B, N, N, d)
        gate_input = Ah_i[:, :, None, :] + Bh_j[:, None, :, :] + Ce_ij
        gate_ij = torch.sigmoid(gate_input)        # (B, N, N, d)

        # 节点聚合：h_i_new = ReLU(U*h_i + sum_j(gate_ij * V*h_j))
        Vh_j = self.V(h)                           # (B, N, d)
        # gate_ij * Vh_j: (B, N, N, d)，在 j 维度求和
        agg = (gate_ij * Vh_j[:, None, :, :]).sum(dim=2)  # (B, N, d)
        h_new = F.relu(self.U(h) + agg)            # (B, N, d)

        # 边特征更新：e_ij_new = ReLU(BN(Ah_i + Bh_j + Ce_ij))
        # 与 DIFUSCO 一致：直接复用 gate_input，无额外线性变换
        # BatchNorm（需要 reshape 为 2D）
        h_new = self.bn_node(h_new.view(B * N, d)).view(B, N, d)
        e_new = F.relu(self.bn_edge(gate_input.view(B * N * N, d)).view(B, N, N, d))

        # 残差连接
        h_new = h_new + h
        e_new = e_new + e

        return h_new, e_new


# ---------------------------------------------------------------------------
# GAT Layer（轻量化变体）
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """多头图注意力层（稠密实现，适合邻接矩阵格式）。"""

    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_e = nn.Linear(hidden_dim, heads, bias=False)   # 边特征 → 注意力偏置
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, d)
            e: (B, N, N, d)
        Returns:
            h_new: (B, N, d)
            e:     (B, N, N, d)  边特征不变（GAT 不更新边特征）
        """
        B, N, d = h.shape
        H, Hd = self.heads, self.head_dim

        Q = self.W_q(h).view(B, N, H, Hd).transpose(1, 2)   # (B, H, N, Hd)
        K = self.W_k(h).view(B, N, H, Hd).transpose(1, 2)
        V = self.W_v(h).view(B, N, H, Hd).transpose(1, 2)

        # 注意力分数 + 边偏置
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Hd)   # (B, H, N, N)
        edge_bias = self.W_e(e).permute(0, 3, 1, 2)          # (B, H, N, N)
        scores = scores + edge_bias

        attn = F.softmax(scores, dim=-1)                       # (B, H, N, N)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, d)  # (B, N, d)

        h_new = F.relu(self.out_proj(out))
        h_new = self.bn(h_new.view(B * N, d)).view(B, N, d) + h

        return h_new, e


# ---------------------------------------------------------------------------
# Simple GCN Layer（最轻量变体）
# ---------------------------------------------------------------------------

class SimpleGCNLayer(nn.Module):
    """标准图卷积层（谱域 GCN，最简版本）。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, d)
            e: (B, N, N, d)   SimpleGCN 只用边的标量权重（取均值近似）
        Returns:
            h_new: (B, N, d)
            e:     (B, N, N, d)  不变
        """
        B, N, d = h.shape

        # 用边特征的 L2 范数作为邻接权重（软注意力）
        w = e.norm(dim=-1, keepdim=True)          # (B, N, N, 1)
        w = F.softmax(w, dim=2)                   # 按出边归一化

        # 加权聚合邻居：sum_j w_{ij} * h_j
        # w: (B,N,N,1), h[:,None,:,:]: (B,1,N,d) → broadcast → (B,N,N,d), sum(dim=2) → (B,N,d)
        agg = (w * h[:, None, :, :]).sum(dim=2)   # (B, N, d)

        h_new = F.relu(self.W(h + agg))
        h_new = self.bn(h_new.view(B * N, d)).view(B, N, d) + h

        return h_new, e


# ---------------------------------------------------------------------------
# 统一 GNN 编码器
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """
    多层 GNN 编码器，预测速度场 v_θ(coords, A_t, t) → (B, N, N)。

    支持三种 encoder_type:
      'gated_gcn'  Gated GCN（默认，最强）
      'gat'        Graph Attention Network
      'gcn'        Simple GCN（最轻量）
    """

    def __init__(
        self,
        n_layers: int = 12,
        hidden_dim: int = 256,
        encoder_type: str = 'gated_gcn',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        # 初始特征嵌入
        self.node_emb = CoordEmbedding(hidden_dim)
        self.edge_emb = EdgeEmbedding(hidden_dim)

        # 时间步嵌入 → 投影到 hidden_dim，加到节点特征上
        self.time_emb_dim = hidden_dim
        self.time_proj = nn.Linear(self.time_emb_dim, hidden_dim)

        # GNN 层
        layer_cls = {
            'gated_gcn': GatedGCNLayer,
            'gat': GATLayer,
            'gcn': SimpleGCNLayer,
        }[encoder_type]
        self.layers = nn.ModuleList([layer_cls(hidden_dim) for _ in range(n_layers)])

        # 输出头：将边特征映射到标量（速度场每个元素）
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,    # (B, N, 2)
        adj_t: torch.Tensor,     # (B, N, N)  插值状态 A_t
        t: torch.Tensor,         # (B,)       时间步 ∈ [0,1]
    ) -> torch.Tensor:           # (B, N, N)  预测速度场
        B, N, _ = coords.shape

        # 1. 初始节点和边特征
        h = self.node_emb(coords)           # (B, N, d)
        e = self.edge_emb(coords, adj_t)    # (B, N, N, d)

        # 2. 时间步嵌入 → 加到每个节点特征（broadcast）
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, d)
        t_feat = self.time_proj(t_emb)                       # (B, d)
        h = h + t_feat[:, None, :]                           # (B, N, d)

        # 3. 多层 GNN
        for layer in self.layers:
            h, e = layer(h, e)

        # 4. 从边特征输出速度场：每条边 (i,j) 对应一个标量
        v = self.output_head(e).squeeze(-1)   # (B, N, N)

        return v


# ---------------------------------------------------------------------------
# 快速单元测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    B, N = 2, 20
    coords = torch.rand(B, N, 2)
    adj_t = torch.rand(B, N, N)
    t = torch.rand(B)

    for enc_type in ['gated_gcn', 'gat', 'gcn']:
        model = GNNEncoder(n_layers=4, hidden_dim=128, encoder_type=enc_type)
        out = model(coords, adj_t, t)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'[{enc_type:12s}] output: {tuple(out.shape)}  params: {n_params:,}')
    print('GNNEncoder OK')
