"""
GNN 编码器模块 — 门控图卷积 (Gated GCN) + GAT + SimpleGCN

参考: DIFUSCO (NeurIPS 2023), Edward-Sun/DIFUSCO

模块结构:
  sinusoidal_embedding  — 标量 → 正弦位置编码向量
  CoordEmbedding        — 节点坐标 (x,y) → sinusoidal → Linear → ReLU
  EdgeEmbedding         — 边特征 [dist, sinusoidal(adj_t)] → Linear → ReLU
  GatedGCNLayer         — 门控图卷积: sigmoid 门控 + LayerNorm + 残差
  GATLayer              — 多头图注意力: scaled dot-product + 边偏置 + 残差
  SimpleGCNLayer        — 轻量图卷积: 边范数软注意力聚合 + 残差
  GNNEncoder            — 完整编码器: 嵌入 + n 层 GNN + 时间逐层注入 + 输出头
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    将标量 t ∈ [0,1] 编码为 dim 维正弦向量。

    与 Transformer 位置编码相同的公式:
        emb[2i]   = sin(t * 10000^(-i / half))
        emb[2i+1] = cos(t * 10000^(-i / half))

    Args:
        t:   (B,) 时间步，浮点数
        dim: 输出维度，必须为偶数
    Returns:
        emb: (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CoordEmbedding(nn.Module):
    """
    节点坐标嵌入。

    将 (x, y) 各自做 sinusoidal 编码后拼接，再经 Linear + ReLU 投影到 hidden_dim。
    输入: coords (B, N, 2)，值域 [0,1]
    输出: (B, N, hidden_dim)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        assert hidden_dim % 4 == 0
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 4
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=coords.device) / half
        )
        x_args = coords[..., 0:1] * freqs          # (B, N, half)
        y_args = coords[..., 1:2] * freqs          # (B, N, half)
        emb = torch.cat([
            torch.sin(x_args), torch.cos(x_args),
            torch.sin(y_args), torch.cos(y_args),
        ], dim=-1)                                  # (B, N, hidden_dim)
        return F.relu(self.proj(emb))


class EdgeEmbedding(nn.Module):
    """
    边特征嵌入。

    对每条边 (i,j) 拼接两部分特征:
      - dist(i,j): 欧氏距离标量，直接作为线性特征
      - sinusoidal(adj_t[i,j]): 当前扩散状态做正弦编码，捕捉连续/离散信号
    拼接后经 Linear(hidden_dim+1, hidden_dim) + ReLU 投影。
    输入: coords (B, N, 2), adj_t (B, N, N)
    输出: (B, N, N, hidden_dim)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, coords: torch.Tensor, adj_t: torch.Tensor) -> torch.Tensor:
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = diff.norm(dim=-1, keepdim=True)          # (B, N, N, 1)
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=coords.device) / half
        )
        adj_args = adj_t.unsqueeze(-1) * freqs          # (B, N, N, half)
        adj_emb = torch.cat([torch.sin(adj_args), torch.cos(adj_args)], dim=-1)  # (B,N,N,hidden_dim)
        edge_feat = torch.cat([dist, adj_emb], dim=-1)  # (B, N, N, hidden_dim+1)
        return F.relu(self.proj(edge_feat))


class GatedGCNLayer(nn.Module):
    """
    门控图卷积层 (Bresson & Laurent, 2017)。

    边门控: gate_ij = sigmoid(A·h_i + B·h_j + C·e_ij)
    节点更新: h_i = ReLU(LN(U·h_i + Σ_j gate_ij · V·h_j)) + h_i
    边更新:   e_ij = ReLU(LN(gate_input_ij)) + e_ij

    所有线性层带 bias，归一化使用 LayerNorm，更新顺序为 linear → LN → ReLU → 残差。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        d = hidden_dim
        self.A = nn.Linear(d, d, bias=True)
        self.B = nn.Linear(d, d, bias=True)
        self.C = nn.Linear(d, d, bias=True)
        self.U = nn.Linear(d, d, bias=True)
        self.V = nn.Linear(d, d, bias=True)
        self.norm_h = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, d = h.shape

        gate_input = (
            self.A(h)[:, :, None, :] +
            self.B(h)[:, None, :, :] +
            self.C(e)
        )
        gate_ij = torch.sigmoid(gate_input)

        # 节点: linear → LN → ReLU → 残差
        agg = (gate_ij * self.V(h)[:, None, :, :]).sum(dim=2)
        h_new = F.relu(self.norm_h(self.U(h) + agg)) + h

        # 边: linear → LN → ReLU → 残差
        e_new = F.relu(self.norm_e(gate_input)) + e

        return h_new, e_new


class GATLayer(nn.Module):
    """
    多头图注意力层（稠密邻接矩阵实现）。

    节点更新: scaled dot-product attention，注意力分数加边特征线性偏置
      scores_ij = (Q_i · K_j) / sqrt(d_head) + W_e · e_ij
      h_i = ReLU(LN(out_proj(Σ_j softmax(scores)_ij · V_j))) + h_i
    边特征透传不更新。
    用于消融实验，对比 GatedGCN vs GAT。
    """
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, heads, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, d = h.shape
        H, Hd = self.heads, self.head_dim
        Q = self.W_q(h).view(B, N, H, Hd).transpose(1, 2)
        K = self.W_k(h).view(B, N, H, Hd).transpose(1, 2)
        V = self.W_v(h).view(B, N, H, Hd).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Hd)
        scores = scores + self.W_e(e).permute(0, 3, 1, 2)
        out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, N, d)
        h_new = F.relu(self.norm(self.out_proj(out))) + h
        return h_new, e


class SimpleGCNLayer(nn.Module):
    """
    最轻量图卷积层，用于消融实验对比 GatedGCN vs SimpleGCN。

    节点更新: h_i = relu(LN(W * (h_i + sum_j(w_ij * h_j)))) + h_i
    聚合权重: w_ij = softmax(||e_ij||)  (边特征 L2 范数作软注意力)
    边特征: 不更新，直接透传
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = F.softmax(e.norm(dim=-1, keepdim=True), dim=2)
        agg = (w * h[:, None, :, :]).sum(dim=2)
        h_new = F.relu(self.norm(self.W(h + agg))) + h
        return h_new, e


class GNNEncoder(nn.Module):
    """
    多层 GNN 编码器，将 TSP 图编码为边概率热图。

    输入:
        coords  (B, N, 2)   城市坐标，值域 [0,1]
        adj_t   (B, N, N)   当前扩散状态（FM: 连续插值；D3PM: {0,1}；DDPM: 连续）
        t       (B,)        归一化时间步，值域 (0,1]
    输出:
        (B, N, N)   逐边预测值（速度场 / 噪声 / x0 logits，由上层 diffusion 模型解释）

    架构（5 步）:
        1. 节点嵌入: CoordEmbedding → (B, N, d)
             sinusoidal(x) ‖ sinusoidal(y) → Linear(d,d) → ReLU
        2. 边嵌入: EdgeEmbedding → (B, N, N, d)
             [dist, sinusoidal(adj_t)] → Linear(d+1, d) → ReLU
        3. 时间嵌入: sinusoidal(t) → Linear(d,d) → ReLU → Linear(d,d) → (B, d)
        4. n_layers × GNN:
             h, e = GNNLayer(h, e)
             e = e + time_proj_i(t_feat)    ← 每层将时间特征加到边上
        5. 输出头: LN → ReLU → Linear(d,1) → squeeze → (B, N, N)

    encoder_type 可选（消融实验用）:
        'gated_gcn'  门控图卷积，节点与边均更新（默认）
        'gat'        多头图注意力，仅更新节点
        'gcn'        轻量图卷积，仅更新节点
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

        self.node_emb = CoordEmbedding(hidden_dim)
        self.edge_emb = EdgeEmbedding(hidden_dim)

        # 时间嵌入: 2 层 MLP，sinusoidal → Linear → ReLU → Linear
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 每层对应一个时间投影（注入到边特征）
        self.time_proj_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        layer_cls = {'gated_gcn': GatedGCNLayer, 'gat': GATLayer, 'gcn': SimpleGCNLayer}[encoder_type]
        self.layers = nn.ModuleList([layer_cls(hidden_dim) for _ in range(n_layers)])

        # 输出头: LN → ReLU → Linear(d,1) → per-edge 标量
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        coords: torch.Tensor,   # (B, N, 2)  城市坐标
        adj_t:  torch.Tensor,   # (B, N, N)  当前扩散状态
        t:      torch.Tensor,   # (B,)       归一化时间步 ∈ (0,1]
    ) -> torch.Tensor:          # (B, N, N)  速度场 / 噪声 / logits
        B, N, _ = coords.shape

        # 1. 初始特征
        h = self.node_emb(coords)           # (B, N, d)
        e = self.edge_emb(coords, adj_t)    # (B, N, N, d)

        # 2. 时间嵌入（2 层 MLP）
        t_feat = self.time_embed(sinusoidal_embedding(t, self.hidden_dim))  # (B, d)

        # 3. 多层 GNN，每层后注入时间到边特征
        for i, layer in enumerate(self.layers):
            h, e = layer(h, e)
            e = e + self.time_proj_layers[i](t_feat)[:, None, None, :]  # (B, N, N, d)

        # 4. 输出
        return self.output_head(e).squeeze(-1)  # (B, N, N)


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
