"""
GNN 编码器模块 — 对齐 DIFUSCO 官方 + Flow Matching 扩展

与 DIFUSCO 官方 (difusco/models/gnn_encoder.py) 的对齐项:
  ✅ PositionEmbeddingSine (normalize=True, 2π 缩放)
  ✅ ScalarEmbeddingSine (边特征正弦编码)
  ✅ time_embed_dim = hidden_dim // 2
  ✅ time_embed: Linear(d, d//2) → ReLU → Linear(d//2, d//2)
  ✅ time_embed_layers: ReLU → Linear(d//2, d) (per-layer)
  ✅ per_layer_out: LayerNorm → SiLU → zero_module(Linear) (per-layer)
  ✅ GNNLayer: mode="direct" (无内部残差) + 外部残差
  ✅ Output head: GroupNorm32(32) → ReLU → Conv2d(d, out_channels, 1)
  ✅ out_channels=2 for categorical, 1 for gaussian/FM

本项目扩展 (DIFUSCO 官方没有的):
  ✅ GATLayer — 多头注意力层 (消融实验)
  ✅ SimpleGCNLayer — 轻量 GCN (消融实验)
  ✅ encoder_type 参数控制层类型

参考:
  - DIFUSCO (NeurIPS 2023) difusco/models/gnn_encoder.py
  - Bresson & Laurent, 2017 (Gated GCN)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import normalization, zero_module, timestep_embedding


# =============================================================================
# 位置编码 (与 DIFUSCO 官方对齐)
# =============================================================================

class PositionEmbeddingSine(nn.Module):
    """
    节点坐标正弦位置编码 — 与 DIFUSCO 官方完全一致。

    将 (x, y) ∈ [0,1]² 编码为 hidden_dim 维向量:
      1. normalize=True 时: coords *= 2π
      2. 对每个坐标分量做正弦编码: sin/cos 交错排列
      3. 拼接 pos_y 和 pos_x

    输入: (B, N, 2) → 输出: (B, N, num_pos_feats*2 = hidden_dim)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x: (B, N, 2), x[:,:,0] = y_coord, x[:,:,1] = x_coord (DIFUSCO 约定)
        y_embed = x[:, :, 0]
        x_embed = x[:, :, 1]
        if self.normalize:
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, None] / dim_t              # (B, N, num_pos_feats)
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
        return pos  # (B, N, num_pos_feats * 2 = hidden_dim)


class ScalarEmbeddingSine(nn.Module):
    """
    标量正弦嵌入 — 用于边特征 (adj_t 值) 的正弦编码。
    与 DIFUSCO 官方完全一致。

    输入: (B, N, N) → 输出: (B, N, N, num_pos_feats)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x: (B, N, N) — 扩散状态 adj_t
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t           # (B, N, N, num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return pos_x  # (B, N, N, num_pos_feats)


# =============================================================================
# GNN 层 (与 DIFUSCO 官方对齐)
# =============================================================================

class GatedGCNLayer(nn.Module):
    """
    门控图卷积层 (Bresson & Laurent, 2017) — 与 DIFUSCO GNNLayer 对齐。

    边门控: gate_ij = sigmoid(A·h_i + B·h_j + C·e_ij)
    节点更新: h = norm(U·h_i + Σ_j gate_ij · V·h_j) → ReLU
    边更新:   e = norm(gate_input) → ReLU

    注意: 不含内部残差连接！残差在 GNNEncoder.forward() 中外部处理。
    这与 DIFUSCO 官方 mode="direct" 一致。
    """
    def __init__(self, hidden_dim: int, norm: str = 'layer', learn_norm: bool = True):
        super().__init__()
        d = hidden_dim
        self.A = nn.Linear(d, d, bias=True)
        self.B = nn.Linear(d, d, bias=True)
        self.C = nn.Linear(d, d, bias=True)
        self.U = nn.Linear(d, d, bias=True)
        self.V = nn.Linear(d, d, bias=True)

        # DIFUSCO 默认 norm="layer" with elementwise_affine=learn_norm
        if norm == 'layer':
            self.norm_h = nn.LayerNorm(d, elementwise_affine=learn_norm)
            self.norm_e = nn.LayerNorm(d, elementwise_affine=learn_norm)
        elif norm == 'batch':
            self.norm_h = nn.BatchNorm1d(d, affine=learn_norm)
            self.norm_e = nn.BatchNorm1d(d, affine=learn_norm)
        else:
            self.norm_h = None
            self.norm_e = None

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple:
        """
        Args:
            h: (B, N, d) 节点特征
            e: (B, N, N, d) 边特征
        Returns:
            h_out, e_out: 更新后的特征 (无残差)
        """
        B, N, d = h.shape

        Uh = self.U(h)   # (B, N, d)

        # V(h) 展开为 (B, 1, N, d) — 表示 "节点 j 的特征，对所有 i"
        Vh = self.V(h).unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, d)

        # 边门控
        gate_input = (
            self.A(h)[:, :, None, :] +  # source: (B, N, 1, d)
            self.B(h)[:, None, :, :] +  # target: (B, 1, N, d)
            self.C(e)                    # edge:   (B, N, N, d)
        )
        gates = torch.sigmoid(gate_input)   # (B, N, N, d)

        # 节点聚合: sum over j
        agg = (gates * Vh).sum(dim=2)   # (B, N, d)
        h_out = Uh + agg

        # 归一化
        if self.norm_h is not None:
            if isinstance(self.norm_h, nn.BatchNorm1d):
                h_out = self.norm_h(h_out.view(B * N, d)).view(B, N, d)
            else:
                h_out = self.norm_h(h_out)
        if self.norm_e is not None:
            if isinstance(self.norm_e, nn.BatchNorm1d):
                gate_input = self.norm_e(
                    gate_input.view(B * N * N, d)
                ).view(B, N, N, d)
            else:
                gate_input = self.norm_e(gate_input)

        # ReLU (在残差之前)
        h_out = F.relu(h_out)
        e_out = F.relu(gate_input)

        return h_out, e_out


class GATLayer(nn.Module):
    """
    多头图注意力层 — 本项目扩展，用于消融实验。
    DIFUSCO 官方没有此层。

    [IMPROVEMENT] 扩展: 多头注意力替代门控 GCN，测试不同聚合机制的效果。
    """
    def __init__(self, hidden_dim: int, heads: int = 4, **kwargs):
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

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple:
        B, N, d = h.shape
        H, Hd = self.heads, self.head_dim
        Q = self.W_q(h).view(B, N, H, Hd).transpose(1, 2)
        K = self.W_k(h).view(B, N, H, Hd).transpose(1, 2)
        V = self.W_v(h).view(B, N, H, Hd).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Hd)
        scores = scores + self.W_e(e).permute(0, 3, 1, 2)
        out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, N, d)
        h_out = F.relu(self.norm(self.out_proj(out)))
        # 不更新边特征，返回原始 e (edge features pass-through)
        return h_out, e


class SimpleGCNLayer(nn.Module):
    """
    轻量图卷积层 — 本项目扩展，用于消融实验。
    DIFUSCO 官方没有此层。

    [IMPROVEMENT] 扩展: 最简 GCN 作为下界基线。
    """
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> tuple:
        w = F.softmax(e.norm(dim=-1, keepdim=True), dim=2)
        agg = (w * h[:, None, :, :]).sum(dim=2)
        h_out = F.relu(self.norm(self.W(h + agg)))
        return h_out, e


# =============================================================================
# GNN 编码器 (与 DIFUSCO 官方对齐)
# =============================================================================

class GNNEncoder(nn.Module):
    """
    多层 GNN 编码器 — 对齐 DIFUSCO 官方 dense_forward 路径。

    架构:
      1. 节点嵌入: PositionEmbeddingSine(d//2, normalize=True) → Linear(d, d)
      2. 边嵌入:   ScalarEmbeddingSine(d) → Linear(d, d)
      3. 时间嵌入: timestep_embedding(d) → Linear(d, d//2) → ReLU → Linear(d//2, d//2)
      4. n_layers × {
            x_in, e_in = x, e
            x, e = GNNLayer(x, e)                    # mode="direct", 无内部残差
            e = e + time_embed_layers[i](t_feat)      # 时间注入到边
            x = x_in + x                              # 外部节点残差
            e = e_in + per_layer_out[i](e)             # 外部边残差 + 学习门控
         }
      5. Output: GroupNorm32(32, d) → ReLU → Conv2d(d, out_channels, 1)

    out_channels:
      - categorical (D3PM): 2 (CrossEntropyLoss 两类输出)
      - gaussian (DDPM): 1 (ε-prediction)
      - flow_matching (FM): 1 (velocity-prediction)

    encoder_type 可选:
      'gated_gcn' — 门控图卷积 (DIFUSCO 默认)
      'gat'       — 多头注意力 (本项目扩展消融)
      'gcn'       — 轻量 GCN (本项目扩展消融)
    """

    def __init__(
        self,
        n_layers: int = 12,
        hidden_dim: int = 256,
        out_channels: int = 1,
        encoder_type: str = 'gated_gcn',
        norm: str = 'layer',
        learn_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.encoder_type = encoder_type
        time_embed_dim = hidden_dim // 2

        # --- 位置编码 (DIFUSCO 一致) ---
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)

        # --- 嵌入投影 (DIFUSCO 一致) ---
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

        # --- 时间嵌入 (DIFUSCO 一致: d → d//2 → ReLU → d//2) ---
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- 每层时间投影 (DIFUSCO 一致: ReLU → d//2 → d) ---
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(time_embed_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])

        # --- 每层边输出门控 (DIFUSCO 一致: LN → SiLU → zero_init Linear) ---
        self.per_layer_out = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
                nn.SiLU(),
                zero_module(nn.Linear(hidden_dim, hidden_dim)),
            ) for _ in range(n_layers)
        ])

        # --- GNN 层 ---
        layer_map = {
            'gated_gcn': lambda: GatedGCNLayer(hidden_dim, norm=norm, learn_norm=learn_norm),
            'gat': lambda: GATLayer(hidden_dim),
            'gcn': lambda: SimpleGCNLayer(hidden_dim),
        }
        if encoder_type not in layer_map:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.layers = nn.ModuleList([layer_map[encoder_type]() for _ in range(n_layers)])

        # --- 输出头 (DIFUSCO 一致: GroupNorm32 → ReLU → Conv2d) ---
        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True),
        )

    def forward(
        self,
        coords: torch.Tensor,   # (B, N, 2) 城市坐标
        adj_t:  torch.Tensor,   # (B, N, N) 扩散状态 (已经过预处理)
        t:      torch.Tensor,   # (B,) 时间步
    ) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) 城市坐标 [0,1]²
            adj_t:  (B, N, N) 当前扩散状态
                    - categorical: {0,1} → ×2-1 → +jitter 后的连续值
                    - gaussian: 连续值
                    - FM: 连续值
            t:      (B,) 时间步（DIFUSCO 用原始整数步，FM 用 [0,1] 浮点）
        Returns:
            categorical: (B, 2, N, N) 两类 logits
            gaussian/FM: (B, 1, N, N) 单通道预测
        """
        # 1. 嵌入
        x = self.node_embed(self.pos_embed(coords))       # (B, N, d)
        e = self.edge_embed(self.edge_pos_embed(adj_t))    # (B, N, N, d)

        # 2. 时间嵌入
        time_emb = self.time_embed(
            timestep_embedding(t, self.hidden_dim)
        )  # (B, time_embed_dim)

        # 3. 图设为全连接 (与 DIFUSCO dense_forward 一致)
        # DIFUSCO: graph = torch.ones_like(graph).long()

        # 4. GNN 层 — 外部残差模式 (DIFUSCO mode="direct")
        for layer, time_layer, out_layer in zip(
            self.layers, self.time_embed_layers, self.per_layer_out
        ):
            x_in, e_in = x, e

            x, e = layer(x, e)                               # 无内部残差

            e = e + time_layer(time_emb)[:, None, None, :]    # 时间注入到边

            x = x_in + x                                      # 外部节点残差
            e = e_in + out_layer(e)                            # 外部边残差 + 学习门控

        # 5. 输出头
        # e: (B, N, N, d) → permute → (B, d, N, N) → Conv2d → (B, out_channels, N, N)
        out = self.out(e.permute(0, 3, 1, 2))
        return out


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    B, N = 2, 20
    coords = torch.rand(B, N, 2)
    adj_t = torch.rand(B, N, N)
    t = torch.rand(B) * 1000  # 模拟整数时间步

    for enc_type in ['gated_gcn', 'gat', 'gcn']:
        for oc in [1, 2]:
            model = GNNEncoder(
                n_layers=4, hidden_dim=128,
                out_channels=oc, encoder_type=enc_type,
            )
            out = model(coords, adj_t, t)
            n_params = sum(p.numel() for p in model.parameters())
            print(
                f'[{enc_type:12s} oc={oc}] '
                f'output: {tuple(out.shape)}  params: {n_params:,}'
            )
    print('GNNEncoder OK')
