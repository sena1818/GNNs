"""
神经网络辅助模块 — 对齐 DIFUSCO 官方 models/nn.py

提供:
  - GroupNorm32       32-group GroupNorm（float32 精度，兼容混合精度训练）
  - normalization()   标准归一化层工厂
  - zero_module()     将模块参数初始化为零（用于 per_layer_out 的残差初始化）
  - timestep_embedding()  正弦时间步嵌入（与 DDPM/Transformer 公式一致）

参考: DIFUSCO difusco/models/nn.py
"""

import math
import torch
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    """GroupNorm 强制在 float32 精度下计算，避免混合精度训练时的数值不稳定。"""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    """
    标准归一化层: 32-group GroupNorm。
    与 DIFUSCO 官方 normalization() 完全一致。
    """
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """
    将模块所有参数初始化为零并返回。
    用于 per_layer_out 的 Linear 层 — 训练初期残差通道不起作用，
    等价于 "先学好主干再慢慢引入辅助通道"（渐进式学习）。

    参考: DIFUSCO difusco/models/nn.py zero_module()
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    正弦时间步嵌入（与 DIFUSCO 官方完全一致）。

    公式:
        freqs = exp(-log(max_period) * i / half)  for i = 0, ..., half-1
        emb = [cos(t * freqs), sin(t * freqs)]

    注意: DIFUSCO 官方的顺序是 [cos, sin]，与标准 Transformer [sin, cos] 相反。
    这不影响性能，但需要匹配以确保权重兼容。

    Args:
        timesteps: (B,) 一维张量
        dim:       输出嵌入维度
        max_period: 控制最低频率
    Returns:
        (B, dim) 嵌入向量
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
