"""
模型代码风格参考示例。

示例目的：
1. 展示科研核心模块手写方式；
2. 展示中文 docstring；
3. 展示参数换行风格；
4. 展示 forward 中的 Tensor Shape 注释。

注意：该文件只是代码风格示例，不代表具体项目必须使用这个模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    手写多头自注意力模块。

    设计说明：
    1. 显式构造 Q、K、V，避免使用 nn.MultiheadAttention 隐藏核心计算；
    2. 将 embedding 维度拆分为多个 attention head；
    3. 使用 scaled dot-product attention 计算 token 间关系。

    输入:
        x: (B, N, D)

    输出:
        x: (B, N, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim 必须能被 num_heads 整除。"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行多头自注意力计算。"""
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)  # (B, N, D) -> (B, N, 3D)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )  # (B, N, 3D) -> (B, N, 3, H, Dh)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (B, N, 3, H, Dh) -> (3, B, H, N, Dh)

        q, k, v = qkv.unbind(dim=0)
        # 每个张量: (B, H, N, Dh)

        attn = q @ k.transpose(-2, -1)
        # (B, H, N, Dh) @ (B, H, Dh, N) -> (B, H, N, N)

        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        # (B, H, N, N) @ (B, H, N, Dh) -> (B, H, N, Dh)

        x = x.transpose(1, 2)
        # (B, H, N, Dh) -> (B, N, H, Dh)

        x = x.reshape(batch_size, num_tokens, embed_dim)
        # (B, N, H, Dh) -> (B, N, D)

        x = self.proj(x)  # (B, N, D) -> (B, N, D)
        x = self.proj_drop(x)

        return x


class PatchEmbedding(nn.Module):
    """
    图像 Patch Embedding。

    将二维图像划分为不重叠 patch，并映射到 token embedding。
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size 必须能被 patch_size 整除。"
            )

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将图像转换为 token 序列。"""
        x = self.proj(x)
        # (B, C, H, W) -> (B, D, H/P, W/P)

        x = x.flatten(2)
        # (B, D, H/P, W/P) -> (B, D, N)

        x = x.transpose(1, 2)
        # (B, D, N) -> (B, N, D)

        return x
