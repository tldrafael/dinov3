# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from dinov3.utils import cat_keep_shapes, uncat_with_shapes
from torch import Tensor, nn


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if rope is None:
            return q, k
        # Accept tuples with extra metadata (e.g., spatial shapes) and gracefully handle missing RoPE values.
        sin, cos, *rest = rope  # noqa: F841
        if sin is None or cos is None:
            return q, k
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class WindowSelfAttention(SelfAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        window_size: int | Tuple[int, int] = 14,
        device=None,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size: Tuple[int, int] = window_size

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # [B, H, N, D]

        sin = cos = None
        H = W = None
        if rope is not None:
            sin, cos, *meta = rope
            if sin is not None and cos is not None:
                q, k = self.apply_rope(q, k, (sin, cos))
            if len(meta) >= 2:
                H, W = meta[:2]
        if sin is not None:
            patch_tokens = sin.shape[-2]
        elif H is not None and W is not None:
            patch_tokens = H * W
        else:
            patch_tokens = N

        prefix_tokens = max(N - patch_tokens, 0)
        patch_tokens = N - prefix_tokens
        if H is None or W is None:
            H = int(math.sqrt(patch_tokens))
            W = patch_tokens // max(H, 1)
        window_h, window_w = self.window_size

        q_prefix = q[:, :, :prefix_tokens, :]
        k_prefix = k[:, :, :prefix_tokens, :]
        v_prefix = v[:, :, :prefix_tokens, :]

        q_patches = q[:, :, prefix_tokens:, :].reshape(B, self.num_heads, H, W, C // self.num_heads)
        k_patches = k[:, :, prefix_tokens:, :].reshape(B, self.num_heads, H, W, C // self.num_heads)
        v_patches = v[:, :, prefix_tokens:, :].reshape(B, self.num_heads, H, W, C // self.num_heads)

        pad_h = (window_h - H % window_h) % window_h
        pad_w = (window_w - W % window_w) % window_w
        q_padded = F.pad(q_patches, (0, 0, 0, pad_w, 0, pad_h))
        k_padded = F.pad(k_patches, (0, 0, 0, pad_w, 0, pad_h))
        v_padded = F.pad(v_patches, (0, 0, 0, pad_w, 0, pad_h))

        H_pad = H + pad_h
        W_pad = W + pad_w
        num_windows_h = H_pad // window_h
        num_windows_w = W_pad // window_w
        num_windows = num_windows_h * num_windows_w

        def reshape_to_windows(t: Tensor) -> Tensor:
            t = t.view(B, self.num_heads, num_windows_h, window_h, num_windows_w, window_w, -1)
            t = t.permute(0, 2, 4, 1, 3, 5, 6).reshape(B * num_windows, self.num_heads, window_h * window_w, -1)
            return t

        q_windows = reshape_to_windows(q_padded)
        k_windows = reshape_to_windows(k_padded)
        v_windows = reshape_to_windows(v_padded)

        if prefix_tokens > 0:
            k_prefix_expanded = (
                k_prefix[:, :, None, :, :]
                .expand(-1, -1, num_windows, -1, -1)
                .reshape(B * num_windows, self.num_heads, prefix_tokens, -1)
            )
            v_prefix_expanded = (
                v_prefix[:, :, None, :, :]
                .expand(-1, -1, num_windows, -1, -1)
                .reshape(B * num_windows, self.num_heads, prefix_tokens, -1)
            )
            k_windows = torch.cat([k_prefix_expanded, k_windows], dim=2)
            v_windows = torch.cat([v_prefix_expanded, v_windows], dim=2)

        attn_windows = torch.nn.functional.scaled_dot_product_attention(q_windows, k_windows, v_windows)
        attn_windows = attn_windows.view(B, num_windows_h, num_windows_w, self.num_heads, window_h, window_w, -1)
        attn_windows = attn_windows.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, self.num_heads, H_pad, W_pad, -1)
        attn_patches = attn_windows[:, :, :H, :W, :].reshape(B, self.num_heads, H * W, -1)

        if prefix_tokens > 0:
            kv_global = torch.cat(
                [k_prefix, k_patches.reshape(B, self.num_heads, H * W, -1)],
                dim=2,
            )
            vv_global = torch.cat(
                [v_prefix, v_patches.reshape(B, self.num_heads, H * W, -1)],
                dim=2,
            )
            out_prefix = torch.nn.functional.scaled_dot_product_attention(q_prefix, kv_global, vv_global)
            out = torch.cat([out_prefix, attn_patches], dim=2)
        else:
            out = attn_patches

        out = out.transpose(1, 2)
        return out.reshape([B, N, C])


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
