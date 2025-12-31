from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

@dataclass(frozen=True)
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

def make_vit_config(size: Literal["base", "large"] = "base", *, img_size: int = 224, patch_size: int = 16, in_chans: int = 3) -> ViTConfig:
    if size == "base":
        return ViTConfig(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12)
    if size == "large":
        return ViTConfig(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16)
    raise ValueError(f"Unknown ViT size: {size}")

def build_vit_encoder(cfg: ViTConfig) -> nn.Module:
    try:
        from timm.models.vision_transformer import VisionTransformer
    except Exception as exc:
        raise RuntimeError(
            "timm is required for this project. Install with: pip install timm"
        ) from exc

    model = VisionTransformer(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        num_classes=0,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        qkv_bias=cfg.qkv_bias,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    return model

def vit_num_patches(vit: nn.Module) -> int:
    if hasattr(vit, "patch_embed") and hasattr(vit.patch_embed, "num_patches"):
        return int(vit.patch_embed.num_patches)
    raise ValueError("Could not infer num_patches from ViT instance")

def vit_embed_dim(vit: nn.Module) -> int:
    if hasattr(vit, "embed_dim"):
        return int(vit.embed_dim)
    raise ValueError("Could not infer embed_dim from ViT instance")

@torch.no_grad()
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, *, cls_token: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sincos embedding")

    device = device or torch.device("cpu")
    grid_h = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid_w = torch.arange(grid_size, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)

    if cls_token:
        cls = torch.zeros([1, embed_dim], device=device, dtype=pos_embed.dtype)
        pos_embed = torch.cat([cls, pos_embed], dim=0)
    return pos_embed

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    pos = pos.reshape(-1)
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)