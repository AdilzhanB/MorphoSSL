from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from vision_transformer import get_2d_sincos_pos_embed


@dataclass
class MAEConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True

    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    decoder_mlp_ratio: float = 4.0

    mask_ratio: float = 0.75
    norm_pix_loss: bool = False


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg

        try:
            from timm.models.vision_transformer import Block, PatchEmbed
        except Exception as exc:  
            raise RuntimeError("timm is required. Install with: pip install timm") from exc

        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = int(num_patches)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, cfg.embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, cfg.decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=cfg.decoder_embed_dim,
                    num_heads=cfg.decoder_num_heads,
                    mlp_ratio=cfg.decoder_mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(cfg.decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_embed_dim, cfg.patch_size * cfg.patch_size * cfg.in_chans, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        grid_size = int((self.cfg.img_size // self.cfg.patch_size))
        pos_embed = get_2d_sincos_pos_embed(self.cfg.embed_dim, grid_size, cls_token=True)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        dec_pos_embed = get_2d_sincos_pos_embed(self.cfg.decoder_embed_dim, grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        n, c, h, w = imgs.shape
        if h != w or h % p != 0:
            raise ValueError(f"Expected square images divisible by patch_size={p}. Got {h}x{w}.")
        gs = h // p
        x = imgs.reshape(n, c, gs, p, gs, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  
        x = x.reshape(n, gs * gs, p * p * c)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        n, l, ppcc = x.shape
        c = self.cfg.in_chans
        if ppcc != p * p * c:
            raise ValueError("Patch dimension mismatch")
        gs = int(l ** 0.5)
        if gs * gs != l:
            raise ValueError("L must be a square number")
        x = x.reshape(n, gs, gs, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(n, c, gs * p, gs * p)
        return imgs

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        x_vis = x[:, 1:, :]
        n, l_vis, d = x_vis.shape
        l = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(n, l - l_vis, 1)
        x_ = torch.cat([x_vis, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d))

        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)

        if self.cfg.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    def forward(self, imgs: torch.Tensor, *, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_ratio = self.cfg.mask_ratio if mask_ratio is None else float(mask_ratio)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) 
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def build_mae_vit_base(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    mask_ratio: float = 0.75,
    norm_pix_loss: bool = False,
) -> MaskedAutoencoderViT:
    cfg = MAEConfig(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=mask_ratio,
        norm_pix_loss=norm_pix_loss,
    )
    return MaskedAutoencoderViT(cfg)


def build_mae_vit_large(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    mask_ratio: float = 0.75,
    norm_pix_loss: bool = False,
) -> MaskedAutoencoderViT:
    cfg = MAEConfig(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=mask_ratio,
        norm_pix_loss=norm_pix_loss,
    )
    return MaskedAutoencoderViT(cfg)