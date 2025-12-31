from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl

from model_mae import MaskedAutoencoderViT, build_mae_vit_base, build_mae_vit_large


@dataclass
class PretrainConfig:
    backbone: str = "vit_base"  
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    mask_ratio: float = 0.75
    norm_pix_loss: bool = False

    lr: float = 1.5e-4
    weight_decay: float = 0.05


class MAEPretrainModule(pl.LightningModule):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg

        if cfg.backbone == "vit_base":
            self.model: MaskedAutoencoderViT = build_mae_vit_base(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_chans=cfg.in_chans,
                mask_ratio=cfg.mask_ratio,
                norm_pix_loss=cfg.norm_pix_loss,
            )
        elif cfg.backbone == "vit_large":
            self.model = build_mae_vit_large(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_chans=cfg.in_chans,
                mask_ratio=cfg.mask_ratio,
                norm_pix_loss=cfg.norm_pix_loss,
            )
        else:
            raise ValueError("backbone must be vit_base or vit_large")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        loss, _, _ = self.model(imgs)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_epoch", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        loss, _, _ = self.model(imgs)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return opt

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        keys_to_drop = [k for k in sd.keys() if k.startswith("decoder_") or k in {"mask_token"}]
        for k in keys_to_drop:
            sd.pop(k, None)
        return sd