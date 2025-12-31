from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl


@dataclass
class EvalConfig:
    num_classes: int
    lr: float = 1e-4
    encoder_lr: float | None = None
    weight_decay: float = 0.01
    freeze_encoder: bool = False


class ViTClassifier(pl.LightningModule):
    def __init__(self, encoder: nn.Module, cfg: EvalConfig):
        super().__init__()
        self.save_hyperparameters({**cfg.__dict__})
        self.cfg = cfg
        self.encoder = encoder
        embed_dim = getattr(encoder, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Could not infer encoder embed_dim")

        self.head = nn.Linear(int(embed_dim), int(cfg.num_classes))
        self.criterion = nn.CrossEntropyLoss()

        for p in self.encoder.parameters():
            p.requires_grad = not bool(cfg.freeze_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(x)
        return self.head(feats)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            feats = self.encoder.forward_features(x)
        else:
            feats = self.encoder(x)

        if feats.dim() == 3:
            feats = feats[:, 0]
        return feats

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.cfg.freeze_encoder:
            params = [{"params": self.head.parameters(), "lr": self.cfg.lr}]
        else:
            enc_lr = self.cfg.encoder_lr if self.cfg.encoder_lr is not None else self.cfg.lr * 0.1
            params = [
                {"params": self.encoder.parameters(), "lr": enc_lr},
                {"params": self.head.parameters(), "lr": self.cfg.lr},
            ]
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return opt