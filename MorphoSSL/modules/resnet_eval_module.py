from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl


@dataclass
class ResNetConfig:
    arch: str
    pretrained: bool
    num_classes: int
    lr: float = 1e-4
    encoder_lr: float | None = None
    weight_decay: float = 0.01
    freeze_encoder: bool = False


class ResNetClassifier(pl.LightningModule):
    def __init__(self, cfg: ResNetConfig):
        super().__init__()
        self.save_hyperparameters({**cfg.__dict__})
        self.cfg = cfg

        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

        if cfg.arch == "resnet18":
            weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
            base = resnet18(weights=weights)
        elif cfg.arch == "resnet50":
            weights = ResNet50_Weights.DEFAULT if cfg.pretrained else None
            base = resnet50(weights=weights)
        else:
            raise ValueError("arch must be resnet18 or resnet50")

        feat_dim = int(base.fc.in_features)
        base.fc = nn.Identity()
        self.encoder = base

        for p in self.encoder.parameters():
            p.requires_grad = not bool(cfg.freeze_encoder)

        self.head = nn.Linear(feat_dim, int(cfg.num_classes))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)

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
        return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)