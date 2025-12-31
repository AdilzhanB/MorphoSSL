from __future__ import annotations

import argparse
import os
import sys
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pretrain_module import MAEPretrainModule, PretrainConfig
from eval_module import ViTClassifier, EvalConfig
from vision_transformer import build_vit_encoder, make_vit_config
from visualizations import mae_reconstruction_triplet
from visualizations import plot_mae_single_image_reconstruction


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print_kv(title: str, kv: Dict[str, object]) -> None:
    print(f"\n[{_now()}] {title}")
    for k, v in kv.items():
        print(f"  - {k}: {v}")


def _env_summary() -> Dict[str, object]:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _dataloader_common_kwargs(num_workers: int) -> Dict[str, object]:
    return {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": bool(num_workers and num_workers > 0),
    }


def _build_trainer(*, out_dir: str, epochs: int, precision: str, has_val: bool) -> pl.Trainer:
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    monitor_key = "val/loss" if has_val else "train/loss_epoch"
    filename_metric = monitor_key.replace("/", "_")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:03d}-{step:06d}" + f"-{{{filename_metric}:.4f}}",
            monitor=monitor_key,
            mode="min",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        _EpochMetricsAndPlotsCallback(out_dir=out_dir, max_epochs=epochs, has_val=has_val),
    ]

    return pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
        default_root_dir=out_dir,
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_checkpointing=True,
        callbacks=callbacks,
    )


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return None


def _pick_metric(metrics: Dict[str, object], names: List[str]) -> Optional[float]:
    for n in names:
        if n in metrics:
            v = _to_float(metrics[n])
            if v is not None:
                return v
    return None


class _EpochMetricsAndPlotsCallback(pl.Callback):
    def __init__(self, *, out_dir: str, max_epochs: int, has_val: bool):
        super().__init__()
        self.out_dir = out_dir
        self.max_epochs = max_epochs
        self.has_val = has_val

        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []

        self._last_train_loss: Optional[float] = None
        self._last_train_acc: Optional[float] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        e = int(trainer.current_epoch)
        print(f"\n[{_now()}] Epoch {e}/{self.max_epochs - 1}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        t_loss = _pick_metric(metrics, ["train/loss_epoch", "train/loss"])  
        t_acc = _pick_metric(metrics, ["train/acc_epoch", "train/acc"])

        self._last_train_loss = t_loss
        self._last_train_acc = t_acc

        if t_loss is not None:
            self.train_loss.append(t_loss)
        if t_acc is not None:
            self.train_acc.append(t_acc)

        if not self.has_val:
            parts = []
            if t_loss is not None:
                parts.append(f"train_loss={t_loss:.6f}")
            if t_acc is not None:
                parts.append(f"train_acc={t_acc:.4f}")
            if parts:
                print(f"[{_now()}] " + "  ".join(parts))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        v_loss = _pick_metric(metrics, ["val/loss", "val/loss_epoch"])
        v_acc = _pick_metric(metrics, ["val/acc", "val/acc_epoch"])

        if v_loss is not None:
            self.val_loss.append(v_loss)
        if v_acc is not None:
            self.val_acc.append(v_acc)

        parts = []
        if self._last_train_loss is not None:
            parts.append(f"train_loss={self._last_train_loss:.6f}")
        if self._last_train_acc is not None:
            parts.append(f"train_acc={self._last_train_acc:.4f}")
        if v_loss is not None:
            parts.append(f"val_loss={v_loss:.6f}")
        if v_acc is not None:
            parts.append(f"val_acc={v_acc:.4f}")
        if parts:
            print(f"[{_now()}] " + "  ".join(parts))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import os
        import csv

        os.makedirs(self.out_dir, exist_ok=True)

        try:
            rows = max(len(self.train_loss), len(self.val_loss), len(self.train_acc), len(self.val_acc))
            csv_path = os.path.join(self.out_dir, "metrics.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
                for e in range(rows):
                    writer.writerow(
                        [
                            e,
                            self.train_loss[e] if e < len(self.train_loss) else "",
                            self.val_loss[e] if e < len(self.val_loss) else "",
                            self.train_acc[e] if e < len(self.train_acc) else "",
                            self.val_acc[e] if e < len(self.val_acc) else "",
                        ]
                    )
            print(f"[{_now()}] Saved metrics CSV: {csv_path}")
        except Exception as exc:
            print(f"[{_now()}] Could not write metrics.csv: {exc}")

        if self.train_loss or self.val_loss:
            try:
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(7, 4))
                ax = fig.add_subplot(1, 1, 1)
                if self.train_loss:
                    ax.plot(range(len(self.train_loss)), self.train_loss, label="train_loss")
                if self.val_loss:
                    ax.plot(range(len(self.val_loss)), self.val_loss, label="val_loss")
                ax.set_title("Loss vs Epoch")
                ax.set_xlabel("epoch")
                ax.set_ylabel("loss")
                ax.grid(True, alpha=0.3)
                ax.legend()
                out_path = os.path.join(self.out_dir, "loss_curve.png")
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[{_now()}] Saved loss plot: {out_path}")
            except Exception as exc:
                print(f"[{_now()}] Could not write loss plot: {exc}")

        if self.train_acc or self.val_acc:
            try:
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(7, 4))
                ax = fig.add_subplot(1, 1, 1)
                if self.train_acc:
                    ax.plot(range(len(self.train_acc)), self.train_acc, label="train_acc")
                if self.val_acc:
                    ax.plot(range(len(self.val_acc)), self.val_acc, label="val_acc")
                ax.set_title("Accuracy vs Epoch")
                ax.set_xlabel("epoch")
                ax.set_ylabel("accuracy")
                ax.grid(True, alpha=0.3)
                ax.legend()
                out_path = os.path.join(self.out_dir, "acc_curve.png")
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"[{_now()}] Saved accuracy plot: {out_path}")
            except Exception as exc:
                print(f"[{_now()}] Could not write accuracy plot: {exc}")


def _summarize_labeled_dataset(ds: "LabeledImageFolder") -> Dict[str, object]:
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    sample_preview = []
    for i in range(min(3, len(ds.samples))):
        p, y = ds.samples[i]
        sample_preview.append({"path": p, "label": int(y), "class": idx_to_class.get(int(y), "?")})
    return {
        "root": ds.root,
        "num_samples": len(ds),
        "num_classes": len(ds.class_to_idx),
        "class_to_idx": ds.class_to_idx,
        "sample_preview": sample_preview,
    }


def _summarize_unlabeled_dataset(ds: "UnlabeledImageFolder") -> Dict[str, object]:
    sample_preview = ds.paths[:3]
    return {
        "root": ds.root,
        "num_samples": len(ds),
        "sample_preview": sample_preview,
    }


class UnlabeledImageFolder(Dataset):
    """Recursively loads images from a folder (no labels required)."""

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMG_EXTS:
                    self.paths.append(os.path.join(dirpath, fn))
        if len(self.paths) == 0:
            raise ValueError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


class LabeledImageFolder(Dataset):
    """Labeled folder format:

    root/
      class_a/xxx.png
      class_b/yyy.png

    Returns (image_tensor, class_index)
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        class_names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        class_names.sort()
        if not class_names:
            raise ValueError(f"No class subfolders found under: {root}")
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        self.samples: List[Tuple[str, int]] = []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            for dirpath, _, filenames in os.walk(cls_dir):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in IMG_EXTS:
                        self.samples.append((os.path.join(dirpath, fn), self.class_to_idx[cls]))
        if len(self.samples) == 0:
            raise ValueError(f"No images found under: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)


def make_transforms(img_size: int, *, mode: str):
    if mode == "pretrain":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
            ]
        )
    if mode == "finetune":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ]
        )
    if mode == "eval":
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )
    raise ValueError("Unknown mode")


def cmd_pretrain(args) -> None:
    _print_kv("Environment", _env_summary())
    cfg = PretrainConfig(
        backbone=args.backbone,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    _print_kv(
        "Pretrain config",
        {
            "backbone": cfg.backbone,
            "img_size": cfg.img_size,
            "patch_size": cfg.patch_size,
            "mask_ratio": cfg.mask_ratio,
            "norm_pix_loss": cfg.norm_pix_loss,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "precision": args.precision,
            "num_workers": args.num_workers,
            "out_dir": args.out_dir,
        },
    )

    dm_train = UnlabeledImageFolder(args.train_dir, transform=make_transforms(args.img_size, mode="pretrain"))
    _print_kv("Pretrain dataset (train)", _summarize_unlabeled_dataset(dm_train))
    train_loader = DataLoader(
        dm_train,
        batch_size=args.batch_size,
        shuffle=True,
        **_dataloader_common_kwargs(args.num_workers),
    )

    val_loader = None
    if args.val_dir:
        dm_val = UnlabeledImageFolder(args.val_dir, transform=make_transforms(args.img_size, mode="eval"))
        _print_kv("Pretrain dataset (val)", _summarize_unlabeled_dataset(dm_val))
        val_loader = DataLoader(
            dm_val,
            batch_size=args.batch_size,
            shuffle=False,
            **_dataloader_common_kwargs(args.num_workers),
        )

    module = MAEPretrainModule(cfg)
    trainer = _build_trainer(out_dir=args.out_dir, epochs=args.epochs, precision=args.precision, has_val=val_loader is not None)

    print(f"\n[{_now()}] Starting MAE pretraining...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_cb = next((c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)), None)
    best_path = ckpt_cb.best_model_path if ckpt_cb is not None else ""
    if best_path and os.path.exists(best_path):
        ckpt_path = os.path.join(args.out_dir, "mae_pretrained.ckpt")
        shutil.copy2(best_path, ckpt_path)
        print(f"[{_now()}] Selected best checkpoint: {best_path}")
        print(f"[{_now()}] Saved best-as-default: {ckpt_path}")
    else:
        ckpt_path = os.path.join(args.out_dir, "mae_pretrained.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"[{_now()}] Saved checkpoint: {ckpt_path}")
    print(f"[{_now()}] Lightning checkpoints: {os.path.join(args.out_dir, 'checkpoints')}")


def _load_encoder_from_mae_ckpt(ckpt_path: str, *, backbone: str, img_size: int, patch_size: int) -> torch.nn.Module:
    if backbone == "vit_base":
        vit_cfg = make_vit_config("base", img_size=img_size, patch_size=patch_size, in_chans=3)
    elif backbone == "vit_large":
        vit_cfg = make_vit_config("large", img_size=img_size, patch_size=patch_size, in_chans=3)
    else:
        raise ValueError("backbone must be vit_base or vit_large")

    encoder = build_vit_encoder(vit_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    
    enc_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            kk = k[len("model.") :]
        else:
            kk = k

        if kk.startswith("patch_embed") or kk.startswith("blocks") or kk.startswith("norm") or kk.startswith("pos_embed") or kk.startswith("cls_token"):
            enc_state[kk] = v

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if unexpected:
        print("Unexpected keys:", unexpected)
    if missing:
        print("Missing keys (often OK):", missing)

    return encoder


def cmd_finetune(args) -> None:
    _print_kv("Environment", _env_summary())
    train_ds = LabeledImageFolder(args.train_dir, transform=make_transforms(args.img_size, mode="finetune"))
    val_ds = LabeledImageFolder(args.val_dir, transform=make_transforms(args.img_size, mode="eval"))

    _print_kv("Finetune dataset (train)", _summarize_labeled_dataset(train_ds))
    _print_kv("Finetune dataset (val)", _summarize_labeled_dataset(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **_dataloader_common_kwargs(args.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_dataloader_common_kwargs(args.num_workers),
    )

    num_classes = len(train_ds.class_to_idx)

    _print_kv(
        "Finetune config",
        {
            "init": args.init,
            "mae_ckpt": args.mae_ckpt if args.init == 'mae' else None,
            "backbone": args.backbone,
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "num_classes": num_classes,
            "freeze_encoder": bool(args.freeze_encoder),
            "encoder_lr": args.encoder_lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "num_workers": args.num_workers,
            "out_dir": args.out_dir,
        },
    )

    if args.init == "mae":
        if not args.mae_ckpt:
            raise ValueError("--mae-ckpt is required when --init mae")
        print(f"[{_now()}] Loading encoder weights from MAE checkpoint: {args.mae_ckpt}")
        encoder = _load_encoder_from_mae_ckpt(args.mae_ckpt, backbone=args.backbone, img_size=args.img_size, patch_size=args.patch_size)
    elif args.init == "imagenet":
        import timm

        model_name = "vit_base_patch16_224" if args.backbone == "vit_base" else "vit_large_patch16_224"
        print(f"[{_now()}] Loading ImageNet pretrained encoder via timm: {model_name}")
        encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
    else:
        raise ValueError("--init must be mae or imagenet")

    module = ViTClassifier(
        encoder=encoder,
        cfg=EvalConfig(
            num_classes=num_classes,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            weight_decay=args.weight_decay,
            freeze_encoder=bool(args.freeze_encoder),
        ),
    )

    trainer = _build_trainer(out_dir=args.out_dir, epochs=args.epochs, precision=args.precision, has_val=True)

    print(f"\n[{_now()}] Starting fine-tuning...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "finetuned.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[{_now()}] Saved checkpoint: {ckpt_path}")
    print(f"[{_now()}] Lightning checkpoints: {os.path.join(args.out_dir, 'checkpoints')}")


def cmd_finetune_resnet(args) -> None:
    _print_kv("Environment", _env_summary())
    train_ds = LabeledImageFolder(args.train_dir, transform=make_transforms(args.img_size, mode="finetune"))
    val_ds = LabeledImageFolder(args.val_dir, transform=make_transforms(args.img_size, mode="eval"))

    _print_kv("Finetune (ResNet) dataset (train)", _summarize_labeled_dataset(train_ds))
    _print_kv("Finetune (ResNet) dataset (val)", _summarize_labeled_dataset(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **_dataloader_common_kwargs(args.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_dataloader_common_kwargs(args.num_workers),
    )

    num_classes = len(train_ds.class_to_idx)
    _print_kv(
        "Finetune (ResNet) config",
        {
            "arch": args.arch,
            "pretrained": bool(args.pretrained),
            "freeze_encoder": bool(args.freeze_encoder),
            "encoder_lr": args.encoder_lr,
            "img_size": args.img_size,
            "num_classes": num_classes,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "num_workers": args.num_workers,
            "out_dir": args.out_dir,
        },
    )

    from resnet_eval_module import ResNetClassifier, ResNetConfig

    module = ResNetClassifier(
        cfg=ResNetConfig(
            arch=args.arch,
            pretrained=bool(args.pretrained),
            num_classes=num_classes,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            weight_decay=args.weight_decay,
            freeze_encoder=bool(args.freeze_encoder),
        )
    )

    trainer = _build_trainer(out_dir=args.out_dir, epochs=args.epochs, precision=args.precision, has_val=True)
    print(f"\n[{_now()}] Starting ResNet linear-probe...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "finetuned.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[{_now()}] Saved checkpoint: {ckpt_path}")
    print(f"[{_now()}] Lightning checkpoints: {os.path.join(args.out_dir, 'checkpoints')}")


def cmd_visualize(args) -> None:
    from pretrain_module import MAEPretrainModule, PretrainConfig

    cfg = PretrainConfig(
        backbone=args.backbone,
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    )
    module = MAEPretrainModule(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device)

    img = Image.open(args.image_path).convert("RGB")

    fig = plot_mae_single_image_reconstruction(
        mae_model=module.model,
        image=img,
        device=device,
        save_path=args.out_path,
        title=args.title,
    )

    import matplotlib.pyplot as plt

    plt.show()


def cmd_reconstruct(args) -> None:
    from pretrain_module import MAEPretrainModule, PretrainConfig

    cfg = PretrainConfig(backbone=args.backbone, img_size=args.img_size, patch_size=args.patch_size, mask_ratio=args.mask_ratio)
    module = MAEPretrainModule(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"], strict=True)

    ds = UnlabeledImageFolder(args.images_dir, transform=make_transforms(args.img_size, mode="eval"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.model.to(device)

    _print_kv(
        "Reconstruction run",
        {
            "images_dir": args.images_dir,
            "num_images": len(ds),
            "ckpt": args.ckpt,
            "device": str(device),
            "mask_ratio": args.mask_ratio,
            "out_dir": args.out_dir,
        },
    )

    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    for i, imgs in enumerate(loader):
        if i >= args.max_batches:
            break
        orig, masked, recon = mae_reconstruction_triplet(mae_model=module.model, imgs=imgs, device=device)
        for b in range(min(orig.shape[0], args.max_items)):
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            for ax, im, title in zip(
                axs,
                [orig[b], masked[b], recon[b]],
                ["original", "masked", "recon"],
            ):
                ax.imshow(im.permute(1, 2, 0).detach().cpu().clamp(0, 1))
                ax.set_title(title)
                ax.axis("off")
            out = os.path.join(args.out_dir, f"recon_{i}_{b}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[{_now()}] Wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_pre = sp.add_parser("pretrain", help="MAE self-supervised pretraining")
    p_pre.add_argument("--train-dir", required=True)
    p_pre.add_argument("--val-dir")
    p_pre.add_argument("--out-dir", default="outputs/pretrain")
    p_pre.add_argument("--backbone", choices=["vit_base", "vit_large"], default="vit_large")
    p_pre.add_argument("--img-size", type=int, default=224)
    p_pre.add_argument("--patch-size", type=int, default=16)
    p_pre.add_argument("--mask-ratio", type=float, default=0.75)
    p_pre.add_argument("--norm-pix-loss", action="store_true")
    p_pre.add_argument("--batch-size", type=int, default=64)
    p_pre.add_argument("--epochs", type=int, default=50)
    p_pre.add_argument("--lr", type=float, default=1.5e-4)
    p_pre.add_argument("--weight-decay", type=float, default=0.05)
    p_pre.add_argument("--num-workers", type=int, default=4)
    p_pre.add_argument("--precision", default="16-mixed")
    p_pre.set_defaults(func=cmd_pretrain)

    p_ft = sp.add_parser("finetune", help="Fine-tune / linear-probe on labeled rare-disease data")
    p_ft.add_argument("--train-dir", required=True)
    p_ft.add_argument("--val-dir", required=True)
    p_ft.add_argument("--out-dir", default="outputs/finetune")
    p_ft.add_argument("--backbone", choices=["vit_base", "vit_large"], default="vit_base")
    p_ft.add_argument("--img-size", type=int, default=224)
    p_ft.add_argument("--patch-size", type=int, default=16)
    p_ft.add_argument("--batch-size", type=int, default=32)
    p_ft.add_argument("--epochs", type=int, default=100)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--encoder-lr", type=float)
    p_ft.add_argument("--weight-decay", type=float, default=0.01)
    p_ft.add_argument("--num-workers", type=int, default=4)
    p_ft.add_argument("--precision", default="16-mixed")
    p_ft.add_argument("--init", choices=["mae", "imagenet"], default="mae")
    p_ft.add_argument("--mae-ckpt", help="Path to mae_pretrained.ckpt")
    p_ft.add_argument("--freeze-encoder", action="store_true")
    p_ft.set_defaults(func=cmd_finetune)

    p_rn = sp.add_parser("finetune_resnet", help="ResNet baseline (frozen encoder + linear head)")
    p_rn.add_argument("--train-dir", required=True)
    p_rn.add_argument("--val-dir", required=True)
    p_rn.add_argument("--out-dir", default="outputs/finetune_resnet")
    p_rn.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet50")
    p_rn.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p_rn.add_argument("--freeze-encoder", action="store_true")
    p_rn.add_argument("--img-size", type=int, default=224)
    p_rn.add_argument("--batch-size", type=int, default=32)
    p_rn.add_argument("--epochs", type=int, default=100)
    p_rn.add_argument("--lr", type=float, default=1e-4)
    p_rn.add_argument("--encoder-lr", type=float)
    p_rn.add_argument("--weight-decay", type=float, default=0.01)
    p_rn.add_argument("--num-workers", type=int, default=4)
    p_rn.add_argument("--precision", default="16-mixed")
    p_rn.set_defaults(func=cmd_finetune_resnet)

    p_rec = sp.add_parser("reconstruct", help="Save MAE reconstructions for a folder")
    p_rec.add_argument("--images-dir", required=True)
    p_rec.add_argument("--ckpt", required=True)
    p_rec.add_argument("--out-dir", default="outputs/recon")
    p_rec.add_argument("--backbone", choices=["vit_base", "vit_large"], default="vit_base")
    p_rec.add_argument("--img-size", type=int, default=224)
    p_rec.add_argument("--patch-size", type=int, default=16)
    p_rec.add_argument("--mask-ratio", type=float, default=0.75)
    p_rec.add_argument("--batch-size", type=int, default=8)
    p_rec.add_argument("--num-workers", type=int, default=2)
    p_rec.add_argument("--max-batches", type=int, default=2)
    p_rec.add_argument("--max-items", type=int, default=4)
    p_rec.set_defaults(func=cmd_reconstruct)

    p_viz = sp.add_parser("visualize", help="Show MAE: original vs corrupted vs reconstruction for one image")
    p_viz.add_argument("--image-path", required=True)
    p_viz.add_argument("--ckpt", required=True)
    p_viz.add_argument("--out-path")
    p_viz.add_argument("--title")
    p_viz.add_argument("--backbone", choices=["vit_base", "vit_large"], default="vit_base")
    p_viz.add_argument("--img-size", type=int, default=224)
    p_viz.add_argument("--patch-size", type=int, default=16)
    p_viz.add_argument("--mask-ratio", type=float, default=0.75)
    p_viz.set_defaults(func=cmd_visualize)

    return p


def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 1 and argv[0].endswith(".json") and os.path.exists(argv[0]):
        build_parser().print_help()
        return

    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()