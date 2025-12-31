# MorphoSSL

MorphoSSL is a two-stage training pipeline designed for label-scarce medical imaging problems (including rare diseases):

1. Self-supervised pretraining with a Masked Autoencoder (MAE) using a Vision Transformer (ViT) encoder
2. Supervised fine-tuning on a small labeled dataset, compared against a supervised ResNet baseline

This repository contains:
- MAE (ViT encoder + lightweight decoder) for masked patch reconstruction
- Fine-tuning (ViT encoder + classification head) with optional encoder freezing
- A ResNet18/ResNet50 baseline fine-tuning path
- A CLI that exports `metrics.csv` plus loss/accuracy curves
- MAE reconstruction visualizations (original / masked / reconstruction)

## Why MAE for medicine
Medical datasets often contain large unlabeled archives but limited expert annotations. MAE learns an image representation by reconstructing masked patches from visible context. The learned encoder can then be fine-tuned with far fewer labeled samples than training from scratch.

## Repository layout

- MorphoSSL/core
  - model_mae.py: MAE architecture (patchify/unpatchify, random masking, encoder/decoder, masked MSE loss)
  - vision_transformer.py: ViT configuration helpers and positional embeddings
- MorphoSSL/modules
  - pretrain_module.py: PyTorch Lightning module for MAE pretraining
  - eval_module.py: PyTorch Lightning module for ViT classification fine-tuning
  - resnet_eval_module.py: PyTorch Lightning module for ResNet baseline
- MorphoSSL/utils
  - visualizations.py: MAE reconstruction utilities and plotting
- MorphoSSL/main.py
  - CLI entrypoint
- MorphoSSL/results
  - results/metrics: example `metrics_*.csv`
  - results/fiqures: example curves and reconstructions (note: folder name is `fiqures`)

## Setup

### Requirements
- Python 3.10+
- PyTorch, torchvision, timm, PyTorch Lightning

### Install
From the repository root:

```bash
python -m pip install -r MorphoSSL/requirements.txt
```

If you need a specific PyTorch build (CPU-only or CUDA), install it first following the official PyTorch instructions, then install the remaining requirements.

## Data layout

### Unlabeled data (MAE pretraining)
Point `--train-dir` to any folder containing images. Images are discovered recursively.

Example:

```
pretrain_dataset/
  patient_0001/
    img001.png
    img002.png
  patient_0002/
    viewA.jpg
```

Optional: pass `--val-dir` with another unlabeled folder.

### Labeled data (fine-tuning)
Use an ImageFolder-style structure:

```
finetune_dataset/
  train/
    class0/
      a.png
      b.png
    class1/
      c.png
  val/
    class0/
      d.png
    class1/
      e.png
```

Class names come from subfolder names.

## Quickstart

All commands run from the repository root.

### 1) MAE pretraining (self-supervised)

```bash
python MorphoSSL/main.py pretrain \
  --train-dir /path/to/pretrain_dataset \
  --out-dir outputs/pretrain \
  --backbone vit_base \
  --img-size 224 \
  --patch-size 16 \
  --mask-ratio 0.75 \
  --batch-size 64 \
  --epochs 50
```

Notes:
- `--backbone` must match what you later use for MAE-initialized fine-tuning.
- The default mask ratio is 0.75 (75% of patches masked).

### 2) Fine-tune a classifier using the MAE checkpoint

```bash
python MorphoSSL/main.py finetune \
  --train-dir /path/to/finetune_dataset/train \
  --val-dir /path/to/finetune_dataset/val \
  --out-dir outputs/finetune_mae \
  --init mae \
  --mae-ckpt outputs/pretrain/mae_pretrained.ckpt \
  --backbone vit_base \
  --epochs 100 \
  --batch-size 32
```

Optional flags:
- `--freeze-encoder`: linear-probe style training (head only)
- `--encoder-lr`: separate (usually smaller) learning rate for the encoder when unfreezing

Important: MAE checkpoints are architecture-specific. A ViT-Base checkpoint cannot be loaded into a ViT-Large encoder.

### 3) ResNet baseline (supervised)

```bash
python MorphoSSL/main.py finetune_resnet \
  --train-dir /path/to/finetune_dataset/train \
  --val-dir /path/to/finetune_dataset/val \
  --out-dir outputs/finetune_resnet50 \
  --arch resnet50 \
  --pretrained \
  --epochs 100 \
  --batch-size 32
```

Optional flags:
- `--no-pretrained`: train the ResNet from scratch
- `--freeze-encoder`: train a linear head only
- `--encoder-lr`: separate learning rate for the encoder when unfreezing

## Outputs

For `pretrain`, `finetune`, and `finetune_resnet`, the CLI writes to `--out-dir`:
- `metrics.csv` (epoch-level: train_loss, val_loss, train_acc, val_acc)
- `loss_curve.png`
- `acc_curve.png` (for runs that log accuracy)
- `checkpoints/` (Lightning checkpoints)

## MAE reconstruction visualization

### Folder reconstruction (batch)
Saves triplets (original / masked / reconstruction) for a folder:

```bash
python MorphoSSL/main.py reconstruct \
  --images-dir /path/to/images \
  --ckpt outputs/pretrain/mae_pretrained.ckpt \
  --out-dir outputs/recon
```

### Single-image visualization

```bash
python MorphoSSL/main.py visualize \
  --image-path /path/to/image.png \
  --ckpt outputs/pretrain/mae_pretrained.ckpt \
  --out-path outputs/mae_viz.png \
  --title "MAE reconstruction"
```

## Included example results

This repo includes example artifacts under:
- MorphoSSL/results/metrics
  - metrics_mae_encoder_pretrain.csv
  - metrics_mae_as_encoder_finetune.csv
  - metrics_resnet_finetune.csv
- MorphoSSL/results/fiqures
  - loss_curves and acc_curves
  - visualization_of_mae_encoder_reconstruction.png

## Reproducibility and evaluation notes

- Avoid data leakage: split at the patient level (when possible) before any SSL pretraining.
- MAE reconstruction loss does not directly equal classification performance; it is a pretext objective.
- For small labeled datasets, overfitting is common. Practical mitigations include stronger augmentation, early stopping, class balancing, and using a smaller encoder learning rate.