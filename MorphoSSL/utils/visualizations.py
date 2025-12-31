from __future__ import annotations

from typing import Optional, Tuple

import torch
from PIL import Image


@torch.no_grad()
def mae_reconstruction_triplet(
    *,
    mae_model,
    imgs: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = device or imgs.device
    mae_model.eval()

    imgs = imgs.to(device)
    _, pred, mask = mae_model(imgs)

    patches = mae_model.patchify(imgs)
    mask_ = mask.unsqueeze(-1)
    visible_patches = patches * (1.0 - mask_)
    masked_vis = mae_model.unpatchify(visible_patches)

    recon_patches = patches * (1.0 - mask_) + pred * mask_
    recon_img = mae_model.unpatchify(recon_patches)

    return imgs, masked_vis, recon_img


def simple_input_gradient_saliency(
    *,
    model,
    imgs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    model.eval()
    imgs = imgs.requires_grad_(True)

    logits = model(imgs)
    if targets is None:
        targets = logits.argmax(dim=1)
    sel = logits.gather(1, targets.view(-1, 1)).sum()

    sel.backward()
    grads = imgs.grad.detach().abs().mean(dim=1, keepdim=True)

    n = grads.shape[0]
    grads_flat = grads.view(n, -1)
    mins = grads_flat.min(dim=1).values.view(n, 1, 1, 1)
    maxs = grads_flat.max(dim=1).values.view(n, 1, 1, 1)
    sal = (grads - mins) / (maxs - mins + 1e-8)
    return sal


@torch.no_grad()
def plot_mae_single_image_reconstruction(
    *,
    mae_model,
    image: "torch.Tensor | Image.Image",
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    mae_model.eval()
    if device is None:
        device = next(mae_model.parameters()).device

    if isinstance(image, Image.Image):
        import torchvision.transforms.functional as TF

        img_t = TF.to_tensor(image.convert("RGB"))
    else:
        img_t = image

    if img_t.dim() != 3:
        raise ValueError("Expected a single image tensor of shape (C,H,W)")

    imgs = img_t.unsqueeze(0).to(device)

    expected = None
    if hasattr(mae_model, "cfg") and hasattr(mae_model.cfg, "img_size"):
        expected = int(mae_model.cfg.img_size)
    elif hasattr(mae_model, "patch_embed") and hasattr(mae_model.patch_embed, "img_size"):
        pe_size = mae_model.patch_embed.img_size
        expected = int(pe_size[0] if isinstance(pe_size, (tuple, list)) else pe_size)

    if expected is not None:
        _, _, h, w = imgs.shape
        if h != expected or w != expected:
            imgs = F.interpolate(imgs, size=(expected, expected), mode="bilinear", align_corners=False)

    orig, masked, recon = mae_reconstruction_triplet(mae_model=mae_model, imgs=imgs, device=device)

    orig0 = orig[0].detach().cpu().clamp(0, 1)
    masked0 = masked[0].detach().cpu().clamp(0, 1)
    recon0 = recon[0].detach().cpu().clamp(0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, im, t in zip(
        axs,
        [orig0, masked0, recon0],
        ["original", "corrupted (masked)", "reconstruction"],
    ):
        ax.imshow(im.permute(1, 2, 0))
        ax.set_title(t)
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig