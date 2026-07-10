"""
Per-channel IoU. Reported separately, never averaged into one number —
parcel_border and tree performance are expected to diverge, and an
averaged metric would hide that (see prior methodology discussion —
same principle as not using SSIM for the GAN-stage layout evaluation).
"""

import torch


@torch.no_grad()
def per_channel_iou(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (0, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = ((preds + targets) > 0).float().sum(dims)
    iou = (intersection + eps) / (union + eps)
    return iou  # shape (C,)
