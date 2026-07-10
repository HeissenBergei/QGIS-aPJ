"""
Per-channel IoU. Reported separately, never averaged into one number —
parcel_border and tree performance are expected to diverge, and an
averaged metric would hide that (see prior methodology discussion —
same principle as not using SSIM for the GAN-stage layout evaluation).

IoU is accumulated as intersection/union pixel COUNTS across the whole
split and divided once at the end (dataset-level IoU), rather than
averaging per-batch ratios. This is both more stable and lets a channel
that never appears in the split be reported as NaN (undefined) instead of
being silently scored 1.0 — the empty-prediction-on-empty-target case that
otherwise inflates the mean on sparse classes like tree.
"""

import torch


@torch.no_grad()
def iou_counts(logits, targets, threshold=0.5):
    """Per-channel intersection and union pixel counts for one batch.
    Accumulate both across an epoch, then call iou_from_counts once."""
    preds = (torch.sigmoid(logits) > threshold).float()
    dims = (0, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = ((preds + targets) > 0).float().sum(dims)
    return intersection, union  # each shape (C,)


def iou_from_counts(intersection, union):
    """Dataset-level per-channel IoU from accumulated counts. Channels with
    union == 0 (class absent from both preds and targets across the entire
    split) are returned as NaN so they can be excluded from the mean via
    torch.nanmean rather than counted as a perfect 1.0."""
    iou = intersection / union.clamp_min(1.0)
    iou[union == 0] = float("nan")
    return iou  # shape (C,)
