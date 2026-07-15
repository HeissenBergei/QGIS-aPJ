"""
Combined BCEWithLogits + Dice loss, computed independently per channel
and summed. Each of the 4 classes is a separate binary segmentation
problem — no cross-channel normalization (that would reintroduce the
mutual-exclusivity assumption we're deliberately avoiding).
"""

import torch
import torch.nn as nn


def dice_loss_per_channel(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)  # reduce over batch + spatial, keep channel dim
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice  # shape (C,)


class MultiLabelSegLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        # pos_weight must broadcast over the CHANNEL dim of (B, C, H, W).
        # A flat (C,) tensor would align with the last axis (W) and fail;
        # reshape to (C, 1, 1) so it broadcasts per channel.
        pw = (torch.tensor(pos_weight, dtype=torch.float32).view(-1, 1, 1)
              if pos_weight else None)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # BCE per-channel mean
        bce_map = self.bce(logits, targets)          # (B, C, H, W)
        bce_per_channel = bce_map.mean(dim=(0, 2, 3))  # (C,)

        dice_per_channel = dice_loss_per_channel(logits, targets)  # (C,)

        total_per_channel = (
            self.bce_weight * bce_per_channel + self.dice_weight * dice_per_channel
        )
        return total_per_channel.sum(), {
            "bce_per_channel": bce_per_channel.detach(),
            "dice_per_channel": dice_per_channel.detach(),
        }
