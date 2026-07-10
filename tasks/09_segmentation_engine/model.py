"""
Model factory. Multi-label output (sigmoid per channel, not softmax) —
see TASK_09_segmentation_engine.md for why. Raw logits are returned;
sigmoid is applied in the loss (BCEWithLogitsLoss) and separately at
inference time, never inside the model itself.
"""

import segmentation_models_pytorch as smp


def build_model(cfg):
    arch = cfg["model"]["arch"]
    kwargs = dict(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
        activation=None,  # logits out — see docstring
    )

    if arch == "Unet":
        model = smp.Unet(**kwargs)
    elif arch == "UnetPlusPlus":
        # Fallback if the tree channel underperforms on plain U-Net —
        # do not switch to this by default, see TASK_09.
        model = smp.UnetPlusPlus(**kwargs)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return model
