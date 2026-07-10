"""
Training loop for the fold-1 autonomous segmentation engine.

Usage:
    python train.py --config config.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import ParcelSegDataset
from model import build_model
from losses import MultiLabelSegLoss
from metrics import iou_counts, iou_from_counts


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None, class_names=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    inter_accum = torch.zeros(len(class_names))
    union_accum = torch.zeros(len(class_names))
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss, _ = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            inter, union = iou_counts(logits, masks)
            inter_accum += inter.cpu()
            union_accum += union.cpu()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    # Dataset-level IoU; channels absent from the whole split come back NaN.
    epoch_iou = iou_from_counts(inter_accum, union_accum)
    return avg_loss, epoch_iou


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    class_names = cfg["data"]["class_names"]

    train_ds = ParcelSegDataset(
        root=cfg["data"]["root"],
        split_file=cfg["data"]["train_split"],
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        tile_size=cfg["data"]["tile_size"],
        augment=True,
        class_names=class_names,
    )
    val_ds = ParcelSegDataset(
        root=cfg["data"]["root"],
        split_file=cfg["data"]["val_split"],
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        tile_size=cfg["data"]["tile_size"],
        augment=False,
        class_names=class_names,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = build_model(cfg).to(device)
    loss_fn = MultiLabelSegLoss(
        pos_weight=cfg["loss"]["pos_weight"],
        bce_weight=cfg["loss"]["bce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
    ).to(device)  # moves the pos_weight buffer onto the training device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    patience_counter = 0

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_iou = run_epoch(
            model, train_loader, loss_fn, device, optimizer, class_names
        )
        val_loss, val_iou = run_epoch(
            model, val_loader, loss_fn, device, optimizer=None,
            class_names=class_names,
        )
        scheduler.step()

        # nanmean so channels absent from the val split (NaN) don't drag the
        # mean — they are excluded, not counted as 1.0.
        mean_val_iou = torch.nanmean(val_iou).item()

        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        for name, iou_val in zip(class_names, val_iou.tolist()):
            if iou_val != iou_val:  # NaN -> class not present in this split
                print(f"  val_iou[{name}] = n/a (absent from val split)")
            else:
                print(f"  val_iou[{name}] = {iou_val:.4f}")
        print(f"  val_iou[mean] = {mean_val_iou:.4f}  (over present classes; "
              f"reported per-class above too — do not rely on mean alone, see TASK_09)")

        if mean_val_iou > best_metric:
            best_metric = mean_val_iou
            patience_counter = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "cfg": cfg},
                ckpt_dir / "best_model.pt",
            )
            print(f"  -> saved new best checkpoint (mean_iou={best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
