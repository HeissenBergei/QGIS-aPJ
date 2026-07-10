"""
Dataset loader for the fold-1 autonomous segmentation engine.

Assumed data contract (see TASK_09_segmentation_engine.md):
  data/images/<id>.tif  -> 3-band RGB uint8
  data/masks/<id>.tif   -> 4-band uint8 {0,1}, band order:
                           [parcel_border, building, hard_surface, tree]

If the addon export format differs (e.g. GeoPackage vector layers instead
of pre-rasterized masks), add a rasterize() step in _load_mask() and leave
everything else unchanged.
"""

import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None


class ParcelSegDataset(Dataset):
    def __init__(self, root, split_file, image_dir="images", mask_dir="masks",
                 tile_size=512, augment=False, class_names=None):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.tile_size = tile_size
        self.augment = augment
        self.class_names = class_names or [
            "parcel_border", "building", "hard_surface", "tree"
        ]

        split_path = self.root / split_file
        with open(split_path) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.transform = self._build_transform() if augment else None

    def _build_transform(self):
        if A is None:
            return None
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.2),
        ])

    def __len__(self):
        return len(self.ids)

    def _load_image(self, tile_id):
        path = self.image_dir / f"{tile_id}.tif"
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
        img = np.transpose(img, (1, 2, 0)).astype(np.float32) / 255.0
        return img

    def _load_mask(self, tile_id):
        path = self.mask_dir / f"{tile_id}.tif"
        with rasterio.open(path) as src:
            mask = src.read()  # (4, H, W), values in {0,1}
        mask = np.transpose(mask, (1, 2, 0)).astype(np.float32)
        assert mask.shape[-1] == len(self.class_names), (
            f"Mask band count {mask.shape[-1]} != expected "
            f"{len(self.class_names)} classes {self.class_names}. "
            f"Check addon export format against TASK_09 data contract."
        )
        return mask

    def __getitem__(self, idx):
        tile_id = self.ids[idx]
        image = self._load_image(tile_id)
        mask = self._load_mask(tile_id)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
        return {"image": image, "mask": mask, "id": tile_id}
