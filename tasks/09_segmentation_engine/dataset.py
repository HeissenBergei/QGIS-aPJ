"""
Dataset loader for the fold-1 autonomous segmentation engine.

Data contract (confirmed against the Zoning Manager addon export,
zoning_manager/export/rasterize.py; see TASK_09_segmentation_engine.md):

  data/images/<id>.png  -> the addon's "<stem>_satellite.png" (RGB, uint8)
  data/masks/<id>.png   -> the addon's "<stem>_mask_index.png"
                           (single-channel Grayscale8 class-index map)

The addon writes a MULTI-CLASS index map (one integer class per pixel), NOT a
4-band multi-label stack. We expand it here into the four independent binary
channels Task 09 trains on, by matching each channel to its addon class index
(ADDON_INDEX below). Classes barely overlap (a thin parcel-border ring + three
mutually-exclusive fills), so this expansion is effectively lossless.

Addon index values (from rasterize.py mask_class_index(); 0 = background):
  1 soft_landscape  2 hard_landscape  3 hard_surface   4 parking_space
  5 building        6 parcel_border   7 main_entrance  8 building_entrance

Only the four channels below are used; every other index (soft_landscape,
parking_space, entrances) collapses to background for this task.
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


# Task 09 channel name -> integer value in the addon's _mask_index.png.
# parcel_border is the thin WALL RING (not the filled interior); tree is the
# addon's "hard_landscape". parking_space (4) is intentionally excluded (a
# future feature), so it is treated as background.
ADDON_INDEX = {
    "parcel_border": 6,
    "building": 5,
    "hard_surface": 3,
    "tree": 2,
}


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
        missing = [c for c in self.class_names if c not in ADDON_INDEX]
        assert not missing, (
            f"No addon _mask_index.png class index mapped for {missing}. "
            f"Known: {sorted(ADDON_INDEX)}."
        )

        split_path = self.root / split_file
        with open(split_path) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.transform = self._build_transform() if augment else None

    def _build_transform(self):
        if A is None:
            return None
        # NOTE: flips + 90° rotations (the dihedral-8 group) are materialized
        # OFFLINE when the dataset is built, so they are intentionally NOT
        # repeated here — doing both would just re-shuffle the same 8 finite
        # orientations. Only non-dihedral transforms run online.
        return A.Compose([
            A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.2),
        ])

    def __len__(self):
        return len(self.ids)

    def _load_image(self, tile_id):
        path = self.image_dir / f"{tile_id}.png"
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W); RGBA if the satellite PNG has alpha
        img = img[:3]  # drop the (opaque) alpha channel if present
        img = np.transpose(img, (1, 2, 0)).astype(np.float32) / 255.0
        return img

    def _load_mask(self, tile_id):
        """Read the addon's single-channel class-index PNG and expand it into
        one binary {0,1} channel per class in self.class_names."""
        path = self.mask_dir / f"{tile_id}.png"
        with rasterio.open(path) as src:
            idx = src.read(1)  # (H, W) uint8 class-index map
        h, w = idx.shape
        mask = np.zeros((h, w, len(self.class_names)), dtype=np.float32)
        for c, name in enumerate(self.class_names):
            mask[..., c] = (idx == ADDON_INDEX[name])
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
