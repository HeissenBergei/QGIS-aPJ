"""
Build the fold-1 training set from raw Zoning Manager addon exports.

Each source is a flat folder of addon exports (one export = many files named
`sample_<n>_<feature>.png` + sidecars). This script keeps only the two files
Task 09 trains on per sample — `_satellite.png` (RGB image) and
`_mask_index.png` (single-channel class-index label) — and produces:

  <out>/images/<tile>.png   downscaled RGB
  <out>/masks/<tile>.png    downscaled class-index map (values unchanged)
  <out>/splits/{train,val}.txt
  <out>/provenance.csv      new tile -> (source, original sample, split)

Steps, in order:
  1. Reorder: sources are concatenated in the order given and tiles are
     renumbered consecutively (tile_0001, tile_0002, ...).
  2. Downscale <src_px> -> <dst_px>: image bilinear, mask nearest (exact
     integer stride when the ratio is integral, so class indices stay exact).
  3. Split by *tile* (before augmentation, so no orientation of a tile can
     leak across train/val).
  4. Augment TRAIN ONLY with the dihedral-8 group (4 rotations x optional
     mirror) — the lossless label-preserving set. Val stays single.
  5. Print per-channel train frequency and the inverse-frequency
     (neg/pos) pos_weight to paste into config.yaml.

Runs under the QGIS-bundled Python (numpy + PIL + GDAL); no extra deps.
Usage:
  python prepare_dataset.py --out data \
      --src /path/SIGRADI_LA --src /path/input --src "/path/input son"
"""

import argparse
import csv
import glob
import os
import random

import numpy as np
from PIL import Image
from osgeo import gdal

gdal.UseExceptions()

# Task 09 channel order -> value in the addon's _mask_index.png (see dataset.py).
CLASS_NAMES = ["parcel_border", "building", "hard_surface", "tree"]
ADDON_INDEX = {"parcel_border": 6, "building": 5, "hard_surface": 3, "tree": 2}


def load_image(path, dst_px):
    im = Image.open(path).convert("RGB").resize((dst_px, dst_px), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_mask(path, dst_px):
    a = gdal.Open(path).ReadAsArray().astype(np.uint8)  # (src, src) raw indices
    step, rem = divmod(a.shape[0], dst_px)
    if rem == 0:
        return a[::step, ::step]                         # exact nearest
    im = Image.fromarray(a, "L").resize((dst_px, dst_px), Image.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def dihedral(a, k):
    """k in 0..7: k>=4 mirrors first, then rotate 90*k degrees."""
    if k >= 4:
        a = np.fliplr(a)
    return np.rot90(a, k % 4)


def enumerate_tiles(sources):
    tiles, gid = [], 0
    for src in sources:
        name = os.path.basename(os.path.normpath(src))
        ids = sorted({os.path.basename(f).split("_mask_index")[0]
                      for f in glob.glob(os.path.join(src, "*_mask_index.png"))})
        for oid in ids:
            sat = os.path.join(src, f"{oid}_satellite.png")
            msk = os.path.join(src, f"{oid}_mask_index.png")
            if not (os.path.exists(sat) and os.path.exists(msk)):
                print(f"  ! skipping {name}/{oid}: missing satellite or mask_index")
                continue
            gid += 1
            tiles.append((gid, name, oid, sat, msk))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="raw export folder; repeat, order is preserved")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dst-px", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    mask_dir = os.path.join(args.out, "masks")
    split_dir = os.path.join(args.out, "splits")
    for d in (img_dir, mask_dir, split_dir):
        os.makedirs(d, exist_ok=True)

    tiles = enumerate_tiles(args.src)
    print(f"Enumerated {len(tiles)} tiles from {len(args.src)} source(s)")

    rng = random.Random(args.seed)
    shuffled = tiles[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_frac))
    val_gids = {t[0] for t in shuffled[:n_val]}

    train_list, val_list, prov = [], [], []
    tot_pix, pos = 0, {c: 0 for c in CLASS_NAMES}

    for gid, src, oid, sat, msk in tiles:
        tile = f"tile_{gid:04d}"
        image = load_image(sat, args.dst_px)
        mask = load_mask(msk, args.dst_px)
        if gid in val_gids:
            Image.fromarray(image, "RGB").save(os.path.join(img_dir, tile + ".png"))
            Image.fromarray(mask, "L").save(os.path.join(mask_dir, tile + ".png"))
            val_list.append(tile)
            split = "val"
        else:
            for k in range(8):
                vid = f"{tile}_d{k}"
                Image.fromarray(dihedral(image, k), "RGB").save(
                    os.path.join(img_dir, vid + ".png"))
                Image.fromarray(dihedral(mask, k), "L").save(
                    os.path.join(mask_dir, vid + ".png"))
                train_list.append(vid)
            split = "train"
            tot_pix += mask.size
            for c in CLASS_NAMES:
                pos[c] += int((mask == ADDON_INDEX[c]).sum())
        prov.append((tile, src, oid, split))

    with open(os.path.join(split_dir, "train.txt"), "w") as f:
        f.write("\n".join(sorted(train_list)) + "\n")
    with open(os.path.join(split_dir, "val.txt"), "w") as f:
        f.write("\n".join(sorted(val_list)) + "\n")
    with open(os.path.join(args.out, "provenance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_tile", "source", "orig_sample", "split"])
        w.writerows(prov)

    print(f"Split: {len(train_list)//8} train tiles (x8 = {len(train_list)}), "
          f"{len(val_list)} val")
    print("Train-set class frequency and inverse-frequency pos_weight:")
    pw = []
    for c in CLASS_NAMES:
        p = pos[c]
        w = round((tot_pix - p) / p, 3) if p > 0 else 1.0
        pw.append(w)
        frac = 100.0 * p / tot_pix if tot_pix else 0.0
        print(f"  {c:14s} {frac:6.3f}% px  pos_weight={w}")
    print(f"pos_weight: {pw}   <- paste into config.yaml loss.pos_weight")


if __name__ == "__main__":
    main()
