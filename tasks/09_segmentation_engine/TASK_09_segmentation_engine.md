# Task 09 — Autonomous Segmentation Engine (Fold 1)

## Status
New. Not a stub — this is the real first training pass, intended to run on a
small mirrored dataset in Google Colab as a validation loop before scaling
annotation effort.

## Objective
Train a single shared-encoder, multi-label decoder model that predicts four
independent binary masks per input tile:

| Channel | Class          | Notes                                                    |
|---------|----------------|-----------------------------------------------------------|
| 0       | parcel_border  | thin wall/ring outline of the parcel, not the filled interior (addon `parcel_border` polyline) |
| 1       | building       | building footprint                                        |
| 2       | hard_surface   | impervious surface + paths, as one class ("geçirimsiz arazi") |
| 3       | tree           | tree canopy (the addon's `hard_landscape` class)          |

These four are the whole task. Other addon classes (`soft_landscape`,
`parking_space`, `main_entrance`, `building_entrance`) are deliberately left
out as future features, not merged into these channels.

`soft_landscape` is NOT a model output. Note that with `parcel_border` now a
thin ring (not the filled interior), it can no longer be derived as
`parcel_border AND NOT (building OR hard_surface OR tree)` — recovering the
soft-landscape area needs the parcel *polygon interior* (from the addon's
vector layer), which the ring channel does not provide. Deferred, not needed
for training.

`building_entrance` / `land_entrance` are explicitly out of scope for this task.

## Why multi-label (sigmoid), not multi-class (softmax)
Classes are not mutually exclusive — a building pixel is also a parcel
pixel. Softmax would force competition between spatially nested classes.
Each channel is trained as its own binary segmentation task, sharing one
encoder. This is a standard pattern (see MultiTalent, arXiv:2303.14444;
`segmentation-models-pytorch` supports it natively via
`activation='sigmoid'`).

## Architecture
- Backbone: `segmentation-models-pytorch` U-Net, encoder = `resnet34`
  (ImageNet pretrained), `classes=4`, `activation=None` (raw logits; sigmoid
  applied in the loss/inference code, not the model, for numerical stability
  with `BCEWithLogitsLoss`).
- Do not switch to a transformer decoder at this dataset size — CNN
  backbones outperform ViT-style decoders below ~a few thousand tiles on
  fine-boundary aerial segmentation tasks. Revisit only if baseline mIoU is
  poor AND dataset size grows substantially.
- If tree-channel IoU is notably worse than the other three after the
  baseline run, swap in `smp.UnetPlusPlus` or add attention gates before
  reaching for more data — flagged in code as a TODO, not implemented yet.

## Loss
Per-channel `BCEWithLogitsLoss` (with `pos_weight` per channel, inverse
frequency) + Dice loss, summed. See `losses.py`. Add boundary-weighted BCE
for the `parcel_border` and `building` channels only, once baseline is
running — not in v1.

## Known hard case — do not treat as a bug
`parcel_border` may show materially lower IoU than the other three classes.
This is expected: parcel boundaries are frequently not visually
distinguishable from imagery alone (see Crommelinck et al., 2019,
10.3390/rs11212505). Do not tune hyperparameters to chase this number without
first checking a handful of failing validation tiles to confirm the boundary
is actually visible in the source imagery. If it isn't, log it and move on —
that's a dataset/scene-selection issue, not a model issue.

## Data contract (confirmed against the addon export)
```
data/
  images/
    tile_0001.png         # addon "<stem>_satellite.png" (RGB uint8)
    ...
  masks/
    tile_0001.png         # addon "<stem>_mask_index.png"
                          #   single-channel Grayscale8 class-index map
    ...
  splits/
    train.txt             # tile ids, one per line
    val.txt
```
The addon (`zoning_manager/export/rasterize.py`) writes a **multi-class index
map** — one integer class per pixel, 0 = background — NOT a 4-band multi-label
stack. Addon index values (`mask_class_index()`):
`1 soft_landscape · 2 hard_landscape · 3 hard_surface · 4 parking_space ·
5 building · 6 parcel_border · 7 main_entrance · 8 building_entrance`.

`dataset.py::_load_mask()` expands that index map into the four independent
binary channels above via `ADDON_INDEX` (parcel_border←6, building←5,
hard_surface←3, tree←2). Because the four classes barely overlap (a thin ring
plus three mutually-exclusive fills) the expansion is effectively lossless, so
Task 09 keeps its multi-label (sigmoid) design unchanged. If overlapping
targets are ever needed (e.g. re-adding the *filled* parcel interior), the
addon must emit per-class binary masks before `resolve_overlaps()`; converting
the argmax index map cannot recover overlaps after the fact.

## Deliverables for this task
1. `dataset.py` — PyTorch `Dataset`, tiling/loading, augmentation hooks
2. `model.py` — SMP model factory
3. `losses.py` — combined per-channel BCE+Dice
4. `train.py` — training loop, per-channel IoU logging, checkpointing
5. `config.yaml` — hyperparameters
6. `colab_trial.ipynb` — minimal Colab notebook to run a short trial on the
   small mirrored dataset

## Explicitly deferred (do not implement in this task)
- Boundary/edge auxiliary loss head
- Distance-transform-to-nearest-road auxiliary input channel (fallback if
  parcel_border IoU is bad — see prior architecture discussion)
- Post-processing polygon extraction / GeoPackage export (belongs in Zoning
  Manager plugin integration, not the training engine)
- building_entrance / land_entrance
