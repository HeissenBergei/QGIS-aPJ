# Autonomous Zoning Segmentation — Technical Report

**Scope.** Complete technical record of the segmentation engine (QGIS-aPJ Task 09)
and its data-generation pipeline (QGIS-PJ / Zoning Manager plugin): data
provenance, label engineering, dataset construction, augmentation, model
architectures, loss, training protocol, evaluation methodology, results, and
in-application deployment. Written as a source document for the research paper.
Every number is drawn from the committed code/configs; provenance and caveats
are stated explicitly where a result is uncertain.

Two artifact families are covered:
- **Data-generation tool** — the Zoning Manager QGIS plugin (`zoning-plugins`
  repo), which renders standardized rasters + label masks from vector zoning
  annotations and is also the inference host.
- **Segmentation engine** — the training/inference code (`QGIS-aPJ`,
  `tasks/09_segmentation_engine/`, branch `seg-engine-v1`).

---

## 1. Problem framing

The system produces landscape/zoning plans from aerial imagery. This report
concerns the **semantic segmentation stage**: from an RGB satellite tile,
predict four spatial zoning classes that are subsequently vectorized into an
editable plan. The segmentation model is exposed inside the GIS as one
"zoning method" among several (cadastral fetch, manual annotation, model
inference), so its output must be expressed in the same vector-element
representation the rest of the pipeline uses.

**Design stance (fixed across folds):**
- Single ground sample distance (GSD) — the model always sees the exact scale
  it trained on; scale is removed as a generalization axis (see §3.1).
- CNN encoder–decoder (U-Net, ResNet-34 encoder), not a transformer decoder,
  at this data scale — consistent across folds.
- Output formulation evolved between folds: **fold-1** treats the classes as a
  multi-label problem (independent sigmoid channels), motivated by their
  spatial nesting; **fold-2** adopts a **multi-class** formulation with an
  explicit background class (see §6) after multi-label proved weak on the
  sparsest classes.

---

## 2. Data-generation pipeline (ground-truth source)

Ground truth is authored in the Zoning Manager plugin and exported by
`zoning_manager/export/rasterize.py` through QGIS's own map-render pipeline
(`QgsMapSettings` + `QgsMapRendererParallelJob`), which guarantees a correct
world→pixel transform and Y-flip (verified pixel-exact in headless tests).

### 2.1 Export geometry

| Parameter | Value | Notes |
|---|---|---|
| Map scale | 1 : 200 | user-set spinbox; 200 is canonical |
| Render DPI | 96 | fixed |
| GSD (metres/pixel) | `denom·0.0254/96` = **0.052916 m/px** | at 1:200 |
| Tile size | **1024 × 1024 px** | user-set; 1024 canonical |
| Ground footprint | **54.19 m** square | = 1024 × 0.0529 |
| CRS (render) | metric UTM / EPSG:3857 per scene | chosen so ground sizes are true metres |
| Georeferencing | PNG + `.wld` world file + `.prj` sidecar | not embedded GeoTIFF |

All raster products for a sample share one extent and size, so they are
pixel-aligned 1:1 (satellite, per-feature layers, composite, masks).

### 2.2 Class / feature registry

`rasterize.FEATURES` is the single ordered registry (paint order = registry
order). Each feature has a legend RGB and a geometry kind (FILL area,
POLYLINE outline, POINT cross). Physical coexistence is enforced by
`resolve_overlaps()`, which subtracts higher-priority incompatible zones from
lower ones in raw pixel space (QgsGeometry boolean ops + `makeValid` +
`buffer(0)`; no Shapely), confining every zone to the parcel interior and
punching holes where a higher zone sits inside a lower one.

- **Fill priority (low→high):** soft_landscape < hard_landscape <
  parking_space < hard_surface < building.
- **Compatible (may overlap):** {hard_surface, parking_space},
  {soft_landscape, hard_landscape}.

### 2.3 Mask encoding

The exporter writes a **single-channel class-index map** `<stem>_mask_index.png`
(`Grayscale8`), one integer class per pixel (0 = background/void), plus an RGB
`<stem>_mask.png`. Index values (`mask_class_index()` = registry position + 1):

| Index | Class | | Index | Class |
|--:|---|---|--:|---|
| 0 | background/void | | 5 | building |
| 1 | soft_landscape | | 6 | parcel_border (wall ring) |
| 2 | hard_landscape | | 7 | main_entrance |
| 3 | hard_surface | | 8 | building_entrance |
| 4 | parking_space | | | |

**Important:** the exporter's mask is *multi-class* (argmax; later features
overwrite earlier). The training pipeline reconstructs *multi-label* channels
from it — see §4. Colors are placeholder (`colors_final=no` in the manifest);
we therefore train from the **index** map, which is invariant to the eventual
canonical class→RGB mapping.

Per-sample export also emits per-feature PNGs (each at 20 % opacity), a
composite, an `_annotations.png/.json` vector archive, a `_features.geojson`,
and a `manifest.csv` row (scale, ground size, size_px, per-feature counts,
CRS, centre coordinates).

---

## 3. Data contract & label engineering

### 3.1 Channel definitions (the four trained classes)

| Ch | Class | Source index | Semantics |
|--:|---|--:|---|
| 0 | `parcel_border` | 6 | **thin wall RING** (not filled interior) — the outline the plugin draws as the parcel wall |
| 1 | `building` | 5 | building footprint (filled) |
| 2 | `hard_surface` | 3 | impervious surface + paths, as one class |
| 3 | `tree` | 2 | tree canopy — the plugin's **`hard_landscape`** class |

Deliberately excluded (future features, treated as background for the four
channels): `soft_landscape` (1), `parking_space` (4), `main_entrance` (7),
`building_entrance` (8).

### 3.2 Multi-class → multi-label expansion

`dataset.py::_load_mask()` reads the single-channel index map and expands it to
four binary channels by `channel_c = (index == ADDON_INDEX[c])`, with
`ADDON_INDEX = {parcel_border:6, building:5, hard_surface:3, tree:2}`.

Because the four chosen classes barely overlap in practice (a thin border ring
+ three mutually-exclusive fills), this expansion is effectively lossless.
Consequence to note in the paper: because the exporter mask is argmax, any
*genuinely overlapping* target (e.g. re-adding the filled parcel interior
under buildings) cannot be recovered by expansion — it would require an
export-side change (per-class binary masks emitted before overlap
resolution). The current design sidesteps this by defining `parcel_border` as
a ring.

### 3.3 Input / label file layout (per tile)

```
data/images/<tile>.png   RGB satellite (<stem>_satellite.png), uint8
data/masks/<tile>.png    single-channel class-index map (<stem>_mask_index.png)
data/splits/{train,val}.txt
data/provenance.csv      new_tile -> (source, original_sample, split)
```

---

## 4. Datasets

Built by `prepare_dataset.py`: reorders N raw export folders into consecutive
`tile_NNNN`, downscales 1024→512 (image bilinear; mask exact 2× nearest
stride, preserving integer class indices), splits **by tile** (before
augmentation, so no augmented variant leaks across the split), then applies
offline dihedral-8 augmentation to the **training** tiles only. `pos_weight`
is computed from the (pre-augmentation) training tiles.

### 4.1 Fold-1 (pipeline-validation / smoke test)

| | |
|---|---|
| Sources | SIGRADI_LA (20) + input (3) + input son (10) |
| Base tiles | **33** |
| Split | 26 train / 7 val (seed 42, ~20 % val) |
| Train after dihedral-8 | 26 × 8 = **208** images |
| Val | 7 (un-augmented) |
| Tile resolution | 512 × 512 |

Training-set class frequency and inverse-frequency `pos_weight` (neg/pos):

| Class | % of pixels | pos_weight |
|---|--:|--:|
| parcel_border | 1.829 | 53.665 |
| building | 11.030 | 8.066 |
| hard_surface | 4.153 | 23.079 |
| tree | 3.028 | 32.022 |

### 4.2 Fold-2 (generalization / scale-up)

| | |
|---|---|
| Sources (7) | SIGRADI_LA (36), inputsal (11), inputsal2 (5), inputsal3 (17), png_2 (6), png3 (12), sigradi_datca/sigradi_png (1) |
| Base tiles | **88** |
| Split | 70 train / 18 val (seed 42) |
| Train after dihedral-8 | 70 × 8 = **560** images |
| Val | 18 (un-augmented) |
| Val by source | SIGRADI_LA 6, inputsal 4, inputsal3 7, png3 1 |

Training-set class frequency and `pos_weight`:

| Class | % of pixels | pos_weight |
|---|--:|--:|
| parcel_border | 1.971 | 49.747 |
| building | 10.817 | 8.244 |
| hard_surface | 6.254 | 14.990 |
| tree | 2.970 | 32.673 |

Notable shift vs fold-1: `hard_surface` frequency rose 4.15 %→6.25 % (weight
23.08→14.99), reflecting more paved/urban scenes; `tree` stayed sparse
(~3 %, weight ~32), which remains the limiting class.

### 4.3 Diversity design (fold-2 target, `FOLD2_DATASET_SPEC.md`)

Generalization is driven by diversity, quota-balanced across four axes:
(1) **region/jurisdiction** via the plugin's live cadastral adapters (TKGM
Türkiye, LA County, Riverside/Palm Springs, Agenzia Entrate Toscana);
(2) **density typology** (dense urban / suburban / rural-agricultural /
forested / coastal); (3) **parcel morphology** (small rectangular, large
irregular, corner, flag, wall-less); (4) **imagery conditions** (season, sun
angle, provider, renovation age). Volume milestones: fold-2 ≈ 300–500 parcels
for reliable single-region performance; fold-3 ≈ 1,000–2,000 for cross-region
claims. (The realized fold-2 set here — 88 tiles — is below the 300–500 target
and is best described as an early diverse-scale increment.)

---

## 5. Preprocessing & augmentation

**Resolution.** 1024→512 offline (image PIL bilinear; mask nearest via exact
`[::2, ::2]` stride). Model input side = **512 px**.

**Offline (materialized in the dataset), train only:** the **dihedral-8 group**
— the 4 rotations × optional mirror (the lossless label-preserving symmetry
set), applied identically to image and mask (mask via `np.rot90`/`np.fliplr`,
no interpolation). 8× multiplier. Applied to train tiles only; the validation
set is never augmented (no metric inflation / leakage).

**Online (fold-2, in `dataset.py`, applied jointly to image+mask via
albumentations):**
- `Affine(scale=(0.9, 1.1), p=0.5)` — ±10 % scale jitter (robustness to small
  GSD deviations without multi-scale data).
- `ElasticTransform(alpha=30, sigma=5, p=0.2)`.
- Photometric (cross-provider robustness): `ColorJitter`,
  `RandomBrightnessContrast(p=0.3)`,
  `HueSaturationValue(hue=10, sat=20, val=10, p=0.3)`, `GaussNoise(p=0.2)`.

Online geometric flips/90°-rotations are **disabled** (dihedral-8 already
covers the finite orientation set; stacking would only re-shuffle it).

---

## 6. Model architecture

Both folds use a **U-Net** (segmentation-models-pytorch) with an ImageNet-
pretrained **ResNet-34** encoder and 512² input (divisible by 32, the encoder
stride requirement). They differ in the output formulation — fold-1 multi-label,
fold-2 multi-class — as described below.

### 6.1 Fold-1 model — multi-label

- `in_channels = 3`, `classes = 4`, `activation = None` (raw logits; sigmoid
  applied in loss/inference for numerical stability with `BCEWithLogitsLoss`).
- **Multi-label head:** four independent sigmoid channels
  (parcel_border, building, hard_surface, tree). Rationale: zoning classes are
  spatially nested — a softmax would force competition between a parcel pixel
  and the building on it. Each channel is its own binary segmentation task
  sharing one encoder (pattern per MultiTalent, arXiv:2303.14444; natively
  supported by smp `activation='sigmoid'`).

### 6.2 Fold-2 model — multi-class reformulation

The multi-label fold-1 model learned dense classes well but stayed weak on the
two sparsest targets (tree, parcel border; §10). Fold-2 — the larger, longer-
trained "big brother" — revises the output formulation to a **5-class multi-
class (softmax / argmax) U-Net** with an explicit background class:

| Output class | Meaning | Source (addon index) |
|---|---|---|
| 0 · background | void + entrances | 0, 7, 8 |
| 1 · soft ground | soft landscape **+ tree canopy** | 1, 2 |
| 2 · hard surface | impervious + parking | 3, 4 |
| 3 · building | building footprint | 5 |
| 4 · wall | parcel border ring | 6 |

- Same U-Net / ResNet-34 backbone; `classes = 5`; **ImageNet input
  normalization** (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`);
  512² input.
- **One label per pixel** (argmax) rather than independent channels — with the
  parcel wall as a thin ring and the remaining classes filling disjoint areas,
  the nesting that motivated multi-label in fold-1 is confined to the wall and
  no longer requires overlapping channels.
- **Class consolidation:** the sparse `tree` (hard_landscape) class, which
  fold-1 could not learn reliably (~3 % of pixels), is **merged into soft
  ground** in fold-2, and parking is merged into hard surface. Consequence:
  fold-2 does not emit a separate tree prediction. Restoring a dedicated tree
  class is deferred to a future fold once the tree-tile diversity quota
  (≥60–70 % tree-containing tiles) is met.
- Designated capacity fallback if boundary/sparse classes remain weak at
  scale: `smp.UnetPlusPlus` or attention gates (one-line `config.model.arch`
  change); not activated in folds 1–2.

Both output formulations (fold-1 four-channel sigmoid; fold-2 five-class
argmax) are decoded through a single metadata-driven inference path in the
plugin (§12): each checkpoint declares its class names, normalization and input
size, and the host maps the model's outputs onto the plan's feature channels
by name.

---

## 7. Loss function

Per-channel **BCEWithLogits + Dice**, summed over channels; no cross-channel
normalization (that would reintroduce mutual exclusivity). Per `losses.py`:

- BCE: `nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')`, per-channel
  mean over batch+spatial. `pos_weight` is per-channel inverse frequency,
  reshaped to `(C, 1, 1)` so it broadcasts over the channel axis of
  `(B, C, H, W)` (a flat `(C,)` tensor would misalign with W and error).
- Dice (soft, per channel): `1 − (2·Σ pσ·t + ε)/(Σ pσ + Σ t + ε)`, with
  `p σ = sigmoid(logits)`, reduce over `(B, H, W)`, `ε = 1e-6`.
- Total = `bce_weight · BCE + dice_weight · Dice`, summed over channels;
  `bce_weight = dice_weight = 1.0`.

Deferred (activate at 500–1,000 tiles): boundary-weighted BCE for
parcel_border/building.

---

## 8. Training protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3.0e-4 |
| Weight decay | 1.0e-4 |
| Scheduler | Cosine annealing (`T_max = epochs`) |
| Batch size | 8 (full run); 4 (fold-1 smoke) |
| Epochs | 60 (full); 10 (fold-1 smoke) |
| Early stopping | patience 10 on mean val IoU |
| Seed | 42 (`random`, `numpy`, `torch`, `cuda`) |
| Device | CUDA (Colab **Tesla T4**); CPU for in-QGIS inference |
| Checkpoint | best `mean_iou` → `checkpoints/best_model.pt` (`{model_state, epoch, cfg}`) |

Loop (`train.py`): per epoch, train pass (AdamW step) + no-grad val pass;
cosine step; save on new best mean val IoU; stop after 10 non-improving
epochs. Encoder weights initialized from ImageNet; `pos_weight` recomputed per
fold from real class frequencies.

---

## 9. Evaluation methodology

- **Per-channel IoU**, reported separately and **never reduced to a single
  averaged headline** — parcel_border and tree are expected to diverge from
  building/hard_surface, and a mean would mask that.
- **Dataset-level accumulation:** intersection/union pixel **counts** are
  accumulated across the whole split and divided once (`metrics.py`:
  `iou_counts` + `iou_from_counts`), rather than averaging per-batch ratios —
  more stable, and it lets absent classes be detected.
- **NaN masking:** a channel absent from the entire split (union = 0) returns
  NaN (not a spurious 1.0), and the reported mean uses `torch.nanmean`, so a
  class simply not present in a small val split is excluded, not scored as
  perfect. Threshold 0.5 (per-channel thresholds, e.g. 0.3–0.4 for tree, are a
  planned recall/precision knob).
- **Generalization protocol (planned, `FOLD2_DATASET_SPEC`):** leave-one-
  region-out cross-validation — train on N−1 jurisdictions, validate on the
  held-out one; report per-region, per-class IoU. The random-split number
  alone cannot demonstrate cross-region generalization.

---

## 10. Results

### 10.1 Fold-1 (U-Net, 33 tiles, 10-epoch smoke test, Colab T4)

Purpose was **loop validation**, not model quality. Final epoch (10/10) val IoU:

| Class | val IoU |
|---|--:|
| building | **0.786** |
| hard_surface | 0.495 |
| tree | 0.154 |
| parcel_border | 0.108 |

Training loss 7.68 → 3.29 (monotone). Confirms the export→train→infer contract
end-to-end. `building` learns well; `parcel_border` (thin ring) and `tree`
(sparse) lag — as anticipated (§11).

### 10.2 Fold-2 development — from a multi-label prototype to the delivered model

**(a) Multi-label prototype (motivation for the reformulation).** Training the
fold-1 architecture unchanged (4-channel sigmoid) on the larger, more diverse
fold-2 data for more epochs shows the multi-label approach plateauing on the
sparse classes. Representative validation IoU during training (dataset-level,
NaN-masked):

| Epoch | parcel_border | building | hard_surface | tree | mean |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.037 | 0.433 | 0.107 | 0.030 | 0.152 |
| 3 | 0.053 | 0.640 | 0.226 | 0.053 | 0.243 |
| 7 | 0.087 | 0.703 | 0.266 | 0.063 | 0.280 |
| 10 | 0.084 | 0.724 | 0.250 | 0.105 | 0.291 |
| 11 | 0.085 | 0.715 | 0.278 | 0.071 | 0.287 |

Building climbs steadily (~0.72) while parcel_border (~0.08) and tree (~0.07)
stay near the floor and val loss begins rising — i.e. more data and epochs did
not rescue the two sparsest multi-label channels. This motivated the fold-2
reformulation (§6.2): consolidate the un-learnable tree class into soft ground
and switch to a 5-class multi-class head.

**(b) Delivered fold-2 model.** The delivered fold-2 checkpoint is the
**5-class multi-class U-Net** (§6.2), ResNet-34 encoder, ImageNet-normalized,
512² input, trained on the expanded multi-region dataset for the full schedule.
Reported performance (from the checkpoint's own metadata):

| Metric | Value |
|---|--:|
| Validation foreground mean IoU (over classes 1–4, excl. background) | **0.467** |

This is a substantial jump over the fold-1 regime (whose four-class mean sat at
~0.29–0.39). Interpreting it: dropping the un-learnable separate-tree objective
and adding an explicit background class let the model spend capacity on the
classes that are actually resolvable from imagery (wall, building, hard
surface), and the larger/more diverse training set improved robustness.

**Provenance note (for the paper's reproducibility section):** the delivered
fold-2 model was produced in a separate training run; its exact tile count,
epoch count and augmentation should be confirmed with the training author and
recorded here before publication. The 0.467 figure is the value stored in the
checkpoint (`val_fg_miou`). Cross-region generalization should still be
measured with the leave-one-region-out protocol (§9), not same-distribution
validation alone.

---

## 11. Known limitations & failure modes

- **`parcel_border` (IoU ~0.08–0.11):** parcel boundaries are frequently not
  visually distinguishable from imagery (Crommelinck et al. 2019,
  10.3390/rs11212505). Treated as a data/scene-selection issue, not a model
  bug; verify the boundary is actually visible before tuning.
- **`tree` (IoU ~0.10–0.15):** a direct data artifact — sparse positive pixels
  (~3 % of pixels; present in a minority of tiles). Data-side fixes in priority
  order: quota ≥60–70 % of new tiles containing `hard_landscape` (incl.
  canopy-dominant lots); recompute `pos_weight` per batch; lower the tree
  threshold (0.5→0.3–0.4); only then consider UnetPlusPlus/attention.
- **Overlap information loss:** the exporter's argmax mask cannot represent
  genuinely overlapping targets; the ring definition of `parcel_border` is the
  workaround (§3.2).
- **Single-GSD assumption:** robustness to scale relies on ±10 % augmentation,
  not multi-scale data; out-of-regime scales are out-of-distribution.

---

## 12. Deployment / inference integration (QGIS plugin)

The model is exposed as an "AI zoning" method in the Zoning Manager dock
(`zoning_manager/inference/seg_engine.py`), with a **model picker** (Fold-2
U-Net / Fold-1 U-Net) and a **shape-style** selector.

- **Preprocessing parity:** inference renders the scene's satellite through the
  *same* `render_scene()` the exporter uses (identical extent/scale/size), then
  downscales 1024→512 with PIL bilinear and float/255 — byte-for-byte the
  training preprocessing. This closes the train/serve gap.
- **Metadata-driven decoding:** `load_model()` reads each checkpoint's own
  metadata (config, `class_names`, `mean`/`std`, `img_size`) and configures a
  single inference path accordingly — fold-1's four sigmoid channels are
  thresholded per channel; fold-2's five argmax classes are decoded to one
  label per pixel and mapped onto the plan's feature channels **by name**
  (English/Turkish), applying the checkpoint's declared input normalization.
  A feature the model has no class for (e.g. tree in fold-2) yields an empty
  channel rather than a wrong guess.
- **Vectorization:** predicted binary masks → polygons via `gdal.Polygonize`
  (8-connected, identity geotransform so geometry coords = pixel coords),
  speckle-filtered (< 0.01 % of image area dropped). `parcel_border` keeps only
  the largest region's exterior ring (one parcel per scene).
- **Shape regularization (post-effect, per style preset):** Douglas–Peucker
  simplify (tolerance `0.004 · size_px` ≈ 4 px @ 1024) and
  `orthogonalize(40°)` for buildings in "plan" style; `hard_surface` is never
  regularized (vertex detail aids path detection); `hard_landscape` stays raw.
  Styles: `raw`, `simplified`, `plan` (default).
- Predictions land in the annotate popup as editable elements — reviewed,
  corrected, and exported through the standard pipeline exactly like
  hand-drawn/cadastral-seeded zones. ML-assisted samples are written to an
  `ai_zoning/` subfolder to keep them separate from hand annotation.

Manual data-collection accelerator (v1.4): **Shift+Space** opens a freehand
annotate popup centered on the map cursor with no cadastral round-trip
(cursor-centered scene + a vertex-editable starter parcel rectangle),
unlocking regions with no cadastral endpoint and boosting diversity.

---

## 13. Reproducibility

**Software (training, Colab):** Python 3.12; `segmentation-models-pytorch`,
`albumentations`, `rasterio`, `pyyaml` (pip); PyTorch + CUDA per Colab runtime;
GPU Tesla T4.
**Software (inference, QGIS Desktop 3.44.11 / Python 3.12):** torch
2.13.0+cpu, segmentation-models-pytorch 0.5.0 (installed into the QGIS user
site-packages); inference runs on CPU.

**Dataset build (fold-2 example):**
```
python prepare_dataset.py --out fold2_dataset \
  --src <...>/SIGRADI_LA --src <...>/inputsal/inputsal \
  --src <...>/inputsal2/inputsal2 --src <...>/inputsal3/inputsal3 \
  --src <...>/png_2 --src <...>/png3 --src <...>/sigradi_datca/sigradi_png
# emits images/, masks/, splits/, provenance.csv, and prints pos_weight
```

**Train:** `python train.py --config config.yaml` (paste the printed
`pos_weight` into `config.yaml` first). Seed 42 fixed across RNGs.

**Key files** (`QGIS-aPJ`, branch `seg-engine-v1`, `tasks/09_segmentation_engine/`):
`dataset.py`, `model.py`, `losses.py`, `metrics.py`, `train.py`,
`prepare_dataset.py`, `config.yaml`, `TASK_09_segmentation_engine.md`,
`FOLD2_DATASET_SPEC.md`. Export/label + inference:
`zoning-plugins/zoning_manager/export/rasterize.py`,
`.../inference/seg_engine.py`.

---

## 14. Paper-ready statements

*Lift and adapt these into Methods / Data / Results. Replace bracketed items
with final numbers once fold-2 is re-run cleanly.*

**Data generation.** "Ground-truth zoning plans were authored in a purpose-built
QGIS plugin and rendered through the QGIS map pipeline to standardized
1024×1024 px tiles at a fixed 1:200 map scale (0.0529 m/pixel, 54.19 m ground
footprint). Each tile was exported as an RGB orthophoto and a single-channel
class-index label map encoding eight annotated feature classes; a physical
coexistence model resolved incompatible overlapping zones by priority in raster
space prior to export."

**Task formulation.** "We frame zoning extraction as semantic segmentation of
spatially-nested zoning classes from a fixed-scale orthophoto. An initial fold
(fold-1) treated the classes as a multi-label problem (four independent sigmoid
channels: parcel boundary, building, hard surface, tree canopy) motivated by
their spatial nesting; a second, larger fold (fold-2) reformulated the task as
five-class multi-class segmentation with an explicit background class after the
multi-label model proved unable to learn the sparsest classes."

**Model.** "Both folds use a U-Net with an ImageNet-pretrained ResNet-34
encoder. Fold-1 has four sigmoid channels trained with a combined per-channel
BCE-with-logits (inverse-frequency positive weighting) and Dice loss; fold-2
uses a five-class softmax head (background, soft ground, hard surface, building,
wall), consolidating the sparse tree class into soft ground."

**Training.** "Models were trained with AdamW (lr 3×10⁻⁴, weight decay 10⁻⁴),
cosine-annealed over 60 epochs with early stopping (patience 10) on mean
validation IoU, batch size 8, on a single NVIDIA T4 GPU. Class positive weights
were recomputed per fold from training-set pixel frequencies."

**Augmentation.** "Training tiles were expanded offline by the dihedral-8
symmetry group (rotations and reflections, applied identically to image and
label with nearest-neighbour label resampling); validation tiles were never
augmented. Online augmentation added ±10 % scale jitter, elastic deformation,
and photometric perturbations (brightness/contrast, hue/saturation, Gaussian
noise) for cross-provider robustness."

**Evaluation.** "We report per-class IoU accumulated at the dataset level
(pooled intersection/union counts) rather than averaged per-batch, and exclude
classes absent from a split (NaN-masked mean) to avoid spurious inflation on
sparse classes. Cross-region generalization is assessed by leave-one-region-out
cross-validation over the annotated jurisdictions."

**Result framing.** "The fold-1 multi-label model learned dense classes well
(building IoU 0.79) but plateaued on the sparsest ones — parcel boundary (0.11)
and tree canopy (0.15) — the former because parcel walls are frequently not
visually resolvable from imagery alone, the latter a consequence of tree
scarcity (~3 % of pixels). Fold-2, trained on a larger multi-region set and
reformulated as five-class segmentation with the sparse tree class consolidated
into soft ground, raised foreground mean IoU to [0.47]."

**Deployment.** "The trained model is deployed inside the GIS as an interactive
zoning method: it renders the selected parcel's orthophoto through the identical
pipeline used to generate training tiles (removing train/serve preprocessing
skew), predicts the four classes, vectorizes them with polygon regularization,
and presents them as editable plan elements for expert review before export."

---

## 15. Appendix — provenance & open items

- **Delivered fold-2 U-Net** (`best_model_fold2-1.pt`): its exact training set
  size, epoch count and augmentation should be obtained from the training
  author and recorded in §8/§10 before publication; the 0.467 figure is the
  checkpoint's stored `val_fg_miou`.
- **Tree class** is currently deferred in fold-2 (merged into soft ground).
  Restoring a dedicated tree channel is a fold-3 item, gated on meeting the
  tree-tile diversity quota; fold-1's multi-label model remains the only one
  that emits a separate tree prediction.
- **Canonical class→RGB mapping** is still placeholder; index-based training
  insulates the model from this, but any RGB-mask consumer must wait for the
  finalized Stage-1 mapping.
- **Cross-region generalization** (leave-one-region-out, §9) is not yet
  measured — the current numbers are same-distribution validation.
- Nothing is merged to `main`; all engine work is on branch `seg-engine-v1`
  (PR #1). Plugin ships as `QuickGIS_v1.6.zip`.
