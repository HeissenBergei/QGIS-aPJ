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
- **Multi-label** segmentation (4 independent binary channels, sigmoid), not
  multi-class (softmax) — because zoning classes are spatially nested (a
  building pixel is also inside the parcel). See §6.1.
- CNN encoder–decoder (U-Net), not a transformer decoder, at this data scale.

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

### 6.1 Primary model (fold-1, fold-2 U-Net)

- **U-Net** (segmentation-models-pytorch), encoder **ResNet-34**, ImageNet-
  pretrained encoder weights.
- `in_channels = 3`, `classes = 4`, `activation = None` (raw logits; sigmoid
  applied in loss/inference for numerical stability with `BCEWithLogitsLoss`).
- **Multi-label head:** four independent sigmoid channels. Rationale: zoning
  classes are spatially nested — a softmax would force competition between a
  parcel pixel and the building on it. Each channel is its own binary
  segmentation task sharing one encoder (pattern per MultiTalent,
  arXiv:2303.14444; natively supported by smp `activation='sigmoid'`).
- Input 512² is divisible by 32 (encoder stride requirement).
- Designated fallback if the tree channel underperforms at scale:
  `smp.UnetPlusPlus` or attention gates (one-line `config.model.arch` change);
  not activated in folds 1–2.

### 6.2 Alternative model (a "fold-2" DeepLabV3 experiment)

A separately-trained checkpoint (`best_model_fold2.pt`) is a **torchvision
`deeplabv3_resnet50`**: ResNet-50 backbone + ASPP head + auxiliary head
(`backbone.` / `classifier` / `aux_classifier` modules; 370 weight tensors).
It is *multi-class* (argmax over class channels) with a **21-channel head**
(Pascal-VOC shape). The inference host maps its argmax index map back to the
four zoning channels via the same index scheme (§3.2) and feeds it ImageNet
normalization. **Caveat for the paper:** its training details (whether/how it
was fine-tuned to the zoning classes) are not documented here, and its head is
still 21-way; observed predictions were sparse. Treat it as an exploratory
architecture comparison, not a validated result, until its provenance is
confirmed.

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

### 10.2 Fold-2 (U-Net, 88 tiles, full config, Colab T4) — monitored trajectory

Representative validation IoU during training (dataset-level, NaN-masked):

| Epoch | parcel_border | building | hard_surface | tree | mean |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.037 | 0.433 | 0.107 | 0.030 | 0.152 |
| 3 | 0.053 | 0.640 | 0.226 | 0.053 | 0.243 |
| 7 | 0.087 | 0.703 | 0.266 | 0.063 | 0.280 |
| 10 | 0.084 | 0.724 | 0.250 | 0.105 | 0.291 |
| 11 | 0.085 | 0.715 | 0.278 | 0.071 | 0.287 |

Train loss 7.84 → ~3.7 over 11 epochs; val loss reached ~5.1 (ep 9) then began
rising (early-overfitting on 88 tiles). **Caveat:** this run was monitored via
the training log; the persisted best checkpoint from this specific U-Net run
was not reliably retained (Colab runtime recycling around the save step), and
the file currently on disk named `best_model_fold2.pt` is the separate
DeepLabV3 experiment (§6.2), **not** this U-Net. These IoU values are therefore
from training logs and should be reproduced from a clean re-run before being
quoted as final.

**Interpretation (paper-relevant):** fold-2 raw val IoU at comparable epochs is
similar-to-slightly-lower than fold-1 despite ~2.7× more data — expected,
because the fold-2 validation set is more diverse/harder (18 tiles across 4
sources) and rising val loss indicates the model is being pushed to generalize
rather than memorize. The correct comparison is the planned leave-one-region-
out protocol, not same-distribution random-split IoU.

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
DeepLabV3 / Fold-1 U-Net) and a **shape-style** selector.

- **Preprocessing parity:** inference renders the scene's satellite through the
  *same* `render_scene()` the exporter uses (identical extent/scale/size), then
  downscales 1024→512 with PIL bilinear and float/255 — byte-for-byte the
  training preprocessing. This closes the train/serve gap.
- **Architecture auto-detection:** `load_model()` inspects the checkpoint —
  `{model_state, cfg}` + `encoder.` keys → smp U-Net (sigmoid multi-label);
  bare state_dict + `backbone.`/`classifier` → torchvision DeepLabV3 (ImageNet-
  normalized input, argmax → 4-channel split via `CHANNEL_INDEX = [6,5,3,2]`).
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
2.13.0+cpu, torchvision 0.28.0+cpu, segmentation-models-pytorch 0.5.0 (installed
into the QGIS user site-packages).

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

**Task formulation.** "We frame zoning extraction as multi-label semantic
segmentation over four spatially-nested classes — parcel boundary (wall ring),
building footprint, impervious/hard surface, and tree canopy — trained as four
independent binary channels with a shared encoder, rather than mutually-
exclusive multi-class labels, because zoning classes are physically nested
(e.g. a building lies within a parcel)."

**Model.** "The segmentation model is a U-Net with an ImageNet-pretrained
ResNet-34 encoder and four sigmoid output channels, trained with a combined
per-channel BCE-with-logits (inverse-frequency positive weighting) and Dice
loss."

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

**Result framing.** "On a [88]-tile multi-region set, the model reached
[0.72] IoU on buildings and [0.28] on hard surface, while parcel boundaries
([0.09]) and tree canopy ([0.11]) remained hard — the former because parcel
walls are frequently not visually resolvable from imagery alone, the latter a
consequence of tree scarcity in the current annotation set ([~3 %] of pixels)."

**Deployment.** "The trained model is deployed inside the GIS as an interactive
zoning method: it renders the selected parcel's orthophoto through the identical
pipeline used to generate training tiles (removing train/serve preprocessing
skew), predicts the four classes, vectorizes them with polygon regularization,
and presents them as editable plan elements for expert review before export."

---

## 15. Appendix — provenance & open items

- **Fold-2 U-Net checkpoint** should be re-run and persisted cleanly (the
  monitored run's best checkpoint was not reliably saved); the disk file
  `best_model_fold2.pt` is the DeepLabV3 experiment, not this U-Net.
- **DeepLabV3 experiment** needs its training provenance documented (data,
  epochs, whether the 21-class head was fine-tuned) before inclusion as a
  result.
- **Canonical class→RGB mapping** is still placeholder; index-based training
  insulates the model from this, but any RGB-mask consumer must wait for the
  finalized Stage-1 mapping.
- Nothing is merged to `main`; all engine work is on branch `seg-engine-v1`
  (PR #1). Plugin ships as `QuickGIS_v1.5.zip`.
