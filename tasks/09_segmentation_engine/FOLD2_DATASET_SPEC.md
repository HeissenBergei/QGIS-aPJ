# Fold-2 Dataset & Training Spec — Toward a Generalizable Model

Answers "what dataset and what training do we need to make this perform on
different types of land?" Written after the fold-1 smoke test (33 tiles,
10 epochs, val IoU: building 0.79 / hard_surface 0.50 / tree 0.15 /
parcel_border 0.11) and the first in-plugin field test.

## What fold-1 proved and what it can't do

Fold-1 validated the *pipeline* (export contract → training → in-QGIS
inference). It cannot generalize: 33 tiles from essentially one imagery
regime is a memorization budget, not a training set. The weak tree channel
is a direct data artifact — only 16/33 tiles contained any tree pixels
(3% of all pixels), so the channel saw ~2.6k positive-pixel batches total.

## Volume targets

| Milestone | Annotated parcels | What it buys |
|---|---|---|
| fold-2 | 300–500 | Reliable within ONE region + imagery source; tree channel functional |
| fold-3 | 1,000–2,000 | Cross-region generalization worth claiming |
| beyond | 3,000+ | Diminishing returns; effort shifts to hard-case mining |

Annotate in ~100-tile batches and retrain/eval between batches — stop
scaling when the val curve flattens; put effort into diversity instead.

## Diversity quotas (the actual driver of generalization)

Volume without diversity just memorizes one land type better. Balance
across four axes, tracked via `manifest.csv` + `provenance.csv`:

1. **Region / jurisdiction** — use the cadastral adapters already live in
   the Zoning Manager: TKGM (Türkiye), LA County, Riverside County (Palm
   Springs), Agenzia Entrate (Toscana). Different countries = different
   parcel morphology, roof styles, vegetation, road materials.
2. **Density typology** — quota each batch: dense urban, suburban
   (fold-1's regime), rural/agricultural, forested, coastal/touristic.
3. **Parcel morphology** — small rectangular lots, large irregular
   parcels, corner lots, flag lots, parcels with no visible wall.
4. **Imagery conditions** — season, sun angle/shadow length, provider
   (Google vs Esri vs national orthophoto), renovation age.

## Tree channel fix (data-side, in priority order)

1. Quota: **≥60–70% of new tiles must contain hard_landscape**, including
   tiles where canopy DOMINATES the parcel (orchards, wooded lots).
2. Recompute `pos_weight` after every batch (`prepare_dataset.py` prints it).
3. Before touching architecture: tune the tree channel's threshold on val
   (default 0.5 → try 0.3–0.4; multi-label heads allow per-channel
   thresholds — trades precision for recall, often the whole fix).
4. Only if still weak at 500+ diverse tiles: consider UnetPlusPlus /
   attention (already flagged as the designated fallback in TASK_09).

## Scale policy

**Keep the canonical 1:200 / 54.19 m / 1024 px export regime.** A single
GSD is a feature: the plugin always feeds the model the exact scale it
trained on, which removes a whole generalization axis. Robustness to
small deviations comes from augmentation, not from multi-scale data:
- ±10% random scale jitter (crop-and-resize) — online, in `dataset.py`
- photometric: RandomBrightnessContrast, HueSaturationValue, GaussNoise
  (cross-provider imagery robustness) — online
- keep the offline dihedral-8 (`prepare_dataset.py`), keep elastic at 0.2

## Split protocol

- Training split: random by tile (already enforced pre-augmentation by
  `prepare_dataset.py` — augmented variants never straddle splits).
- **Generalization measurement: leave-one-region-out cross-validation** —
  train on N−1 regions, validate on the held-out one. Report per-region,
  per-class IoU (the NaN-masked metric already handles absent classes).
  A model that scores 0.7 building IoU on a region it never saw is
  generalizing; the random-split number alone can't tell you that.

## Training progression

| Dataset size | Recipe |
|---|---|
| now–500 tiles | resnet34 U-Net (current), 60 epochs cosine + early stop (already in `config.yaml`), recompute pos_weight per fold |
| 500–1,000 | + boundary-weighted BCE for parcel_border/building (deferred in TASK_09, activate here); per-channel threshold tuning on val |
| 1,000+ | try encoder upgrade — resnet50 / efficientnet-b3 (one-line change, `config.yaml` `model.encoder`); revisit only with evidence, CNN > ViT at this data size |

## Non-goals (unchanged from TASK_09)

- No entrance-point channels, no parking channel — future features.
- No multi-scale model, no NIR band, no transformer decoder for now.
