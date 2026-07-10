# Fold-1 Segmentation Engine — Scaffold

Drop this into the monorepo (e.g. as `tasks/09_segmentation_engine/`) alongside
the existing Task 01–07 structure, and point Claude Code at
`TASK_09_segmentation_engine.md` for the full spec and rationale.

## Files
- `TASK_09_segmentation_engine.md` — spec, architecture rationale, data
  contract, known hard cases. Read this first.
- `config.yaml` — hyperparameters. `pos_weight` values are placeholders,
  recompute from real class frequencies once data lands.
- `dataset.py` — PyTorch Dataset. `_load_mask()` is the one function likely
  to need editing once the real addon export format is confirmed.
- `model.py` — SMP U-Net factory, sigmoid multi-label output.
- `losses.py` — combined BCE + Dice, per-channel.
- `metrics.py` — per-channel IoU (not averaged in the primary report).
- `train.py` — training loop, entry point.
- `colab_trial.ipynb` — smoke-test notebook for the small mirrored dataset.

## Data contract (confirmed against the addon export)
The contract has been reconciled with the Zoning Manager addon
(`zoning_manager/export/rasterize.py`). Images = the addon's
`<stem>_satellite.png` (RGB); masks = `<stem>_mask_index.png`, a
**single-channel multi-class index map** (NOT a 4-band stack).
`dataset.py::_load_mask()` expands that index map into the four binary
channels (parcel_border ring, building, hard_surface, tree←`hard_landscape`)
via the `ADDON_INDEX` mapping — the only place format assumptions live.
Before running the Colab trial:
1. Export one sample tile from the addon (`_satellite.png` + `_mask_index.png`)
2. Run the sanity-check cell in `colab_trial.ipynb` — it prints the index
   map's unique class values so you can confirm they match `ADDON_INDEX`
3. If the index values differ, adjust `ADDON_INDEX` in `dataset.py`

## Suggested Claude Code workflow
Given the existing git worktree pattern:
1. New worktree branch for this task (`seg-engine-v1` or similar)
2. Confirm data contract against real export (above) before writing more
   code
3. Run `colab_trial.ipynb` smoke test (10 epochs, small batch) on the
   mirrored dataset — this validates the loop, not model quality
4. Review per-channel IoU, especially `parcel_border` — check a few
   validation tiles by eye before assuming a bug
5. Only then scale to full epoch count / full small-mirrored dataset
