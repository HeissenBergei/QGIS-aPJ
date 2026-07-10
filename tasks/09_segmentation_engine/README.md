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

## First thing to actually verify before training
The data contract in `TASK_09_segmentation_engine.md` assumes 4-band
GeoTIFF masks. This has NOT been confirmed against the addon's actual
export format. Before running the Colab trial:
1. Export one sample tile + mask from the addon
2. Run the sanity-check cell in `colab_trial.ipynb`
3. If band count/order doesn't match, fix `dataset.py::_load_mask()` — this
   is the only place format assumptions live

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
