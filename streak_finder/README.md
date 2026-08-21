# Streak Finder

A per-shot binary "streak / no-streak" CNN over the **raw** `Ximg`
(16 detectors × N_E energy bins). Runs pre-denoising, in parallel with
the zero-pulse gate, and is sized for FPGA inference (~3-4k params,
matching the Rahimifar 2024 FCNN precedent for the deployed LCLS-II
CookieBox). Full design rationale is in `streak_finder_design.md`.

## Where it sits in the pipeline

```
Ximg ──┬─► 0-or-1 pulse classifier   (0classifier/)
       └─► streak_finder             (this folder)   ─┐
                                                       │
                                                       ▼
              denoise AE ─► how_many ─► phase_{single, two}
```

Every downstream stage assumes the streak / no-streak decision has
already been made on the raw image.

## Model at a glance

- Input: `Ximg`, shape `(16, N_E)` — 16 azimuthal detectors, energy bins
  along the last axis (0.1 eV/bin on the current realistic dataset;
  older runs used 0.25 eV/bin).
- Physics: the streak is a sinusoid `ΔE(φ) ∝ cos(φ − φ_streak)` across
  the 16 detectors. Detecting it is a "find-a-sinusoid-in-a-sinogram"
  problem; the first conv is a learned matched-filter bank.
- **Circular padding on the φ axis, zero padding on E.** φ is periodic
  (detector 15 wraps to 0); E has real boundaries (photoline can leave
  the window). This is why `CircularAngularPad2d` exists instead of
  `Conv2d(padding_mode="circular")`, which would wrap both axes.
- First conv `(k_phi=8, k_e=12)` covers half the ring in φ and the
  8-bin streak footprint plus margin in E.
- Depthwise-separable second stage → energy pool → GAP → tiny FC head.
  Output is a single logit; sigmoid gives P(streak).
- Even-kernel crop: `k_phi=8 → pad=4` on both sides makes the "valid"
  conv output 17 rows in φ; the network drops the extra to restore 16
  (`streak_finder_training.py:316-319`). If you flip `k_phi` to odd, the
  crop becomes a no-op.

## Data expectations

Every shot is an h5 group with at least:

- `Ximg` (or whatever `--input_key` names), shape `(16, N_E)`
- an attribute carrying the streak label (default
  `attrs["streak_amplitude"]`)

The label is binarized two ways:

1. If `streak_attr` is boolean/int, used directly.
2. If real-valued, `label = 1 if amplitude >= --streak_threshold_eV
   else 0`.

**Naming caveat.** The flag is `--streak_threshold_eV`, but the units
are **kick strength** in whatever the generator emits, not eV.
`plot_kamp_vs_ev.py` is the utility for converting kick amplitude to
the physical eV shift.

### Two `Ximg` conventions

Two on-disk conventions coexist in this repo and loaders do **not**
rescale further:

- MinMax-scaled float, produced by
  `realistic_data_gen/universal_cookiesimslim_processor.py` or
  `realistic_data_gen/postprocess/minmax_shard.py`.
- Raw uint16 counts, produced by `generate_streak_data_realistic.py`.

`--input_norm log1p` (default) is applied on top of whichever
convention was used — so `log1p` of MinMax-scaled data is not the same
tensor as `log1p` of raw counts. **Train and eval must feed the same
convention the checkpoint was trained on**, or accuracy silently
collapses.

### Class balance

Either generate a 50/50 dataset with the recipe in
`s3df_generate_streak_finder_realistic_v2.sh`, or leave
`--balanced_sampler 1` (default) which uses a `WeightedRandomSampler`
to draw 50/50 batches from an unbalanced pool.

## Data generation

Local generators, roughly in order of "what to use now":

- `generate_streak_data_realistic.py` — realistic detector noise on top
  of the simulator; the default for training the shipped model.
  Launcher: `s3df_generate_streak_finder_realistic_v2.sh`.
- `generate_streak_data.py`, `_v2/_v3/_v4` — earlier iterations, kept
  for reproducibility of older checkpoints.
- `build_real_duck_goose_h5.py`, `build_real_lowgmd_h5.py`,
  `build_real_topkamp_h5.py` — repackage real TMO runs into the shot-
  group h5 layout for held-out real evaluation.
- `minmax_realistic_shards.py` (`s3df_minmax_realistic.sh`) —
  post-process a realistic-generator output tree into the MinMax
  convention expected by the split-bottleneck encoder.

## Train / eval

**SLURM**

```
sbatch s3df_streak_finder_train_realistic_v2.sh
sbatch s3df_streak_finder_eval_realistic.sh
```

Top of each `.sh` is user-config (`TRAIN_DATA_DIRS`, `MODEL_SAVE_DIR`,
`MODEL_IDENTIFIER`, `INPUT_KEY`, hyperparams). The bottom builds the
python command.

**Interactive**

```
python3 streak_finder_training.py \
    --data_dirs /path/to/streak/train:/path/to/more/train \
    --input_key Ximg \
    --streak_attr streak_amplitude \
    --streak_threshold_eV 0.5 \
    --input_norm log1p \
    --save_dir /path/to/runs --save_model streak.pth

python3 streak_finder_eval.py \
    --data_dirs /path/to/streak/test \
    --model_path /path/to/runs/streak.pth
```

`--data_dirs` is `":"`-separated and accepts a mix of `.h5` files and
directories.

### Eval flavors

Several launchers wrap `streak_finder_eval.py` with pre-set data and
flags:

- `s3df_streak_finder_eval_realistic.sh` — held-out realistic sim.
- `s3df_streak_finder_eval_kickmix.sh` — sweep kick amplitude, plots
  sensitivity vs streak magnitude.
- `s3df_streak_finder_eval_real_holdout*.sh` — real TMO shots
  (duck-goose / low-GMD / top-kick). The `_v2_tta_mask` variant enables
  rotation-TTA and an energy-column mask to match training-time aug.
- `s3df_streak_finder_eval_realtopk.sh` — top-K real shots.

## Checkpoint format

`torch.save` dict with:

- `state_dict`
- `input_shape`, `input_key`, `input_norm`
- `streak_attr`, `streak_threshold_eV`
- `arch` — `{c1, c2, k_phi, k_e, pool_e, hidden}` so eval rebuilds the
  exact model without extra flags
- `temperature` (default 1.0; set post-hoc for calibration)
- `n_params`

Do **not** save a raw `state_dict` on its own — the eval script relies
on the packed metadata.

## Key flags (training)

| Flag | Default | Notes |
|---|---|---|
| `--input_key` | `Ximg` | Raw image; do not point at `Ypdf_denoised`. |
| `--streak_attr` | `streak_amplitude` | Per-shot label attr. |
| `--streak_threshold_eV` | `None` | If unset, expects 0/1 attr. Units = kick strength (see caveat). |
| `--input_norm` | `log1p` | Or `none`. Baked into checkpoint. |
| `--loss` | `bce` | Or `focal` (`--focal_alpha`, `--focal_gamma`). |
| `--balanced_sampler` | `1` | 0 = natural prior. |
| `--c1 / --c2 / --k_phi / --k_e / --pool_e / --hidden` | 16 / 32 / 8 / 12 / 4 / 32 | Sized to ~3k params. Trim for tighter FPGA budgets. |
| `--aug_angular_roll` | `1` | Cheap symmetry-consistent aug. |
| `--aug_energy_shift_bins` | `2` | Simulates BW jitter / KE drift. |
| `--aug_detector_gain_sigma` | `0.05` | Log-normal per-channel gain. |
| `--aug_mask_*` | off | Zero out an energy column range during training; pair with `--energy_col_lo/hi` at eval time. |

See `streak_finder_training.py` for the complete list.

## Figures

`./figures/` next to the entrypoint by default, or wherever
`--figures_dir` points. Training writes a 3-panel loss / ROC-AUC /
TPR-at-FPR curve; eval writes confusion matrix, ROC, and (for
kick-mix runs) sensitivity vs streak magnitude.
