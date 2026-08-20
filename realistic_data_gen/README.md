# realistic_data

Generate cookiebox training shards whose (kick amplitude, kick angle),
per-port hit rate, dead-time gaps, dark-count ROI, and per-port energy
jitter are matched to a real MRCO run. Everything downstream of this
folder — the split-bottleneck AE, the `how_many` classifier, the phase
regressors — reads shards with the exact same schema as the older
sim-only generators, so plugging this dataset in is a path swap, not a
loader change.

The output distribution (`Ximg (16, 512)` per shot, `Ypdf` per shot,
attrs for `npulses` / `phases` / `streak_amplitude` / `kickangle`) is
schema-compatible with `src/data_processing/svd_dataset_generator.py`
and `src/denoising/`; the changes here are all inside the generator
(pull kicks from a real river file, calibrate `drawscale` to a target
hit count, apply dead-time / dark-ROI / per-port jitter / localized
ghosts, drop pulses that overlap in phase).

## What's here

```
realistic_data/
  generate.py                  # main generator (was streak_finder/generate_streak_data_realistic.py)
  slurm/
    generate.sh                # sbatch wrapper for generate.py
    minmax.sh                  # sbatch wrapper for postprocess/minmax_shards.py
  postprocess/
    minmax_shards.py           # fit/apply the per-column MinMax(0,1) that the sibling forks expect
  calibrate/
    fit_realism_params.py      # Nelder-Mead fit of (dead_time, loc_noise_rate, loc_noise_sigma)
  validate/
    real_vs_sim_metrics.py     # per-port band-width / gap / hit-count metrics, real vs sim
    compare_real_vs_sim.py     # side-by-side pcolormesh + FWHM sharpness metric
    measure_streak_sharpness.py# streak-band column-profile sharpness (peak, FWHM, shoulder ratio)
    plot_kamp_vs_ev.py         # kangle-aligned mean shot: real duck/goose vs sim clean-truth
  kick_pool/
    build_topkamp.py           # sim-schema shard from the top-N real duck kamps (positive class)
    build_lowgmd.py            # sim-schema shard from the bottom-N gmd_energy shots (no-beam / 0-pulse)
    build_duck_goose.py        # sim-schema shard splitting top-duck (streak=1) / bottom-goose (streak=0)
```

`generate.py` is the entrypoint. The other subdirs are optional:

- `postprocess/` runs after generation if you want the per-column
  MinMax(0, 1) scaler that `src/data_processing/universal_cookiesimslim_processor.py`
  produces — the split-bottleneck AE fork consumes that scaler.
- `calibrate/` and `validate/` need a real MRCO run to run against;
  they're the loop I used to tune the noise knobs and won't be useful
  without access to the SLAC data paths (see [Real-data paths](#real-data-paths)).
- `kick_pool/` produces small real-data shards that share the sim shard
  schema, so you can point eval scripts at real duck/goose shots without
  writing a new loader.

## Prerequisites

### 1. CookieSimSlim (external dependency)

`generate.py` imports `Params` and `build_XY` from
[CookieSimSlim](https://github.com/) — the underlying physics simulator.
Point the `COOKIESIMSLIM_PATH` env var at your local checkout:

```
export COOKIESIMSLIM_PATH=$HOME/CookieSimSlim
```

The generator will `sys.exit` with a clear error if that var is unset.

### 2. Python packages

```
pip install numpy h5py scipy scikit-learn joblib matplotlib
```

No GPU / torch dependency — the generator runs on CPU only, multiprocessed
across `--n_workers`.

### 3. Real-data paths

`generate.py` **needs a real MRCO river file** to sample (kamp, kangle)
from. On the SLAC S3DF filesystem, the default is:

```
/sdf/data/lcls/ds/tmo/tmol1043723/results/streaking_results/run[99]_vjtw_river.h5
```

If you have a different river file elsewhere, pass it via `--river_path`.
The generator only reads the `duck_kamp` and `duck_kangle` datasets from
it — see `_load_river_kicks` in `generate.py:178`.

The `calibrate/` and `validate/` scripts additionally need:

- **Preproc h5** (per-shot detector traces + gmd + timestamps) — default:
  `/sdf/data/lcls/ds/tmo/tmol1043723/scratch/preproc/vjtw/run99_vjtw.h5`
- **MRCO calibration h5** (per-angle `t0` and alphas) — default:
  `/sdf/home/m/miaed/copied_ipynbs/tmo_utils/calibration/mrco_calib_tmox101_205-213_simon_t0.h5`

All three paths are exposed as CLI flags (`--preproc_path`, `--river_path`,
`--calib_path`) on every script that needs them, so you can point at your
own copies without editing the code.

## Quick start

Generate a smoke-test dataset (2 k train / 200 val / 200 test) locally:

```
export COOKIESIMSLIM_PATH=$HOME/CookieSimSlim

python3 generate.py \
    --outdir /tmp/realistic_smoke \
    --n_train 2000 --n_val 200 --n_test 200 \
    --shots_per_shard 100 \
    --n_workers 4 \
    --river_path /path/to/your/river.h5 \
    --recipe pulse_123_1of3 \
    --min_dphi_rad 0.7854 \
    --write_ypdf
```

Output layout:

```
/tmp/realistic_smoke/
  train/streak_realistic_train_00000.h5   # 100 shots per shard
  train/streak_realistic_train_00001.h5
  ...
  val/streak_realistic_val_00000.h5
  test/streak_realistic_test_00000.h5
```

Each shard is one h5 with per-shot groups; every group carries
`Ximg (16, 512) uint16`, `Ypdf (16, 512) float32` (when `--write_ypdf`
is on), and attrs `npulses`, `phases` (radians), `streak_amplitude`,
`kickstrength`, `kickangle`.

## Full pipeline

### Step 1 — generate raw shards

Under SLURM (S3DF):

```
sbatch slurm/generate.sh                 # 200k / 20k / 20k default
sbatch slurm/generate.sh smoke           # 2000 / 200 / 200
sbatch slurm/generate.sh dry             # plan shards, don't write
```

Override any knob by exporting it before `sbatch`:

```
OUT_DIR=/scratch/me/my_dataset N_TRAIN=100000 sbatch slurm/generate.sh
```

Interactively, `python3 generate.py --help` lists every flag. The key
realism knobs (all exposed on both CLI and `slurm/generate.sh`):

| flag                          | what it controls                                        |
|-------------------------------|---------------------------------------------------------|
| `--river_path`                | real MRCO river h5 (source of (kamp, kangle) pairs)     |
| `--recipe`                    | pulse-count recipe (`natural`, `pulse_123_1of3`, ...)   |
| `--min_dphi_rad`              | reject 2-pulse shots whose wrapped Δφ is below this     |
| `--target_hits`               | mean sum(Ximg) per shot (drawscale is auto-calibrated)  |
| `--dead_time_eV`              | per-port non-paralyzable dead-time gap                  |
| `--loc_noise_rate` / `--loc_noise_sigma_bins` | signal-localized ghost hits             |
| `--per_port_e_jitter_bins`    | Gaussian per-port energy-axis wobble                    |
| `--dark_min` / `--dark_max`   | uniform dark background scale range                     |
| `--dark_roi_bin_lo` / `--dark_roi_bin_hi` | energy window that dark counts live in      |
| `--frac_unstreaked`           | fraction of shots forced to kamp=0                      |
| `--p_boost` / `--boost_min` / `--boost_max` | up-sample the high-kamp band on top of the empirical draw |

### Step 2 (optional) — corpus-column MinMax(0, 1)

The split-bottleneck AE was trained against Ximg scaled by
`universal_cookiesimslim_processor.py`'s per-column MinMax(0, 1). To feed
this dataset into that AE you have to fit + apply the same scaler:

```
sbatch slurm/minmax.sh
```

or interactively:

```
python3 postprocess/minmax_shards.py fit \
    --in_dir  /tmp/realistic_smoke/train \
    --scaler_path /tmp/realistic_smoke_minmax/scaler.joblib

for split in train val test; do
    python3 postprocess/minmax_shards.py transform \
        --in_dir  /tmp/realistic_smoke/$split \
        --out_dir /tmp/realistic_smoke_minmax/$split \
        --scaler_path /tmp/realistic_smoke_minmax/scaler.joblib
done
```

The joblib is fit train-only (no test leakage) and reused verbatim for
val/test and for real inference.

Skip this step if you're training something that already normalizes
inputs internally (batch norm, layer norm, etc.).

### Step 3 (optional) — calibrate noise knobs to a real run

If you're generating for a *different* MRCO run than run 99, the default
`dead_time_eV`, `loc_noise_rate`, `loc_noise_sigma_bins` won't match.
`calibrate/fit_realism_params.py` does a Nelder-Mead fit of those three
scalars against band-width / gap-length / hit-count summary stats
extracted from the real data.

```
python3 validate/real_vs_sim_metrics.py \
    --n_real 2000 \
    --kamp_min 1.5 \
    --outdir /tmp/real_vs_sim/                   # caches metrics_real.npz

python3 calibrate/fit_realism_params.py \
    --real_npz /tmp/real_vs_sim/metrics_real.npz \
    --n_sim 800 --max_iter 30 \
    --outdir /tmp/real_vs_sim/
```

The fit prints the winning `(dead_time_eV, loc_noise_rate,
loc_noise_sigma_bins)`; feed those back into `generate.py` (or export
them into `slurm/generate.sh`).

### Step 4 (optional) — visual sanity check

```
python3 validate/compare_real_vs_sim.py \
    --sim_h5 /tmp/realistic_smoke/train/streak_realistic_train_00000.h5 \
    --n 10 \
    --out compare.png

python3 validate/measure_streak_sharpness.py \
    --sim_h5 /tmp/realistic_smoke/train/streak_realistic_train_00000.h5 \
    --out sharpness.png

python3 validate/plot_kamp_vs_ev.py \
    --outdir /tmp/kamp_scan/
```

These all need real MRCO data (river + preproc + calib) to draw the
"real" side of the comparison.

### Step 5 (optional) — real-data eval shards

`kick_pool/` produces small real-data shards in the *sim schema* so any
eval script that already reads sim shards can eat them without changes.
Handy for spot-checking a trained model on real shots.

```
# Streaked positives (top-N duck shots by kamp)
python3 kick_pool/build_topkamp.py \
    --top_n 500 \
    --output /tmp/real_topkamp/topkamp_00000.h5

# Streaked positives + unstreaked negatives split by kamp
python3 kick_pool/build_duck_goose.py \
    --top_n_duck 500 --bot_n_goose 500 \
    --output /tmp/real_duck_goose/duck_goose \
    --n_train_shards 5

# No-beam / dropped-pulse shots (npulses = 0)
python3 kick_pool/build_lowgmd.py \
    --bot_n 2000 \
    --output /tmp/real_lowgmd/lowgmd_00000.h5
```

## SLURM (S3DF) notes

Both `slurm/generate.sh` and `slurm/minmax.sh` default to:

- partition: `milano`
- account: `lcls:tmox42619`
- logs: `/sdf/home/m/miaed/slurm_logs/output-%j.txt`

Change the `#SBATCH` header block at the top of each script for your
account / partition / log dir.

## Caveats

- **Default paths are SLAC-specific.** The `--river_path`, `--preproc_path`,
  and `--calib_path` defaults point at `/sdf/data/lcls/...` and
  `/sdf/home/m/miaed/...`. Off-SLAC, override all three flags.
- **`generate.py` does not ship CookieSimSlim.** Set `COOKIESIMSLIM_PATH`
  before running.
- **The generator uses `multiprocessing`.** `--n_workers > 1` will spawn
  that many worker processes; each imports CookieSimSlim independently.
  If import is slow on your setup, set `--n_workers 1` for interactive
  debugging.
- **Recipe `pulse_123_1of3`** hard-splits 33/33/33 into 1/2/3+ pulse
  classes, and rejection-samples 2-pulse shots until |arccos(cos Δφ)|
  ≥ `--min_dphi_rad`. Use `--recipe natural` to fall back to the
  empirical `npulses` distribution the simulator produces.

## Provenance

These files were assembled from the working copy under
`streak_finder/` and `realist_data_figures/`; the intent of moving them
into their own folder is to make the "generate a realistic dataset"
pipeline standalone from the rest of `COOKIE_ML`. The originals in
`streak_finder/` were the tuning scratch space and may still diverge
from what's here.
