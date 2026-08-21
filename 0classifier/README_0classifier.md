# Zero-Pulse Classifier (`0classifier/`)

> This README is written for the GitHub `0classifier/` folder. The
> source lives locally in `src/denoising/` — move the README with the
> `.py` files if you consolidate the layout.

A per-shot binary "0 vs ≥1 pulses" MLP over the raw `Ximg`. Runs
**first** on the FPGA path — every downstream stage (streak finder,
denoise AE, `how_many`, phase regressors) is only meaningful for shots
this net calls "≥1". Trimmed sibling of `train_how_many.py`; keeps
the same checkpoint layout so the two live in the same eval world.

## Where it sits in the pipeline

```
Ximg ──┬─► 0-or-1 pulse classifier   (this folder)   ─┐
       └─► streak_finder                              │
                                                       │
                                (only if ≥1) ◄─────────┘
              denoise AE ─► how_many ─► phase_{single, two}
```

## Model

`SimpleMLP` — three hidden layers with ReLU + dropout(0.3):

```
input (flat 16 × N_E) ─► 512 ─► 256 ─► 128 ─► 2 logits
```

Conv layers were intentionally left out per the plain-MLP baseline
request. If you need a conv variant, `train_how_many.py` is the sibling
to fork.

## Data expectations

Standard shot-group h5:

- Each shot is an h5 group.
- Dataset `--input_key` (default `Ximg`), any 2-D shape — flattened
  before the MLP.
- Attribute `npulses` (int) drives the label:
  `class 0 = npulses == 0`, `class 1 = npulses >= 1`.

**No normalization is applied by the loader.** Feed the model the same
`Ximg` convention the checkpoint was trained on:

- MinMax-scaled float from
  `realistic_data_gen/universal_cookiesimslim_processor.py` or
  `postprocess/minmax_shard.py`, or
- Raw uint16 counts from `generate_streak_data_realistic.py`.

Silently feeding the wrong convention will not error — it will just
give bad numbers.

## Data generation

The zero-pulse-specific recipe currently lives (for legacy reasons) in
the `streak_finder/` folder — move it into `0classifier/` when you
next reshuffle:

- `s3df_generate_zero_pulse_realistic.sh` — 50/50 split, "≥1" half is
  drawn from the standard multi-pulse pool. Recipe:
  `zero_pulse_5050`.
- `s3df_generate_zero_pulse_realistic_retargeted.sh` — same recipe with
  a retargeted photoline / kick pool for the currently deployed
  configuration.

## Train / eval

**Interactive**

```
python3 train_0or1_mlp.py \
    --data_dirs /path/to/zero_pulse/train \
    --input_key Ximg \
    --save_dir /path/to/runs --save_model zero_or_one.pth

python3 eval_0or1_mlp.py \
    --data_dirs /path/to/zero_pulse/test \
    --model_path /path/to/runs/zero_or_one.pth \
    --threshold 0.5           # tune on the val PR curve
    --cm_percent              # render CM as row-normalized %
```

`--data_dirs` is `":"`-separated and accepts a mix of `.h5` files and
directories.

**SLURM.** No launchers ship in `0classifier/` yet — either add one
mirroring `src/denoising/s3df_how_many_train.sh`, or run interactively
inside an `srun` shell.

## Key flags

### `train_0or1_mlp.py`

| Flag | Default | Notes |
|---|---|---|
| `--data_dirs` | required | `":"`-separated. |
| `--input_key` | `Ximg` | Per-shot dataset name. |
| `--val_frac` | 0.2 | Random split, seed 42. |
| `--batch_size` | 128 | |
| `--lr` | 1e-3 | Adam. |
| `--epochs` | 10 | |
| `--patience` | 5 | Early stopping on val loss. |
| `--save_dir / --save_model` | — | If unset, no checkpoint is written. |
| `--figures_dir` | `./figures/` | Training-curve PNG. |

### `eval_0or1_mlp.py`

| Flag | Default | Notes |
|---|---|---|
| `--data_dirs` | required | |
| `--model_path` | required | Checkpoint from `train_0or1_mlp.py`. |
| `--input_key` | (from ckpt) | Override the stored key. |
| `--threshold` | 0.5 | Prob threshold for the "≥1" class; lower to trade FN for FP. |
| `--identifier` | ckpt basename | Prefix for output PNGs. |
| `--cm_title` | — | Override the confusion-matrix title. |
| `--cm_percent` | off | Row-normalized percentages on a fixed 0-100 colormap. |
| `--figures_dir` | `./figures/` | |

Eval prints accuracy, elapsed time, TN/FP/FN/TP, precision, recall, F1
and writes:

- `confusion_matrix_for_{identifier}.png`
- `images_for_{identifier}.png` — 12 sample shots with true / pred /
  P(≥1).

## Checkpoint format

`torch.save` dict with:

- `state_dict`
- `input_shape`
- `num_classes` (must equal 2 — eval refuses otherwise)
- `min_pulses` (= 0), `max_pulses` (= 1)
- `input_key`

The `min_pulses / max_pulses` fields are carried purely so
`evaluate_how_many.py` can load a `0or1_mlp` checkpoint without knowing
it was binary — treat the top class as "≥ `max_pulses`".

Do **not** save a raw `state_dict` on its own.
