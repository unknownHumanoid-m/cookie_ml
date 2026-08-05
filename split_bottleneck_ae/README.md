# split_bottleneck_ae

Learned split-bottleneck autoencoder for the LCLS attosecond cookiebox
pulse-count / phase problem. Replaces the SVD-based pipeline
(`src/denoising_1D_svd/`, `src/ml_backbone/regressions_1D_SVD/`) with a
single joint model whose deployed subgraph is designed to be small and
cheap for FPGA inference.

## Design

Input is a raw simulated per-shot pulse image of shape **`(16, 512)`**
(pulled from the same HDF5 tree the rest of the repo already uses;
`Ximg` per shot, plus `attrs["npulses"]` and `attrs["phases"]`). No SVD
compression anywhere in this pipeline.

```
input (N, 16, 512)
  -> encoder (2D conv, 1 -> 16 -> 32 -> 64) -> feat (N, 64, 18, 514)
  -> splits into:
       (a) decoder (mirror ConvTranspose2d) -> reconstructed (N, 16, 512)
             [training only; discarded on export]
       (b) count branch:  feat -> GAP over (H,W) -> (N, 64)
             -> bottleneck_count -> count_head (pulse-count classifier)
       (c) phase branch:  feat -> TimeAdaptivePoolAdapter -> (N, 64*L)
             -> bottleneck_phase -> {phase_head_single (phi0),
                                     phase_head_two   (arccos(cos Δφ))}
  [encoder + GAP/adapter + bottlenecks + heads = deployed subgraph]
```

Reconstruction shares no bottleneck of its own — feat feeds the decoder
directly so the pretrained raw-AE inverse holds from step zero. Count and
phase get their own task-side bottlenecks: count sits on a plain 0-param
GAP (it doesn't need time-axis structure and paying the C*L flatten dim
on FPGA for it would be waste), while phase sits on the adaptive time
pool so both phase heads see coarse "where along the pulse train" info.

## Losses

Trained jointly through the shared trunk:

* **reconstruction** — pixel-wise MSE between the decoder output and the
  clean `Ypdf` target.
* **count** — cross entropy on the classifier.
* **phase** — periodic. Default is a `(sin, cos)` MSE parametrization
  which avoids the wraparound discontinuity of plain angle regression.
  A von Mises negative-log-likelihood alternative is available behind
  `LOSSES['phase_parametrization'] = 'vonmises'`.

Weighting is either static (`w_recon`, `w_count`, `w_phase` in
`config.py`) or Kendall & Gal (2018) uncertainty-weighted (learned
per-task log-variance scalars) — toggle with
`LOSSES['uncertainty_weighting']`.

All three loss components are logged separately every epoch — the
diagnostic for encoder-level gradient conflict between the recon and
task branches is watching whether any single component starts diverging
while the others improve.

## Training schedule

Two-phase:

1. **Warmup** (`TRAIN['warmup_epochs']` epochs) — recon only. Trains
   encoder + decoder. The task branch is inactive.
2. **Joint** — all three losses active simultaneously through the shared
   trunk for the remainder of `TRAIN['epochs']`.

Set `TRAIN['warmup_epochs'] = 0` to skip the warmup and start joint
training from epoch 0.

## Layout

```
config.py            # all hyperparams live here — no argparse in train.py
dataset.py           # HDF5 loader adapted from DataMilking_HalfAndHalf,
                     #   with the SVD step stripped out. Returns
                     #   (raw_input, clean_target, count_label, phase_rad).
model.py             # SplitBottleneckAE + InferenceSubgraph
losses.py            # recon MSE, CE count, sin/cos or von Mises phase,
                     #   Kendall multi-task uncertainty combiner
train.py             # warm-start + joint training with per-loss logging
eval.py              # recon MSE, count accuracy + confusion, phase
                     #   circular error, sanity figures
export_inference.py  # strip decoder; save the deployed subgraph as
                     #   TorchScript + state_dict + meta JSON
figures/             # training / eval figures (default landing dir)
s3df_train.sh        # SLURM launcher (ampere partition, 32 GB)
```

## Running

Everything is driven by `config.py`. Edit paths, arch dims, and loss
weights there before launching.

### Interactive

```bash
cd split_bottleneck_ae
python3 train.py
python3 eval.py           # uses IO['save_dir']/IO['run_name']/model.pth
python3 export_inference.py
```

### SLURM

```bash
sbatch split_bottleneck_ae/s3df_train.sh
```

The launcher requests the `ampere` partition (GPU), 32 GB RAM, and writes
logs to `/sdf/home/m/miaed/slurm_logs/`. Adjust the request in the
launcher header if the job is OOM'ing.

## Data expectations

Each HDF5 file is a flat set of per-shot groups. Each group must contain:

* `Ximg`  — raw noisy pulse image, shape `(16, 512)`, dtype float
* `Ypdf`  — clean truth for the same shot, shape `(16, 512)`
* `attrs["npulses"]` — integer pulse count
* `attrs["phases"]`  — array of phase values, in *fraction-of-a-turn*
  units (radians = value * 2π). The loader wraps the first phase into
  `[0, 2π)`.

The loader errors out with an explicit message if the input array does
not have the expected `(16, 512)` shape — this is the guardrail against
accidentally pointing it at an SVD-compressed tree.
