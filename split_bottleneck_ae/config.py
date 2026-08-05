"""
Hardcoded configuration for the split-bottleneck autoencoder pipeline.

Everything the training / eval / export scripts consume lives in the CONFIG
dict below. To sweep a hyperparam, edit the dict entry directly — this file
is intentionally not driven by argparse or environment variables so that a
config is a single reviewable diff.

Input assumption: the HDF5 loader produces raw per-shot pulse arrays of
shape (16, 512) (i.e. 8192 elements when flattened). The encoder trunk
flattens; downstream heads consume the bottleneck_task representation.
"""

import math
import os


_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_PROJ_DIR, "figures")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
DATA = {
    # ":"-separated list of h5 files or directories, mirroring the convention
    # used elsewhere in the repo. Point these at the raw processed h5 tree
    # (per-shot groups with Ximg + Ypdf + attrs["npulses"] / attrs["phases"]).
    "train_dirs": (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/"
        "mrco_h5/train/"
    ),
    "val_dirs":  "",   # if empty, val is carved out of train_dirs at file level
    "test_dirs": (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/"
        "mrco_h5/test/"
    ),

    # Which HDF5 dataset per shot feeds the encoder. Raw pulse data lives in
    # "Ximg" in this project's h5 layout. "Ypdf" is the clean-truth
    # reconstruction target used only by the recon head.
    "input_key":  "Ximg",
    "target_key": "Ypdf",

    # Expected per-shot image shape (rows, cols). The encoder flattens to
    # rows*cols. If the loader delivers a different shape, training will
    # error out early on the shape check in dataset.py.
    "image_shape": (16, 512),

    # File-level val split (used when val_dirs == ""); deterministic seed.
    "val_frac": 0.2,
    "split_seed": 42,

    # Pulse-count label range. The count classifier has
    # num_classes = max_pulses - min_pulses + 1, and the top class means
    # ">= max_pulses" — labels are remapped to 0 .. num_classes-1.
    "min_pulses": 1,
    "max_pulses": 4,
}


# --------------------------------------------------------------------------
# Sample budgets (TRAIN_SAMPLES-style hardcoded schedule)
# --------------------------------------------------------------------------
# Cap on the number of shots pulled per split. None = load everything under
# the given dirs. These are hard caps applied after file-level splitting;
# lower them for smoke-testing, raise for a full run.
TRAIN_SAMPLES = {
    "train": None,
    "val":   None,
    "test":  None,
}


# --------------------------------------------------------------------------
# Model architecture
# --------------------------------------------------------------------------
MODEL = {
    # Per-task bottlenecks: count goes through a plain global-avg-pool
    # (feat -> (N, 64) -> Linear) while phase goes through the
    # TimeAdaptivePoolAdapter (feat -> (N, 64*L) -> Linear). Only these two
    # (plus the encoder + adapter) ship on FPGA. The phase bottleneck feeds
    # BOTH the single-pulse head (phi0) and the two-pulse head
    # (arccos(cos Δφ)).
    "bottleneck_count": 32,

    # Widen knob #2 (phase). The Linear(tapped_dim -> B_phase) output width.
    # 32 was the initial baseline; widening this gives BOTH phase heads
    # more room downstream of the adapter. Bumped 32 -> 48 for the
    # double-pulse case; try 64 if 48 still leaves phase_two under-fit.
    "bottleneck_phase": 48,

    # Widen knob #1 (phase). adaptive_avg_pool1d over W maps the encoder's
    # time axis (~514) down to this many bins so the phase heads see coarse
    # "where along the pulse train" info instead of a scalar. Flattened
    # adapter output is (encoder_channels * adapter_pool_len)-dim, so 64*24
    # = 1536 at the current setting. Bumped 16 -> 24 to give the phase
    # heads finer time-axis resolution; try 32 next if 24 doesn't unlock
    # phase_two.
    "adapter_pool_len": 24,

    # Head widths. Each head is: bottleneck -> hidden -> output.
    # Set hidden to [] to skip the hidden layer (pure linear head).
    "count_head_hidden":        [64],
    "phase_head_single_hidden": [64],
    "phase_head_two_hidden":    [64],

    # Final decoder activation. "sigmoid" clamps to [0, 1] (matches the
    # zero-to-one target rescale). Sigmoid mirrors the raw-input denoiser
    # so the pretrained decoder weights are consistent.
    "decoder_output_activation": "sigmoid",

    # Path to the pretrained raw-input autoencoder .pth. Encoder and
    # decoder weights are loaded from here at train startup; leave as ""
    # to skip and train from scratch.
    "pretrained_denoiser_path": (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/"
        "raw_autoencoder/autoencoder_raw_best_model.pth"
    ),
}


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
LOSSES = {
    # Static loss weights used when uncertainty_weighting is False.
    # The phase weight is applied to both phase heads (single & two).
    "w_recon":        1.0,
    "w_count":        1.0,
    "w_phase_single": 1.0,
    "w_phase_two":    1.0,

    # If True, replace the static weights above with Kendall & Gal (2018)
    # uncertainty-weighted multi-task learning: each loss L_i is scaled by
    # exp(-log_var_i) plus 0.5 * log_var_i, and the log_var_i are learned
    # scalar parameters. Static weights are ignored while this is on.
    "uncertainty_weighting": False,
}


# --------------------------------------------------------------------------
# Training schedule
# --------------------------------------------------------------------------
TRAIN = {
    "batch_size":  128,
    "num_workers": 4,

    # Recon-only warm start. During these epochs the task branch is not
    # touched and only encoder + decoder receive gradients. Set to 0 to
    # skip warm-up and start joint training on epoch 0.
    "warmup_epochs": 5,

    # Total number of epochs (warmup + joint). Joint training runs for
    # (epochs - warmup_epochs) epochs.
    "epochs": 60,

    # Adam lr for both phases. Kept as one number to stay FPGA-friendly:
    # no schedule tuning stage that would balloon the training-time graph.
    "lr": 5e-4,

    # Early stopping on the joint validation loss (after warmup). Set to
    # None to disable.
    "patience": 8,

    # Seed for torch / numpy.
    "seed": 42,
}


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
IO = {
    # Where checkpoints and per-epoch metric logs are written.
    "save_dir": (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/"
        "split_bottleneck_ae/"
    ),
    # v1: dropped recon bottleneck (feat -> decoder directly), split count
    # onto its own plain-GAP path off feat, widened phase adapter_pool_len
    # 16 -> 24 and bottleneck_phase 32 -> 48. Bumped from v0 so nothing old
    # gets overwritten — every figure filename below keys off run_name.
    "run_name": "split_bottleneck_ae_v1_gap_count_wide_phase",

    # Figures always land in split_bottleneck_ae/figures/. The absolute
    # path is resolved from this file's location so the target is stable
    # regardless of where the training / eval script is launched from
    # (interactive shell, SLURM, etc).
    "figures_dir": FIGURES_DIR,
}


# --------------------------------------------------------------------------
# Convenience getters
# --------------------------------------------------------------------------
def input_dim():
    r, c = DATA["image_shape"]
    return int(r) * int(c)


def num_count_classes():
    return int(DATA["max_pulses"]) - int(DATA["min_pulses"]) + 1


# Fixed output dims for the two phase heads:
#   single -> (sin(phi0), cos(phi0)) 2-vector, decoded via atan2 to [0, 2*pi)
#   two    -> scalar in [0, pi] matching arccos(cos(phi0 - phi1))
PHASE_SINGLE_OUTPUT_DIM = 2
PHASE_TWO_OUTPUT_DIM = 1


PHASE_MAX_RADIANS = 2.0 * math.pi
