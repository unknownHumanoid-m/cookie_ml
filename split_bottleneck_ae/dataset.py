"""
HDF5 dataset for the split-bottleneck autoencoder.

Adapted from src/ml_backbone/utils/custom_dataloader.py::DataMilking_HalfAndHalf
and src/denoising/train_how_many.py::load_how_many_from_files, with the SVD
compression step stripped: this loader now returns raw per-shot pulse arrays
(shape (rows, cols) — 16 x 512 by default) alongside the count and phase
labels stored in attrs.

Each shot in the source h5 is a group containing at least:
    * ``Ximg``    — noisy per-shot pulse image (the encoder's input)
    * ``Ypdf``    — clean-truth reconstruction target (used by decoder loss)
    * ``attrs["npulses"]``      — integer pulse count
    * ``attrs["phases"]``       — array of phase values, in "fraction of a turn"
                                  (so radians = phases[i] * 2*pi)

The loader preloads everything into memory as tensors (mirrors how the
existing train_how_many / train_phase_single scripts work — this dataset is
small enough that streaming isn't worth the complexity).
"""

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PHASE_TWO_PI = 2.0 * np.pi


def collect_h5_files(paths: Iterable[str]) -> List[str]:
    """Return a sorted list of .h5 files, accepting a mix of files and dirs.

    Ported verbatim in spirit from src/denoising/train_how_many.py so file
    discovery behaves identically to the rest of the pipeline.
    """
    files: List[str] = []
    for p in paths:
        if not p:
            continue
        if os.path.isfile(p) and p.endswith(".h5"):
            files.append(p)
        elif os.path.isdir(p):
            files.extend(
                os.path.join(p, n)
                for n in sorted(os.listdir(p))
                if n.endswith(".h5")
            )
        else:
            raise FileNotFoundError(f"Not an .h5 file or directory: {p}")
    if not files:
        raise RuntimeError(f"No .h5 files found under {list(paths)}")
    return files


def split_files_train_val(
    files: Sequence[str], val_frac: float, seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """File-level (not shot-level) split, matching train_how_many.py.

    A shot-level split leaks val provenance into train because shots inside
    the same h5 share instrument state.
    """
    n = len(files)
    if n < 2 or val_frac <= 0:
        return list(files), []
    idx = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    n_val = min(n_val, n - 1)
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])
    return [files[i] for i in train_idx], [files[i] for i in val_idx]


class PulseH5Dataset(Dataset):
    """In-memory dataset of (raw_input, clean_target, count_label, phi0, dphi).

    ``phi0`` is the first pulse's phase in [0, 2*pi), fed to the single-pulse
    phase head. ``dphi`` is the wrapped phase difference ``arccos(cos(phi0 -
    phi1))`` in [0, pi] for 2-pulse shots and NaN otherwise; the two-pulse
    head is trained/eval'd only on shots where dphi is finite.

    Parameters
    ----------
    files
        Explicit list of .h5 file paths (already split at file-level).
    input_key
        Which per-shot dataset feeds the encoder. Default is ``Ximg``
        (the raw noisy pulse image); pass ``Ypdf`` to train an idealized
        clean-in / clean-out variant.
    target_key
        Which per-shot dataset the decoder tries to reconstruct. Default is
        ``Ypdf`` (clean truth).
    min_pulses, max_pulses
        Pulse-count label range. Shots with fewer than ``min_pulses`` are
        dropped; shots with more are clamped to the top class
        (``max_pulses - min_pulses``), matching the how-many convention.
    image_shape
        Expected per-shot 2D shape (rows, cols). Loader errors out if the
        h5 data doesn't match — this is the check that flags SVD-shaped
        residual inputs sneaking in.
    max_samples
        Optional hard cap on the total number of shots pulled in.
    tag
        Print label so it's clear which split is loading.
    """

    def __init__(
        self,
        files: Sequence[str],
        input_key: str,
        target_key: str,
        min_pulses: int,
        max_pulses: int,
        image_shape: Tuple[int, int],
        max_samples: Optional[int] = None,
        tag: str = "",
    ):
        if not files:
            raise ValueError("PulseH5Dataset given empty file list.")

        exp_rows, exp_cols = int(image_shape[0]), int(image_shape[1])

        inputs: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        counts: List[int] = []
        phi0s: List[float] = []
        dphis: List[float] = []

        dropped_low = 0
        clipped_high = 0

        for path in files:
            print(f"pulse_h5[{tag}]: reading {path}")
            with h5py.File(path, "r") as f:
                for shot in f.keys():
                    grp = f[shot]
                    n = int(grp.attrs["npulses"])
                    if n < min_pulses:
                        dropped_low += 1
                        continue
                    if n > max_pulses:
                        clipped_high += 1
                    n_label = min(n, max_pulses) - min_pulses

                    x = np.asarray(grp[input_key][()], dtype=np.float32)
                    y = np.asarray(grp[target_key][()], dtype=np.float32)

                    if x.shape != (exp_rows, exp_cols):
                        raise RuntimeError(
                            f"{path}::{shot}/{input_key} has shape {x.shape}, "
                            f"expected {(exp_rows, exp_cols)}. If this is an "
                            f"SVD-compressed input, this project intentionally "
                            f"skips that step — point the loader at the raw "
                            f"processed h5 tree instead."
                        )
                    if y.shape != (exp_rows, exp_cols):
                        raise RuntimeError(
                            f"{path}::{shot}/{target_key} has shape {y.shape}, "
                            f"expected {(exp_rows, exp_cols)}."
                        )

                    # phases attr stores fraction-of-turn; convert to radians.
                    phases_all = np.asarray(grp.attrs["phases"], dtype=np.float64) * PHASE_TWO_PI
                    phi0 = float(np.mod(phases_all[0], PHASE_TWO_PI))
                    if n == 2:
                        dphi = float(np.arccos(np.cos(phases_all[0] - phases_all[1])))
                    else:
                        dphi = float("nan")

                    inputs.append(x)
                    targets.append(y)
                    counts.append(n_label)
                    phi0s.append(phi0)
                    dphis.append(dphi)

                    if max_samples is not None and len(inputs) >= max_samples:
                        break
            if max_samples is not None and len(inputs) >= max_samples:
                break

        if not inputs:
            raise RuntimeError(
                f"No shots with npulses >= {min_pulses} found in the given "
                f"files ({len(files)} files, tag={tag!r})."
            )

        x = torch.from_numpy(np.stack(inputs))    # (N, rows, cols)
        y = torch.from_numpy(np.stack(targets))   # (N, rows, cols)
        c = torch.tensor(counts, dtype=torch.long)
        phi = torch.tensor(phi0s, dtype=torch.float32)
        dphi = torch.tensor(dphis, dtype=torch.float32)

        self.inputs = x
        self.targets = y
        self.counts = c
        self.phases = phi
        self.dphi = dphi
        self.input_shape = tuple(x.shape[1:])
        self.num_classes = max_pulses - min_pulses + 1
        self.min_pulses = min_pulses
        self.max_pulses = max_pulses

        n_two = int(torch.isfinite(dphi).sum().item())
        print(
            f"pulse_h5[{tag}]: loaded {len(c)} shots, input_key={input_key}, "
            f"target_key={target_key}, shape={self.input_shape}, "
            f"classes={self.num_classes} (pulses {min_pulses}..{max_pulses}), "
            f"dropped(<{min_pulses})={dropped_low}, "
            f"clipped(>{max_pulses})={clipped_high}, "
            f"two-pulse shots (finite dphi)={n_two}"
        )

    def __len__(self):
        return int(self.inputs.shape[0])

    def __getitem__(self, idx):
        return (
            self.inputs[idx],   # (rows, cols) — flattened downstream by model
            self.targets[idx],  # (rows, cols)
            self.counts[idx],   # long, in [0, num_classes)
            self.phases[idx],   # phi0 in radians, in [0, 2*pi)
            self.dphi[idx],     # arccos(cos Δφ) in [0, pi], or NaN if n != 2
        )


def build_datasets(cfg_data, cfg_train_samples):
    """Build (train, val) datasets from a CONFIG-style dict.

    If ``cfg_data["val_dirs"]`` is empty, val is carved out of
    ``cfg_data["train_dirs"]`` at the file level using the split_seed.
    """
    train_paths = [d for d in cfg_data["train_dirs"].split(":") if d]
    train_files_all = collect_h5_files(train_paths)

    val_dirs = cfg_data.get("val_dirs", "") or ""
    if val_dirs.strip():
        val_paths = [d for d in val_dirs.split(":") if d]
        val_files = collect_h5_files(val_paths)
        train_files = train_files_all
    else:
        train_files, val_files = split_files_train_val(
            train_files_all,
            val_frac=cfg_data["val_frac"],
            seed=cfg_data["split_seed"],
        )
        if not val_files:
            raise RuntimeError(
                "File-level split produced no val files; either point "
                "cfg.DATA['val_dirs'] at a separate directory or provide "
                "enough h5 files under train_dirs."
            )

    print(
        f"[dataset] train files ({len(train_files)}): "
        f"{[os.path.basename(f) for f in train_files]}"
    )
    print(
        f"[dataset] val files ({len(val_files)}): "
        f"{[os.path.basename(f) for f in val_files]}"
    )

    train_ds = PulseH5Dataset(
        train_files,
        input_key=cfg_data["input_key"],
        target_key=cfg_data["target_key"],
        min_pulses=cfg_data["min_pulses"],
        max_pulses=cfg_data["max_pulses"],
        image_shape=cfg_data["image_shape"],
        max_samples=cfg_train_samples.get("train"),
        tag="train",
    )
    val_ds = PulseH5Dataset(
        val_files,
        input_key=cfg_data["input_key"],
        target_key=cfg_data["target_key"],
        min_pulses=cfg_data["min_pulses"],
        max_pulses=cfg_data["max_pulses"],
        image_shape=cfg_data["image_shape"],
        max_samples=cfg_train_samples.get("val"),
        tag="val",
    )
    if val_ds.input_shape != train_ds.input_shape:
        raise RuntimeError(
            f"train vs val input_shape mismatch: "
            f"{train_ds.input_shape} vs {val_ds.input_shape}"
        )
    return train_ds, val_ds


def build_test_dataset(cfg_data, cfg_train_samples):
    test_dirs = cfg_data.get("test_dirs", "") or ""
    if not test_dirs.strip():
        raise RuntimeError("cfg.DATA['test_dirs'] is empty.")
    test_paths = [d for d in test_dirs.split(":") if d]
    test_files = collect_h5_files(test_paths)
    return PulseH5Dataset(
        test_files,
        input_key=cfg_data["input_key"],
        target_key=cfg_data["target_key"],
        min_pulses=cfg_data["min_pulses"],
        max_pulses=cfg_data["max_pulses"],
        image_shape=cfg_data["image_shape"],
        max_samples=cfg_train_samples.get("test"),
        tag="test",
    )
