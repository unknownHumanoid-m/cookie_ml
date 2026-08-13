# -*- coding: utf-8 -*-
"""
Train a binary streak / no-streak classifier on the raw CookieBox Ximg.

Design reference: streak_finder/streak_finder_design.md.

- Input:  per-shot raw `Ximg`, shape (16, N_E), 16 azimuthal detectors x N_E
          energy bins (0.1 eV / bin on the current realistic dataset;
          earlier runs used 0.25 eV / bin).
- Output: single logit -> sigmoid = P(streak).
- Label:  `streak` (0/1) attribute per shot, or thresholded from a real-valued
          `streak_amplitude` (eV) via --streak_threshold_eV.
- Balancing: 50/50 via WeightedRandomSampler (oversample the minority class
             with augmentation-induced variety rather than raw repetition;
             see design doc s3).
- Loss:   BCE-with-logits by default; --loss focal switches to focal loss.
- Arch:   small CNN with a matched-filter-shaped first layer and circular
          angular padding.  Checkpoints pack arch / label semantics for eval.

Example
-------
    python3 streak_finder_training.py \\
        --data_dirs /path/to/mrco_h5_svd/train/ \\
        --input_key Ximg \\
        --streak_attr streak_amplitude --streak_threshold_eV 0.5 \\
        --save_dir /path/to/runs/ --save_model streak_finder.pth
"""

import argparse
import copy
import math
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


##############################################################################
# I/O helpers
##############################################################################
def collect_h5_files(paths):
    files = []
    for p in paths:
        if not p:
            continue
        if os.path.isfile(p) and p.endswith(".h5"):
            files.append(p)
        elif os.path.isdir(p):
            files.extend(
                os.path.join(p, n) for n in sorted(os.listdir(p)) if n.endswith(".h5")
            )
        else:
            raise FileNotFoundError(f"Not an .h5 file or directory: {p}")
    if not files:
        raise RuntimeError(f"No .h5 files found under {paths}")
    return files


def split_files_train_val(files, val_frac, seed=42):
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


def _read_streak_label(attrs, streak_attr, threshold_eV):
    """Return (label:int, amplitude:float|None). Falls through the two
    supported conventions:
      1. attrs[streak_attr] is boolean/int (0/1)  -> direct label
      2. attrs[streak_attr] is a real (eV)        -> label = (amp >= threshold)
    Amplitude is also returned so eval can compute sensitivity vs magnitude.
    """
    if streak_attr not in attrs:
        raise KeyError(
            f"h5 group is missing attr '{streak_attr}'. Adjust --streak_attr "
            f"or add the label to the dataset generator."
        )
    v = attrs[streak_attr]
    v = float(np.asarray(v).reshape(-1)[0])
    if threshold_eV is None:
        return int(v > 0.5), v
    return (1 if v >= threshold_eV else 0), v


def _apply_input_norm(x, input_norm):
    """Stateless per-pixel input normalization for the raw Ximg. Applied at
    load time (once) so the training loop and augmentation see the normalized
    tensor; the same choice is baked into the checkpoint so eval mirrors it.

    - 'none'  : pass through
    - 'log1p' : natural for sparse Poisson counts on Ximg; compresses the
                dynamic range and lifts the many exact zeros off the input
                floor. FPGA-friendly (fixed pointwise op, no fitted params).
    """
    if input_norm == "none":
        return x
    if input_norm == "log1p":
        return np.log1p(x, dtype=np.float32)
    raise ValueError(f"Unknown --input_norm: {input_norm}")


def load_streak_h5(files, input_key, streak_attr, threshold_eV, input_norm,
                   tag=""):
    """Preload all shots. Returns numpy arrays so we can plug into a lightweight
    torch Dataset with per-item augmentation.
    """
    inputs, labels, amps = [], [], []
    for path in files:
        print(f"streak[{tag}]: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                y, a = _read_streak_label(grp.attrs, streak_attr, threshold_eV)
                inputs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                labels.append(y)
                amps.append(a)
    if not inputs:
        raise RuntimeError(f"No shots loaded from {files}")

    x = np.stack(inputs).astype(np.float32)
    x = _apply_input_norm(x, input_norm).astype(np.float16)
    y = np.asarray(labels, dtype=np.float32)
    a = np.asarray(amps, dtype=np.float32)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(
        f"streak[{tag}]: {len(y)} shots, input_key={input_key}, "
        f"input_norm={input_norm}, shape={tuple(x.shape[1:])}, "
        f"pos={n_pos}, neg={n_neg}, "
        f"x[min={x.min():.3f} mean={x.mean():.3f} max={x.max():.3f}]"
    )
    return x, y, a


##############################################################################
# Dataset with physics-grounded augmentation
##############################################################################
class StreakDataset(Dataset):
    """Wraps preloaded (X, y, amp) with per-item augmentation.
    Augmentations enabled by default on training set; leave disabled on val.
    """

    def __init__(
        self,
        x,
        y,
        amp,
        augment=False,
        angular_roll=True,
        energy_shift_bins=0,
        detector_gain_sigma=0.0,
        noise_sigma=0.0,
        channel_dropout_p=0.0,
        rng_seed=None,
    ):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.amp = torch.from_numpy(amp)
        self.augment = augment
        self.angular_roll = angular_roll
        self.energy_shift_bins = int(energy_shift_bins)
        self.detector_gain_sigma = float(detector_gain_sigma)
        self.noise_sigma = float(noise_sigma)
        self.channel_dropout_p = float(channel_dropout_p)
        # per-worker rng seeded on first access to avoid identical draws
        self._seed = rng_seed

    def __len__(self):
        return self.x.shape[0]

    def _rng(self):
        # torch.Generator per call is fine here — augmentation is cheap.
        g = torch.Generator()
        if self._seed is not None:
            g.manual_seed(int(self._seed))
        return g

    def __getitem__(self, i):
        img = self.x[i].float()  # stored fp16 to halve host memory; model wants fp32
        y = self.y[i]
        amp = self.amp[i]

        if self.augment:
            if self.angular_roll:
                k = int(torch.randint(0, img.shape[0], (1,)).item())
                img = torch.roll(img, shifts=k, dims=0)
            if self.energy_shift_bins > 0:
                s = int(
                    torch.randint(
                        -self.energy_shift_bins,
                        self.energy_shift_bins + 1,
                        (1,),
                    ).item()
                )
                if s != 0:
                    img = torch.roll(img, shifts=s, dims=1)
            if self.detector_gain_sigma > 0:
                # log-normal gain per detector, mean 0 in log-space
                log_g = self.detector_gain_sigma * torch.randn(img.shape[0])
                img = img * torch.exp(log_g)[:, None]
            if self.noise_sigma > 0:
                img = img + self.noise_sigma * torch.randn_like(img)
            if self.channel_dropout_p > 0:
                if torch.rand(()).item() < self.channel_dropout_p:
                    j = int(torch.randint(0, img.shape[0], (1,)).item())
                    img[j] = 0.0

        return img.unsqueeze(0), y, amp


##############################################################################
# Model
##############################################################################
class CircularAngularPad2d(nn.Module):
    """Circular padding on the angular (H = 16 detectors) axis; zero padding
    on the energy axis. Keeps the physics: streak wraps around in phi.
    """

    def __init__(self, pad_phi, pad_e):
        super().__init__()
        self.pad_phi = int(pad_phi)
        self.pad_e = int(pad_e)

    def forward(self, x):
        # x: (B, C, H, W); pad energy first, then wrap phi
        if self.pad_e > 0:
            x = F.pad(x, (self.pad_e, self.pad_e, 0, 0), mode="constant", value=0.0)
        if self.pad_phi > 0:
            x = F.pad(x, (0, 0, self.pad_phi, self.pad_phi), mode="circular")
        return x


class StreakCNN(nn.Module):
    """Small CNN: matched-filter-shaped first layer, angular-circular padding,
    energy pooling, global-avg-pool head. Sized to ~3-4k params (see design
    doc s4.1, matched to the Rahimifar 2024 3,433-param FCNN precedent).
    """

    def __init__(
        self,
        input_shape,      # (H, W) = (16, N_E)
        c1=4,
        c2=8,
        k_phi=8,
        k_e=12,
        pool_e=4,
        hidden=16,
    ):
        super().__init__()
        h, w = input_shape
        self.input_shape = (h, w)
        self.c1, self.c2 = c1, c2
        self.k_phi, self.k_e = k_phi, k_e
        self.pool_e = pool_e
        self.hidden = hidden

        pad_phi = k_phi // 2
        pad_e = k_e // 2
        self.pad1 = CircularAngularPad2d(pad_phi=pad_phi, pad_e=pad_e)
        # 'valid' after our own padding; extra half-pixel for even kernel
        self.conv1 = nn.Conv2d(1, c1, kernel_size=(k_phi, k_e), padding=0)
        self.bn1 = nn.BatchNorm2d(c1)

        self.pool_e_layer = nn.AvgPool2d(kernel_size=(1, pool_e))

        # depthwise-separable second stage: cheap on FPGA
        self.pad2 = CircularAngularPad2d(pad_phi=1, pad_e=1)
        self.conv2_dw = nn.Conv2d(c1, c1, kernel_size=3, groups=c1, padding=0)
        self.bn2_dw = nn.BatchNorm2d(c1)
        self.conv2_pw = nn.Conv2d(c1, c2, kernel_size=1)

        self.fc1 = nn.Linear(c2, hidden)
        self.bn_fc1 = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, 1, 16, N_E)
        x = self.pad1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # crop out the +1 from even-kernel circular pad (k_phi=8 -> pad=4 =>
        # output has 16+8-8+1 = 17 rows; we drop the last to restore 16)
        if x.shape[2] > self.input_shape[0]:
            x = x[:, :, : self.input_shape[0], :]
        if x.shape[3] > self.input_shape[1]:
            x = x[:, :, :, : self.input_shape[1]]

        x = self.pool_e_layer(x)
        x = self.pad2(x)
        x = F.relu(self.bn2_dw(self.conv2_dw(x)))
        x = F.relu(self.conv2_pw(x))

        # GAP over phi and energy
        x = x.mean(dim=(2, 3))
        x = F.relu(self.bn_fc1(self.fc1(x)))
        return self.fc_out(x).squeeze(-1)  # (B,)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##############################################################################
# Losses
##############################################################################
class FocalLossWithLogits(nn.Module):
    """Focal loss for binary classification (Lin et al. 2017)."""

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, target):
        # target: 0/1 float
        p = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, p, 1 - p)
        alpha_t = torch.where(target > 0.5, self.alpha, 1 - self.alpha)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (alpha_t * (1 - pt).pow(self.gamma) * ce).mean()


##############################################################################
# Train / eval loops
##############################################################################
def _tpr_at_fpr(scores, labels, target_fpr):
    """Empirical TPR at the highest threshold whose FPR <= target_fpr.
    Returns nan if there aren't enough negatives to resolve target_fpr.
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    fp = np.cumsum(labels == 0)
    tp = np.cumsum(labels == 1)
    fpr = fp / n_neg
    tpr = tp / n_pos
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(tpr[ok[-1]])


def _accuracy_at_half(scores, labels):
    """Fraction of shots whose sigmoid score agrees with the label at thr=0.5."""
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if len(scores) == 0:
        return float("nan")
    pred = (scores >= 0.5).astype(np.float32)
    return float((pred == labels).mean())


def _roc_auc(scores, labels):
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    order = np.argsort(-scores)
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    return float(np.trapz(tpr, fpr))


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    scores = []
    labels_all = []
    for img, y, _ in loader:
        img = img.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_all.append(y.detach().cpu().numpy())
    return (
        total_loss / max(1, n_batches),
        np.concatenate(scores),
        np.concatenate(labels_all),
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    scores = []
    labels_all = []
    with torch.no_grad():
        for img, y, _ in loader:
            img = img.to(device)
            y = y.to(device)
            logits = model(img)
            total_loss += float(criterion(logits, y).item())
            n_batches += 1
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels_all.append(y.cpu().numpy())
    return (
        total_loss / max(1, n_batches),
        np.concatenate(scores),
        np.concatenate(labels_all),
    )


##############################################################################
# Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--input_key", type=str, default="Ximg",
                        help="Per-shot h5 dataset (default 'Ximg' -- raw, "
                             "pre-denoising; the streak finder is designed "
                             "to run on raw input, not Ypdf_denoised).")
    parser.add_argument("--streak_attr", type=str, default="streak_amplitude",
                        help="Per-shot attr carrying the streak label. If "
                             "float, thresholded by --streak_threshold_eV; "
                             "if 0/1, used directly.")
    parser.add_argument("--streak_threshold_eV", type=float, default=None,
                        help="Threshold in eV to turn --streak_attr into a "
                             "binary label. Leave unset if the attr is 0/1.")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--input_norm", type=str, default="log1p",
                        choices=["none", "log1p"],
                        help="Per-pixel input normalization applied once at "
                             "load time. 'log1p' compresses Poisson counts "
                             "(default; sparse Ximg has many exact zeros). "
                             "Baked into the checkpoint so eval mirrors it.")

    # training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--balanced_sampler", type=int, default=1,
                        help="1 = WeightedRandomSampler to 50/50 (default). "
                             "Set 0 to sample at the natural prior.")

    # architecture. Defaults target ~3k params to match the Rahimifar 2024
    # 3,433-param FCNN precedent for the deployed LCLS-II CookieBox
    # (design doc s4.4). Adjust down for tighter FPGA budgets.
    parser.add_argument("--c1", type=int, default=16)
    parser.add_argument("--c2", type=int, default=32)
    parser.add_argument("--k_phi", type=int, default=8,
                        help="Angular kernel width. Should be <= 16.")
    parser.add_argument("--k_e", type=int, default=12,
                        help="Energy kernel width (bins). 12 covers the "
                             "~8-bin streak footprint with margin.")
    parser.add_argument("--pool_e", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)

    # augmentation
    parser.add_argument("--aug_angular_roll", type=int, default=1)
    parser.add_argument("--aug_energy_shift_bins", type=int, default=2)
    parser.add_argument("--aug_detector_gain_sigma", type=float, default=0.05)
    parser.add_argument("--aug_noise_sigma", type=float, default=0.0,
                        help="Additive Gaussian std in Ximg units. 0 disables.")
    parser.add_argument("--aug_channel_dropout_p", type=float, default=0.02)

    # I/O
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_paths = [d for d in args.data_dirs.split(":") if d]
    all_files = collect_h5_files(data_paths)
    train_files, val_files = split_files_train_val(
        all_files, args.val_frac, seed=args.seed,
    )
    print(f"Train files ({len(train_files)}): "
          f"{[os.path.basename(f) for f in train_files]}")
    print(f"Val files ({len(val_files)}): "
          f"{[os.path.basename(f) for f in val_files]}")
    if not val_files:
        raise RuntimeError("File-level split produced no val files.")

    x_tr, y_tr, a_tr = load_streak_h5(
        train_files, args.input_key, args.streak_attr,
        args.streak_threshold_eV, args.input_norm, tag="train",
    )
    x_va, y_va, a_va = load_streak_h5(
        val_files, args.input_key, args.streak_attr,
        args.streak_threshold_eV, args.input_norm, tag="val",
    )
    if x_tr.shape[1:] != x_va.shape[1:]:
        raise RuntimeError(
            f"train vs val input_shape mismatch: {x_tr.shape[1:]} vs {x_va.shape[1:]}"
        )
    input_shape = tuple(x_tr.shape[1:])  # (16, N_E)
    if input_shape[0] != 16:
        print(f"WARNING: expected 16 angular detectors, got shape={input_shape}")

    train_ds = StreakDataset(
        x_tr, y_tr, a_tr,
        augment=True,
        angular_roll=bool(args.aug_angular_roll),
        energy_shift_bins=args.aug_energy_shift_bins,
        detector_gain_sigma=args.aug_detector_gain_sigma,
        noise_sigma=args.aug_noise_sigma,
        channel_dropout_p=args.aug_channel_dropout_p,
    )
    val_ds = StreakDataset(x_va, y_va, a_va, augment=False)

    if args.balanced_sampler:
        # WeightedRandomSampler forces 50/50 in expectation over an epoch.
        pos = (y_tr > 0.5).astype(np.float32)
        neg = 1.0 - pos
        n_pos = pos.sum()
        n_neg = neg.sum()
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(
                f"Cannot balance-sample: train has n_pos={int(n_pos)}, "
                f"n_neg={int(n_neg)}."
            )
        w = pos / n_pos + neg / n_neg
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(w).double(),
            num_samples=len(y_tr),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model / loss
    # ------------------------------------------------------------------
    model = StreakCNN(
        input_shape=input_shape,
        c1=args.c1, c2=args.c2,
        k_phi=args.k_phi, k_e=args.k_e,
        pool_e=args.pool_e, hidden=args.hidden,
    ).to(device)
    n_params = count_params(model)
    print(f"StreakCNN params: {n_params}")

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = FocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    val_tpr_1e2, val_tpr_1e3 = [], []
    actual_epochs = 0

    start_time = time.time()
    for epoch in range(args.epochs):
        actual_epochs += 1
        tl, tr_scores, tr_labels = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        vl, va_scores, va_labels = evaluate(
            model, val_loader, criterion, device,
        )
        tr_auc = _roc_auc(tr_scores, tr_labels)
        va_auc = _roc_auc(va_scores, va_labels)
        tr_acc = _accuracy_at_half(tr_scores, tr_labels)
        va_acc = _accuracy_at_half(va_scores, va_labels)
        va_tpr_2 = _tpr_at_fpr(va_scores, va_labels, 1e-2)
        va_tpr_3 = _tpr_at_fpr(va_scores, va_labels, 1e-3)
        train_losses.append(tl); val_losses.append(vl)
        train_aucs.append(tr_auc); val_aucs.append(va_auc)
        val_tpr_1e2.append(va_tpr_2); val_tpr_1e3.append(va_tpr_3)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"| Train Loss {tl:.4f} AUC {tr_auc:.4f} Acc {tr_acc:.4f} "
            f"| Val Loss {vl:.4f} AUC {va_auc:.4f} Acc {va_acc:.4f} "
            f"TPR@FPR1e-2 {va_tpr_2:.3f} TPR@FPR1e-3 {va_tpr_3:.3f}"
        )
        stopper(vl, model)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    best_epoch = int(np.argmin(val_losses)) if val_losses else 0
    if stopper.best_weights is not None:
        model.load_state_dict(stopper.best_weights)
        print(f"Rolled back to best weights (Val Loss {stopper.best_loss:.4f}).")

    print(f"Training time: {time.time() - start_time:.1f}s")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    if args.save_model is not None:
        save_dir = args.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, args.save_model)
        torch.save({
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "input_key": args.input_key,
            "input_norm": args.input_norm,
            "streak_attr": args.streak_attr,
            "streak_threshold_eV": args.streak_threshold_eV,
            "arch": {
                "c1": args.c1, "c2": args.c2,
                "k_phi": args.k_phi, "k_e": args.k_e,
                "pool_e": args.pool_e, "hidden": args.hidden,
            },
            "temperature": 1.0,
            "n_params": n_params,
        }, save_path)
        print(f"Best model saved to {save_path}")

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    epochs_range = range(1, actual_epochs + 1)
    plt.figure(figsize=(14, 4))
    plt.suptitle(
        f"Streak finder | input_key={args.input_key} "
        f"| loss={args.loss} | params={n_params}",
        fontsize=13,
    )
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_aucs, label="Train")
    plt.plot(epochs_range, val_aucs, label="Val")
    plt.title("ROC-AUC"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_tpr_1e2, label="TPR@FPR=1e-2")
    plt.plot(epochs_range, val_tpr_1e3, label="TPR@FPR=1e-3")
    plt.title("Val TPR at fixed FPR"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()

    if args.save_model is not None:
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures"
        )
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(
            figures_dir, f"training_data_for_{args.save_model}.png"
        )
        plt.savefig(fig_path)
        plt.close()
        print(f"Figure saved to {fig_path}")

        # Standalone train-vs-val loss PDF, matching the naming and style
        # used elsewhere in COOKIE_ML (see e.g.
        # src/denoising/figures/autoencoder_*_losses.pdf).
        identifier = os.path.splitext(args.save_model)[0]
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        if val_losses:
            plt.scatter(
                best_epoch, val_losses[best_epoch],
                marker="*", color="red", label="Best Epoch",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        losses_path = os.path.join(figures_dir, f"{identifier}_losses.pdf")
        plt.savefig(losses_path)
        plt.close()
        print(f"Losses plot saved to {losses_path}")


if __name__ == "__main__":
    main()
