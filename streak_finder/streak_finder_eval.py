# -*- coding: utf-8 -*-
"""
Evaluate the binary streak / no-streak classifier.

Reads the checkpoint dict written by streak_finder_training.py (arch, input
shape, streak_attr / threshold), runs on a test set drawn at the *deployment*
prior, and produces:
  * ROC curve + PR curve
  * TPR at fixed FPRs {1e-2, 1e-3, 1e-4}
  * Sensitivity vs streak-magnitude (TPR bucketed by ΔE_max)
  * Reliability diagram (before / after temperature scaling)
  * Confusion matrix at the chosen operating point
  * Threshold-moving hint for a target deployment prior

Example
-------
    python3 streak_finder_eval.py \\
        --data_dirs /path/to/mrco_h5_svd/test/ \\
        --model_path /path/to/runs/streak_finder.pth
"""

import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


##############################################################################
# Model — mirror of training script so eval doesn't import training internals.
##############################################################################
class CircularAngularPad2d(nn.Module):
    def __init__(self, pad_phi, pad_e):
        super().__init__()
        self.pad_phi = int(pad_phi)
        self.pad_e = int(pad_e)

    def forward(self, x):
        if self.pad_e > 0:
            x = F.pad(x, (self.pad_e, self.pad_e, 0, 0), mode="constant", value=0.0)
        if self.pad_phi > 0:
            x = F.pad(x, (0, 0, self.pad_phi, self.pad_phi), mode="circular")
        return x


class StreakCNN(nn.Module):
    def __init__(self, input_shape, c1=4, c2=8, k_phi=8, k_e=12,
                 pool_e=4, hidden=16):
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
        self.conv1 = nn.Conv2d(1, c1, kernel_size=(k_phi, k_e), padding=0)
        self.bn1 = nn.BatchNorm2d(c1)

        self.pool_e_layer = nn.AvgPool2d(kernel_size=(1, pool_e))

        self.pad2 = CircularAngularPad2d(pad_phi=1, pad_e=1)
        self.conv2_dw = nn.Conv2d(c1, c1, kernel_size=3, groups=c1, padding=0)
        self.bn2_dw = nn.BatchNorm2d(c1)
        self.conv2_pw = nn.Conv2d(c1, c2, kernel_size=1)

        self.fc1 = nn.Linear(c2, hidden)
        self.bn_fc1 = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.pad1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        if x.shape[2] > self.input_shape[0]:
            x = x[:, :, : self.input_shape[0], :]
        if x.shape[3] > self.input_shape[1]:
            x = x[:, :, :, : self.input_shape[1]]
        x = self.pool_e_layer(x)
        x = self.pad2(x)
        x = F.relu(self.bn2_dw(self.conv2_dw(x)))
        x = F.relu(self.conv2_pw(x))
        x = x.mean(dim=(2, 3))
        x = F.relu(self.bn_fc1(self.fc1(x)))
        return self.fc_out(x).squeeze(-1)


##############################################################################
# Data
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


def _read_streak_label(attrs, streak_attr, threshold_eV):
    if streak_attr not in attrs:
        raise KeyError(f"h5 group missing attr '{streak_attr}'.")
    v = float(np.asarray(attrs[streak_attr]).reshape(-1)[0])
    if threshold_eV is None:
        return int(v > 0.5), v
    return (1 if v >= threshold_eV else 0), v


def parse_kick_mix(spec):
    """Parse '0:0.5,0-5:0.25,5-15:0.25' into a list of (lo, hi, frac) tuples.

    A bare integer or float means "exact-equal": the bucket matches shots with
    `kick == value` (used for the kick == 0 unstreaked shots). A 'lo-hi' range
    matches `lo <= kick < hi` — half-open so contiguous ranges don't double-count.
    Fractions must sum to ~1.0.
    """
    if not spec:
        return None
    buckets = []
    total = 0.0
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        head, frac_s = token.split(":")
        frac = float(frac_s)
        head = head.strip()
        if "-" in head:
            lo_s, hi_s = head.split("-")
            lo, hi = float(lo_s), float(hi_s)
            if hi <= lo:
                raise ValueError(f"Bad kick_mix range '{token}': hi <= lo")
            buckets.append(("range", lo, hi, frac))
        else:
            v = float(head)
            buckets.append(("exact", v, v, frac))
        total += frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"kick_mix fractions must sum to 1.0, got {total:.6f}: {spec!r}"
        )
    return buckets


def build_kick_mix_indices(kick, buckets, seed=0):
    """Return an index array carving `kick` into the requested mix.

    For each bucket, gather matching shot indices, then find N_max such that
    frac_i * N_max <= count_i for every bucket. Warns if any bucket was the
    bottleneck. Sampling is without replacement using np.random.RandomState.
    """
    kick = np.asarray(kick, dtype=np.float64)
    rng = np.random.RandomState(seed)

    bucket_indices = []
    for kind, lo, hi, frac in buckets:
        if kind == "exact":
            mask = kick == lo
            desc = f"kick == {lo:g}"
        else:
            mask = (kick >= lo) & (kick < hi)
            desc = f"{lo:g} <= kick < {hi:g}"
        idxs = np.where(np.isfinite(kick) & mask)[0]
        bucket_indices.append((desc, frac, idxs))
        print(f"kick_mix: {desc:32s} frac={frac:.3f}  available={len(idxs)}")

    per_bucket_max = []
    for desc, frac, idxs in bucket_indices:
        if frac <= 0:
            per_bucket_max.append(np.inf)
            continue
        per_bucket_max.append(len(idxs) / frac)
    n_max = min(per_bucket_max)
    n_total = int(np.floor(n_max))
    if n_total <= 0:
        raise RuntimeError(
            "kick_mix produced 0 shots -- at least one bucket is empty. "
            f"Buckets: {[(d, len(i)) for d, _, i in bucket_indices]}"
        )

    bottleneck = np.argmin(per_bucket_max)
    bottleneck_desc = bucket_indices[bottleneck][0]
    print(f"kick_mix: capping total to {n_total} shots "
          f"(bottleneck bucket: {bottleneck_desc})")

    selected = []
    for desc, frac, idxs in bucket_indices:
        n_take = int(round(frac * n_total))
        n_take = min(n_take, len(idxs))
        pick = rng.choice(idxs, size=n_take, replace=False)
        print(f"kick_mix: {desc:32s} took {n_take}")
        selected.append(pick)

    return np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)


def _apply_input_norm(x, input_norm):
    """Mirror of streak_finder_training._apply_input_norm."""
    if input_norm == "none":
        return x
    if input_norm == "log1p":
        return np.log1p(x, dtype=np.float32)
    raise ValueError(f"Unknown input_norm: {input_norm}")


def load_streak_h5(files, input_key, streak_attr, threshold_eV, input_norm):
    inputs, labels, amps, kicks, sws = [], [], [], [], []
    for path in files:
        print(f"streak[test]: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                y, a = _read_streak_label(grp.attrs, streak_attr, threshold_eV)
                inputs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                labels.append(y)
                amps.append(a)
                kicks.append(
                    float(np.asarray(grp.attrs["kickstrength"]).reshape(-1)[0])
                    if "kickstrength" in grp.attrs else np.nan
                )
                sws.append(
                    float(np.asarray(grp.attrs["sasewidth"]).reshape(-1)[0])
                    if "sasewidth" in grp.attrs else np.nan
                )
    x = np.stack(inputs).astype(np.float32)
    x = _apply_input_norm(x, input_norm).astype(np.float16)
    y = np.asarray(labels, dtype=np.float32)
    a = np.asarray(amps, dtype=np.float32)
    kick = np.asarray(kicks, dtype=np.float32)
    sw = np.asarray(sws, dtype=np.float32)
    return x, y, a, kick, sw


class TensorSetWithAmp(Dataset):
    def __init__(self, x, y, a):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.a = torch.from_numpy(a)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i].float().unsqueeze(0), self.y[i], self.a[i]


##############################################################################
# Metrics
##############################################################################
def roc_curve(scores, labels):
    """Return (fpr, tpr, thr) sorted by descending threshold."""
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # prepend the (0,0) origin so trapz works cleanly
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thr = np.concatenate([[scores.max() + 1e-6], scores])
    return fpr, tpr, thr


def pr_curve(scores, labels):
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    n_pos = labels.sum()
    if n_pos == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return precision, recall, scores


def auc(x, y):
    return float(np.trapz(y, x))


def tpr_at_fpr(scores, labels, target_fpr):
    fpr, tpr, _ = roc_curve(scores, labels)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return 0.0
    return float(tpr[ok[-1]])


def expected_calibration_error(scores, labels, n_bins=15):
    """Standard ECE (Guo et al. 2017)."""
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi) if hi < 1.0 else (scores >= lo) & (scores <= hi)
        m = mask.sum()
        if m == 0:
            continue
        acc = labels[mask].mean()
        conf = scores[mask].mean()
        ece += (m / n) * abs(acc - conf)
    return float(ece)


##############################################################################
# Temperature scaling (Guo et al. 2017)
##############################################################################
def fit_temperature(logits, labels, max_iter=200, lr=0.01):
    """Fit a scalar T > 0 that minimizes BCE(logits/T, labels)."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    log_T = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = F.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss

    opt.step(_closure)
    T = float(torch.exp(log_T.detach()))
    return T


##############################################################################
# Plot helpers
##############################################################################
def plot_roc_pr(scores, labels, save_path, title=""):
    fpr, tpr, _ = roc_curve(scores, labels)
    prec, rec, _ = pr_curve(scores, labels)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prec)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axs[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].set_title("ROC")
    axs[0].legend()

    axs[1].plot(rec, prec, label=f"AUC = {pr_auc:.4f}")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[1].set_title("Precision-Recall"); axs[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"ROC/PR saved to {save_path}")
    return roc_auc, pr_auc


def plot_sensitivity(amp, scores, labels, threshold, save_path,
                     bin_edges_eV=None, title=""):
    """TPR bucketed by streak magnitude ΔE_max, at a fixed threshold."""
    pred = (scores >= threshold).astype(np.float32)
    if bin_edges_eV is None:
        # Design doc s2: streak of interest ~2 eV; sweep 0-4 eV in 0.25 eV bins.
        bin_edges_eV = np.arange(0.0, 4.01, 0.25)

    centers = 0.5 * (bin_edges_eV[:-1] + bin_edges_eV[1:])
    tpr_bins = np.full(len(centers), np.nan)
    n_bins = np.zeros(len(centers), dtype=int)

    pos_mask = labels > 0.5
    for i, (lo, hi) in enumerate(zip(bin_edges_eV[:-1], bin_edges_eV[1:])):
        mask = pos_mask & (amp >= lo) & (amp < hi)
        m = int(mask.sum())
        n_bins[i] = m
        if m > 0:
            tpr_bins[i] = pred[mask].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centers, tpr_bins, "o-", label="Classifier")
    ax.set_xlabel("Streak magnitude ΔE_max [eV]")
    ax.set_ylabel(f"TPR at threshold={threshold:.3f}")
    ax.set_title(title or "Sensitivity vs streak magnitude")
    ax.set_ylim(-0.05, 1.05)
    for i, m in enumerate(n_bins):
        if m > 0:
            ax.annotate(str(m), (centers[i], tpr_bins[i]),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color="gray")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Sensitivity curve saved to {save_path}")


def plot_reliability(scores, labels, save_path, n_bins=15, title=""):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (scores >= lo) & (scores < hi) if hi < 1.0 else (scores >= lo) & (scores <= hi)
        m = int(mask.sum())
        counts[i] = m
        if m > 0:
            accs[i] = labels[mask].mean()
            confs[i] = scores[mask].mean()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.plot(confs, accs, "o-", label="Empirical")
    ax.set_xlabel("Predicted P(streak)")
    ax.set_ylabel("Empirical fraction of streaks")
    ax.set_title(title or "Reliability diagram")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Reliability diagram saved to {save_path}")


def plot_error_hist(values, y_true, y_pred, save_path, xlabel, title,
                    bin_edges=None, n_bins=40):
    """Overlay histograms of `values` split into (correct, mislabeled) shots.

    Also splits the mislabeled bucket into false-positives and false-negatives
    so it's obvious which direction the model is failing in.
    """
    values = np.asarray(values, dtype=np.float32)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    valid = np.isfinite(values)
    values = values[valid]; y_true = y_true[valid]; y_pred = y_pred[valid]

    correct = y_true == y_pred
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)

    if bin_edges is None:
        if values.size == 0:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            lo, hi = float(values.min()), float(values.max())
            if lo == hi:
                hi = lo + 1.0
            bin_edges = np.linspace(lo, hi, n_bins + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axs[0].hist(values[correct], bins=bin_edges, color="tab:green",
                alpha=0.75, label=f"correct (n={int(correct.sum())})")
    axs[0].set_title("Correctly labeled")
    axs[0].set_xlabel(xlabel); axs[0].set_ylabel("count")
    axs[0].grid(alpha=0.3); axs[0].legend()

    axs[1].hist(values[fp], bins=bin_edges, color="tab:orange",
                alpha=0.6, label=f"FP  y=0,pred=1 (n={int(fp.sum())})")
    axs[1].hist(values[fn], bins=bin_edges, color="tab:red",
                alpha=0.6, label=f"FN  y=1,pred=0 (n={int(fn.sum())})")
    axs[1].set_title("Mislabeled")
    axs[1].set_xlabel(xlabel); axs[1].set_ylabel("count")
    axs[1].grid(alpha=0.3); axs[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Error histogram saved to {save_path}")


def plot_kick_bin_correctness(kick, scores, y_true, op_thr, save_path,
                              k_bins=8, kick_range=(0.0, 15.0), title=""):
    """Per-kick-bin correctness at the operating threshold.

    For each bin of true kickstrength, count how many shots were classified
    correctly (score >= op_thr matches the truth label) vs incorrectly.
    Renders as a horizontal stacked bar chart: green = correct, red = wrong,
    with the bin's total count and % correct annotated inline. That answers
    "at this kick strength, does the classifier get it right?" — which is
    the actual quantity of interest for a streak / no-streak decision.

    The old KxK true-kick vs predicted-score matrix conflated calibration
    with classification; this plot separates them by collapsing the
    score-axis into the binary decision at the operating threshold.
    """
    kick = np.asarray(kick, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    valid = np.isfinite(kick) & np.isfinite(scores)
    kick = kick[valid]; scores = scores[valid]; y_true = y_true[valid]
    if kick.size == 0:
        print("plot_kick_bin_correctness: no valid shots, skipping.")
        return None

    pred = (scores >= op_thr).astype(np.int64)
    correct_all = (pred == y_true).astype(np.int64)

    kick_edges = np.linspace(kick_range[0], kick_range[1], k_bins + 1)
    # Clip overflow into the last bin so nothing falls off the axis.
    k_clipped = np.clip(kick, kick_edges[0], kick_edges[-1] - 1e-9)
    k_idx = np.clip(np.digitize(k_clipped, kick_edges) - 1, 0, k_bins - 1)

    # Per-shot certainty = distance from the 0.5 midpoint, i.e. max(p, 1-p).
    # 0.5 = model is undecided, 1.0 = model is maximally confident in whichever
    # direction it picked. Reported per-bin as the mean over shots in the bin.
    certainty_all = np.maximum(scores, 1.0 - scores)

    correct = np.zeros(k_bins, dtype=np.int64)
    total = np.zeros(k_bins, dtype=np.int64)
    certainty_sum = np.zeros(k_bins, dtype=np.float64)
    prob_sum = np.zeros(k_bins, dtype=np.float64)
    for i, ok, c, s in zip(k_idx, correct_all, certainty_all, scores):
        total[i] += 1
        correct[i] += int(ok)
        certainty_sum[i] += float(c)
        prob_sum[i] += float(s)
    wrong = total - correct
    mean_certainty = np.where(
        total > 0, certainty_sum / np.maximum(total, 1), np.nan,
    )
    # Mean predicted P(streak) per bin -- the model's raw post-T sigmoid
    # output averaged over shots in the bin. Complements mean_certainty
    # (which folds no-streak / streak sides together): for a well-calibrated
    # classifier this should track the fraction of true streaks in the bin.
    mean_prob = np.where(
        total > 0, prob_sum / np.maximum(total, 1), np.nan,
    )

    bin_labels = [f"[{kick_edges[i]:.1f}, {kick_edges[i+1]:.1f})"
                  for i in range(k_bins)]
    pct = np.where(total > 0, 100.0 * correct / np.maximum(total, 1), 0.0)

    print(f"plot_kick_bin_correctness: K={k_bins}, thr={op_thr:.3f}")
    print(f"  {'bin':>18s}  {'N':>6s}  {'%correct':>8s}  "
          f"{'mean_certainty':>14s}  {'mean_P(streak)':>15s}")
    for i in range(k_bins):
        cert_s = f"{mean_certainty[i]:.4f}" if total[i] > 0 else "   --"
        prob_s = f"{mean_prob[i]:.4f}" if total[i] > 0 else "   --"
        print(f"  {bin_labels[i]:>18s}  {total[i]:>6d}  {pct[i]:>7.2f}%  "
              f"{cert_s:>14s}  {prob_s:>15s}")

    fig, ax = plt.subplots(figsize=(9, 0.55 * k_bins + 1.5))
    y_pos = np.arange(k_bins)
    ax.barh(y_pos, correct, color="#2ca02c", label="Correct")
    ax.barh(y_pos, wrong, left=correct, color="#d62728", label="Wrong")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bin_labels)
    ax.invert_yaxis()  # smallest-kick bin at the top
    ax.set_xlabel("Shots in bin")
    ax.set_ylabel("True kickstrength [eV] bin")
    ax.set_title(title or
                 f"Per-kick-bin correctness at op thr={op_thr:.3f} (K={k_bins})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")

    # Inline "N=<total>  <pct>% correct" label past the bar's right edge.
    xmax = max(1, int(total.max()))
    for i in range(k_bins):
        if total[i] == 0:
            ax.text(0.02 * xmax, i, "no shots",
                    va="center", ha="left", fontsize=9, color="gray")
            continue
        ax.text(total[i] + 0.01 * xmax, i,
                f"N={total[i]}  {pct[i]:.1f}% correct",
                va="center", ha="left", fontsize=9)
    ax.set_xlim(0, xmax * 1.30)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Per-kick-bin correctness plot saved to {save_path}")
    return {"edges": kick_edges, "total": total, "correct": correct,
            "mean_certainty": mean_certainty, "mean_prob": mean_prob}


def plot_kick_bin_score_distribution(kick, scores, y_true, op_thr, save_path,
                                     k_bins=16, kick_range=(0.0, 15.0),
                                     title=""):
    """Per-kick-bin distribution of the *predicted probability* (post-T sigmoid).

    Complements plot_kick_bin_correctness: instead of collapsing the score axis
    into a binary decision at op_thr, this shows the full per-bin score
    distribution as a box+scatter. That surfaces edge cases — bins where the
    classifier "gets it right" but sits near the threshold, or bins where a
    handful of outliers drag accuracy down.

    Points are colored by truth (streak / no-streak). The operating threshold
    is drawn as a horizontal dashed line so it's obvious how far each bin's
    score cloud sits from the decision boundary.
    """
    kick = np.asarray(kick, dtype=np.float32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    valid = np.isfinite(kick) & np.isfinite(scores)
    kick = kick[valid]; scores = scores[valid]; y_true = y_true[valid]
    if kick.size == 0:
        print("plot_kick_bin_score_distribution: no valid shots, skipping.")
        return None

    kick_edges = np.linspace(kick_range[0], kick_range[1], k_bins + 1)
    k_clipped = np.clip(kick, kick_edges[0], kick_edges[-1] - 1e-9)
    k_idx = np.clip(np.digitize(k_clipped, kick_edges) - 1, 0, k_bins - 1)

    per_bin_scores = [scores[k_idx == i] for i in range(k_bins)]
    per_bin_y = [y_true[k_idx == i] for i in range(k_bins)]
    centers = 0.5 * (kick_edges[:-1] + kick_edges[1:])
    width = float(kick_edges[1] - kick_edges[0])

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * k_bins + 3), 5))

    # Boxplot per bin — captures the score distribution shape without needing
    # violin kernels (which are misleading at low counts).
    box_positions = centers
    non_empty = [i for i in range(k_bins) if len(per_bin_scores[i]) > 0]
    if non_empty:
        ax.boxplot(
            [per_bin_scores[i] for i in non_empty],
            positions=[box_positions[i] for i in non_empty],
            widths=0.7 * width,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#cfd8dc", alpha=0.6, edgecolor="#455a64"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="#455a64"),
            capprops=dict(color="#455a64"),
        )

    # Overlay individual shots with a small horizontal jitter, colored by
    # ground truth. Jitter is deterministic (index-based) — Math.random is
    # forbidden in workflow scripts anyway, and this keeps re-runs identical.
    rng = np.random.RandomState(0)
    for i in range(k_bins):
        s = per_bin_scores[i]
        y = per_bin_y[i]
        if len(s) == 0:
            continue
        jitter = rng.uniform(-0.35 * width, 0.35 * width, size=len(s))
        x_pts = centers[i] + jitter
        pos_mask = y == 1
        neg_mask = ~pos_mask
        if pos_mask.any():
            ax.scatter(x_pts[pos_mask], s[pos_mask], s=6, alpha=0.35,
                       color="#c62828", label="streak" if i == non_empty[0] else None)
        if neg_mask.any():
            ax.scatter(x_pts[neg_mask], s[neg_mask], s=6, alpha=0.35,
                       color="#1565c0", label="no streak" if i == non_empty[0] else None)

    ax.axhline(op_thr, color="black", linestyle="--", alpha=0.7,
               label=f"op thr={op_thr:.3f}")
    ax.set_xlabel("True kickstrength [eV] bin center")
    ax.set_ylabel("Predicted P(streak)  (post-T sigmoid)")
    ax.set_title(title or
                 f"Per-kick-bin score distribution (K={k_bins}, thr={op_thr:.3f})")
    ax.set_xlim(kick_edges[0] - 0.5 * width, kick_edges[-1] + 0.5 * width)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8)

    # Annotate per-bin counts along the top of the plot.
    for i in range(k_bins):
        n = len(per_bin_scores[i])
        if n == 0:
            continue
        ax.annotate(f"N={n}", (centers[i], 1.005),
                    ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Per-kick-bin score distribution saved to {save_path}")
    return {"edges": kick_edges, "per_bin_scores": per_bin_scores,
            "per_bin_y_true": per_bin_y}


def plot_confusion(cm, save_path, accuracy, elapsed, title=""):
    """2x2 confusion matrix (rows = true, cols = pred), styled to match the
    how_many/evaluate_how_many.py output in src/denoising/figures/: Blues
    colormap, per-cell text auto-colored (white on dark, black on light) so
    the labels stay legible against any blue.

    Each cell shows count and row-normalized percentage (per-true-class
    recall / miss-rate for row 0, TPR / FNR for row 1).
    """
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(1.6 * n + 4, 1.0 * n + 3))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    class_labels = ["No streak", "Streak"]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{title}\n"
        f"Acc: {accuracy:.2f}%  |  Time: {elapsed:.3f}s"
        if title else
        f"Confusion Matrix\n"
        f"Acc: {accuracy:.2f}%  |  Time: {elapsed:.3f}s"
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = 100.0 * count / row_totals[i, 0] if row_totals[i, 0] > 0 else 0.0
            text = f"{count}\n{pct:.1f}%"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if count > thresh else "black",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


##############################################################################
# Main
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--model_path", type=str, required=True)

    # overrides
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--input_norm", type=str, default=None,
                        choices=["none", "log1p"],
                        help="Override the input normalization stored in the "
                             "checkpoint. Only set this if you know what you "
                             "are doing -- the model was trained expecting a "
                             "specific transform.")
    parser.add_argument("--streak_attr", type=str, default=None)
    parser.add_argument("--streak_threshold_eV", type=float, default=None)

    # calibration split — carve off a fraction of the test set to fit
    # temperature scaling; the reported metrics use the rest.
    parser.add_argument("--calib_frac", type=float, default=0.2,
                        help="Fraction of the test set to fit temperature "
                             "scaling on. The remainder is used for reported "
                             "metrics. Set 0 to skip temperature scaling.")

    # kick-strength distribution shaping — carve a subset with a target
    # kickstrength mix so the same fixed dataset can compare models trained
    # on different unstreaked-label definitions.
    parser.add_argument("--kick_mix", type=str, default=None,
                        help="Comma-separated bucket:fraction spec, e.g. "
                             "'0:0.5,0-5:0.25,5-15:0.25'. Bare '0' matches "
                             "kick == 0 exactly; 'a-b' matches a<=kick<b. "
                             "Fractions must sum to 1.0. Applied BEFORE the "
                             "calibration/report split.")
    parser.add_argument("--mix_seed", type=int, default=0,
                        help="Seed for kick_mix sampling. Keep fixed so "
                             "multiple models see the same shots.")

    # kick-strength binning granularity for the per-kick-bin correctness plot
    parser.add_argument("--kick_bins", type=int, default=16,
                        help="Number of kickstrength bins in the per-kick-bin "
                             "correctness plot.")
    parser.add_argument("--kick_max_eV", type=float, default=4.0,
                        help="Upper edge of the kickstrength axis for the "
                             "per-kick-bin correctness plot. Shots above this "
                             "value are clipped into the top bin.")

    # threshold selection
    parser.add_argument("--operating_fpr", type=float, default=1e-3,
                        help="Reported confusion matrix uses the threshold "
                             "achieving this FPR on the reported set.")
    parser.add_argument("--simple_accuracy", action="store_true",
                        help="Skip temperature scaling and ROC-based "
                             "threshold selection. Just report "
                             "accuracy at logit>=0 (i.e. p>=0.5) and exit. "
                             "Meant for single-class holdouts (e.g. all-"
                             "streaked real duck shots) where the ROC/FPR "
                             "calc is degenerate.")
    parser.add_argument("--deployment_prior", type=float, default=None,
                        help="If set, prints the Bayes-optimal logit shift "
                             "for a deployment prior p_pos vs. balanced "
                             "training.")

    # io
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--identifier", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            f"{args.model_path} is not a streak_finder checkpoint."
        )

    input_shape = tuple(ckpt["input_shape"])
    arch = ckpt.get("arch", {})
    input_key = args.input_key or ckpt.get("input_key", "Ximg")
    input_norm = args.input_norm or ckpt.get("input_norm", "none")
    streak_attr = args.streak_attr or ckpt.get("streak_attr", "streak_amplitude")
    threshold_eV = (args.streak_threshold_eV
                    if args.streak_threshold_eV is not None
                    else ckpt.get("streak_threshold_eV"))
    trained_T = float(ckpt.get("temperature", 1.0))

    print(f"input_key={input_key}, input_norm={input_norm}, "
          f"input_shape={input_shape}, "
          f"streak_attr={streak_attr}, threshold_eV={threshold_eV}, "
          f"trained temperature={trained_T}")

    model = StreakCNN(input_shape=input_shape, **arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_paths = [d for d in args.data_dirs.split(":") if d]
    files = collect_h5_files(data_paths)
    x, y, a, kick_all, sw_all = load_streak_h5(
        files, input_key, streak_attr, threshold_eV, input_norm,
    )
    if tuple(x.shape[1:]) != input_shape:
        raise RuntimeError(
            f"Test input_shape {x.shape[1:]} != checkpoint input_shape {input_shape}"
        )
    print(f"Loaded {len(y)} shots, pos={int(y.sum())}, neg={int(len(y) - y.sum())}")

    # Optional kick-strength mix: carve the loaded shots into the requested
    # fractional buckets over kickstrength. Fixed seed so multiple models can
    # be compared on the exact same subset.
    if args.kick_mix:
        buckets = parse_kick_mix(args.kick_mix)
        mix_idx = build_kick_mix_indices(kick_all, buckets, seed=args.mix_seed)
        x = x[mix_idx]
        y = y[mix_idx]
        a = a[mix_idx]
        kick_all = kick_all[mix_idx]
        sw_all = sw_all[mix_idx]
        print(f"kick_mix: kept {len(y)} shots, pos={int(y.sum())}, "
              f"neg={int(len(y) - y.sum())}")

    # calibration / reporting split
    n = len(y)
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    calib_frac = 0.0 if args.simple_accuracy else args.calib_frac
    n_calib = int(round(calib_frac * n)) if calib_frac > 0 else 0
    calib_idx = idx[:n_calib]
    report_idx = idx[n_calib:]

    ds_report = TensorSetWithAmp(x[report_idx], y[report_idx], a[report_idx])
    loader_report = DataLoader(ds_report, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    start_time = time.time()
    logits_report = []
    with torch.no_grad():
        for img, _, _ in loader_report:
            img = img.to(device)
            logits_report.append(model(img).cpu().numpy())
    logits_report = np.concatenate(logits_report)
    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed:.3f}s on {len(logits_report)} shots "
          f"({1e3 * elapsed / max(1, len(logits_report)):.3f} ms/shot).")

    # ------------------------------------------------------------------
    # Simple-accuracy shortcut: skip temperature + ROC threshold.
    # Just threshold logits at 0 (equivalent to p>=0.5) and report accuracy.
    # Used for single-class holdouts where ROC/FPR is undefined.
    # ------------------------------------------------------------------
    if args.simple_accuracy:
        y_report = y[report_idx]
        pred = (logits_report >= 0.0).astype(np.int64)
        y_int = y_report.astype(np.int64)
        n_pos = int((y_int == 1).sum())
        n_neg = int((y_int == 0).sum())
        n_tp = int(((pred == 1) & (y_int == 1)).sum())
        n_tn = int(((pred == 0) & (y_int == 0)).sum())
        n_fp = int(((pred == 1) & (y_int == 0)).sum())
        n_fn = int(((pred == 0) & (y_int == 1)).sum())
        acc = (n_tp + n_tn) / max(1, len(y_int))
        print(f"[simple_accuracy] n={len(y_int)}  pos={n_pos}  neg={n_neg}")
        print(f"[simple_accuracy] TP={n_tp}  TN={n_tn}  FP={n_fp}  FN={n_fn}")
        print(f"[simple_accuracy] accuracy @ logit>=0 = {acc:.4f}")
        if n_pos > 0:
            print(f"[simple_accuracy] TPR (pos recall) = {n_tp / n_pos:.4f}")
        if n_neg > 0:
            print(f"[simple_accuracy] TNR (neg recall) = {n_tn / n_neg:.4f}")
        return

    # ------------------------------------------------------------------
    # Temperature scaling (fit on calibration split)
    # ------------------------------------------------------------------
    T = trained_T
    if n_calib > 0:
        ds_calib = TensorSetWithAmp(x[calib_idx], y[calib_idx], a[calib_idx])
        loader_calib = DataLoader(ds_calib, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
        logits_calib = []
        with torch.no_grad():
            for img, _, _ in loader_calib:
                img = img.to(device)
                logits_calib.append(model(img).cpu().numpy())
        logits_calib = np.concatenate(logits_calib)
        T = fit_temperature(logits_calib, y[calib_idx])
        print(f"Fitted temperature T = {T:.4f} on {n_calib} calibration shots.")

    scores_pre = 1.0 / (1.0 + np.exp(-logits_report))
    scores_post = 1.0 / (1.0 + np.exp(-logits_report / T))
    y_report = y[report_idx]
    a_report = a[report_idx]
    kick_report = kick_all[report_idx]
    sw_report = sw_all[report_idx]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    fpr, tpr, thr = roc_curve(scores_post, y_report)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = pr_curve(scores_post, y_report)
    pr_auc = auc(rec, prec)

    print(f"ROC-AUC = {roc_auc:.4f}   PR-AUC = {pr_auc:.4f}")
    for f in (1e-2, 1e-3, 1e-4):
        print(f"TPR @ FPR = {f:.0e}: {tpr_at_fpr(scores_post, y_report, f):.4f}")

    # Pick operating threshold from the reported set at target FPR.
    ok = np.where(fpr <= args.operating_fpr)[0]
    if len(ok) == 0:
        op_thr = float(thr[0])
    else:
        op_thr = float(thr[ok[-1]])
    op_tpr = tpr_at_fpr(scores_post, y_report, args.operating_fpr)
    print(f"Operating point @ FPR={args.operating_fpr}: threshold={op_thr:.4f}, "
          f"TPR={op_tpr:.4f}")

    ece_pre = expected_calibration_error(scores_pre, y_report)
    ece_post = expected_calibration_error(scores_post, y_report)
    print(f"ECE (pre-T)  = {ece_pre:.4f}")
    print(f"ECE (post-T) = {ece_post:.4f}")

    if args.deployment_prior is not None:
        p = float(args.deployment_prior)
        if not (0 < p < 1):
            raise ValueError("--deployment_prior must be in (0, 1).")
        shift = np.log(p / (1 - p))
        print(f"Deployment prior {p:.4f} -> add logit_shift = {shift:+.4f} "
              f"to model logits before sigmoid (or subtract "
              f"{shift:+.4f} from the score threshold on the logit scale).")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    identifier = args.identifier or os.path.splitext(os.path.basename(args.model_path))[0]
    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures",
    )
    os.makedirs(figures_dir, exist_ok=True)

    plot_roc_pr(
        scores_post, y_report,
        os.path.join(figures_dir, f"roc_pr_for_{identifier}.png"),
        title=f"{identifier} | ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}",
    )

    # Sensitivity vs streak magnitude — only meaningful if we have real-
    # valued amplitudes AND the label came from a threshold on that amp.
    if not np.allclose(a_report, y_report):
        plot_sensitivity(
            a_report, scores_post, y_report, threshold=op_thr,
            save_path=os.path.join(figures_dir, f"sensitivity_for_{identifier}.png"),
            title=f"{identifier} | TPR vs ΔE_max @ FPR={args.operating_fpr:g}",
        )
    else:
        print("Skipping sensitivity plot: streak_attr looks like a plain 0/1 "
              "label, no per-shot magnitudes available.")

    plot_reliability(
        scores_pre, y_report,
        os.path.join(figures_dir, f"reliability_pre_T_for_{identifier}.png"),
        title=f"{identifier} | pre-T | ECE={ece_pre:.4f}",
    )
    plot_reliability(
        scores_post, y_report,
        os.path.join(figures_dir, f"reliability_post_T_for_{identifier}.png"),
        title=f"{identifier} | post-T (T={T:.3f}) | ECE={ece_post:.4f}",
    )

    pred_op = (scores_post >= op_thr).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_report.astype(int), pred_op):
        cm[t, p] += 1
    op_accuracy = 100.0 * (pred_op == y_report.astype(int)).mean()
    plot_confusion(
        cm,
        os.path.join(figures_dir, f"confusion_matrix_for_{identifier}.png"),
        accuracy=op_accuracy,
        elapsed=elapsed,
        title=f"Confusion Matrix | {identifier} "
              f"| FPR={args.operating_fpr:g} thr={op_thr:.3f}",
    )

    # Per-kick-bin correctness at the operating threshold. Replaces the
    # old KxK true-kick vs predicted-score matrix, which conflated
    # calibration with the actual streak / no-streak decision.
    if np.isfinite(kick_report).any():
        plot_kick_bin_correctness(
            kick_report, scores_post, y_report, op_thr,
            save_path=os.path.join(
                figures_dir,
                f"kick_bin_correctness_k{args.kick_bins}_for_{identifier}.png",
            ),
            k_bins=args.kick_bins,
            kick_range=(0.0, args.kick_max_eV),
            title=f"{identifier} | per-kick-bin correctness | "
                  f"K={args.kick_bins} thr={op_thr:.3f}",
        )
        # Same binning, but shows the full predicted-probability distribution
        # rather than collapsing to the binary decision. Useful for finding
        # edge cases where the classifier is right but sits close to the
        # threshold, or where a few outliers drag accuracy down.
        plot_kick_bin_score_distribution(
            kick_report, scores_post, y_report, op_thr,
            save_path=os.path.join(
                figures_dir,
                f"kick_bin_score_distribution_k{args.kick_bins}_for_{identifier}.png",
            ),
            k_bins=args.kick_bins,
            kick_range=(0.0, args.kick_max_eV),
            title=f"{identifier} | per-kick-bin P(streak) distribution | "
                  f"K={args.kick_bins} thr={op_thr:.3f}",
        )
    else:
        print("Skipping kick-bin correctness plot: kickstrength missing on all shots.")

    # Per-attribute correct/mislabeled breakdowns at the operating point.
    # Kickstrength: unstreaked shots (kick=0) dominate the low end; keep them
    # in the histogram so it's obvious how FPs cluster there.
    if np.isfinite(kick_report).any():
        plot_error_hist(
            kick_report, y_report, pred_op,
            save_path=os.path.join(
                figures_dir, f"error_hist_kickstrength_for_{identifier}.png"),
            xlabel="kickstrength [eV]",
            title=f"{identifier} | kickstrength | thr={op_thr:.3f}",
            bin_edges=np.linspace(0.0, 4.0, 41),
        )
    else:
        print("Skipping kickstrength error histogram: attr missing on all shots.")

    if np.isfinite(sw_report).any():
        plot_error_hist(
            sw_report, y_report, pred_op,
            save_path=os.path.join(
                figures_dir, f"error_hist_sasewidth_for_{identifier}.png"),
            xlabel="sasewidth [eV]",
            title=f"{identifier} | sasewidth | thr={op_thr:.3f}",
            bin_edges=np.linspace(2.5, 5.5, 61),
        )
    else:
        print("Skipping sasewidth error histogram: attr missing on all shots.")

    # Per-shot dump — raw logits, pre-T and post-T probabilities, truth,
    # kickstrength, sasewidth, streak amplitude, and the report-set index in
    # the loaded dataset. Lets us inspect edge cases (near-threshold FPs/FNs,
    # per-bin outliers) without re-running inference.
    preds_path = os.path.join(figures_dir, f"predictions_for_{identifier}.npz")
    np.savez(
        preds_path,
        logits=logits_report.astype(np.float32),
        prob_pre_T=scores_pre.astype(np.float32),
        prob_post_T=scores_post.astype(np.float32),
        y_true=y_report.astype(np.int8),
        streak_amplitude=a_report.astype(np.float32),
        kickstrength=kick_report.astype(np.float32),
        sasewidth=sw_report.astype(np.float32),
        report_idx=report_idx.astype(np.int64),
        temperature=np.float32(T),
        operating_threshold=np.float32(op_thr),
        operating_fpr=np.float32(args.operating_fpr),
    )
    print(f"Per-shot predictions saved to {preds_path}")

    # Terse summary file so eval runs leave a machine-readable trail.
    summary_path = os.path.join(figures_dir, f"summary_for_{identifier}.txt")
    with open(summary_path, "w") as f:
        f.write(f"model_path: {args.model_path}\n")
        f.write(f"n_shots: {len(y)}\n")
        f.write(f"n_calibration: {n_calib}\n")
        f.write(f"n_report: {len(y_report)}\n")
        f.write(f"pos_frac_report: {y_report.mean():.4f}\n")
        f.write(f"temperature: {T:.4f}\n")
        f.write(f"roc_auc: {roc_auc:.6f}\n")
        f.write(f"pr_auc: {pr_auc:.6f}\n")
        for tgt in (1e-2, 1e-3, 1e-4):
            f.write(f"tpr_at_fpr_{tgt:g}: {tpr_at_fpr(scores_post, y_report, tgt):.6f}\n")
        f.write(f"operating_fpr: {args.operating_fpr:g}\n")
        f.write(f"operating_threshold: {op_thr:.6f}\n")
        f.write(f"operating_tpr: {op_tpr:.6f}\n")
        f.write(f"ece_pre: {ece_pre:.6f}\n")
        f.write(f"ece_post: {ece_post:.6f}\n")
        f.write(f"inference_ms_per_shot: {1e3 * elapsed / max(1, len(logits_report)):.6f}\n")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
