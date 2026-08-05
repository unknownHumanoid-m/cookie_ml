"""
Joint training loop for the split-bottleneck autoencoder.

Two-phase schedule
------------------
* **Warmup** (``TRAIN['warmup_epochs']`` epochs) — encoder trunk + decoder
  train on reconstruction MSE alone. The task branch (bottleneck_count /
  bottleneck_phase + count/phase heads) is not touched.
* **Joint** (remaining epochs) — all three losses active simultaneously
  through the shared trunk. Uncertainty-weighted or static weights per
  the config.

Every epoch, all three loss components are logged separately so that
gradient conflict between reconstruction and the task branch is visible
in the log stream. This is the primary diagnostic for the design.

Run:
    python3 train.py

Everything is driven by config.py — no argparse. Override paths there.
"""

import copy
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # SLURM: no X server
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from dataset import build_datasets
from losses import MultiTaskLoss
from model import build_model_from_config


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_figures_dir():
    # cfg.IO["figures_dir"] is pinned to split_bottleneck_ae/figures/ in
    # config.py; keep the fallback so a caller can null it out for a
    # one-off run without editing the module.
    return cfg.IO["figures_dir"] or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )


def resolve_run_dir():
    save_dir = cfg.IO["save_dir"]
    run_name = cfg.IO["run_name"]
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def step(self, val_loss, snapshot_state):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = snapshot_state()
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def run_epoch(model, criterion, loader, device, optimizer, *,
              active_recon: bool, active_task: bool, train: bool):
    if train:
        model.train()
        criterion.train()
    else:
        model.eval()
        criterion.eval()

    running_total = 0.0
    running = {"recon": 0.0, "count": 0.0, "phase_single": 0.0, "phase_two": 0.0}
    n_recon_batches = 0
    n_task_batches = 0
    n_batches = 0
    # Track shot-weighted denominators for the two phase heads separately —
    # 2-pulse shots don't appear in every batch.
    n_single_shots = 0
    n_two_shots = 0
    sum_single_shots = 0.0
    sum_two_shots = 0.0

    min_pulses = int(cfg.DATA["min_pulses"])

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y_clean, count_label, phi0_rad, dphi in loader:
            x = x.to(device, non_blocking=True)
            y_clean = y_clean.to(device, non_blocking=True)
            count_label = count_label.to(device, non_blocking=True)
            phi0_rad = phi0_rad.to(device, non_blocking=True)
            dphi = dphi.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out = model(x, run_recon=active_recon, run_task=active_task)
            total, parts = criterion(
                recon=out["recon"],
                recon_target=y_clean,
                count_logits=out["count_logits"],
                count_label=count_label,
                phase_single_out=out["phase_single_out"],
                phi0_rad=phi0_rad,
                phase_two_out=out["phase_two_out"],
                dphi=dphi,
                npulses_label=count_label,
                min_pulses=min_pulses,
                active_recon=active_recon,
                active_task=active_task,
            )
            if train:
                total.backward()
                optimizer.step()

            running_total += float(total.detach().cpu())
            if active_recon and parts["recon"] is not None:
                running["recon"] += float(parts["recon"].cpu())
                n_recon_batches += 1
            if active_task:
                running["count"] += float(parts["count"].cpu())
                # Weight phase losses by the number of contributing shots so
                # sparse batches (few 2-pulse shots) don't count the same as
                # dense ones.
                nsingle = int(parts["n_single"])
                ntwo = int(parts["n_two"])
                if nsingle > 0:
                    sum_single_shots += float(parts["phase_single"].cpu()) * nsingle
                    n_single_shots += nsingle
                if ntwo > 0:
                    sum_two_shots += float(parts["phase_two"].cpu()) * ntwo
                    n_two_shots += ntwo
                n_task_batches += 1
            n_batches += 1

    def _avg(sum_val, n):
        return sum_val / n if n > 0 else float("nan")

    return {
        "total": _avg(running_total, n_batches),
        "recon": _avg(running["recon"], n_recon_batches),
        "count": _avg(running["count"], n_task_batches),
        "phase_single": _avg(sum_single_shots, n_single_shots),
        "phase_two": _avg(sum_two_shots, n_two_shots),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    set_seed(cfg.TRAIN["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")
    if device.type == "cuda":
        print(f"[train] gpu   = {torch.cuda.get_device_name(0)}")

    # -- data --------------------------------------------------------------
    train_ds, val_ds = build_datasets(cfg.DATA, cfg.TRAIN_SAMPLES)
    print(f"[train] train size = {len(train_ds)}, val size = {len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.TRAIN["batch_size"],
        shuffle=True,
        num_workers=cfg.TRAIN["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.TRAIN["batch_size"],
        shuffle=False,
        num_workers=cfg.TRAIN["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # -- model + loss ------------------------------------------------------
    model = build_model_from_config(cfg).to(device)

    # Warm-start encoder + decoder from the trained raw-input denoiser so
    # the three bottleneck fanouts inherit a working autoencoder. Bottleneck
    # layers themselves keep their fresh init.
    pretrained_path = cfg.MODEL.get("pretrained_denoiser_path", "") or ""
    if pretrained_path:
        model.load_pretrained_denoiser(pretrained_path, map_location=device)
    else:
        print("[train] no pretrained denoiser configured; training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    n_task_params = (
        sum(p.numel() for p in model.encoder.parameters())
        + sum(p.numel() for p in model.bottleneck_count.parameters())
        + sum(p.numel() for p in model.bottleneck_phase.parameters())
        + sum(p.numel() for p in model.count_head.parameters())
        + sum(p.numel() for p in model.phase_head_single.parameters())
        + sum(p.numel() for p in model.phase_head_two.parameters())
    )
    print(f"[train] total params = {n_params:,}  |  deployed subgraph params = {n_task_params:,}")

    criterion = MultiTaskLoss(
        w_recon=cfg.LOSSES["w_recon"],
        w_count=cfg.LOSSES["w_count"],
        w_phase_single=cfg.LOSSES["w_phase_single"],
        w_phase_two=cfg.LOSSES["w_phase_two"],
        uncertainty_weighting=cfg.LOSSES["uncertainty_weighting"],
    ).to(device)

    # -- optimizer ---------------------------------------------------------
    trainable = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.Adam(trainable, lr=cfg.TRAIN["lr"])

    patience = cfg.TRAIN["patience"]
    early_stop = EarlyStopping(patience) if patience else None

    def snapshot_best_state():
        return {
            "model": copy.deepcopy(model.state_dict()),
            "criterion": copy.deepcopy(criterion.state_dict()),
        }

    # -- history + run dir -------------------------------------------------
    history: List[Dict] = []
    run_dir = resolve_run_dir()
    figures_dir = resolve_figures_dir()
    os.makedirs(figures_dir, exist_ok=True)
    print(f"[train] run_dir     = {run_dir}")
    print(f"[train] figures_dir = {figures_dir}")

    total_epochs = int(cfg.TRAIN["epochs"])
    warmup = int(cfg.TRAIN["warmup_epochs"])
    if warmup > total_epochs:
        raise ValueError(f"warmup_epochs ({warmup}) > epochs ({total_epochs})")

    print(f"[train] warmup epochs = {warmup}, joint epochs = {total_epochs - warmup}")
    start = time.time()

    for epoch in range(total_epochs):
        in_warmup = epoch < warmup
        active_recon = True
        active_task = not in_warmup
        phase_tag = "warmup" if in_warmup else "joint"

        train_metrics = run_epoch(
            model, criterion, train_loader, device, optimizer,
            active_recon=active_recon, active_task=active_task, train=True,
        )
        # Val loss: during warmup, only recon is meaningful. In joint,
        # everything is meaningful and the total is the early-stop signal.
        val_metrics = run_epoch(
            model, criterion, val_loader, device, optimizer,
            active_recon=active_recon, active_task=active_task, train=False,
        )

        print(
            f"[epoch {epoch+1:03d}/{total_epochs}] [{phase_tag}] "
            f"train: total={train_metrics['total']:.4f} "
            f"recon={train_metrics['recon']:.4f} "
            f"count={train_metrics['count']:.4f} "
            f"phase_single={train_metrics['phase_single']:.4f} "
            f"phase_two={train_metrics['phase_two']:.4f}  ||  "
            f"val: total={val_metrics['total']:.4f} "
            f"recon={val_metrics['recon']:.4f} "
            f"count={val_metrics['count']:.4f} "
            f"phase_single={val_metrics['phase_single']:.4f} "
            f"phase_two={val_metrics['phase_two']:.4f}"
        )
        history.append({
            "epoch": epoch + 1,
            "phase": phase_tag,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Early-stop only on joint validation loss — during warmup the val
        # total is entirely reconstruction, so it's a very different signal.
        if early_stop is not None and not in_warmup:
            early_stop.step(val_metrics["total"], snapshot_best_state)
            if early_stop.early_stop:
                print("[train] early stopping triggered.")
                break

    if early_stop is not None and early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state["model"])
        criterion.load_state_dict(early_stop.best_state["criterion"])
        print(f"[train] rolled back to best (val total = {early_stop.best_loss:.4f})")

    elapsed = time.time() - start
    print(f"[train] elapsed = {elapsed:.1f}s")

    # -- persist everything ------------------------------------------------
    ckpt_path = os.path.join(run_dir, "model.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "criterion_state": criterion.state_dict(),
        "input_shape": model.input_shape,
        "num_count_classes": model.num_count_classes,
        "phase_single_output_dim": model.phase_single_output_dim,
        "phase_two_output_dim": model.phase_two_output_dim,
        "bottleneck_count_dim": model.bottleneck_count_dim,
        "bottleneck_phase_dim": model.bottleneck_phase_dim,
        "min_pulses": int(cfg.DATA["min_pulses"]),
        "max_pulses": int(cfg.DATA["max_pulses"]),
        "config": {
            "DATA": cfg.DATA,
            "MODEL": cfg.MODEL,
            "LOSSES": cfg.LOSSES,
            "TRAIN": cfg.TRAIN,
            "TRAIN_SAMPLES": cfg.TRAIN_SAMPLES,
        },
    }, ckpt_path)
    print(f"[train] checkpoint saved -> {ckpt_path}")

    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    print(f"[train] history saved    -> {history_path}")

    # -- figures -----------------------------------------------------------
    _plot_history(history, figures_dir, cfg.IO["run_name"])


def _plot_history(history, figures_dir, run_name):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    parts = ["total", "recon", "count", "phase_single", "phase_two"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f"Split-bottleneck AE | run={run_name}", fontsize=13)
    ax_iter = iter(axes.flatten())
    for key in parts:
        ax = next(ax_iter)
        tr = [h["train"][key] for h in history]
        va = [h["val"][key] for h in history]
        ax.plot(epochs, tr, label="train")
        ax.plot(epochs, va, label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        transitions = [
            e for e, h in zip(epochs, history)
            if h["phase"] == "joint" and h["epoch"] == min(
                (hh["epoch"] for hh in history if hh["phase"] == "joint"),
                default=None,
            )
        ]
        for e in transitions:
            ax.axvline(e - 0.5, color="k", linestyle=":", linewidth=1)
    # Blank the leftover subplot.
    for ax in ax_iter:
        ax.axis("off")
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"training_curves_convae_{run_name}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[train] figure saved     -> {fig_path}")


if __name__ == "__main__":
    main()
