"""
Evaluation for a trained split-bottleneck autoencoder checkpoint.

Reports four families of metrics on the test split:
    * reconstruction MSE (decoder branch)
    * count classifier accuracy + per-class confusion
    * single-pulse phase circular error (mean + median absolute, in radians),
      routed by the classifier's "1"-pulse predictions
    * two-pulse arccos(cos Δφ) MSE, routed by the classifier's "2"-pulse
      predictions (mirrors src/ml_backbone/regressions/eval_phase_two.py)

Also saves a reconstruction sanity figure (magma_r), a confusion-matrix
figure (percentages per row, magma), and two phase scatter plots — matching
the convention in src/ml_backbone/regressions/eval_phase_{single,two}.py.

Run:
    python3 eval.py [--ckpt /path/to/model.pth]

If ``--ckpt`` is omitted, it uses ``IO['save_dir']/IO['run_name']/model.pth``.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import build_test_dataset
from losses import (
    circular_error, count_accuracy, decode_phase_two, decode_sincos_phase,
    recon_mse,
)
from model import build_model_from_config


RECON_CMAP = "magma_r"
CONFUSION_CMAP = "Blues"


def default_ckpt_path():
    return os.path.join(cfg.IO["save_dir"], cfg.IO["run_name"], "model.pth")


def resolve_figures_dir():
    # cfg.IO["figures_dir"] is pinned to split_bottleneck_ae/figures/ in
    # config.py; keep the fallback so a caller can null it out for a
    # one-off run without editing the module.
    return cfg.IO["figures_dir"] or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )


def load_model(ckpt_path, device):
    print(f"[eval] loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the .pth checkpoint. Defaults to "
                             "IO['save_dir']/IO['run_name']/model.pth.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")

    ckpt_path = args.ckpt or default_ckpt_path()
    model, _ = load_model(ckpt_path, device)

    test_ds = build_test_dataset(cfg.DATA, cfg.TRAIN_SAMPLES)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.TRAIN["batch_size"],
        shuffle=False,
        num_workers=cfg.TRAIN["num_workers"],
    )
    print(f"[eval] test size = {len(test_ds)}")

    num_classes = cfg.num_count_classes()

    total_recon = 0.0
    total_count_correct = 0
    total_samples = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    circ_errs_single = []
    pred_phi0_all = []
    true_phi0_all = []
    pred_dphi_all = []
    true_dphi_all = []
    pred_counts = []

    # Grab a handful of examples for a reconstruction sanity figure.
    example_x, example_yhat, example_ytrue = [], [], []
    example_ncap = 8

    with torch.no_grad():
        for x, y_clean, count_label, phi0_rad, dphi in test_loader:
            x = x.to(device, non_blocking=True)
            y_clean = y_clean.to(device, non_blocking=True)
            count_label = count_label.to(device, non_blocking=True)
            phi0_rad = phi0_rad.to(device, non_blocking=True)
            dphi = dphi.to(device, non_blocking=True)

            out = model(x, run_recon=True, run_task=True)

            b = x.size(0)
            total_recon += float(recon_mse(out["recon"], y_clean).cpu()) * b
            total_samples += b

            preds = out["count_logits"].argmax(dim=1)
            total_count_correct += int((preds == count_label).sum().cpu())
            for t, p in zip(count_label.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(t), int(p)] += 1

            phi0_pred = decode_sincos_phase(out["phase_single_out"])
            dphi_pred = decode_phase_two(out["phase_two_out"])

            circ_errs_single.append(
                circular_error(phi0_pred, phi0_rad).cpu().numpy()
            )
            pred_phi0_all.append(phi0_pred.cpu().numpy())
            true_phi0_all.append(phi0_rad.cpu().numpy())
            pred_dphi_all.append(dphi_pred.cpu().numpy())
            true_dphi_all.append(dphi.cpu().numpy())
            pred_counts.append(preds.cpu().numpy())

            if len(example_x) < example_ncap:
                take = min(example_ncap - len(example_x), b)
                example_x.append(x[:take].cpu().numpy())
                example_yhat.append(out["recon"][:take].cpu().numpy())
                example_ytrue.append(y_clean[:take].cpu().numpy())

    if total_samples == 0:
        raise RuntimeError("eval: test loader produced no samples.")

    mean_recon = total_recon / total_samples
    count_acc = total_count_correct / total_samples
    circ_errs_single = np.concatenate(circ_errs_single, axis=0) if circ_errs_single else np.zeros(0)
    pred_phi0_all = np.concatenate(pred_phi0_all, axis=0) if pred_phi0_all else np.zeros(0)
    true_phi0_all = np.concatenate(true_phi0_all, axis=0) if true_phi0_all else np.zeros(0)
    pred_dphi_all = np.concatenate(pred_dphi_all, axis=0) if pred_dphi_all else np.zeros(0)
    true_dphi_all = np.concatenate(true_dphi_all, axis=0) if true_dphi_all else np.zeros(0)
    pred_counts = np.concatenate(pred_counts, axis=0) if pred_counts else np.zeros(0, dtype=np.int64)

    phase_single_mean_err = float(np.mean(circ_errs_single)) if circ_errs_single.size else float("nan")
    phase_single_med_err = float(np.median(circ_errs_single)) if circ_errs_single.size else float("nan")

    print(f"[eval] mean recon MSE                = {mean_recon:.6f}")
    print(f"[eval] count accuracy                = {count_acc*100:.2f}% "
          f"({total_count_correct}/{total_samples})")
    print(f"[eval] phi0 mean|err| (rad, all)     = {phase_single_mean_err:.4f}")
    print(f"[eval] phi0 med |err| (rad, all)     = {phase_single_med_err:.4f}")

    # Per-class breakdown of the confusion matrix.
    print("[eval] confusion (rows=true, cols=pred):")
    print(confusion)

    figures_dir = resolve_figures_dir()
    os.makedirs(figures_dir, exist_ok=True)
    run_name = cfg.IO["run_name"]

    # -- reconstruction sanity figure --------------------------------------
    example_x = np.concatenate(example_x, axis=0) if example_x else np.zeros((0,))
    example_yhat = np.concatenate(example_yhat, axis=0) if example_yhat else np.zeros((0,))
    example_ytrue = np.concatenate(example_ytrue, axis=0) if example_ytrue else np.zeros((0,))
    if example_x.size:
        n = example_x.shape[0]
        fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 5))
        if n == 1:
            axes = axes[:, None]
        for i in range(n):
            axes[0, i].imshow(example_x[i], aspect="auto", cmap=RECON_CMAP)
            axes[1, i].imshow(example_yhat[i], aspect="auto", cmap=RECON_CMAP)
            axes[2, i].imshow(example_ytrue[i], aspect="auto", cmap=RECON_CMAP)
            for a in axes[:, i]:
                a.set_xticks([]); a.set_yticks([])
        axes[0, 0].set_ylabel("input")
        axes[1, 0].set_ylabel("recon")
        axes[2, 0].set_ylabel("truth")
        fig.suptitle(f"Reconstruction sanity | run={run_name}")
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f"eval_recon_convae_{run_name}.png")
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"[eval] recon figure     -> {fig_path}")

    # -- confusion matrix figure (row-normalized percentages) --------------
    row_totals = confusion.sum(axis=1, keepdims=True)
    row_totals_safe = np.where(row_totals > 0, row_totals, 1)
    pct = confusion.astype(np.float64) / row_totals_safe * 100.0  # rows sum to 100
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pct, cmap=CONFUSION_CMAP, vmin=0.0, vmax=100.0)
    ax.set_xlabel("predicted pulses"); ax.set_ylabel("true pulses")
    min_p = int(cfg.DATA["min_pulses"])
    max_p = int(cfg.DATA["max_pulses"])
    tick_labels = [str(min_p + i) for i in range(num_classes)]
    tick_labels[-1] = f"≥{max_p}"
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(tick_labels)
    for i in range(num_classes):
        for j in range(num_classes):
            val = pct[i, j]
            # Blues cmap is light at 0 and dark at 100 — dark cells need
            # white text, light cells need black text so the numbers stay
            # legible against the background.
            color = "white" if val > 50.0 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of true class")
    ax.set_title(f"Count confusion | acc={count_acc*100:.1f}%")
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"eval_confusion_convae_{run_name}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[eval] confusion figure -> {fig_path}")

    # -- phase scatters, routed by the classifier --------------------------
    # Mirrors src/ml_backbone/regressions/eval_phase_{single,two}.py: only
    # shots the classifier calls "1" go on the single-pulse scatter, and
    # only shots it calls "2" go on the two-pulse (arccos(cos Δφ)) scatter.
    min_p = int(cfg.DATA["min_pulses"])
    single_cls = 1 - min_p
    two_cls = 2 - min_p

    single_mse = None
    two_mse = None

    # single-pulse scatter: phi0 vs. decoded phi0_pred, both in [0, 2*pi).
    if 0 <= single_cls < num_classes:
        mask = pred_counts == single_cls
        if mask.any():
            t = true_phi0_all[mask]
            p = pred_phi0_all[mask]
            single_mse = float(np.mean((p - t) ** 2))
            plt.figure(figsize=(8, 6))
            plt.scatter(t, p, s=4, alpha=0.4, c="C0", label="Predicted vs True")
            plt.plot([0, 2 * np.pi], [0, 2 * np.pi], "r--", label="Ideal")
            plt.xlabel("True phase [rad]"); plt.ylabel("Predicted phase [rad]")
            plt.title(
                f"Single-pulse phase | run={run_name} | routed: {int(mask.sum())}"
                f"/{total_samples} | MSE={single_mse:.4f}"
            )
            plt.legend(); plt.grid(True); plt.tight_layout()
            fig_path = os.path.join(
                figures_dir, f"eval_phase_scatter_single_convae_{run_name}.png"
            )
            plt.savefig(fig_path, dpi=120)
            plt.close()
            print(f"[eval] single scatter -> {fig_path}")
        else:
            print("[eval] no shots routed to single; skipping")

    # two-pulse scatter: arccos(cos Δφ) truth vs. arccos(cos Δφ) pred, [0, pi].
    # Drop shots that were routed to "2" by the classifier but weren't
    # actually 2-pulse (dphi is NaN) — mirrors eval_phase_two.py's labeled
    # subset.
    if 0 <= two_cls < num_classes:
        mask = (pred_counts == two_cls) & np.isfinite(true_dphi_all)
        n_routed = int((pred_counts == two_cls).sum())
        n_mislabel = n_routed - int(mask.sum())
        if mask.any():
            t = true_dphi_all[mask]
            p = pred_dphi_all[mask]
            two_mse = float(np.mean((p - t) ** 2))
            plt.figure(figsize=(8, 6))
            plt.scatter(t, p, s=4, alpha=0.4, c="C0", label="Pred vs True")
            plt.plot([0, np.pi], [0, np.pi], "r--", label="Ideal")
            plt.xlabel("True arccos(cos(Δφ))"); plt.ylabel("Pred arccos(cos(Δφ))")
            plt.title(
                f"Two-pulse arccos(cos(Δφ)) | run={run_name} | "
                f"kept {int(mask.sum())}/{n_routed} | MSE={two_mse:.4f}"
            )
            plt.legend(); plt.grid(True); plt.tight_layout()
            fig_path = os.path.join(
                figures_dir, f"eval_phase_scatter_two_convae_{run_name}.png"
            )
            plt.savefig(fig_path, dpi=120)
            plt.close()
            print(f"[eval] two scatter    -> {fig_path}  "
                  f"(dropped {n_mislabel} misrouted-no-label)")
        else:
            print("[eval] no shots routed to two (with labels); skipping")

    # -- persist a summary json --------------------------------------------
    summary = {
        "ckpt": os.path.abspath(ckpt_path),
        "test_size": total_samples,
        "recon_mse": mean_recon,
        "count_accuracy": count_acc,
        "count_confusion": confusion.tolist(),
        "count_confusion_pct": pct.tolist(),
        "phi0_mean_abs_err_rad": phase_single_mean_err,
        "phi0_median_abs_err_rad": phase_single_med_err,
        "phase_mse_single_rad2": single_mse,
        "phase_mse_two_rad2": two_mse,
    }
    summary_path = os.path.join(
        cfg.IO["save_dir"], cfg.IO["run_name"], "eval_summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[eval] summary saved    -> {summary_path}")


if __name__ == "__main__":
    main()
