# -*- coding: utf-8 -*-
"""
Evaluate the binary 0-or-1 MLP classifier trained by train_0or1_mlp.py.

The checkpoint format matches train_how_many.py (state_dict + input_shape +
num_classes + min_pulses + max_pulses + input_key), so this script is a
trimmed clone of evaluate_how_many.py — same load path, same confusion-matrix
and sample-image figures — with binary-only class labels and a threshold
CLI knob for the "1+" decision boundary.

Example
-------
    python3 eval_0or1_mlp.py \\
        --data_dirs /path/to/mrco_h5/test/ \\
        --model_path /path/to/runs/zero_or_one_mlp.pth
"""

import os
import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the training-file's loader + model so the two scripts can't drift.
# Same-directory sibling; runs via `python3 eval_0or1_mlp.py` (see the
# accompanying .sh launcher, which cd's into src/denoising first).
from train_0or1_mlp import SimpleMLP, load_binary_h5


def plot_confusion_matrix(cm, save_path, accuracy, elapsed, class_labels,
                          input_key, title=None, percent=False):
    """Rows = true, cols = pred. `percent=True` renders row-normalized %
    cells with a fixed 0-100 colormap; otherwise raw counts, matching the
    original evaluate_how_many.plot_confusion_matrix style.
    """
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.0 * n + 3))

    if percent:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_pct = np.where(row_sums > 0, 100.0 * cm / row_sums, 0.0)
        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues",
                       vmin=0.0, vmax=100.0)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("% of true class")
        for i in range(n):
            for j in range(n):
                color = "white" if cm_pct[i, j] > 50.0 else "black"
                ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n(n={int(cm[i, j])})",
                        ha="center", va="center", color=color)
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    if title is not None:
        ax.set_title(f"{title}\nAcc: {accuracy:.2f}%")
    else:
        ax.set_title(
            f"0-or-1 Confusion Matrix | input_key={input_key}\n"
            f"Acc: {accuracy:.2f}%  |  Time: {elapsed:.3f}s"
        )

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained .pth (from train_0or1_mlp.py).")
    parser.add_argument("--input_key", type=str, default=None,
                        help="Override the input_key stored in the checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prob threshold for the '1+' class. Default 0.5. "
                             "Lower it to trade false negatives for false positives.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="Directory for output PNGs. Defaults to ./figures/.")
    parser.add_argument("--identifier", type=str, default=None,
                        help="Prefix for output figure filenames. "
                             "Defaults to the .pth basename (no extension).")
    parser.add_argument("--cm_title", type=str, default=None,
                        help="Override the confusion-matrix figure title. "
                             "When set, the input_key / elapsed-time suffix "
                             "is also dropped.")
    parser.add_argument("--cm_percent", action="store_true",
                        help="Render CM cells as row-normalized percentages "
                             "with a 0-100 colormap instead of raw counts.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            f"{args.model_path} is not a train_0or1_mlp checkpoint "
            f"(expected dict with 'state_dict' + metadata)."
        )

    input_key = args.input_key or ckpt["input_key"]
    input_shape = tuple(ckpt["input_shape"])
    num_classes = int(ckpt["num_classes"])
    if num_classes != 2:
        raise RuntimeError(
            f"eval_0or1_mlp.py expected a 2-class checkpoint, got "
            f"num_classes={num_classes}. Use evaluate_how_many.py instead."
        )
    input_size = int(np.prod(input_shape))

    print(f"input_key={input_key}, input_shape={input_shape}, "
          f"num_classes={num_classes}, threshold={args.threshold}")

    model = SimpleMLP(input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_paths = [d for d in args.data_dirs.split(":") if d]
    test_dataset, _ = load_binary_h5(data_paths, input_key)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    start_time = time.time()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            # Softmax + threshold on p(class=1) so the user can shift the
            # operating point (default 0.5 == argmax).
            probs = F.softmax(logits, dim=1)
            preds = (probs[:, 1] >= args.threshold).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    elapsed = time.time() - start_time
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Time to evaluate test cases: {elapsed:.3f}s")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Extra binary-classifier metrics beyond raw accuracy.
    tp = int(cm[1, 1]); fn = int(cm[1, 0])
    tn = int(cm[0, 0]); fp = int(cm[0, 1])
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall else float("nan"))
    print(f"Confusion (rows=true, cols=pred): TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    identifier = args.identifier or os.path.splitext(os.path.basename(args.model_path))[0]
    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)

    class_labels = ["0", "≥1"]
    cm_path = os.path.join(figures_dir, f"confusion_matrix_for_{identifier}.png")
    plot_confusion_matrix(cm, cm_path, accuracy, elapsed, class_labels,
                          input_key, title=args.cm_title,
                          percent=args.cm_percent)

    # ------------------------------------------------------------------
    # Sample visualization: a few test images with true / pred labels.
    # ------------------------------------------------------------------
    num_samples = min(12, len(test_dataset))
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    indices = random.sample(range(len(test_dataset)), num_samples)

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = np.atleast_2d(axs).ravel()

    fig.suptitle(
        f"0-or-1 MLP | input_key={input_key} "
        f"| Test Acc: {accuracy:.2f}% | Time: {elapsed:.3f}s "
        f"| threshold={args.threshold}",
        fontsize=14,
    )

    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            prob1 = F.softmax(logits, dim=1)[0, 1].cpu().item()
        pred = 1 if prob1 >= args.threshold else 0
        true_str = "≥1" if int(label.item()) == 1 else "0"
        pred_str = "≥1" if pred == 1 else "0"

        axs[i].imshow(img.numpy(), aspect="auto", cmap="magma_r")
        axs[i].axis("off")
        axs[i].set_title(f"True: {true_str}\nPred: {pred_str} (p={prob1:.2f})")

    for ax in axs[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"images_for_{identifier}.png")
    plt.savefig(fig_path)
    print(f"Evaluation figure saved to {fig_path}")


if __name__ == "__main__":
    main()
