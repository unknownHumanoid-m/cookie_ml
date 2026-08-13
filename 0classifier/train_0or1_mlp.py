# -*- coding: utf-8 -*-
"""
Binary "zero vs. one-or-more pulses" MLP classifier over the raw 16x512
Ximg image. Trimmed sibling of `train_how_many.py` — same load / checkpoint
convention, so `evaluate_how_many.py` can load a run from here without
knowing it was binary.

Labels: 0 for shots with npulses == 0, 1 for shots with npulses >= 1. The
classifier is a 3-hidden-layer MLP over the flattened image; conv layers
were intentionally left out per the request for a plain MLP baseline.

Example
-------
    python3 train_0or1_mlp.py \\
        --data_dirs /path/to/mrco_h5/train/ \\
        --input_key Ximg \\
        --save_dir  /path/to/runs/ \\
        --save_model zero_or_one_mlp.pth
"""

import os
import copy
import argparse
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# Hard-coded label range so the checkpoint format matches train_how_many.py
# (evaluate_how_many.py reads min_pulses / max_pulses and treats the top
# class as ">= max_pulses"). Here max_pulses=1 means the "1" class collapses
# every shot with 1+ pulses.
MIN_PULSES = 0
MAX_PULSES = 1
NUM_CLASSES = MAX_PULSES - MIN_PULSES + 1  # = 2


def load_binary_h5(paths, input_key):
    """Preload every h5 group under `paths` into a TensorDataset with binary
    labels: 0 if npulses == 0, 1 if npulses >= 1. Accepts a mix of
    directories and .h5 files.
    """
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

    inputs, labels = [], []
    for path in files:
        print(f"0or1_mlp: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                y = 0 if n == 0 else 1
                inputs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                labels.append(y)
    if not inputs:
        raise RuntimeError(f"No shots found in {files}")

    x = torch.from_numpy(np.stack(inputs))
    y = torch.tensor(labels, dtype=torch.long)
    input_shape = tuple(x.shape[1:])
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(
        f"0or1_mlp: loaded {len(y)} shots, input_key={input_key}, "
        f"shape={input_shape}, classes=2 "
        f"(npulses==0: {n_neg} | npulses>=1: {n_pos})"
    )
    return TensorDataset(x, y), input_shape


##############################################################################
# Early Stopping Tracker Class
##############################################################################
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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


##############################################################################
# Model — same SimpleMLP as train_how_many.py, kept in-file so this script
# is self-contained (no risk of an import-name drift silently changing the
# 0-or-1 model when the how-many trunk gets tweaked).
##############################################################################
class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super().__init__()
        print(f"Input size: {input_size}, num_classes: {num_classes}")
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        return self.out(x)


def mse_metric(logits, labels):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
    return F.mse_loss(probs, one_hot, reduction="mean")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_correct = 0
    total_elements = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mse += mse_metric(logits, labels).item()

        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_elements += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_mse = running_mse / len(loader)
    epoch_acc = 100.0 * running_correct / total_elements
    return epoch_loss, epoch_mse, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_correct = 0
    total_elements = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            running_loss += criterion(logits, labels).item()
            running_mse += mse_metric(logits, labels).item()

            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_elements += labels.size(0)

    avg_loss = running_loss / len(loader)
    epoch_mse = running_mse / len(loader)
    avg_acc = 100.0 * running_correct / total_elements
    return avg_loss, epoch_mse, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--input_key", type=str, default="Ximg",
                        help="Per-shot h5 dataset to feed the MLP. "
                             "Default 'Ximg' (raw 16x512 image).")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to write the .pth into. "
                             "If unset, the model is not saved.")
    parser.add_argument("--save_model", type=str, default=None,
                        help="Filename (relative to --save_dir) for the model.")
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="Directory for training-curve PNG. "
                             "Defaults to ./figures/ next to this file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_paths = [d for d in args.data_dirs.split(":") if d]
    dataset, input_shape = load_binary_h5(data_paths, args.input_key)

    val_size = int(args.val_frac * len(dataset))
    train_size = len(dataset) - val_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    input_size = int(np.prod(input_shape))
    model = SimpleMLP(input_size=input_size, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience)

    train_losses, train_accs, train_mses = [], [], []
    val_losses, val_accs, val_mses = [], [], []
    actual_epochs = 0

    start_time = time.time()
    print("Training the 0-or-1 pulse MLP classifier...")
    for epoch in range(args.epochs):
        actual_epochs += 1
        tl, tm, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, vm, va = evaluate(model, val_loader, criterion, device)

        train_losses.append(tl); train_mses.append(tm); train_accs.append(ta)
        val_losses.append(vl);   val_mses.append(vm);   val_accs.append(va)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"| Train Loss: {tl:.4f}, MSE: {tm:.4f}, Acc: {ta:.2f}% "
              f"| Val Loss: {vl:.4f}, MSE: {vm:.4f}, Acc: {va:.2f}%")

        early_stopping(vl, model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Cutting training short.")
            break

    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)
        print(f"Rolled back to best weights (Best Val Loss: {early_stopping.best_loss:.4f})")

    print(f"Training time: {time.time() - start_time:.2f}s")

    if args.save_model is not None:
        save_dir = args.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, args.save_model)
        torch.save({
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "num_classes": NUM_CLASSES,
            "min_pulses": MIN_PULSES,
            "max_pulses": MAX_PULSES,
            "input_key": args.input_key,
        }, save_path)
        print(f"Best model saved to {save_path}")

    # ------------------------------------------------------------------
    # Training-curve figure — same 3-panel layout as train_how_many.py.
    # ------------------------------------------------------------------
    epochs_range = range(1, actual_epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.suptitle(
        f"0-or-1 MLP Classifier | input_key={args.input_key} "
        f"| classes {MIN_PULSES}..{MAX_PULSES}",
        fontsize=14,
    )
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.title("CE Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_mses, label="Train")
    plt.plot(epochs_range, val_mses, label="Val")
    plt.title("MSE"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_accs, label="Train")
    plt.plot(epochs_range, val_accs, label="Val")
    plt.title("Accuracy (%)"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()

    if args.save_model is not None:
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures"
        )
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f"training_data_for_{args.save_model}.png")
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
