"""
Loss functions for the split-bottleneck autoencoder.

Four components:
    * reconstruction — pixel-wise MSE between decoder output and clean truth
    * count          — cross entropy on the pulse-count classifier
    * phase_single   — MSE on the sincos-unit-circle target for phi0.
                       Masked to shots with npulses == 1 (mirrors
                       train_phase_single.py).
    * phase_two      — MSE on arccos(cos(phi0 - phi1)) in [0, pi]. Masked
                       to shots with npulses == 2 (mirrors
                       train_phase_two.py).

Total loss is either a weighted sum (static weights) or the Kendall-Gal
uncertainty-weighted combination (learned per-task log_var scalars).

None of the losses are called on tensors that live on a different device
than the model, so no device juggling is done here — the caller is
responsible for that (see train.py).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Phase-single: sincos of phi0
# --------------------------------------------------------------------------
def sincos_phase_loss(phase_out: torch.Tensor, phase_rad: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE on the unit-circle target for phi0.

    ``phase_out`` is a raw 2-vector per sample. Target is
    ``(sin(phi), cos(phi))``. MSE will naturally push the head to unit-norm
    outputs, and skipping the L2-normalize saves ops on FPGA.

    If ``mask`` is provided (bool tensor of shape (N,)), the loss is
    computed only over ``mask == True`` shots and averaged over those.
    Returns 0 (with a device-matching zero tensor) if the mask is empty.
    """
    target = torch.stack([torch.sin(phase_rad), torch.cos(phase_rad)], dim=1)
    if mask is None:
        return F.mse_loss(phase_out, target)
    if mask.any():
        return F.mse_loss(phase_out[mask], target[mask])
    return phase_out.sum() * 0.0  # keep on-device, no grad contribution


def decode_sincos_phase(phase_out: torch.Tensor) -> torch.Tensor:
    """(sin, cos) 2-vector -> phase in [0, 2*pi)."""
    s = phase_out[:, 0]
    c = phase_out[:, 1]
    phi = torch.atan2(s, c)
    return torch.remainder(phi, 2.0 * math.pi)


# --------------------------------------------------------------------------
# Phase-two: scalar arccos(cos Δφ) in [0, pi]
# --------------------------------------------------------------------------
def phase_two_loss(phase_out: torch.Tensor, dphi: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """MSE on ``arccos(cos(phi0 - phi1))`` for 2-pulse shots.

    ``phase_out`` is (N, 1); ``dphi`` is (N,) and must be finite where
    ``mask`` is True (upstream loader stores NaN for non-2-pulse shots).
    Zero-averaged shots -> zero on-device tensor.
    """
    pred = phase_out.view(-1)
    if mask is None:
        return F.mse_loss(pred, dphi)
    if mask.any():
        return F.mse_loss(pred[mask], dphi[mask])
    return phase_out.sum() * 0.0


def decode_phase_two(phase_out: torch.Tensor) -> torch.Tensor:
    """Scalar 2-pulse head output folded into [0, pi] via arccos(cos(.))."""
    return torch.arccos(torch.cos(phase_out.view(-1)))


# --------------------------------------------------------------------------
# Eval metric on the wrap-aware absolute error (rad)
# --------------------------------------------------------------------------
def circular_error(phase_pred: torch.Tensor, phase_true: torch.Tensor) -> torch.Tensor:
    """Absolute wrap-aware angular error in radians.

    Useful as an eval-time metric; not itself used in the training loss.
    """
    diff = torch.remainder(phase_pred - phase_true + math.pi, 2.0 * math.pi) - math.pi
    return diff.abs()


# --------------------------------------------------------------------------
# Multi-task combiner
# --------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """Combine (recon, count, phase_single, phase_two) losses into a scalar.

    Two modes:
      * static — use CONFIG weights w_recon / w_count / w_phase_single /
        w_phase_two directly
      * uncertainty — Kendall & Gal (2018). Each task gets a learnable
        ``log_var_i``; the combined term for task i is
        ``0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i``. Static weights
        are ignored while this is on.

    ``forward`` returns (total_loss, dict of per-task unweighted scalars)
    plus the per-batch valid-shot counts for each phase head (so logging
    can average correctly across batches with different counts).
    """

    def __init__(
        self,
        w_recon: float,
        w_count: float,
        w_phase_single: float,
        w_phase_two: float,
        uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.w_recon = float(w_recon)
        self.w_count = float(w_count)
        self.w_phase_single = float(w_phase_single)
        self.w_phase_two = float(w_phase_two)
        self.uncertainty_weighting = bool(uncertainty_weighting)

        if self.uncertainty_weighting:
            # log_var_i initialized to 0 (i.e. var=1). Learned scalars.
            self.log_var_recon = nn.Parameter(torch.zeros(()))
            self.log_var_count = nn.Parameter(torch.zeros(()))
            self.log_var_phase_single = nn.Parameter(torch.zeros(()))
            self.log_var_phase_two = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_var_recon", None)
            self.register_parameter("log_var_count", None)
            self.register_parameter("log_var_phase_single", None)
            self.register_parameter("log_var_phase_two", None)

        self.ce = nn.CrossEntropyLoss()

    def _device_probe(self):
        if self.log_var_recon is not None:
            return self.log_var_recon.device
        return torch.device("cpu")

    def _combine(self, l_recon, l_count, l_phase_single, l_phase_two,
                 active_recon, active_task, has_single, has_two):
        dev = self._device_probe()
        total = torch.zeros((), device=dev)
        if self.uncertainty_weighting:
            if active_recon and l_recon is not None:
                total = total + 0.5 * torch.exp(-self.log_var_recon) * l_recon \
                              + 0.5 * self.log_var_recon
            if active_task:
                total = total + 0.5 * torch.exp(-self.log_var_count) * l_count \
                              + 0.5 * self.log_var_count
                if has_single:
                    total = total + 0.5 * torch.exp(-self.log_var_phase_single) * l_phase_single \
                                  + 0.5 * self.log_var_phase_single
                if has_two:
                    total = total + 0.5 * torch.exp(-self.log_var_phase_two) * l_phase_two \
                                  + 0.5 * self.log_var_phase_two
        else:
            if active_recon and l_recon is not None:
                total = total + self.w_recon * l_recon
            if active_task:
                total = total + self.w_count * l_count
                if has_single:
                    total = total + self.w_phase_single * l_phase_single
                if has_two:
                    total = total + self.w_phase_two * l_phase_two
        return total

    def forward(
        self,
        recon: Optional[torch.Tensor],
        recon_target: Optional[torch.Tensor],
        count_logits: Optional[torch.Tensor],
        count_label: Optional[torch.Tensor],
        phase_single_out: Optional[torch.Tensor],
        phi0_rad: Optional[torch.Tensor],
        phase_two_out: Optional[torch.Tensor],
        dphi: Optional[torch.Tensor],
        npulses_label: Optional[torch.Tensor],   # count labels remapped to [0, num_classes)
        min_pulses: int,
        active_recon: bool = True,
        active_task: bool = True,
    ):
        dev = self._device_probe()
        l_recon = None
        l_count = torch.zeros((), device=dev)
        l_phase_single = torch.zeros((), device=dev)
        l_phase_two = torch.zeros((), device=dev)
        n_single = 0
        n_two = 0

        if active_recon:
            if recon is None or recon_target is None:
                raise ValueError("active_recon=True but recon or target is None.")
            l_recon = F.mse_loss(recon, recon_target)

        has_single = False
        has_two = False
        if active_task:
            if count_logits is None or count_label is None:
                raise ValueError("active_task=True but count logits/labels are None.")
            if phase_single_out is None or phi0_rad is None:
                raise ValueError("active_task=True but phase_single_out or phi0_rad is None.")
            if phase_two_out is None or dphi is None:
                raise ValueError("active_task=True but phase_two_out or dphi is None.")
            if npulses_label is None:
                raise ValueError("active_task=True but npulses_label is None.")

            l_count = self.ce(count_logits, count_label)

            # phase_single trains on npulses == 1 (count label 1 - min_pulses)
            single_label = 1 - int(min_pulses)
            single_mask = (count_label == single_label)
            n_single = int(single_mask.sum().item())
            has_single = n_single > 0
            l_phase_single = sincos_phase_loss(
                phase_single_out, phi0_rad, mask=single_mask,
            )

            # phase_two trains on shots where dphi is finite (npulses == 2)
            two_mask = torch.isfinite(dphi)
            n_two = int(two_mask.sum().item())
            has_two = n_two > 0
            l_phase_two = phase_two_loss(phase_two_out, dphi, mask=two_mask)

        total = self._combine(
            l_recon, l_count, l_phase_single, l_phase_two,
            active_recon, active_task, has_single, has_two,
        )

        parts = {
            "recon": (l_recon.detach() if l_recon is not None else None),
            "count": l_count.detach(),
            "phase_single": l_phase_single.detach(),
            "phase_two": l_phase_two.detach(),
            "n_single": n_single,
            "n_two": n_two,
        }
        if self.uncertainty_weighting:
            parts["log_var_recon"] = self.log_var_recon.detach()
            parts["log_var_count"] = self.log_var_count.detach()
            parts["log_var_phase_single"] = self.log_var_phase_single.detach()
            parts["log_var_phase_two"] = self.log_var_phase_two.detach()
        return total, parts


# --------------------------------------------------------------------------
# Eval metrics (not in the training path — used by eval.py)
# --------------------------------------------------------------------------
def count_accuracy(count_logits: torch.Tensor, count_label: torch.Tensor) -> torch.Tensor:
    preds = count_logits.argmax(dim=1)
    return (preds == count_label).float().mean()


def recon_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)
