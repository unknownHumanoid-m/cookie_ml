#!/usr/bin/env python3
"""Fit (dead_time_eV, loc_noise_rate, loc_noise_sigma_bins) to minimize the
distance between real and sim summary statistics from real_vs_sim_metrics.

Preconditions:
  1. Run `real_vs_sim_metrics.py` once to produce
     <outdir>/metrics_real.npz. That file is the fixed target for the fit
     (the real data is expensive to reload, so we cache it).
  2. This script iterates: for each parameter guess, resample N_SIM shots
     from generate_streak_data_realistic.sample_shot with those knobs,
     compute the same summary metrics, and score against the real target.

Loss = weighted L2 over three scalars from metrics_scalars.json:
  - gap_mean_length              (weight 1.0)
  - bandwidth_roi_mean_fwhm      (weight 1.0)
  - hits_per_port_median         (weight 0.25 -- already matched to
                                  --target_hits at generator start-up)

Nelder-Mead: cheap for 3 params, no gradient required, and the metric
surface is likely non-convex + noisy so scipy.optimize.minimize with
gradient methods would be unreliable anyway.

Typical use (assumes real metrics have been cached to figures/real_vs_sim/):

    python3 fit_realism_params.py \\
        --real_npz figures/real_vs_sim/metrics_real.npz \\
        --n_sim 800 \\
        --max_iter 30 \\
        --outdir figures/real_vs_sim/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

# Import in-tree modules directly by manipulating sys.path so the fit
# script doesn't require the streak_finder/ dir to be on PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_streak_data_realistic import (  # noqa: E402
    GenConfig,
    _load_river_kicks,
    calibrate_drawscale,
    sample_shot,
)
from real_vs_sim_metrics import (  # noqa: E402
    NBINS,
    ROI_BIN_LO,
    ROI_BIN_HI,
    bandwidth_vs_energy,
    gap_length_hist,
    hitcount_hist,
)


# Same 180..340 default as generate_streak_data_realistic; padded around
# ROI_BIN_LO/HI so bins outside get zero dark. Clamped to the axis so a
# smaller nenergies never overshoots.
_DARK_PAD_BINS = 24
_DEFAULT_DARK_ROI_LO = max(0, ROI_BIN_LO - _DARK_PAD_BINS)
_DEFAULT_DARK_ROI_HI = min(NBINS, ROI_BIN_HI + _DARK_PAD_BINS)


def _build_cfg(args, kamp_pool, kangle_pool, dead_time_eV,
               loc_noise_rate, loc_noise_sigma_bins) -> GenConfig:
    return GenConfig(
        nangles=16,
        nenergies=512,
        drawscale=1.0,  # calibrated below
        secondaryscale=0.0,
        e_center_min=args.e_center_min,
        e_center_max=args.e_center_max,
        sase_width_min=3.0,
        sase_width_max=5.0,
        ce_var=8.0,
        dark_min=0.007,
        dark_max=0.03,
        kamp_pool=kamp_pool,
        kangle_pool=kangle_pool,
        boost_min=1.0,
        boost_max=2.0,
        p_boost=0.0,             # fit uses the raw real distribution
        frac_unstreaked=0.0,     # keep every shot streaked for the fit
        per_port_e_jitter_bins=1.0,
        streak_threshold_eV=2.0,
        high_pulse_min=4,
        high_pulse_max=8,
        dead_time_eV=float(dead_time_eV),
        loc_noise_rate=float(loc_noise_rate),
        loc_noise_sigma_bins=float(loc_noise_sigma_bins),
        dark_roi_bin_lo=_DEFAULT_DARK_ROI_LO,
        dark_roi_bin_hi=_DEFAULT_DARK_ROI_HI,
    )


def _simulate_batch(cfg: GenConfig, n_sim: int, base_seed: int) -> np.ndarray:
    out = np.zeros((n_sim, cfg.nangles, cfg.nenergies), dtype=np.uint16)
    for i in range(n_sim):
        img, _ = sample_shot(base_seed + i, cfg)
        out[i] = img
    return out


def _summarise(x: np.ndarray, gap_edges, hit_edges):
    bw = bandwidth_vs_energy(x)
    gh = gap_length_hist(x, gap_edges, ROI_BIN_LO, ROI_BIN_HI)
    hh = hitcount_hist(x, hit_edges)
    per_port = x.sum(axis=2)
    return {
        "gap_mean_length": float(
            (gh.sum(axis=0)
             * (0.5 * (gap_edges[:-1] + gap_edges[1:]))).sum()
            / max(1.0, float(gh.sum()))
        ),
        "bandwidth_roi_mean_fwhm": float(
            bw[0, ROI_BIN_LO:ROI_BIN_HI + 1].mean()
        ),
        "hits_per_port_median": float(np.median(per_port)),
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real_npz", type=str, required=True,
                    help="Path to metrics_real.npz produced by "
                         "real_vs_sim_metrics.py. The fit uses its summary "
                         "scalars as the target.")
    ap.add_argument("--river_path", type=str,
                    default="/sdf/data/lcls/ds/tmo/tmol1043723/results/"
                            "streaking_results/run[99]_vjtw_river.h5")
    ap.add_argument("--e_center_min", type=float, default=50.0)
    ap.add_argument("--e_center_max", type=float, default=200.0)
    ap.add_argument("--target_hits", type=float, default=250.0)
    ap.add_argument("--n_sim", type=int, default=800,
                    help="Sim shots resampled per parameter guess.")
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    # Initial params.
    ap.add_argument("--x0_dead_time_eV", type=float, default=1.0)
    ap.add_argument("--x0_loc_noise_rate", type=float, default=0.5)
    ap.add_argument("--x0_loc_noise_sigma_bins", type=float, default=1.5)
    # Loss weights.
    ap.add_argument("--w_gap", type=float, default=1.0)
    ap.add_argument("--w_bw",  type=float, default=1.0)
    ap.add_argument("--w_hits",type=float, default=0.25)
    ap.add_argument("--outdir", type=str, default="figures/real_vs_sim")
    return ap.parse_args()


def _target_from_real(real_npz_path: str, gap_edges, hit_edges):
    d = np.load(real_npz_path)
    bw = d["bandwidth"]
    gh = d["gap_hist"]
    per_port = None  # not saved directly; recompute proxy from hit_hist:
    # hit_hist is a per-port histogram of hits/shot -- take the histogram
    # median as an approximation of hits_per_port_median.
    hh = d["hit_hist"]
    centers = 0.5 * (hit_edges[:-1] + hit_edges[1:])
    counts = hh.sum(axis=0).astype(float)
    if counts.sum() == 0:
        med = 0.0
    else:
        cum = np.cumsum(counts) / counts.sum()
        med = float(centers[np.searchsorted(cum, 0.5)])
    return {
        "gap_mean_length": float(
            (gh.sum(axis=0)
             * (0.5 * (gap_edges[:-1] + gap_edges[1:]))).sum()
            / max(1.0, float(gh.sum()))
        ),
        "bandwidth_roi_mean_fwhm": float(
            bw[0, ROI_BIN_LO:ROI_BIN_HI + 1].mean()
        ),
        "hits_per_port_median": med,
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gap_edges = np.arange(1, 41)
    hit_edges = np.linspace(0, 80, 41)

    target = _target_from_real(args.real_npz, gap_edges, hit_edges)
    print("target (from real):", target)

    # Load the real kamp/kangle pool once so each iteration doesn't reopen h5.
    kamp_pool, kangle_pool = _load_river_kicks(args.river_path, min_kamp=1.5)
    if kamp_pool.size == 0:
        raise RuntimeError("river pool empty")
    print(f"kamp pool size: {kamp_pool.size}")

    # Calibrate drawscale ONCE at the initial params. If dead-time / noise
    # substantially shift mean(sum(Ximg)) we accept the residual and let
    # the hits weight pull it back.
    init_cfg = _build_cfg(
        args, kamp_pool, kangle_pool,
        args.x0_dead_time_eV, args.x0_loc_noise_rate,
        args.x0_loc_noise_sigma_bins,
    )
    ds = calibrate_drawscale(init_cfg, args.target_hits, args.seed)

    def make_cfg(theta):
        dt, rate, sigma = theta
        cfg = _build_cfg(args, kamp_pool, kangle_pool,
                         max(0.0, dt), max(0.0, rate), max(1e-3, sigma))
        # Reuse the calibrated drawscale (linear in build_XY output, so
        # dead-time and localized noise only shift hits mildly).
        return GenConfig(**{**cfg.__dict__, "drawscale": ds})

    history = []

    def loss(theta):
        cfg = make_cfg(theta)
        seed_base = int(args.seed + 1_000 * len(history))
        x = _simulate_batch(cfg, args.n_sim, seed_base)
        s = _summarise(x, gap_edges, hit_edges)
        # Normalised squared errors so the scales are comparable.
        def rel(k):
            t = target[k]
            return ((s[k] - t) / max(abs(t), 1e-6)) ** 2
        L = (args.w_gap * rel("gap_mean_length")
             + args.w_bw  * rel("bandwidth_roi_mean_fwhm")
             + args.w_hits* rel("hits_per_port_median"))
        history.append({"theta": list(map(float, theta)),
                        "sim": s, "loss": float(L)})
        print(f"  iter {len(history):3d}: dt={theta[0]:.3f}  rate={theta[1]:.3f}  "
              f"sigma={theta[2]:.3f}  ->  gap={s['gap_mean_length']:.2f} "
              f"bw={s['bandwidth_roi_mean_fwhm']:.2f} hits={s['hits_per_port_median']:.1f} "
              f" loss={L:.4f}", flush=True)
        return L

    from scipy.optimize import minimize

    x0 = np.array([args.x0_dead_time_eV, args.x0_loc_noise_rate,
                   args.x0_loc_noise_sigma_bins], dtype=float)
    t0 = time.time()
    res = minimize(
        loss, x0, method="Nelder-Mead",
        options={"maxiter": args.max_iter, "xatol": 5e-3, "fatol": 5e-4},
    )
    dt = time.time() - t0

    print(f"\nfit done in {dt:.1f} s over {len(history)} evals")
    print(f"best theta = {res.x}")
    print(f"best loss  = {res.fun:.4f}")

    out = {
        "target": target,
        "best": {
            "dead_time_eV": float(res.x[0]),
            "loc_noise_rate": float(res.x[1]),
            "loc_noise_sigma_bins": float(res.x[2]),
            "loss": float(res.fun),
        },
        "history": history,
        "elapsed_s": dt,
    }
    with open(os.path.join(args.outdir, "fit_realism_params.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote fit history to {args.outdir}/fit_realism_params.json")


if __name__ == "__main__":
    main()
