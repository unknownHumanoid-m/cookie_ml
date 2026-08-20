#!/usr/bin/env python3
"""Extract per-port summary statistics that discriminate the two real-vs-sim
mismatches called out in the task brief:

  (a) SIGNAL BAND WIDTH:  for each port and energy bin, the mean cross-port
      row profile is measured and its FWHM across the ports axis is treated
      as a rough band-width proxy. Real data has a *tighter* core signal
      band than sim; this metric should surface that.
  (b) GAP-LENGTH DISTRIBUTION:  histogram of consecutive-zero runs inside
      the region-of-interest (ROI) window of the Ximg, per port. Real data
      shows dead-time-like gaps that sim doesn't produce.
  (c) HIT-COUNT DISTRIBUTION:  per-port total hit counts per shot.

Everything is measured in Ximg bin-space so the fact that real and sim
live on different KE offsets doesn't matter -- only the *shape* of the
per-port row does.

Real data is rebuilt on the fly from the notebook's preproc + calibration
pipeline (documented in copied_ipynbs/MRCO_Streaking_rawData.ipynb). To
keep runtime bounded the script defaults to a random subset of duck shots
biased to high kamp (the interesting regime for feasibility).

Outputs:
  <outdir>/metrics_real.npz      -- raw per-shot metric arrays
  <outdir>/metrics_sim.npz       -- same schema for sim
  <outdir>/metrics_scalars.json  -- compact scalar summary (fit target)
  <outdir>/metrics_bandwidth.png -- band-width vs energy bin, real vs sim
  <outdir>/metrics_gaps.png      -- gap-length histogram, real vs sim
  <outdir>/metrics_hits.png      -- per-port hit-count histogram, real vs sim

Typical use:

    python3 real_vs_sim_metrics.py \\
        --sim_glob '/sdf/.../streak_finder_realistic_sanity/train/*.h5' \\
        --n_real 2000 --n_sim 2000 \\
        --kamp_min 1.5 \\
        --outdir figures/real_vs_sim/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import h5py
import numpy as np


# --- Notebook-derived constants; keep in sync with MRCO_Streaking_rawData.ipynb.
# 2026-08-11 retune: dropped from 0.25 eV/bin to 0.1 eV/bin and anchored the
# RIGHT edge at 43.11 eV (where the old 0.25 eV grid's bin 320 sat, past which
# inspection showed essentially no hits). This matches the grid the
# realistic-sim data + checkpoints (*_01eV.pth) were trained against.
HSD_CHANNELS: Sequence[int] = (
    0, 22, 45, 67, 90, 112, 135, 157,
    180, 202, 225, 247, 270, 292, 315, 337,
)
V_RET = 280.0            # retardation magnitude, eV
NBINS = 512
ESTEP = 0.1              # eV / bin

E_AU  = 27.2114
P_ROI = (4.65, 4.85)
P_ROI_KE_IP_LO  = P_ROI[0] ** 2 / 2 * E_AU
P_ROI_KE_IP_HI  = P_ROI[1] ** 2 / 2 * E_AU
P_ROI_KE_DET_LO = P_ROI_KE_IP_LO - V_RET
P_ROI_KE_DET_HI = P_ROI_KE_IP_HI - V_RET
KE_CENTER_DET   = 0.5 * (P_ROI_KE_DET_LO + P_ROI_KE_DET_HI)
EMAX = 43.11
EMIN = EMAX - NBINS * ESTEP
EBIN_EDGES = np.linspace(EMIN, EMAX, NBINS + 1)
EBIN_CENTERS = 0.5 * (EBIN_EDGES[:-1] + EBIN_EDGES[1:])

# ROI in bins (kept as the "signal band" for gap-length statistics; the
# notebook computes 204..307 with V_RET=280 eV).
ROI_BIN_LO = int(np.argmin(np.abs(EBIN_CENTERS - P_ROI_KE_DET_LO)))
ROI_BIN_HI = int(np.argmin(np.abs(EBIN_CENTERS - P_ROI_KE_DET_HI)))

TOF_KE_GATE_MARGIN = 20.0  # eV -- matches the notebook


@dataclass
class RealPaths:
    preproc: str
    river: str
    calib: str


def _default_paths() -> RealPaths:
    return RealPaths(
        preproc="/sdf/data/lcls/ds/tmo/tmol1043723/scratch/preproc/vjtw/run99_vjtw.h5",
        river="/sdf/data/lcls/ds/tmo/tmol1043723/results/streaking_results/"
              "run[99]_vjtw_river.h5",
        calib="/sdf/home/m/miaed/copied_ipynbs/tmo_utils/calibration/"
              "mrco_calib_tmox101_205-213_simon_t0.h5",
    )


# --------------------------------------------------------------------------
# Real-side loading (notebook logic, condensed)
# --------------------------------------------------------------------------
def _load_alpha_t0(calib_path: str, v_ret: float):
    """Per-port alpha(v_ret) + t0. Linear inside the calibrated v_r range,
    quadratic extrapolation outside. Mirrors the notebook."""
    from scipy.interpolate import interp1d

    with h5py.File(calib_path, "r") as hf:
        alphas = np.asarray(hf["alphas"])   # (16, 5)
        v_r    = np.asarray(hf["v_r"])       # (5,)
        t0s    = np.asarray(hf["t0s"])       # (16,)
        angs   = np.asarray(hf["angles"])    # (16,)

    a_at, t0_at = {}, {}
    for i, ang in enumerate(angs):
        row = alphas[i].astype(float)
        if v_r.min() <= v_ret <= v_r.max():
            a = float(interp1d(v_r, row, kind="linear")(v_ret))
        else:
            a = float(np.polyval(np.polyfit(v_r.astype(float), row, deg=2), v_ret))
        a_at[int(ang)] = a
        t0_at[int(ang)] = float(t0s[i])
    return a_at, t0_at


def _tof_to_ke(tof, alpha, t0):
    dt = tof - t0
    ke = np.zeros_like(tof, dtype=np.float64)
    m = dt > 0
    ke[m] = alpha / (dt[m] ** 2)
    return ke


def _tof_gate(alpha, t0, ke_lo, ke_hi):
    """Return (t_lo, t_hi) inclusive of the calibrated KE window."""
    ke_hi_eff = max(ke_hi, 1.0)
    ke_lo_eff = max(ke_lo, 1e-3)
    return t0 + np.sqrt(alpha / ke_hi_eff), t0 + np.sqrt(alpha / ke_lo_eff)


def _match_duck_rows(preproc_ts: np.ndarray, duck_ts: np.ndarray):
    """Row indices in preproc for the duck shots. Copies the notebook."""
    order = np.argsort(preproc_ts)
    sorted_ts = preproc_ts[order]
    pos = np.searchsorted(sorted_ts, duck_ts)
    ok = (pos < sorted_ts.size) & (
        sorted_ts[np.clip(pos, 0, sorted_ts.size - 1)] == duck_ts
    )
    rows = order[pos[ok]]
    river_idx = np.flatnonzero(ok)
    sort2 = np.argsort(rows)
    return rows[sort2], river_idx[sort2]


def load_real_ximgs(paths: RealPaths, n_shots: int, kamp_min: float,
                    seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Ximg, kamp) arrays of shape (n_shots, 16, NBINS), (n_shots,)."""
    print(f"[real] loading river ({paths.river}) ...")
    with h5py.File(paths.river, "r") as rf:
        duck_ts    = rf["duck_timestamps"][:]
        duck_kamp  = rf["duck_kamp"][:]

    print(f"[real] loading preproc timestamps ({paths.preproc}) ...")
    with h5py.File(paths.preproc, "r") as pf:
        preproc_ts = pf["timestamp"][:]

    rows, river_idx = _match_duck_rows(preproc_ts, duck_ts)
    kamp = duck_kamp[river_idx]

    # Filter to shots at or above kamp_min, then random subsample down to
    # n_shots. Deterministic given `seed`.
    mask = np.isfinite(kamp) & (kamp >= kamp_min)
    pool_rows = rows[mask]
    pool_kamp = kamp[mask]
    if pool_rows.size == 0:
        raise RuntimeError(
            f"no real shots with kamp >= {kamp_min}. Try lowering --kamp_min."
        )
    print(f"[real] duck shots with kamp>={kamp_min}: {pool_rows.size}")

    take = min(n_shots, pool_rows.size)
    rng = np.random.default_rng(seed)
    sel = rng.choice(pool_rows.size, size=take, replace=False)
    sel = np.sort(sel)  # h5py fancy indexing needs sorted
    sel_rows = pool_rows[sel]
    sel_kamp = pool_kamp[sel]

    # Load per-port hits only for the selected rows. We can't slice a
    # variable-length dataset by fancy index efficiently, so read `flat`
    # + `lens` per channel and slice numpy in-memory.
    print(f"[real] loading hits for {take} shots x {len(HSD_CHANNELS)} ports ...")
    alpha_at, t0_at = _load_alpha_t0(paths.calib, V_RET)
    tof_gate = {
        ang: _tof_gate(alpha_at[ang], t0_at[ang],
                       EMIN - TOF_KE_GATE_MARGIN, EMAX + TOF_KE_GATE_MARGIN)
        for ang in HSD_CHANNELS
    }
    x = np.zeros((take, len(HSD_CHANNELS), NBINS), dtype=np.uint16)

    with h5py.File(paths.preproc, "r") as pf:
        for i, ang in enumerate(HSD_CHANNELS):
            lens = pf[f"var_hsd_hf_times_{ang}_len"][:]
            flat = pf[f"var_hsd_hf_times_{ang}"][:]
            ends = np.cumsum(lens)
            starts = ends - lens
            t_lo, t_hi = tof_gate[ang]
            for k, row in enumerate(sel_rows):
                hits = flat[starts[row]:ends[row]]
                if hits.size == 0:
                    continue
                good = (hits > t_lo) & (hits < t_hi)
                if not good.any():
                    continue
                ke = _tof_to_ke(hits[good], alpha_at[ang], t0_at[ang])
                x[k, i], _ = np.histogram(ke, bins=EBIN_EDGES)

    return x, sel_kamp.astype(np.float64)


# --------------------------------------------------------------------------
# Sim-side loading
# --------------------------------------------------------------------------
def load_sim_ximgs(sim_glob: str, n_shots: int, kamp_min: float, seed: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Read Ximg from any h5 shard matching sim_glob (uses the realistic
    generator's per-shot layout: /shot_XXXXXX/Ximg + attrs['kickstrength'])."""
    paths = sorted(glob.glob(sim_glob))
    if not paths:
        raise RuntimeError(f"no sim files match {sim_glob!r}")
    print(f"[sim ] scanning {len(paths)} shard(s) matching {sim_glob!r}")

    imgs: List[np.ndarray] = []
    kamps: List[float] = []
    rng = np.random.default_rng(seed)
    for path in paths:
        with h5py.File(path, "r") as f:
            keys = sorted(k for k in f.keys() if k.startswith("shot_"))
            rng.shuffle(keys)
            for k in keys:
                grp = f[k]
                kk = float(grp.attrs.get("kickstrength", 0.0))
                if kk < kamp_min:
                    continue
                imgs.append(np.asarray(grp["Ximg"][:], dtype=np.uint16))
                kamps.append(kk)
                if len(imgs) >= n_shots:
                    break
        if len(imgs) >= n_shots:
            break
    if not imgs:
        raise RuntimeError(
            f"sim shards had 0 shots with kamp >= {kamp_min}. Regenerate with "
            f"--dev_kamp_min <= {kamp_min} or lower --kamp_min."
        )
    x = np.stack(imgs, axis=0)
    print(f"[sim ] loaded {x.shape[0]} shots  shape={x.shape}")
    return x, np.asarray(kamps, dtype=np.float64)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def bandwidth_vs_energy(x: np.ndarray) -> np.ndarray:
    """(nports, nbins) -> (nports, nbins) FWHM proxy across the ports axis.

    For each energy bin, look at the intensity profile across the 16 ports
    (mean over shots first). FWHM is measured as: number of ports whose
    intensity is at least half the max port intensity for that bin. Cheap,
    binning-free, and doesn't care about the sinusoid geometry.

    A tighter core signal band -> smaller FWHM at bins where the signal is.
    """
    if x.ndim != 3:
        raise ValueError(f"expected (n, ports, bins), got {x.shape}")
    m = x.mean(axis=0).astype(np.float64)   # (ports, bins)
    peak = m.max(axis=0, keepdims=True)     # (1, bins)
    peak = np.maximum(peak, 1e-12)
    above = m >= 0.5 * peak                  # (ports, bins)
    fwhm_ports = above.sum(axis=0).astype(np.float64)   # (bins,)
    # broadcast to the (ports, bins) contract for downstream flexibility.
    return np.broadcast_to(fwhm_ports, m.shape).copy()


def gap_length_hist(x: np.ndarray, bin_edges: np.ndarray,
                    roi_bin_lo: int, roi_bin_hi: int) -> np.ndarray:
    """Per-port histogram of consecutive-zero run lengths in the ROI window.

    Real data shows dead-time-driven gaps as long stretches of empty bins
    inside the signal band. Sim (pre-dead-time) has few runs longer than
    a couple of bins -- that difference is what step 4 fits against.
    """
    nports = x.shape[1]
    out = np.zeros((nports, bin_edges.size - 1), dtype=np.int64)
    for shot in x:
        for p in range(nports):
            row = shot[p, roi_bin_lo:roi_bin_hi + 1]
            zero = row == 0
            # Compress runs: find where the state changes.
            edges = np.diff(np.concatenate(([0], zero.view(np.int8), [0])))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            runs = ends - starts
            if runs.size:
                out[p] += np.histogram(runs, bins=bin_edges)[0]
    return out


def hitcount_hist(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Per-port histogram of total hits per shot (summed over energy bins)."""
    per_port_totals = x.sum(axis=2)   # (n, ports)
    nports = per_port_totals.shape[1]
    out = np.zeros((nports, bin_edges.size - 1), dtype=np.int64)
    for p in range(nports):
        out[p], _ = np.histogram(per_port_totals[:, p], bins=bin_edges)
    return out


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def _plot_bandwidth(bw_real: np.ndarray, bw_sim: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    # Both arrays are broadcast; take row 0 (it's identical across ports).
    ax.plot(bw_real[0], color="tab:blue", label="real")
    ax.plot(bw_sim[0],  color="tab:orange", label="sim")
    ax.axvspan(ROI_BIN_LO, ROI_BIN_HI, color="red", alpha=0.10, label="ROI")
    ax.set_xlabel("energy bin index")
    ax.set_ylabel("FWHM proxy (ports above half-max)")
    ax.set_title("Signal band width vs energy bin  (lower = tighter core band)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_gap_hist(gh_real: np.ndarray, gh_sim: np.ndarray,
                   bin_edges: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(centers, gh_real.sum(axis=0), color="tab:blue",
            label=f"real (sum {gh_real.sum():.0f})")
    ax.plot(centers, gh_sim.sum(axis=0),  color="tab:orange",
            label=f"sim  (sum {gh_sim.sum():.0f})")
    ax.set_xlabel("gap length (bins of zeros inside ROI)")
    ax.set_ylabel("count of gaps")
    ax.set_yscale("log")
    ax.set_title("Gap-length distribution across ports (ROI-only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_hit_hist(hh_real: np.ndarray, hh_sim: np.ndarray,
                   bin_edges: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fig, ax = plt.subplots(figsize=(10, 4))
    # Sum across ports for the summary panel; per-port breakdown is in
    # metrics_scalars.json.
    ax.plot(centers, hh_real.sum(axis=0), color="tab:blue",  label="real")
    ax.plot(centers, hh_sim.sum(axis=0),  color="tab:orange", label="sim")
    ax.set_xlabel("hits per shot per port")
    ax.set_ylabel("count")
    ax.set_title("Per-port hit-count distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preproc_path", type=str, default=_default_paths().preproc)
    ap.add_argument("--river_path",   type=str, default=_default_paths().river)
    ap.add_argument("--calib_path",   type=str, default=_default_paths().calib)
    ap.add_argument("--sim_glob", type=str,
                    default="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/"
                            "miaed_mnis_data/streak_finder_realistic_sanity/"
                            "train/*.h5",
                    help="Glob of sim h5 shards to compare against.")
    ap.add_argument("--n_real", type=int, default=2000)
    ap.add_argument("--n_sim",  type=int, default=2000)
    ap.add_argument("--kamp_min", type=float, default=1.5,
                    help="Restrict both sides to shots with kamp >= this "
                         "(eV). Real shots come from duck_kamp, sim shots "
                         "from grp.attrs['kickstrength'].")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="figures/real_vs_sim")
    ap.add_argument("--sim_only", action="store_true",
                    help="Skip real-side loading. Useful when the preproc "
                         "file isn't reachable and you only want sim stats.")
    return ap.parse_args()


def _scalar_summary(name: str, x: np.ndarray, kamp: np.ndarray,
                    gap_hist: np.ndarray, gap_edges: np.ndarray,
                    hit_hist: np.ndarray, bandwidth: np.ndarray) -> dict:
    per_port_totals = x.sum(axis=2)   # (n, ports)
    return {
        "name": name,
        "n_shots": int(x.shape[0]),
        "n_ports": int(x.shape[1]),
        "kamp_median": float(np.median(kamp)) if kamp.size else None,
        "kamp_p90":    float(np.percentile(kamp, 90)) if kamp.size else None,
        "hits_per_port_median": float(np.median(per_port_totals)),
        "hits_per_port_p90":    float(np.percentile(per_port_totals, 90)),
        "gap_hist_edges": gap_edges.astype(float).tolist(),
        "gap_hist_all_ports_summed": gap_hist.sum(axis=0).astype(int).tolist(),
        # A single scalar for the fit loop: mean gap length weighted by
        # the histogram. Real ~ big (dead-time), sim ~ small.
        "gap_mean_length": float(
            (gap_hist.sum(axis=0)
             * (0.5 * (gap_edges[:-1] + gap_edges[1:]))).sum()
            / max(1.0, float(gap_hist.sum()))
        ),
        # For band width, report the mean FWHM proxy inside the ROI.
        "bandwidth_roi_mean_fwhm": float(
            bandwidth[0, ROI_BIN_LO:ROI_BIN_HI + 1].mean()
        ),
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[cfg] outdir={args.outdir}")
    print(f"[cfg] n_real={args.n_real} n_sim={args.n_sim} kamp_min={args.kamp_min}")
    print(f"[cfg] ROI bins: [{ROI_BIN_LO}, {ROI_BIN_HI}] "
          f"(KE [{P_ROI_KE_DET_LO:.2f}, {P_ROI_KE_DET_HI:.2f}] eV)")

    gap_edges = np.arange(1, 41)   # gap lengths 1..40 bins
    hit_edges = np.linspace(0, 80, 41)

    real_summary = None
    if not args.sim_only:
        x_real, kamp_real = load_real_ximgs(
            RealPaths(args.preproc_path, args.river_path, args.calib_path),
            args.n_real, args.kamp_min, args.seed,
        )
        bw_real = bandwidth_vs_energy(x_real)
        gh_real = gap_length_hist(x_real, gap_edges, ROI_BIN_LO, ROI_BIN_HI)
        hh_real = hitcount_hist(x_real, hit_edges)

        np.savez_compressed(
            os.path.join(args.outdir, "metrics_real.npz"),
            bandwidth=bw_real, gap_hist=gh_real, hit_hist=hh_real,
            gap_edges=gap_edges, hit_edges=hit_edges,
            kamp=kamp_real,
        )
        real_summary = _scalar_summary(
            "real", x_real, kamp_real, gh_real, gap_edges, hh_real, bw_real,
        )
        print(f"[real] scalars: hits/port median="
              f"{real_summary['hits_per_port_median']:.1f}  "
              f"gap_mean={real_summary['gap_mean_length']:.2f}  "
              f"bw_roi_fwhm={real_summary['bandwidth_roi_mean_fwhm']:.2f}")
    else:
        bw_real = gh_real = hh_real = None

    x_sim, kamp_sim = load_sim_ximgs(
        args.sim_glob, args.n_sim, args.kamp_min, args.seed,
    )
    bw_sim = bandwidth_vs_energy(x_sim)
    gh_sim = gap_length_hist(x_sim, gap_edges, ROI_BIN_LO, ROI_BIN_HI)
    hh_sim = hitcount_hist(x_sim, hit_edges)

    np.savez_compressed(
        os.path.join(args.outdir, "metrics_sim.npz"),
        bandwidth=bw_sim, gap_hist=gh_sim, hit_hist=hh_sim,
        gap_edges=gap_edges, hit_edges=hit_edges,
        kamp=kamp_sim,
    )
    sim_summary = _scalar_summary(
        "sim", x_sim, kamp_sim, gh_sim, gap_edges, hh_sim, bw_sim,
    )
    print(f"[sim ] scalars: hits/port median="
          f"{sim_summary['hits_per_port_median']:.1f}  "
          f"gap_mean={sim_summary['gap_mean_length']:.2f}  "
          f"bw_roi_fwhm={sim_summary['bandwidth_roi_mean_fwhm']:.2f}")

    summaries = {"sim": sim_summary}
    if real_summary is not None:
        summaries["real"] = real_summary
    with open(os.path.join(args.outdir, "metrics_scalars.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    if bw_real is not None:
        _plot_bandwidth(bw_real, bw_sim,
                        os.path.join(args.outdir, "metrics_bandwidth.png"))
        _plot_gap_hist(gh_real, gh_sim, gap_edges,
                       os.path.join(args.outdir, "metrics_gaps.png"))
        _plot_hit_hist(hh_real, hh_sim, hit_edges,
                       os.path.join(args.outdir, "metrics_hits.png"))
        print(f"wrote plots + npz + scalars to {args.outdir}")
    else:
        print(f"wrote sim npz + scalars to {args.outdir} (skipped real-side plots)")


if __name__ == "__main__":
    main()
