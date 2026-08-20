#!/usr/bin/env python3
"""Compare kangle-aligned averages of real duck/goose shots to noise-free
sim clean-truth (ymat) at several `kickstrength` eV values.

Goal: pick where to draw the streaked/unstreaked boundary for training
and choose sim kickstrength values that visually match the real duck
class.

Approach:

  * Real side: rebuild every duck and every goose shot via the notebook's
    preproc + calibration pipeline. For each shot, roll the port axis so
    the shot's kangle lands at port 0 -- otherwise the arcs at different
    kangles cancel when averaged. Sum aligned shots per class, divide by
    N, and plot the mean (16, NBINS) map.
  * Sim side: use build_XY's clean 2D PDF (`ymat`), NOT the drawn hit
    histogram. That's noise-free -- pure `saseamps * pol * cossq(...,
    c + kickstrength * cos(...))`. One pulse, kangle=0 (so the sim arc's
    reference orientation matches the aligned-real reference).

Writes: streak_finder/figures/kamp_vs_kickstrength_ev.png
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from real_vs_sim_metrics import (  # noqa: E402
    HSD_CHANNELS, V_RET, NBINS, EMIN, EMAX, EBIN_EDGES, TOF_KE_GATE_MARGIN,
    _default_paths, _match_duck_rows, _load_alpha_t0, _tof_gate, _tof_to_ke,
)

# Reuse sim primitives directly (bypass sample_shot's noise stages).
sys.path.insert(0, "/sdf/home/m/miaed/CookieSimSlim_src")
from simulation import Params, build_XY  # noqa: E402


# ---------------------------------------------------------------------------
# Real side
# ---------------------------------------------------------------------------
def _load_river(river_path):
    with h5py.File(river_path, "r") as rf:
        return {
            "duck_ts":   rf["duck_timestamps"][:],
            "duck_kamp": rf["duck_kamp"][:],
            "duck_kang": rf["duck_kangle"][:],
            "goose_ts":   rf["goose_timestamps"][:],
            "goose_kamp": rf["goose_kamp"][:],
            "goose_kang": rf["goose_kangle"][:],
        }


def _kangle_to_port_roll(kangle_rad: float, n_ports: int) -> int:
    """Ports sit at even spacing round the ring. Rolling the port axis by
    -round(kangle / dtheta) sends this shot's kangle to port index 0.
    """
    dtheta = 2.0 * np.pi / n_ports
    return int(np.round(kangle_rad / dtheta)) % n_ports


def _build_aligned_mean_for_class(paths, class_name, alpha_at, t0_at,
                                  tof_gate, kamp_cut=None):
    """Load every shot of the class (optionally filtered by kamp_cut), build
    its (16, NBINS) histogram, roll ports so kangle -> port 0, and return
    the per-shot mean image plus the (kamp, kangle) arrays used.
    """
    river = _load_river(paths.river)
    with h5py.File(paths.preproc, "r") as pf:
        preproc_ts = pf["timestamp"][:]

    ts   = river[f"{class_name}_ts"]
    kamp = river[f"{class_name}_kamp"]
    kang = river[f"{class_name}_kang"]

    rows, river_idx = _match_duck_rows(preproc_ts, ts)
    k = kamp[river_idx]
    a = kang[river_idx]
    m = np.isfinite(k) & np.isfinite(a)
    rows = rows[m]; k = k[m]; a = a[m]

    if kamp_cut is not None:
        if class_name == "duck":
            keep = k >= kamp_cut
        else:
            keep = k <= kamp_cut
        rows = rows[keep]; k = k[keep]; a = a[keep]

    print(f"[real:{class_name}] using {rows.size} shots  "
          f"kamp range [{k.min():.6f}, {k.max():.6f}]  "
          f"kangle range [{np.degrees(a.min()):.1f}, {np.degrees(a.max()):.1f}] deg")

    n_ports = len(HSD_CHANNELS)
    acc = np.zeros((n_ports, NBINS), dtype=np.float64)

    with h5py.File(paths.preproc, "r") as pf:
        # Preload flat + starts/ends per port once; O(1) per-shot slice.
        starts_by_port = {}
        ends_by_port = {}
        flat_by_port = {}
        for ang in HSD_CHANNELS:
            lens = pf[f"var_hsd_hf_times_{ang}_len"][:]
            flat_by_port[ang] = pf[f"var_hsd_hf_times_{ang}"][:]
            cumsum = np.cumsum(lens)
            starts_by_port[ang] = cumsum - lens
            ends_by_port[ang] = cumsum

        for shot_i, row in enumerate(rows):
            shot_img = np.zeros((n_ports, NBINS), dtype=np.float64)
            for i, ang in enumerate(HSD_CHANNELS):
                start = int(starts_by_port[ang][row])
                end = int(ends_by_port[ang][row])
                if end <= start:
                    continue
                hits = flat_by_port[ang][start:end]
                t_lo, t_hi = tof_gate[ang]
                good = (hits > t_lo) & (hits < t_hi)
                if not good.any():
                    continue
                ke = _tof_to_ke(hits[good], alpha_at[ang], t0_at[ang])
                shot_img[i], _ = np.histogram(ke, bins=EBIN_EDGES)

            roll = _kangle_to_port_roll(a[shot_i], n_ports)
            acc += np.roll(shot_img, -roll, axis=0)

            if (shot_i + 1) % 500 == 0:
                print(f"  [{class_name}] processed {shot_i + 1}/{rows.size}")

    mean_img = acc / max(rows.size, 1)
    return mean_img, k, a


# ---------------------------------------------------------------------------
# Sim clean-truth side
# ---------------------------------------------------------------------------
def _sim_clean_ymat(kickstrength_ev, nangles=16, nenergies=NBINS,
                    e_center_bin=170.0, sase_width_bins=13.5):
    """Return build_XY's noise-free 2D PDF (`ymat`) for a single pulse at
    kangle=0 (so the streak arc's reference lines up with the aligned-real
    port axis) at the requested kickstrength.

    Uses the realistic-generator's default e_center / sase_width knobs so
    the arc lives in the same KE region as the real duck concentration.
    """
    p = Params(".", "streak_ev_plot", 1)
    p.setnangles(nangles)
    p.setnenergies(nenergies)
    p.setdrawscale(0.0)          # we don't need drawn hits
    p.setdarkscale(0.0)          # no background in ymat anyway
    p.setsecondaryscale(0.0)
    p.setcentralenergy(e_center_bin)
    p.centralenergyvar = 0.0
    p.setcentralenergywidth(sase_width_bins)
    p.setsasescale(1)
    p.setsasewidth(sase_width_bins)
    p.setkickstrength(kickstrength_ev)
    p.setkickstrengthvar(0.0)
    p.setfixedlinear()
    if kickstrength_ev > 0.0:
        p.setstreaking()
    else:
        p.setspectroscopy()

    # Deterministic single-pulse setup: kangle=0, e_center exact, amp=1,
    # linear polarization with strength 1.0 pointing along a=0.
    class _FixedRng:
        """Minimal shim: build_XY only calls poisson (ncenters) and
        normal/random for jitter. Force ncenters=1 and zero jitter."""
        def __init__(self):
            self._seq = np.random.default_rng(0)

        def poisson(self, _lam):
            return 1

        def normal(self, loc=0.0, scale=0.0, size=None):
            # For sasecenters (1 draw) and kickstrength (1 draw) build_XY
            # calls normal(mean, var). scale is zero so this is a no-op
            # for kick; for centers we pin to loc so the pulse lands
            # exactly at e_center_bin.
            if size is None:
                return float(loc)
            return np.full(size, loc, dtype=float)

        def random(self, size=None):
            # build_XY uses this to sample phases in [0, 1); we override
            # phases downstream, so 0 is fine.
            if size is None:
                return 0.0
            return np.zeros(size, dtype=float)

    p.rng = _FixedRng()

    # Force pulse phase to 0 (streak arc peaks at port 0). setphases is
    # called inside build_XY; monkey-patch it to a no-op after we set the
    # value we want.
    _orig_setphases = p.setphases

    def _forced_setphases(_phases, _orig=_orig_setphases):
        _orig([0.0])
    p.setphases = _forced_setphases

    _hits, ymat = build_XY(p)
    return ymat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    outdir = os.path.join(_HERE, "figures")
    os.makedirs(outdir, exist_ok=True)

    paths = _default_paths()
    print(f"[real] river:   {paths.river}")
    print(f"[real] preproc: {paths.preproc}")

    alpha_at, t0_at = _load_alpha_t0(paths.calib, V_RET)
    tof_gate = {
        ang: _tof_gate(alpha_at[ang], t0_at[ang],
                       EMIN - TOF_KE_GATE_MARGIN, EMAX + TOF_KE_GATE_MARGIN)
        for ang in HSD_CHANNELS
    }

    # kamp cut: keep the top-1000 duck and bottom-1000 goose we already
    # know from the earlier river inspection. duck kamp >= 0.0228 gets
    # you exactly ~1000 shots; goose <= 0.00244 same.
    duck_mean, duck_kamp, duck_kang = _build_aligned_mean_for_class(
        paths, "duck", alpha_at, t0_at, tof_gate, kamp_cut=0.0228,
    )
    goose_mean, goose_kamp, goose_kang = _build_aligned_mean_for_class(
        paths, "goose", alpha_at, t0_at, tof_gate, kamp_cut=0.00244,
    )

    # Sim clean sweep. Include 0 so the "no streak" reference sits next to
    # the goose mean. Widen the low end to catch sub-eV kicks; extend past
    # the current training threshold of 5 eV so we can see if that's over-
    # or under-kicked relative to the real duck arc.
    sim_evs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5]
    sim_imgs = []
    for kev in sim_evs:
        ymat = _sim_clean_ymat(kev)
        sim_imgs.append(ymat)
        print(f"[sim ] kickstrength={kev:.2f} eV  ymat.max={ymat.max():.4g}  "
              f"peak_bin={int(np.argmax(ymat.sum(axis=0)))}")

    # ------- plotting -------
    # Zoom into the KE signal band so the streak arc dominates the panel.
    zoom_lo, zoom_hi = 100, 320
    ke_lo = EMIN + zoom_lo * (EMAX - EMIN) / NBINS
    ke_hi = EMIN + zoom_hi * (EMAX - EMIN) / NBINS

    n_panels = 2 + len(sim_evs)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.8 * nrows),
                             constrained_layout=True)
    axes_flat = np.asarray(axes).ravel()

    def _show(ax, img, title, is_sim):
        zoom = img[:, zoom_lo:zoom_hi]
        # Per-panel p99 vmax: real means (per-shot averaged hit count) and
        # sim clean ymat (density-like) live on totally different scales.
        vmax = float(np.percentile(zoom, 99))
        if vmax <= 0.0:
            vmax = float(zoom.max()) or 1.0
        pm = ax.imshow(zoom, aspect="auto", origin="lower",
                       vmin=0.0, vmax=vmax,
                       extent=(ke_lo, ke_hi, 0, zoom.shape[0]),
                       cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("KE (eV)", fontsize=9)
        ax.set_ylabel("port idx (kangle-aligned)" if not is_sim else "port idx",
                      fontsize=9)
        ax.tick_params(labelsize=8)
        return pm

    _show(axes_flat[0], goose_mean,
          (f"real goose mean  (N={goose_kamp.size})\n"
           f"kamp <= 0.00244  (bottom-~1000)"),
          is_sim=False)
    _show(axes_flat[1], duck_mean,
          (f"real duck mean  (N={duck_kamp.size})\n"
           f"kamp >= 0.0228  (top-~1000)"),
          is_sim=False)
    for i, (kev, img) in enumerate(zip(sim_evs, sim_imgs)):
        _show(axes_flat[2 + i], img,
              f"sim clean ymat  kickstrength = {kev:.2f} eV\n(1 pulse, kang=0)",
              is_sim=True)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Real duck/goose (kangle-aligned averages) vs sim noise-free ymat "
        "at a kickstrength sweep — pick training boundary from the visual "
        "match to real duck",
        fontsize=12,
    )

    outpath = os.path.join(outdir, "kamp_vs_kickstrength_ev.png")
    fig.savefig(outpath, dpi=140)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
