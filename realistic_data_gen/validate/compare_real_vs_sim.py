#!/usr/bin/env python3
"""Side-by-side of the top-K real duck shots (by kamp) vs the sim dev
shots we just generated. Same 16 x 512 pcolormesh, same magma_r, same
vmax so streak visibility is directly comparable.

Also computes a simple sharpness metric per shot: for each port, take the
argmax bin (in the KE window that contains the streak), then compute the
FWHM of the intensity around it. Report the median FWHM across the top-K
shots for real vs sim.
"""
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

HSD_CHANNELS = [0, 22, 45, 67, 90, 112, 135, 157,
                180, 202, 225, 247, 270, 292, 315, 337]

PREPROC_PATH = '/sdf/data/lcls/ds/tmo/tmol1043723/scratch/preproc/vjtw/run99_vjtw.h5'
RIVER_PATH   = '/sdf/data/lcls/ds/tmo/tmol1043723/results/streaking_results/run[99]_vjtw_river.h5'
CALIB_PATH   = '/sdf/home/m/miaed/copied_ipynbs/tmo_utils/calibration/mrco_calib_tmox101_205-213_simon_t0.h5'
V_RET  = 280.0
NBINS  = 512
ESTEP  = 0.1
E_AU   = 27.2114
P_ROI  = (4.65, 4.85)
# Anchor the RIGHT edge at the old 0.25 eV/bin bin-320 KE — real hits die
# off past that point. Keeps the full ROI in view and preserves the left
# tail. Mirror of MRCO_Streaking_rawData.ipynb config cell.
EMAX_ANCHOR = 43.11


def _ke_grid():
    ke_ip_lo = P_ROI[0] ** 2 / 2 * E_AU
    ke_ip_hi = P_ROI[1] ** 2 / 2 * E_AU
    ke_det_lo = ke_ip_lo - V_RET
    ke_det_hi = ke_ip_hi - V_RET
    emax = EMAX_ANCHOR
    emin = emax - NBINS * ESTEP
    edges = np.linspace(emin, emax, NBINS + 1)
    return edges, ke_det_lo, ke_det_hi


def _load_calib():
    with h5py.File(CALIB_PATH, 'r') as hf:
        alphas = np.array(hf['alphas'])
        vr = np.array(hf['v_r'])
        t0s = np.array(hf['t0s'])
        angs = np.array(hf['angles'])
    alpha_by_ang, t0_by_ang = {}, {}
    for i, ang in enumerate(angs):
        row = alphas[i].astype(float)
        if vr.min() <= V_RET <= vr.max():
            val = float(interp1d(vr, row, kind='linear')(V_RET))
        else:
            poly = np.polyfit(vr.astype(float), row, deg=2)
            val = float(np.polyval(poly, V_RET))
        alpha_by_ang[int(ang)] = val
        t0_by_ang[int(ang)] = float(t0s[i])
    return alpha_by_ang, t0_by_ang


def _tof_to_ke(tof_s, alpha, t0):
    dt = tof_s - t0
    ke = np.zeros_like(tof_s, dtype=np.float64)
    good = dt > 0
    ke[good] = alpha / (dt[good] ** 2)
    return ke


def _load_river_kamps():
    with h5py.File(RIVER_PATH, 'r') as rf:
        ts = rf['duck_timestamps'][:]
        kamp = rf['duck_kamp'][:]
    return ts, kamp


def _load_preproc(top_ts):
    with h5py.File(PREPROC_PATH, 'r') as pf:
        preproc_ts = pf['timestamp'][:]
    sort_idx = np.argsort(preproc_ts)
    sorted_ts = preproc_ts[sort_idx]
    pos = np.searchsorted(sorted_ts, top_ts)
    hit = (pos < sorted_ts.size) & (sorted_ts[np.clip(pos, 0, sorted_ts.size - 1)] == top_ts)
    rows = sort_idx[pos[hit]]
    river_idx = np.flatnonzero(hit)
    order = np.argsort(rows)
    return rows[order], river_idx[order]


def _load_hsd_for_rows(rows):
    with h5py.File(PREPROC_PATH, 'r') as pf:
        out = {}
        for ch in HSD_CHANNELS:
            lens = pf[f'var_hsd_hf_times_{ch}_len'][:]
            flat = pf[f'var_hsd_hf_times_{ch}'][:]
            ends = np.cumsum(lens); starts = ends - lens
            out[ch] = [flat[starts[r]:ends[r]] for r in rows]
    return out


def _real_ke_image(hsd_by_ang, shot_idx, alpha_by_ang, t0_by_ang, edges):
    img = np.zeros((len(HSD_CHANNELS), NBINS), dtype=np.float32)
    for i, ang in enumerate(HSD_CHANNELS):
        hits = hsd_by_ang[ang][shot_idx]
        if hits.size == 0:
            continue
        ke = _tof_to_ke(hits, alpha_by_ang[ang], t0_by_ang[ang])
        img[i], _ = np.histogram(ke, bins=edges)
    return img


def _sharpness(img, band_lo, band_hi):
    """Median per-port FWHM (in bins) inside a window."""
    n_ports, n_bins = img.shape
    fwhms = []
    win = img[:, band_lo:band_hi]
    for p in range(n_ports):
        row = win[p]
        if row.max() <= 0:
            continue
        peak_bin = int(np.argmax(row))
        half = row.max() / 2.0
        left = peak_bin
        while left > 0 and row[left] >= half:
            left -= 1
        right = peak_bin
        while right < row.size - 1 and row[right] >= half:
            right += 1
        fw = right - left
        if fw > 0:
            fwhms.append(fw)
    if not fwhms:
        return np.nan
    return float(np.median(fwhms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_h5", default=os.path.join(os.path.dirname(__file__),
                                                     "streak_realistic_dev_00000.h5"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "real_vs_sim.png"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--vmax", type=float, default=2.0)
    args = ap.parse_args()

    # Real side
    print("loading river + preproc + calib ...")
    duck_ts, duck_kamp = _load_river_kamps()
    top_river = np.argsort(duck_kamp)[::-1][:args.n * 3]
    top_ts = duck_ts[top_river]

    rows, river_idx = _load_preproc(top_ts)
    matched_kamp = duck_kamp[top_river[river_idx]] if river_idx.size else np.array([])
    order = np.argsort(matched_kamp)[::-1]
    rows = rows[order][:args.n]
    matched_kamp = matched_kamp[order][:args.n]

    hsd_by_ang = _load_hsd_for_rows(rows)
    alpha_by_ang, t0_by_ang = _load_calib()
    edges, ke_det_lo, ke_det_hi = _ke_grid()
    band_lo = int(np.searchsorted(edges, ke_det_lo))
    band_hi = int(np.searchsorted(edges, ke_det_hi))

    real_imgs = [_real_ke_image(hsd_by_ang, i, alpha_by_ang, t0_by_ang, edges)
                 for i in range(len(rows))]

    # Sim side
    print(f"loading sim shots from {args.sim_h5}")
    with h5py.File(args.sim_h5, 'r') as f:
        shot_keys = sorted(k for k in f.keys() if k.startswith('shot_'))
        sim_shots = []
        for k in shot_keys:
            g = f[k]
            sim_shots.append({
                "key": k,
                "Ximg": np.asarray(g['Ximg']),
                "kamp": float(g.attrs['kickstrength']),
                "kang": float(g.attrs['kickangle']),
                "npulses": int(g.attrs['npulses']),
                "sw": float(g.attrs['sasewidth']),
            })
    sim_shots = sorted(sim_shots, key=lambda s: -s["kamp"])[:args.n]

    # ---- Sharpness ----
    # Real streak sits in the P_ROI KE window ([ke_det_lo, ke_det_hi]).
    real_fwhm = [_sharpness(img, band_lo, band_hi) for img in real_imgs]
    # Sim streak: center bin ~ centralenergy in 0.25 eV bins. Sim's
    # centralenergy is drawn in [200, 210] eV, so the streak lands within
    # roughly bins [800, 840] on a 0.25 eV/bin axis... but the sim generator
    # writes the raw bin index; the center is around bin 200 in practice
    # (sim uses eV as the bin unit here in its 512-bin axis for the dev
    # stream). Look inside a broad window around the max column instead.
    def _sim_band(img):
        col_sum = img.sum(axis=0)
        peak = int(np.argmax(col_sum))
        return max(0, peak - 40), min(img.shape[1], peak + 40)
    sim_fwhm = []
    for s in sim_shots:
        lo, hi = _sim_band(s['Ximg'])
        sim_fwhm.append(_sharpness(s['Ximg'], lo, hi))

    def _fmt(vals):
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return "nan"
        return f"median={np.median(vals):.2f} bins  n={len(vals)}"
    print(f"streak per-port FWHM (bins @ {ESTEP} eV/bin):")
    print(f"  real (top {args.n} by kamp):  {_fmt(real_fwhm)}")
    print(f"  sim  (top {args.n} by kamp):  {_fmt(sim_fwhm)}")

    # ---- Figure ----
    ncols = args.n
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.4),
                             sharex=False, sharey=True)
    for c, (img, k) in enumerate(zip(real_imgs, matched_kamp)):
        ax = axes[0, c]
        ax.pcolormesh(edges, np.arange(17) - 0.5, img,
                      cmap='magma_r', shading='flat',
                      vmin=0, vmax=args.vmax)
        ax.set_title(f"real kamp={k:.3g}\nFWHM={real_fwhm[c]:.1f}",
                     fontsize=8)
        ax.set_yticks(np.arange(16))
        ax.set_yticklabels([str(a) for a in HSD_CHANNELS], fontsize=5)
        # highlight the P_ROI band
        ax.axvline(ke_det_lo, color='cyan', lw=0.5, alpha=0.4)
        ax.axvline(ke_det_hi, color='cyan', lw=0.5, alpha=0.4)

    for c, s in enumerate(sim_shots):
        ax = axes[1, c]
        ax.pcolormesh(np.arange(s['Ximg'].shape[1] + 1) - 0.5,
                      np.arange(17) - 0.5, s['Ximg'],
                      cmap='magma_r', shading='flat',
                      vmin=0, vmax=args.vmax)
        ax.set_title(f"sim kamp={s['kamp']:.2f}\n"
                     f"n={s['npulses']} sw={s['sw']:.1f}  FWHM={sim_fwhm[c]:.1f}",
                     fontsize=8)
        ax.set_yticks(np.arange(16))
        ax.set_yticklabels([str(i) for i in range(16)], fontsize=5)

    for ax in axes[0, :]:
        ax.set_xlabel('KE at det (eV)', fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel('sim energy bin', fontsize=7)

    axes[0, 0].set_ylabel('port angle (deg)')
    axes[1, 0].set_ylabel('port idx')

    fig.suptitle(
        f"Real top-{ncols} by kamp (top row) vs sim dev (bottom row), sasewidth=5.0 eV.\n"
        f"Median per-port streak FWHM: "
        f"real={np.nanmedian(real_fwhm):.2f} bins, "
        f"sim={np.nanmedian(sim_fwhm):.2f} bins",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
