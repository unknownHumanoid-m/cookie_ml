#!/usr/bin/env python3
"""Measure streak sharpness in two ways:

1) Column-sum profile: sum(img, axis=0) inside the streak band. Then per
   shot: peak, FWHM (in bins), and 'shoulder ratio' = peak / mean-of-
   shoulders. Sharper stripe -> higher ratio, narrower FWHM.

2) Per-port hit spread: for each port whose row has any hits inside the
   band, compute the std of hit-bin positions (weighted). Ghost hits from
   loc_noise widen this. Median over ports/shots reports smear.

Compares real duck shots (top by kamp) vs the sim dev shots we just
generated. Renders overlaid mean profiles + FWHM/ratio histograms.
"""
import os
import argparse
import h5py
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

HSD_CHANNELS = [0, 22, 45, 67, 90, 112, 135, 157,
                180, 202, 225, 247, 270, 292, 315, 337]
PREPROC_PATH = '/sdf/data/lcls/ds/tmo/tmol1043723/scratch/preproc/vjtw/run99_vjtw.h5'
RIVER_PATH   = '/sdf/data/lcls/ds/tmo/tmol1043723/results/streaking_results/run[99]_vjtw_river.h5'
CALIB_PATH   = '/sdf/home/m/miaed/copied_ipynbs/tmo_utils/calibration/mrco_calib_tmox101_205-213_simon_t0.h5'
V_RET, NBINS, ESTEP, E_AU = 280.0, 512, 0.1, 27.2114
P_ROI = (4.65, 4.85)
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
    return np.linspace(emin, emax, NBINS + 1), ke_det_lo, ke_det_hi


def _load_calib():
    with h5py.File(CALIB_PATH, 'r') as hf:
        alphas = np.array(hf['alphas']); vr = np.array(hf['v_r'])
        t0s = np.array(hf['t0s']); angs = np.array(hf['angles'])
    a_by, t_by = {}, {}
    for i, ang in enumerate(angs):
        row = alphas[i].astype(float)
        if vr.min() <= V_RET <= vr.max():
            val = float(interp1d(vr, row, kind='linear')(V_RET))
        else:
            val = float(np.polyval(np.polyfit(vr.astype(float), row, deg=2), V_RET))
        a_by[int(ang)] = val
        t_by[int(ang)] = float(t0s[i])
    return a_by, t_by


def _tof_to_ke(tof_s, alpha, t0):
    dt = tof_s - t0
    out = np.zeros_like(tof_s, dtype=np.float64)
    m = dt > 0
    out[m] = alpha / (dt[m] ** 2)
    return out


def _colsum_metrics(img, band_lo, band_hi, half_win=40):
    """peak col in band, FWHM around peak, and shoulder ratio.

    Shoulder ratio = peak_val / mean(intensity at [-half_win..-half_win/2]
                                     & [half_win/2..half_win]).
    Higher = sharper stripe.
    """
    lo = max(0, band_lo); hi = min(img.shape[1], band_hi)
    win_cols = img[:, lo:hi].sum(axis=0)
    peak_local = int(np.argmax(win_cols))
    peak_bin = lo + peak_local
    n = img.shape[1]
    a = max(0, peak_bin - half_win); b = min(n, peak_bin + half_win + 1)
    strip = img[:, a:b].sum(axis=0).astype(float)
    x_peak = peak_bin - a
    peak = strip[x_peak] if 0 <= x_peak < strip.size else 0.0
    if peak <= 0:
        return dict(peak=0.0, fwhm=np.nan, ratio=np.nan, strip=strip, x_peak=x_peak)
    half = peak / 2.0
    left = x_peak
    while left > 0 and strip[left] >= half:
        left -= 1
    right = x_peak
    while right < strip.size - 1 and strip[right] >= half:
        right += 1
    fwhm = right - left
    q = half_win // 2
    shoulder_slices = np.concatenate([
        strip[max(0, x_peak - half_win):max(0, x_peak - q)],
        strip[min(strip.size, x_peak + q + 1):min(strip.size, x_peak + half_win + 1)],
    ])
    shoulder = float(shoulder_slices.mean()) if shoulder_slices.size else 0.0
    ratio = peak / max(shoulder, 1e-6)
    return dict(peak=float(peak), fwhm=float(fwhm), ratio=float(ratio),
                strip=strip, x_peak=int(x_peak))


def _per_port_spread(img, band_lo, band_hi):
    """Weighted std of hit-bin position per port, inside the streak band.
    Returns list of (port_idx, std_bins). Ports with < 2 hits are skipped."""
    lo = max(0, band_lo); hi = min(img.shape[1], band_hi)
    out = []
    for p in range(img.shape[0]):
        row = img[p, lo:hi].astype(float)
        tot = row.sum()
        if tot < 2:
            continue
        x = np.arange(row.size)
        mean = (x * row).sum() / tot
        var = (row * (x - mean) ** 2).sum() / tot
        out.append((p, float(np.sqrt(var))))
    return out


def _load_real_shots(n):
    with h5py.File(RIVER_PATH, 'r') as rf:
        ts = rf['duck_timestamps'][:]; kamp = rf['duck_kamp'][:]
    top = np.argsort(kamp)[::-1][:n * 3]
    top_ts = ts[top]
    with h5py.File(PREPROC_PATH, 'r') as pf:
        pts = pf['timestamp'][:]
    sidx = np.argsort(pts); sts = pts[sidx]
    pos = np.searchsorted(sts, top_ts)
    hit = (pos < sts.size) & (sts[np.clip(pos, 0, sts.size - 1)] == top_ts)
    rows = sidx[pos[hit]]; ridx = np.flatnonzero(hit)
    order = np.argsort(rows)
    rows = rows[order][:n]; matched = kamp[top[ridx[order]]][:n]
    with h5py.File(PREPROC_PATH, 'r') as pf:
        hsd = {}
        for ch in HSD_CHANNELS:
            lens = pf[f'var_hsd_hf_times_{ch}_len'][:]
            flat = pf[f'var_hsd_hf_times_{ch}'][:]
            ends = np.cumsum(lens); starts = ends - lens
            hsd[ch] = [flat[starts[r]:ends[r]] for r in rows]
    return rows, matched, hsd


def _real_img(hsd, i, alpha_by, t0_by, edges):
    img = np.zeros((16, NBINS), dtype=np.float32)
    for ii, ang in enumerate(HSD_CHANNELS):
        h = hsd[ang][i]
        if h.size == 0:
            continue
        ke = _tof_to_ke(h, alpha_by[ang], t0_by[ang])
        img[ii], _ = np.histogram(ke, bins=edges)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_h5", default=os.path.join(os.path.dirname(__file__),
                                                     "streak_realistic_dev_00000.h5"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--half_win", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "sharpness_profile.png"))
    args = ap.parse_args()

    edges, ke_lo, ke_hi = _ke_grid()
    band_real = (int(np.searchsorted(edges, ke_lo)),
                 int(np.searchsorted(edges, ke_hi)))

    print("loading real top-kamp shots ...")
    rows, kamps, hsd = _load_real_shots(args.n)
    alpha_by, t0_by = _load_calib()
    real_imgs = [_real_img(hsd, i, alpha_by, t0_by, edges)
                 for i in range(len(rows))]
    real_metrics = [_colsum_metrics(img, *band_real, args.half_win)
                    for img in real_imgs]

    print("loading sim shots ...")
    with h5py.File(args.sim_h5, 'r') as f:
        keys = sorted(k for k in f.keys() if k.startswith('shot_'))
        sim_imgs = [np.asarray(f[k]['Ximg']) for k in keys]
    # Sim: band centered on the argmax column-sum, +/- 40 bins wide
    sim_metrics = []
    for img in sim_imgs:
        col = img.sum(axis=0)
        peak_bin = int(np.argmax(col))
        sim_metrics.append(_colsum_metrics(
            img, max(0, peak_bin - 40), min(img.shape[1], peak_bin + 40),
            args.half_win))

    # Per-port spread (only over ports that have hits in the band)
    real_spreads = []
    for img in real_imgs:
        for _, s in _per_port_spread(img, *band_real):
            real_spreads.append(s)
    sim_spreads = []
    for img in sim_imgs:
        col = img.sum(axis=0)
        peak_bin = int(np.argmax(col))
        for _, s in _per_port_spread(img, max(0, peak_bin - 40),
                                     min(img.shape[1], peak_bin + 40)):
            sim_spreads.append(s)
    real_spreads = np.asarray(real_spreads)
    sim_spreads = np.asarray(sim_spreads)

    def _fwhm_arr(mm):
        return np.asarray([m['fwhm'] for m in mm], dtype=float)
    def _ratio_arr(mm):
        return np.asarray([m['ratio'] for m in mm], dtype=float)

    r_fwhm = _fwhm_arr(real_metrics); s_fwhm = _fwhm_arr(sim_metrics)
    r_rat  = _ratio_arr(real_metrics); s_rat = _ratio_arr(sim_metrics)

    def fmt(v):
        v = v[np.isfinite(v)]
        return (f"median={np.median(v):.2f}  mean={v.mean():.2f}  "
                f"p10={np.percentile(v,10):.2f}  p90={np.percentile(v,90):.2f}"
                if v.size else "empty")
    print(f"---- column-sum (port-integrated) streak stripe ----")
    print(f"REAL FWHM (bins @ {ESTEP} eV/bin):  {fmt(r_fwhm)}")
    print(f" SIM FWHM (bins @ {ESTEP} eV/bin):  {fmt(s_fwhm)}")
    print(f"REAL peak/shoulder ratio:         {fmt(r_rat)}")
    print(f" SIM peak/shoulder ratio:         {fmt(s_rat)}")
    print(f"---- per-port hit spread (std of bin positions) ----")
    print(f"REAL per-port spread (bins):      {fmt(real_spreads)}   "
          f"n_ports={real_spreads.size}")
    print(f" SIM per-port spread (bins):      {fmt(sim_spreads)}   "
          f"n_ports={sim_spreads.size}")

    # ---- Figure ----
    # Overlay mean column-sum strips (peak-aligned), FWHM hist, ratio hist,
    # per-port spread hist.
    strip_len = 2 * args.half_win + 1
    def align(mm):
        out = np.zeros(strip_len, dtype=np.float64)
        n = 0
        for m in mm:
            strip = m['strip']; xp = m['x_peak']
            if strip.max() <= 0:
                continue
            # place strip so its peak lands at half_win
            centered = np.zeros(strip_len)
            for k in range(strip.size):
                dst = k - xp + args.half_win
                if 0 <= dst < strip_len:
                    centered[dst] += strip[k]
            centered = centered / centered.max()
            out += centered
            n += 1
        return out / max(n, 1)

    r_align = align(real_metrics); s_align = align(sim_metrics)
    x = np.arange(-args.half_win, args.half_win + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(x, r_align, label=f'real  (FWHM_med={np.nanmedian(r_fwhm):.1f})')
    axs[0, 0].plot(x, s_align, label=f'sim   (FWHM_med={np.nanmedian(s_fwhm):.1f})')
    axs[0, 0].axhline(0.5, color='k', lw=0.4, alpha=0.4)
    axs[0, 0].set_xlabel(f'bin offset from stripe peak ({ESTEP} eV/bin)')
    axs[0, 0].set_ylabel('shot-averaged, peak-normalized')
    axs[0, 0].set_title('Column-sum stripe profile (port-integrated)')
    axs[0, 0].grid(alpha=0.3); axs[0, 0].legend()

    bins_fwhm = np.arange(0, 45, 2)
    axs[0, 1].hist(r_fwhm[np.isfinite(r_fwhm)], bins=bins_fwhm, alpha=0.5, label='real')
    axs[0, 1].hist(s_fwhm[np.isfinite(s_fwhm)], bins=bins_fwhm, alpha=0.5, label='sim')
    axs[0, 1].set_xlabel('stripe FWHM (bins)'); axs[0, 1].set_ylabel('shots')
    axs[0, 1].set_title('per-shot stripe FWHM'); axs[0, 1].grid(alpha=0.3); axs[0, 1].legend()

    bins_ratio = np.linspace(0, 20, 30)
    axs[1, 0].hist(r_rat[np.isfinite(r_rat)], bins=bins_ratio, alpha=0.5, label='real')
    axs[1, 0].hist(s_rat[np.isfinite(s_rat)], bins=bins_ratio, alpha=0.5, label='sim')
    axs[1, 0].set_xlabel('peak / shoulder ratio (higher = sharper)')
    axs[1, 0].set_ylabel('shots')
    axs[1, 0].set_title('per-shot stripe contrast')
    axs[1, 0].grid(alpha=0.3); axs[1, 0].legend()

    bins_spread = np.linspace(0, 20, 30)
    axs[1, 1].hist(real_spreads[np.isfinite(real_spreads)], bins=bins_spread, alpha=0.5, label='real')
    axs[1, 1].hist(sim_spreads[np.isfinite(sim_spreads)], bins=bins_spread, alpha=0.5, label='sim')
    axs[1, 1].set_xlabel('per-port hit-position std (bins)')
    axs[1, 1].set_ylabel('ports (pooled over shots)')
    axs[1, 1].set_title('per-port hit spread inside streak band')
    axs[1, 1].grid(alpha=0.3); axs[1, 1].legend()

    fig.suptitle('Streak sharpness: real vs sim', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
