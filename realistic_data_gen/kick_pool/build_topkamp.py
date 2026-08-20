#!/usr/bin/env python3
"""Build a sim-schema h5 shard from the top-N real duck-shot kamp shots.

Rebuilds the same per-shot Ximg (16, 512) that ``generate_streak_data_realistic.py``
writes, but sourced from the tmol1043723 run 99 duck shots via the notebook's
preproc + calibration pipeline. Emits per-shot groups with:

  * ``Ximg`` — (16, 512) uint16 histogram of hits over
    (port angle idx, detector-KE bin).
  * attrs used by ``streak_finder_eval.load_streak_h5``:
      - ``streak_amplitude`` (float, eV) — see --label_kamp below.
      - ``streak``           (0/1)       — thresholded copy of the above.
      - ``kickstrength``     (float)     — same as streak_amplitude (for the
                                           kick-bin plots).
      - ``kickangle``        (float, rad)— for downstream inspection.
      - ``duck_kamp_real``   (float)     — the actual raw duck_kamp for the
                                           shot (in real-data units), kept so
                                           the sim/real scale mismatch is
                                           auditable.
      - ``npulses``          (uint8) = 1  — placeholder; real pulse count
                                           isn't available for these shots
                                           and the streak finder doesn't
                                           consume it.
      - ``phases``           = [0.0]     — placeholder; same reasoning.
      - ``sasewidth``        = NaN       — real shots have no sase_width;
                                           eval treats NaN as "unknown".

Real ``duck_kamp`` is in a *different unit* than the sim ``kickstrength``
(max ~0.05 for real vs 0-3 eV for sim). If we copied duck_kamp verbatim
into ``streak_amplitude`` every real shot would end up labeled "unstreaked"
against the sim-trained threshold. The intended semantic here is "the top-N
kamp shots are the streaked shots we want to test on" — so we clamp
``streak_amplitude`` above the sim threshold (see --label_kamp), giving a
positive-only test set where the confusion matrix / TPR is directly
interpretable.

Usage:

    python3 build_real_topkamp_h5.py --top_n 25 \\
        --output /tmp/miaed_real_topkamp/topkamp_00000.h5

Runs in ~10 s (single file I/O + histogramming for 25 shots).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Sequence

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the notebook-derived constants + loaders from the metrics module.
from real_vs_sim_metrics import (  # noqa: E402
    HSD_CHANNELS,
    V_RET,
    NBINS,
    EMIN,
    EMAX,
    EBIN_EDGES,
    TOF_KE_GATE_MARGIN,
    _default_paths,
    _match_duck_rows,
    _load_alpha_t0,
    _tof_gate,
    _tof_to_ke,
    RealPaths,
)


def load_top_kamp_shots(paths: RealPaths, top_n: int):
    """Return (Ximg[N, 16, NBINS] uint16, kamp[N] float, kangle[N] float).

    Selects the ``top_n`` shots with the largest ``duck_kamp`` in the river
    file, then rebuilds each shot's (port, KE) histogram from the raw HSD
    hit times using the notebook's alpha(v_ret) + t0 calibration.
    """
    print(f"[real] loading river ({paths.river}) ...")
    with h5py.File(paths.river, "r") as rf:
        duck_ts    = rf["duck_timestamps"][:]
        duck_kamp  = rf["duck_kamp"][:]
        duck_kang  = rf["duck_kangle"][:]

    print(f"[real] loading preproc timestamps ({paths.preproc}) ...")
    with h5py.File(paths.preproc, "r") as pf:
        preproc_ts = pf["timestamp"][:]

    rows, river_idx = _match_duck_rows(preproc_ts, duck_ts)
    kamp   = duck_kamp[river_idx]
    kangle = duck_kang[river_idx]
    mask = np.isfinite(kamp) & np.isfinite(kangle)
    rows = rows[mask]
    kamp = kamp[mask]
    kangle = kangle[mask]

    take = min(int(top_n), rows.size)
    if take == 0:
        raise RuntimeError("no finite duck-kamp rows found")
    top = np.argsort(kamp)[::-1][:take]
    sel_rows_unsorted = rows[top]
    sel_kamp_unsorted = kamp[top]
    sel_kang_unsorted = kangle[top]

    # Reorder to ascending row index so h5py fancy-slicing works efficiently.
    order = np.argsort(sel_rows_unsorted)
    sel_rows = sel_rows_unsorted[order]
    sel_kamp = sel_kamp_unsorted[order]
    sel_kang = sel_kang_unsorted[order]

    print(f"[real] selected top-{take} kamp shots: "
          f"kamp range [{sel_kamp.min():.4f}, {sel_kamp.max():.4f}]")

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

    return x, sel_kamp.astype(np.float32), sel_kang.astype(np.float32)


def _write_real_group(grp, img, label_kamp, streak_threshold_eV,
                      kamp_real, kangle_real):
    grp.create_dataset("Ximg", data=img, dtype=np.uint16, compression=None)
    grp.attrs.create("streak_amplitude", np.float32(label_kamp))
    grp.attrs.create("streak",
                     np.uint8(1 if label_kamp >= streak_threshold_eV else 0))
    grp.attrs.create("kickstrength", np.float32(label_kamp))
    grp.attrs.create("kickangle", np.float32(kangle_real))
    grp.attrs.create("duck_kamp_real", np.float32(kamp_real))
    grp.attrs.create("npulses", np.uint8(1))
    grp.attrs.create("phases", np.array([0.0], dtype=np.float32))
    grp.attrs.create("sasewidth", np.float32(np.nan))
    grp.attrs.create("source", np.bytes_(b"real"))


def _write_sim_neg_group(grp, img, kamp_sim, kangle_sim,
                         streak_threshold_eV):
    grp.create_dataset("Ximg", data=img, dtype=np.uint16, compression=None)
    grp.attrs.create("streak_amplitude", np.float32(kamp_sim))
    grp.attrs.create("streak",
                     np.uint8(1 if kamp_sim >= streak_threshold_eV else 0))
    grp.attrs.create("kickstrength", np.float32(kamp_sim))
    grp.attrs.create("kickangle", np.float32(kangle_sim))
    grp.attrs.create("duck_kamp_real", np.float32(np.nan))
    grp.attrs.create("npulses", np.uint8(0))
    grp.attrs.create("phases", np.array([0.0], dtype=np.float32))
    grp.attrs.create("sasewidth", np.float32(np.nan))
    grp.attrs.create("source", np.bytes_(b"sim_unstreaked"))


def write_shard(x_real: np.ndarray,
                kamp_real: np.ndarray,
                kangle_real: np.ndarray,
                label_kamp: float,
                streak_threshold_eV: float,
                out_path: str,
                x_sim_neg: np.ndarray = None,
                kamp_sim_neg: np.ndarray = None,
                kangle_sim_neg: np.ndarray = None):
    """Write the sim-schema per-shot h5 the streak-finder eval expects.
    Real top-kamp shots are labeled streak=1 via the hard-set label_kamp.
    Optional sim negatives (kamp < threshold) are appended with their
    original attrs so the ROC/FPR calc has both classes.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    n_real = x_real.shape[0]
    n_neg = 0 if x_sim_neg is None else x_sim_neg.shape[0]
    with h5py.File(tmp, "w") as f:
        for k in range(n_real):
            _write_real_group(
                f.create_group("shot_%06d" % k),
                x_real[k], label_kamp, streak_threshold_eV,
                kamp_real[k], kangle_real[k],
            )
        for k in range(n_neg):
            _write_sim_neg_group(
                f.create_group("shot_%06d" % (n_real + k)),
                x_sim_neg[k], kamp_sim_neg[k], kangle_sim_neg[k],
                streak_threshold_eV,
            )
    os.replace(tmp, out_path)
    print(f"wrote {n_real} real + {n_neg} sim-negative shots to {out_path}")


def load_sim_unstreaked(sim_glob: str,
                        n: int,
                        streak_threshold_eV: float,
                        seed: int = 0):
    """Return (Ximg[n, 16, NBINS] uint16, kamp[n], kangle[n]) sampled from
    the sim h5 shards matching sim_glob, restricted to shots with
    kickstrength < streak_threshold_eV (i.e. unstreaked in the model's
    world). Used to inject negatives into the real-top-kamp eval so the
    ROC/FPR/threshold calc has both classes.
    """
    files = sorted(glob.glob(sim_glob))
    if not files:
        raise RuntimeError(f"no sim files match {sim_glob!r}")
    imgs, kamps, kangs = [], [], []
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    for path in files:
        with h5py.File(path, "r") as f:
            keys = sorted(k for k in f.keys() if k.startswith("shot_"))
            rng.shuffle(keys)
            for k in keys:
                grp = f[k]
                kk = float(np.asarray(grp.attrs.get("kickstrength", 0.0))
                           .reshape(-1)[0])
                if kk >= streak_threshold_eV:
                    continue
                imgs.append(np.asarray(grp["Ximg"][()], dtype=np.uint16))
                kamps.append(kk)
                kangs.append(
                    float(np.asarray(grp.attrs.get("kickangle", 0.0))
                          .reshape(-1)[0])
                )
                if len(imgs) >= n:
                    break
        if len(imgs) >= n:
            break
    if not imgs:
        raise RuntimeError(
            f"no unstreaked sim shots (kickstrength < {streak_threshold_eV}) "
            f"under {sim_glob!r}. Regenerate with --recipe natural or a "
            f"recipe that produces unstreaked examples."
        )
    return (np.stack(imgs, axis=0),
            np.asarray(kamps, dtype=np.float32),
            np.asarray(kangs, dtype=np.float32))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top_n", type=int, default=25,
                    help="Number of real duck shots to select by descending "
                         "duck_kamp.")
    ap.add_argument("--output", type=str, required=True,
                    help="Output h5 path. Parent dir is created if missing.")
    ap.add_argument("--label_kamp", type=float, default=2.5,
                    help="Value written into each shot's streak_amplitude / "
                         "kickstrength attrs. Real duck_kamp is on a "
                         "different scale than the sim kickstrength the "
                         "model was trained on, so we hard-set the label "
                         "kamp above the checkpoint threshold (default sim "
                         "threshold = 2.0 eV) to mark every shot as a "
                         "positive streak example.")
    ap.add_argument("--streak_threshold_eV", type=float, default=2.0,
                    help="Threshold applied when writing the convenience "
                         "'streak' uint8 attr. Should match the checkpoint "
                         "threshold.")
    # Negative-injection knobs. All-positive shards make the FPR-based
    # threshold degenerate; adding sim unstreaked shots pins the operating
    # threshold to something meaningful.
    ap.add_argument("--n_sim_neg", type=int, default=0,
                    help="If >0, also copy this many unstreaked shots from "
                         "--sim_neg_glob into the output shard. Their attrs "
                         "are preserved verbatim (kickstrength < threshold "
                         "= 0 label).")
    ap.add_argument("--sim_neg_glob", type=str,
                    default="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/"
                            "miaed_mnis_data/streak_finder_realistic/test/*.h5",
                    help="Glob for sim h5 shards to pull unstreaked "
                         "negatives from. Only used when --n_sim_neg > 0.")
    ap.add_argument("--sim_neg_seed", type=int, default=0)
    ap.add_argument("--preproc_path", type=str,
                    default=_default_paths().preproc)
    ap.add_argument("--river_path", type=str,
                    default=_default_paths().river)
    ap.add_argument("--calib_path", type=str,
                    default=_default_paths().calib)
    return ap.parse_args()


def main():
    args = parse_args()
    paths = RealPaths(
        preproc=args.preproc_path,
        river=args.river_path,
        calib=args.calib_path,
    )
    x_real, kamp_real, kangle_real = load_top_kamp_shots(paths, args.top_n)

    x_neg = kamp_neg = kang_neg = None
    if args.n_sim_neg > 0:
        print(f"[neg ] pulling {args.n_sim_neg} unstreaked sim shots "
              f"(kickstrength < {args.streak_threshold_eV}) from "
              f"{args.sim_neg_glob!r}")
        x_neg, kamp_neg, kang_neg = load_sim_unstreaked(
            args.sim_neg_glob, args.n_sim_neg,
            args.streak_threshold_eV, args.sim_neg_seed,
        )
        if x_neg.shape[1:] != x_real.shape[1:]:
            raise RuntimeError(
                f"sim negative shape {x_neg.shape[1:]} != real shape "
                f"{x_real.shape[1:]}; check --sim_neg_glob points at a "
                f"(16, {NBINS}) dataset."
            )

    write_shard(
        x_real, kamp_real, kangle_real,
        label_kamp=args.label_kamp,
        streak_threshold_eV=args.streak_threshold_eV,
        out_path=args.output,
        x_sim_neg=x_neg,
        kamp_sim_neg=kamp_neg,
        kangle_sim_neg=kang_neg,
    )
    per_port_totals = x_real.sum(axis=2)
    print(f"summary: real n={x_real.shape[0]}  "
          f"hits/shot median={np.median(x_real.sum(axis=(1,2))):.0f}  "
          f"hits/port median={np.median(per_port_totals):.1f}")
    if x_neg is not None:
        print(f"summary: sim-neg n={x_neg.shape[0]}  "
              f"hits/shot median={np.median(x_neg.sum(axis=(1,2))):.0f}")


if __name__ == "__main__":
    main()
