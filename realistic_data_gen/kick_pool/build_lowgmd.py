#!/usr/bin/env python3
"""Build a sim-schema h5 shard from the bottom-N preproc shots by gmd_energy.

Companion to ``build_real_topkamp_h5.py``. Where that script pulls the
highest-``duck_kamp`` real shots (streaked positives), this one pulls the
lowest-``gmd_energy`` real shots from the FULL preproc file — no-beam /
dropped-pulse events. The vast majority are literally empty (median 0
hits across all 16 ports at the bottom-2000 tail; see the diagnostic in
MRCO_Streaking_rawData.ipynb), which is exactly the "0-pulse" real
distribution the 0-or-1 MLP eval needs.

Selection is against ``preproc/timestamp`` directly, NOT the river
duck/goose subset (the river file only labels ~28k / 1.25M preproc
shots, and its designations are only meaningful for shots with actual
electrons in them).

Output per-shot schema mirrors ``build_real_topkamp_h5.py`` so
``eval_0or1_mlp.py`` can consume both shards side-by-side, with:

  * ``Ximg``           — (16, 512) uint16 KE histogram (same tof->KE
                          calibration as the topkamp builder).
  * ``npulses`` = 0     — the class label load_binary_h5 reads.
  * ``streak`` = 0, ``streak_amplitude`` = 0, ``kickstrength`` = 0 —
                          consistent no-streak attrs.
  * ``kickangle`` = NaN — real no-beam shots have no kick.
  * ``gmd_energy`` (float) — the raw gmd value that selected the shot.
  * ``source`` = ``b"real_lowgmd"``.

Usage:

    python3 build_real_lowgmd_h5.py --bot_n 2000 \\
        --output /tmp/miaed_real_lowgmd/lowgmd_00000.h5
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from real_vs_sim_metrics import (  # noqa: E402
    HSD_CHANNELS,
    V_RET,
    NBINS,
    EMIN,
    EMAX,
    EBIN_EDGES,
    TOF_KE_GATE_MARGIN,
    _default_paths,
    _load_alpha_t0,
    _tof_gate,
    _tof_to_ke,
    RealPaths,
)


def load_bot_gmd_shots(paths: RealPaths, bot_n: int):
    """Return (Ximg[N, 16, NBINS] uint16, gmd[N] float).

    Selects the ``bot_n`` preproc rows with the smallest ``gmd_energy`` and
    rebuilds each shot's (port, KE) histogram from the raw HSD hit times.
    """
    print(f"[real] loading preproc gmd ({paths.preproc}) ...")
    with h5py.File(paths.preproc, "r") as pf:
        gmd = pf["gmd_energy"][:]

    n_avail = int(gmd.size)
    take = min(int(bot_n), n_avail)
    if take == 0:
        raise RuntimeError("no preproc rows to select from")
    bot_rows_unsorted = np.argsort(gmd)[:take]
    order = np.argsort(bot_rows_unsorted)
    sel_rows = bot_rows_unsorted[order]
    sel_gmd = gmd[sel_rows].astype(np.float32)
    print(f"[real] selected bottom-{take} gmd shots out of {n_avail:,}: "
          f"gmd range [{sel_gmd.min():.4g}, {sel_gmd.max():.4g}], "
          f"median={float(np.median(sel_gmd)):.4g}")

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

    return x, sel_gmd


def _write_lowgmd_group(grp, img, gmd_value):
    grp.create_dataset("Ximg", data=img, dtype=np.uint16, compression=None)
    grp.attrs.create("streak_amplitude", np.float32(0.0))
    grp.attrs.create("streak", np.uint8(0))
    grp.attrs.create("kickstrength", np.float32(0.0))
    grp.attrs.create("kickangle", np.float32(np.nan))
    grp.attrs.create("duck_kamp_real", np.float32(np.nan))
    grp.attrs.create("npulses", np.uint8(0))
    grp.attrs.create("phases", np.array([0.0], dtype=np.float32))
    grp.attrs.create("sasewidth", np.float32(np.nan))
    grp.attrs.create("gmd_energy", np.float32(gmd_value))
    grp.attrs.create("source", np.bytes_(b"real_lowgmd"))


def write_shard(x, gmd, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    n = x.shape[0]
    with h5py.File(tmp, "w") as f:
        for k in range(n):
            _write_lowgmd_group(f.create_group("shot_%06d" % k), x[k], gmd[k])
    os.replace(tmp, out_path)
    print(f"wrote {n} zero-pulse (low-gmd) real shots to {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bot_n", type=int, default=2000,
                    help="Number of lowest-gmd preproc shots to write.")
    ap.add_argument("--output", type=str, required=True,
                    help="Output h5 path. Parent dir is created if missing.")
    ap.add_argument("--preproc_path", type=str,
                    default=_default_paths().preproc)
    ap.add_argument("--calib_path", type=str,
                    default=_default_paths().calib)
    # River path isn't consumed here (we pick directly from preproc), but
    # keep the arg for API parity with build_real_topkamp_h5.py so both
    # scripts accept the same environment variables in shared launchers.
    ap.add_argument("--river_path", type=str,
                    default=_default_paths().river)
    return ap.parse_args()


def main():
    args = parse_args()
    paths = RealPaths(
        preproc=args.preproc_path,
        river=args.river_path,
        calib=args.calib_path,
    )
    x, gmd = load_bot_gmd_shots(paths, args.bot_n)
    write_shard(x, gmd, args.output)
    tot_hits = x.sum(axis=(1, 2))
    print(f"summary: n={x.shape[0]}  "
          f"hits/shot: min={int(tot_hits.min())} max={int(tot_hits.max())} "
          f"median={int(np.median(tot_hits))} mean={tot_hits.mean():.2f}  "
          f"fraction with 0 hits={(tot_hits==0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
