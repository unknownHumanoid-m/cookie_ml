#!/usr/bin/env python3
"""Build sim-schema h5 shards from real duck (highest-kamp) and goose
(lowest-kamp) shots.

Positive class = top-N duck shots by ``duck_kamp`` — well above every goose
kamp in the river file. Negative class = bottom-M goose shots by
``goose_kamp`` — all zero (upstream fit-failed sentinel), which is what
we're treating as "no streak" for training.

Written to match ``build_real_topkamp_h5.py`` on schema:

  * ``Ximg`` — (16, 512) uint16 histogram of hits over (port angle idx, KE
    detector-bin), rebuilt from the notebook's preproc + calibration
    pipeline. Same tof_to_ke / t0 / alpha(v_ret=280) extrapolation as
    ``build_real_topkamp_h5.py`` (both scripts pull the helpers out of
    ``real_vs_sim_metrics.py``).
  * per-shot attrs:
      - ``streak_amplitude`` (float) — copy of the shot's original
        ``duck_kamp`` (ducks) or ``goose_kamp`` (geese). Real fit
        amplitudes, NOT the placeholder LABEL_KAMP that ``build_real_top
        kamp_h5.py`` writes.
      - ``streak`` (uint8) — thresholded copy of the above at
        ``--streak_threshold``. Default 0.02 sits between the goose max
        (0.0269) and the top-500-duck min (0.0256) so every duck lands as
        1 and every kamp=0 goose lands as 0. Set below the goose max if
        widening the duck count past top-371.
      - ``kickstrength`` — mirror of ``streak_amplitude`` (per streak_
        finder_eval convention).
      - ``kickangle`` (rad) — from ``duck_kangle`` / ``goose_kangle``.
      - ``duck_kamp_real`` — the same raw kamp; kept for schema
        consistency with ``build_real_topkamp_h5.py``.
      - ``npulses`` — 1 for ducks, 0 for geese (placeholder; the streak
        finder doesn't consume it, but it lets the 0-or-1 MLP see the
        expected label if we ever reuse this shard there).
      - ``phases`` / ``sasewidth`` — placeholders (0.0 / NaN); real shots
        have no fit-per-pulse phase.
      - ``source`` — ``b"duck"`` or ``b"goose"``.

Outputs per call:
  * ``<output>_train_<NN>.h5``  — one shard per --n_train_shards. Shots
    are class-interleaved *within* each shard, and the train set is
    round-robin-partitioned across shards so every shard is class-
    balanced. Multiple shards exist so streak_finder_training.py's
    file-level val split works (it refuses to run with n_files < 2).
  * ``<output>_holdout.h5``     — small eval-only set carved off
    deterministically BEFORE the training shards are written. No
    overlap between train and holdout by construction.

Usage:

    python3 build_real_duck_goose_h5.py \\
        --top_n_duck 500 --bot_n_goose 500 \\
        --n_holdout_per_class 25 \\
        --n_train_shards 5 \\
        --output /path/to/real_duck_goose/duck_goose

writes /path/to/real_duck_goose/duck_goose_train_00.h5 ... _04.h5
   and /path/to/real_duck_goose/duck_goose_holdout.h5.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence, Tuple

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the notebook-derived calibration + real-loader helpers so the
# (16, 512) images we write are byte-identical to what
# ``build_real_topkamp_h5.py`` produces on the same shot.
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


def _load_river(paths: RealPaths):
    """Pull the duck_* and goose_* arrays out of the river file."""
    with h5py.File(paths.river, "r") as rf:
        return {
            "duck_ts": rf["duck_timestamps"][:],
            "duck_kamp": rf["duck_kamp"][:],
            "duck_kang": rf["duck_kangle"][:],
            "goose_ts": rf["goose_timestamps"][:],
            "goose_kamp": rf["goose_kamp"][:],
            "goose_kang": rf["goose_kangle"][:],
        }


def _match_river_to_preproc(preproc_ts, river_ts):
    """Same sorted-searchsorted match as _match_duck_rows, but the helper
    is duck-named. Reuse it verbatim -- 'duck' just means 'row I care
    about' inside the function body.
    """
    return _match_duck_rows(preproc_ts, river_ts)


def _select_shots(kamp, kang, order_desc: bool, n_take: int):
    """Return indices (into the river array) for the shots we want.

    ``order_desc=True`` picks the top-N by kamp; ``False`` picks the
    bottom-N. Non-finite kamps are dropped up front. Ties are broken by
    array order (numpy default), which is deterministic across runs.
    """
    finite = np.isfinite(kamp) & np.isfinite(kang)
    if not finite.any():
        raise RuntimeError("no finite kamp/kangle rows")
    idx = np.where(finite)[0]
    values = kamp[idx]
    order = np.argsort(values)
    if order_desc:
        order = order[::-1]
    n_take = min(int(n_take), order.size)
    return idx[order[:n_take]]


def _build_ximg_for_rows(preproc_path: str, rows: np.ndarray,
                         alpha_at: dict, t0_at: dict,
                         tof_gate: dict) -> np.ndarray:
    """(N, 16, NBINS) uint16 histogram of hits per (port, KE bin).

    Same channel-preload + per-row slice trick used by
    ``build_real_topkamp_h5.load_top_kamp_shots``.
    """
    n = int(rows.size)
    x = np.zeros((n, len(HSD_CHANNELS), NBINS), dtype=np.uint16)
    if n == 0:
        return x
    with h5py.File(preproc_path, "r") as pf:
        for i, ang in enumerate(HSD_CHANNELS):
            lens = pf[f"var_hsd_hf_times_{ang}_len"][:]
            flat = pf[f"var_hsd_hf_times_{ang}"][:]
            ends = np.cumsum(lens)
            starts = ends - lens
            t_lo, t_hi = tof_gate[ang]
            for k, row in enumerate(rows):
                hits = flat[starts[row]:ends[row]]
                if hits.size == 0:
                    continue
                good = (hits > t_lo) & (hits < t_hi)
                if not good.any():
                    continue
                ke = _tof_to_ke(hits[good], alpha_at[ang], t0_at[ang])
                x[k, i], _ = np.histogram(ke, bins=EBIN_EDGES)
    return x


def _write_group(grp, img, streak_amp, streak_threshold,
                 kamp_real, kangle_real, npulses, source_tag):
    grp.create_dataset("Ximg", data=img, dtype=np.uint16, compression=None)
    grp.attrs.create("streak_amplitude", np.float32(streak_amp))
    grp.attrs.create("streak",
                     np.uint8(1 if streak_amp >= streak_threshold else 0))
    grp.attrs.create("kickstrength", np.float32(streak_amp))
    grp.attrs.create("kickangle", np.float32(kangle_real))
    grp.attrs.create("duck_kamp_real", np.float32(kamp_real))
    grp.attrs.create("npulses", np.uint8(npulses))
    grp.attrs.create("phases", np.array([0.0], dtype=np.float32))
    grp.attrs.create("sasewidth", np.float32(np.nan))
    grp.attrs.create("streak_threshold_eV", np.float32(streak_threshold))
    grp.attrs.create("source", np.bytes_(source_tag))


def _write_shard(out_path: str, groups: Sequence[dict]):
    """`groups` is a list of dicts with keys img/amp/kamp/kang/npulses/source."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for k, g in enumerate(groups):
            _write_group(
                f.create_group("shot_%06d" % k),
                g["img"],
                streak_amp=g["amp"],
                streak_threshold=g["streak_threshold"],
                kamp_real=g["kamp"],
                kangle_real=g["kang"],
                npulses=g["npulses"],
                source_tag=g["source"],
            )
    os.replace(tmp, out_path)
    return len(groups)


def build_shots_for_class(
    paths: RealPaths,
    class_name: str,       # "duck" or "goose"
    n_take: int,
    order_desc: bool,
    alpha_at: dict,
    t0_at: dict,
    tof_gate: dict,
    river: dict,
    preproc_ts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ximg[N, 16, NBINS] uint16, kamp[N], kangle[N], preproc_rows[N])
    for the selected shots of one class."""
    ts_key = f"{class_name}_ts"
    kamp_key = f"{class_name}_kamp"
    kang_key = f"{class_name}_kang"

    river_ts = river[ts_key]
    river_kamp = river[kamp_key]
    river_kang = river[kang_key]

    # Match every river shot of this class to preproc rows. Some river
    # shots may not appear in preproc; drop those before selection so we
    # don't pick a top-N shot that we can't reconstruct.
    rows, river_idx = _match_river_to_preproc(preproc_ts, river_ts)
    # rows[i] is a preproc row; river_idx[i] is the index into river_ts.
    matched_kamp = river_kamp[river_idx]
    matched_kang = river_kang[river_idx]

    sel_within_matched = _select_shots(
        matched_kamp, matched_kang, order_desc=order_desc, n_take=n_take,
    )
    sel_rows = rows[sel_within_matched]
    sel_kamp = matched_kamp[sel_within_matched]
    sel_kang = matched_kang[sel_within_matched]

    # Order by ascending row index so h5py fancy-slicing is efficient.
    order = np.argsort(sel_rows)
    sel_rows = sel_rows[order]
    sel_kamp = sel_kamp[order]
    sel_kang = sel_kang[order]

    print(f"[{class_name}] selected {sel_rows.size} shots  "
          f"kamp range [{sel_kamp.min():.4f}, {sel_kamp.max():.4f}]  "
          f"kang range [{sel_kang.min():.3f}, {sel_kang.max():.3f}] rad")

    x = _build_ximg_for_rows(paths.preproc, sel_rows, alpha_at, t0_at, tof_gate)
    return x, sel_kamp, sel_kang, sel_rows


def _partition_train_holdout(
    x: np.ndarray,
    amp: np.ndarray,
    kang: np.ndarray,
    n_holdout: int,
    rng: np.random.Generator,
):
    """Random-partition the shots into (train, holdout). Fixed seed so
    two runs of this builder produce identical splits."""
    n = int(x.shape[0])
    if n_holdout < 0 or n_holdout > n:
        raise ValueError(f"n_holdout={n_holdout} out of range for n={n}")
    perm = rng.permutation(n)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    return (
        x[train_idx], amp[train_idx], kang[train_idx],
        x[holdout_idx], amp[holdout_idx], kang[holdout_idx],
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top_n_duck", type=int, default=500,
                    help="Number of highest-kamp duck shots to select as "
                         "positives (streak=1).")
    ap.add_argument("--bot_n_goose", type=int, default=500,
                    help="Number of lowest-kamp goose shots to select as "
                         "negatives (streak=0). At the current defaults "
                         "these are all kamp=0 (upstream fit-failed).")
    ap.add_argument("--n_holdout_per_class", type=int, default=25,
                    help="How many shots per class to reserve for an "
                         "eval-only holdout shard. The training shard gets "
                         "the remainder. Same seed -> identical partition "
                         "across runs.")
    ap.add_argument("--holdout_only", action="store_true",
                    help="Skip the train-shard split entirely: write every "
                         "selected shot into <output>_holdout.h5 in on-disk "
                         "top-N duck / bot-N goose order (no random "
                         "permutation). Use this to build an eval-only shard "
                         "when the model was trained on other data (e.g. "
                         "sim). --n_holdout_per_class is ignored in this "
                         "mode.")
    ap.add_argument("--streak_threshold", type=float, default=0.02,
                    help="Threshold on the streak_amplitude attr used to "
                         "write the binary 'streak' attr. Default 0.02 sits "
                         "between goose max (0.0269) and top-500-duck min "
                         "(0.0256) so every top-500 duck is streak=1 and "
                         "every zero-kamp goose is streak=0. Also passed "
                         "through as an attr on each shot for eval-time "
                         "auditing.")
    ap.add_argument("--output", type=str, required=True,
                    help="Output prefix. The builder writes "
                         "<output>_train_<NN>.h5 (one per --n_train_shards) "
                         "and <output>_holdout.h5.")
    ap.add_argument("--n_train_shards", type=int, default=5,
                    help="Split the train set into this many shards, "
                         "class-balanced within each shard. Must be >= 2 "
                         "so streak_finder_training.py's file-level val "
                         "split can produce >= 1 val file.")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="Seed for the train/holdout permutation.")
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
    river = _load_river(paths)
    with h5py.File(paths.preproc, "r") as pf:
        preproc_ts = pf["timestamp"][:]

    alpha_at, t0_at = _load_alpha_t0(paths.calib, V_RET)
    tof_gate = {
        ang: _tof_gate(alpha_at[ang], t0_at[ang],
                       EMIN - TOF_KE_GATE_MARGIN, EMAX + TOF_KE_GATE_MARGIN)
        for ang in HSD_CHANNELS
    }

    if args.holdout_only:
        print(f"[build] top-{args.top_n_duck} duck + bot-{args.bot_n_goose} "
              f"goose -> {args.output}_holdout.h5 (holdout-only)")
    else:
        print(f"[build] top-{args.top_n_duck} duck + bot-{args.bot_n_goose} "
              f"goose -> {args.output}_train.h5 + {args.output}_holdout.h5")
        print(f"[build] holding out {args.n_holdout_per_class} shots per class "
              f"(seed={args.split_seed})")

    x_duck, amp_duck, kang_duck, _ = build_shots_for_class(
        paths, "duck", args.top_n_duck, order_desc=True,
        alpha_at=alpha_at, t0_at=t0_at, tof_gate=tof_gate,
        river=river, preproc_ts=preproc_ts,
    )
    x_goose, amp_goose, kang_goose, _ = build_shots_for_class(
        paths, "goose", args.bot_n_goose, order_desc=False,
        alpha_at=alpha_at, t0_at=t0_at, tof_gate=tof_gate,
        river=river, preproc_ts=preproc_ts,
    )

    if args.holdout_only:
        # Every selected shot goes into the holdout shard; no random
        # partition. Training shards aren't written.
        x_duck_tr = np.empty((0,) + x_duck.shape[1:], dtype=x_duck.dtype)
        amp_duck_tr = np.empty((0,), dtype=amp_duck.dtype)
        kang_duck_tr = np.empty((0,), dtype=kang_duck.dtype)
        x_duck_ho, amp_duck_ho, kang_duck_ho = x_duck, amp_duck, kang_duck

        x_goose_tr = np.empty((0,) + x_goose.shape[1:], dtype=x_goose.dtype)
        amp_goose_tr = np.empty((0,), dtype=amp_goose.dtype)
        kang_goose_tr = np.empty((0,), dtype=kang_goose.dtype)
        x_goose_ho, amp_goose_ho, kang_goose_ho = x_goose, amp_goose, kang_goose
    else:
        rng = np.random.default_rng(args.split_seed)
        (x_duck_tr, amp_duck_tr, kang_duck_tr,
         x_duck_ho, amp_duck_ho, kang_duck_ho) = _partition_train_holdout(
            x_duck, amp_duck, kang_duck, args.n_holdout_per_class, rng,
        )
        # Fresh rng draw for the goose split so the two class-permutations
        # aren't correlated (would be a no-op statistically but the code was
        # right there so may as well keep them independent).
        rng_g = np.random.default_rng(args.split_seed + 1)
        (x_goose_tr, amp_goose_tr, kang_goose_tr,
         x_goose_ho, amp_goose_ho, kang_goose_ho) = _partition_train_holdout(
            x_goose, amp_goose, kang_goose, args.n_holdout_per_class, rng_g,
        )

    def _make_groups(x, amp, kang, npulses, source):
        return [
            {
                "img": x[i],
                "amp": float(amp[i]),
                "kamp": float(amp[i]),
                "kang": float(kang[i]),
                "npulses": npulses,
                "source": source,
                "streak_threshold": args.streak_threshold,
            }
            for i in range(x.shape[0])
        ]

    # Interleave duck & goose so class labels aren't all-front-loaded --
    # not strictly necessary (streak_finder_training shuffles), but keeps
    # any downstream tool that reads groups in on-disk order well-mixed.
    def _interleave(a, b):
        out = []
        for i in range(max(len(a), len(b))):
            if i < len(a):
                out.append(a[i])
            if i < len(b):
                out.append(b[i])
        return out

    train_groups = _interleave(
        _make_groups(x_duck_tr,  amp_duck_tr,  kang_duck_tr,  npulses=1, source=b"duck"),
        _make_groups(x_goose_tr, amp_goose_tr, kang_goose_tr, npulses=0, source=b"goose"),
    )
    holdout_groups = _interleave(
        _make_groups(x_duck_ho,  amp_duck_ho,  kang_duck_ho,  npulses=1, source=b"duck"),
        _make_groups(x_goose_ho, amp_goose_ho, kang_goose_ho, npulses=0, source=b"goose"),
    )

    holdout_path = f"{args.output}_holdout.h5"
    n_ho = _write_shard(holdout_path, holdout_groups)

    train_shard_paths = []
    if train_groups:
        n_shards = max(2, int(args.n_train_shards))
        # Round-robin across shards so each shard stays class-balanced (the
        # groups are already class-interleaved above). Any extra shots after
        # even division just land in the earlier shards.
        per_shard_groups = [[] for _ in range(n_shards)]
        for i, g in enumerate(train_groups):
            per_shard_groups[i % n_shards].append(g)
        for i, groups in enumerate(per_shard_groups):
            p = f"{args.output}_train_{i:02d}.h5"
            _write_shard(p, groups)
            train_shard_paths.append(p)

    n_pos_tr = int((amp_duck_tr >= args.streak_threshold).sum())
    n_neg_tr = int((amp_goose_tr < args.streak_threshold).sum())
    n_pos_ho = int((amp_duck_ho >= args.streak_threshold).sum())
    n_neg_ho = int((amp_goose_ho < args.streak_threshold).sum())
    if train_shard_paths:
        print(f"[build] train  : {len(train_groups)} shots across "
              f"{len(train_shard_paths)} shards "
              f"(pos={n_pos_tr}  neg={n_neg_tr})")
        for p in train_shard_paths:
            with h5py.File(p, "r") as f:
                n = sum(1 for k in f.keys() if k.startswith("shot_"))
            print(f"[build]   wrote {p}  ({n} shots)")
    else:
        print("[build] train  : (skipped, --holdout_only)")
    print(f"[build] holdout: {n_ho} shots  (pos={n_pos_ho}  neg={n_neg_ho})")
    print(f"[build] wrote {holdout_path}")


if __name__ == "__main__":
    main()
