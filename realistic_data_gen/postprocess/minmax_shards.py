#!/usr/bin/env python3
"""Fit and apply a per-column MinMax(0,1) scaler to realistic-generator
Ximg shards, in the style of src/data_processing/universal_cookiesimslim_
processor.py.

Why this file exists
--------------------
The two sibling forks (split_bottleneck_ae, split_bottleneck_ae_320) train
on Ximg that was rescaled by universal_cookiesimslim_processor.py: a single
sklearn MinMaxScaler(feature_range=(0,1)) is partial_fit per shot on the
raw (16, 512) array so each of the 512 energy columns is one sklearn
"feature", aggregated across every row of every training shot. The fit is
saved to a joblib and applied to every shot via `.transform()`.

split_bottleneck_ae_real_data instead uses generate_streak_data_realistic.py
which writes raw uint16 Ximg — no scaler. That generator carries knobs
(target_hits calibration, dead-time, dark ROI, per-port jitter, recipe
pulse_123_1of3, min_dphi_rad gate) that we want to keep, so we can't just
swap it out for universal_cookiesimslim_processor.

This script is the post-processing stage that bridges the two: takes the
realistic-generator shards as-is, fits the same corpus-column MinMax(0,1)
scaler that the other forks use, and writes a parallel set of scaled
shards + saves the joblib next to them. The joblib is what
split_bottleneck_ae_real_data/eval_real.py loads to transform real
duck-shot Ximg the same way at inference time.

Fit is train-only (no test leakage). Test and val are transformed with
the train-fit joblib.

Ypdf handling
-------------
The realistic generator already writes Ypdf as per-shot _normalize_ypdf'd
float32 in [0, 1] — matches the AE decoder target. We copy Ypdf through
untouched. Attrs are copied through unchanged.

CLI
---
    python3 minmax_realistic_shards.py fit \
        --in_dir  <realistic>/train \
        --scaler_path <out_root>/min_max_scaler_realistic_ximg.joblib

    python3 minmax_realistic_shards.py transform \
        --in_dir  <realistic>/{train,val,test} \
        --out_dir <out_root>/{train,val,test} \
        --scaler_path <out_root>/min_max_scaler_realistic_ximg.joblib
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _collect_h5(in_dir: str) -> list[str]:
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"not a directory: {in_dir}")
    files = sorted(
        os.path.join(in_dir, n)
        for n in os.listdir(in_dir)
        if n.endswith(".h5")
    )
    if not files:
        raise RuntimeError(f"no .h5 files under {in_dir}")
    return files


def _iter_shot_groups(f: h5py.File):
    # generate_streak_data_realistic.py writes groups "shot_000000",
    # "shot_000001", ...; universal_cookiesimslim_processor.py-style
    # data uses arbitrary group names. Match either.
    for k in sorted(f.keys()):
        obj = f[k]
        if isinstance(obj, h5py.Group) and "Ximg" in obj:
            yield k, obj


def cmd_fit(args: argparse.Namespace) -> None:
    files = _collect_h5(args.in_dir)
    scaler = MinMaxScaler(feature_range=(0, 1))
    total_shots = 0
    for path in files:
        with h5py.File(path, "r") as f:
            for _, grp in _iter_shot_groups(f):
                img = np.asarray(grp["Ximg"][()])
                # partial_fit expects 2D (n_samples, n_features). Each of the
                # 512 columns is a feature; the 16 rows are samples.
                scaler.partial_fit(img.astype(np.float64))
                total_shots += 1
        print(f"[fit] scanned {path}  (shots so far: {total_shots})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.scaler_path)), exist_ok=True)
    joblib.dump(scaler, args.scaler_path)

    dmin = np.asarray(scaler.data_min_, dtype=np.float32)
    dmax = np.asarray(scaler.data_max_, dtype=np.float32)
    print(f"[fit] wrote {args.scaler_path}", flush=True)
    print(
        f"[fit] N shots={total_shots}  cols={dmin.size}  "
        f"data_min: min={dmin.min():.3g} median={np.median(dmin):.3g} max={dmin.max():.3g}  "
        f"data_max: min={dmax.min():.3g} median={np.median(dmax):.3g} max={dmax.max():.3g}",
        flush=True,
    )


def cmd_transform(args: argparse.Namespace) -> None:
    scaler: MinMaxScaler = joblib.load(args.scaler_path)
    files = _collect_h5(args.in_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    for path in files:
        base = os.path.basename(path)
        out_path = os.path.join(args.out_dir, base)
        tmp_path = out_path + ".tmp"
        n_shots = 0
        with h5py.File(path, "r") as src, h5py.File(tmp_path, "w") as dst:
            for k, grp in _iter_shot_groups(src):
                img = np.asarray(grp["Ximg"][()]).astype(np.float32)
                scaled = scaler.transform(img).astype(np.float32)

                dgrp = dst.create_group(k)
                dgrp.create_dataset(
                    "Ximg", data=scaled, dtype=np.float32, compression=None,
                )
                if "Ypdf" in grp:
                    dgrp.create_dataset(
                        "Ypdf",
                        data=np.asarray(grp["Ypdf"][()], dtype=np.float32),
                        dtype=np.float32,
                        compression=None,
                    )
                for name, val in grp.attrs.items():
                    dgrp.attrs.create(name, val)
                n_shots += 1
        os.replace(tmp_path, out_path)
        print(f"[transform] {path}  ->  {out_path}  ({n_shots} shots)", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="Fit MinMax scaler on train shards")
    p_fit.add_argument("--in_dir", required=True,
                       help="Directory of realistic-generator train shards.")
    p_fit.add_argument("--scaler_path", required=True,
                       help="Output joblib path for the fitted scaler.")
    p_fit.set_defaults(func=cmd_fit)

    p_tx = sub.add_parser("transform",
                          help="Apply a fitted scaler to a directory of shards")
    p_tx.add_argument("--in_dir", required=True,
                      help="Directory of realistic-generator shards to scale.")
    p_tx.add_argument("--out_dir", required=True,
                      help="Where to write scaled shards.")
    p_tx.add_argument("--scaler_path", required=True,
                      help="Joblib path of the fitted scaler.")
    p_tx.set_defaults(func=cmd_transform)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
