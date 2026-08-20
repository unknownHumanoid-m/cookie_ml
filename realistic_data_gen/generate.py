#!/usr/bin/env python3
"""Generate raw-Ximg training data whose (kamp, kangle), per-port hit rate,
and per-port energy-scale spread are matched to a real river run.

Same output schema as generate_streak_data_v4.py so existing training / eval
code (streak_finder_training.py, streak_finder_eval.py) reads it unchanged.

Realistic knobs vs v4:
  * (kamp, kangle) sampled with replacement from a real river h5 (duck_kamp
    + duck_kangle). p_boost re-weights the [boost_min, boost_max) band on
    top of that empirical distribution so we still up-sample the band the
    model struggles on. Set --p_boost 0 to sample straight from the real
    joint.
  * Sim `drawscale` is adjusted so mean(sum(Ximg)) matches --target_hits.
    Calibrated on 32 dry shots at start-up.
  * Per-port energy-axis jitter (Gaussian, sigma in bins) added at
    histogram time to mimic the +/- 15-100% alpha-extrapolation spread we
    see per port in the real cell-7 diagnostic.
  * kick == 0 unstreaked shots inserted at a fixed fraction (--frac_unstreaked).
  * Per-port dead-time (--dead_time_eV): after build_XY, any hit landing
    within N bins (N = eV / 0.25) of an earlier accepted hit on the same
    port is dropped. Non-paralyzable model. Mimics the sinusoid-trace gaps
    seen in real duck shots when a channel misses consecutive hits.
  * Dark background window (--dark_roi_bin_lo, --dark_roi_bin_hi): the
    sim's flat dark floor is applied *only* inside this energy-bin window
    before build_XY draws hits. Real MRCO shots have essentially zero
    counts outside a ~200-bin band around the streak sinusoid; a uniform
    dark across all 512 bins is unphysical for that data. Passing lo>=hi
    disables the window (falls back to the flat 512-bin dark).
  * Signal-localized noise (--loc_noise_rate, --loc_noise_sigma_bins): for
    every accepted (post-deadtime) hit, draw Poisson(rate) extra hits at
    real_bin + Normal(0, sigma_bins). Models a narrower-but-noisier core
    band (space-charge / secondary-electron / crosstalk-ish spread).
  * `dev_20` split: --n_dev writes exactly N shots to <outdir>/dev/ using a
    high-kamp bias so you get streaked shots for eyeballing without
    generating 200k shots first.

Everything else (SASE width, central energy, dark scale, pulse-count
bucketing) is inherited from v4.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

import h5py
import numpy as np

_SIM_SRC = os.environ.get("COOKIESIMSLIM_PATH")
if not _SIM_SRC:
    raise RuntimeError(
        "COOKIESIMSLIM_PATH is not set. Point it at your local checkout of "
        "CookieSimSlim (contains simulation.py exposing Params, build_XY). "
        "Example: export COOKIESIMSLIM_PATH=$HOME/CookieSimSlim"
    )
if _SIM_SRC not in sys.path:
    sys.path.insert(0, _SIM_SRC)

from simulation import Params, build_XY  # noqa: E402


def _normalize_ypdf(ymat: np.ndarray) -> np.ndarray:
    """Rescale the clean 2D PDF into [0, 1] to match src/data_processing/
    universal_cookiesimslim_processor.py's per-shot MinMax(0, 1) target.

    The AE's decoder ends in a sigmoid + BCE-style recon loss (see
    split_bottleneck_ae/model.py), and mrco_h5 Ypdf lands in [0, ~0.33]
    after per-column MinMax over the full training corpus. We can't fit
    that column-wise scaler here without a two-pass, so per-shot MinMax
    is the closest online approximation. Both squash to [0, 1] but the
    per-shot version rescales each shot to fill its own dynamic range —
    plot cell 17 side-by-side against the mrco_h5 Ypdf to confirm the
    difference is only cosmetic.
    """
    lo = float(ymat.min())
    hi = float(ymat.max())
    if hi <= lo:
        return np.zeros_like(ymat, dtype=np.float32)
    return ((ymat - lo) / (hi - lo)).astype(np.float32)


class _ExactSpikeRng:
    """Force the first rng.poisson() call to return an exact value so
    build_XY draws a chosen pulse count. Every other call passes through.
    """

    def __init__(self, rng, exact_first_poisson: int):
        object.__setattr__(self, "_rng", rng)
        object.__setattr__(self, "_exact", int(exact_first_poisson))
        object.__setattr__(self, "_used", False)

    def poisson(self, lam, *args, **kwargs):
        if not self._used:
            object.__setattr__(self, "_used", True)
            return self._exact
        return self._rng.poisson(lam, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


@dataclass
class GenConfig:
    nangles: int
    nenergies: int
    drawscale: float
    secondaryscale: float
    e_center_min: float
    e_center_max: float
    sase_width_min: float
    sase_width_max: float
    ce_var: float
    dark_min: float
    dark_max: float
    # Realistic sampling: pool of (kamp, kangle) drawn from a river run.
    kamp_pool: np.ndarray
    kangle_pool: np.ndarray
    # Boost sampling: with probability p_boost, draw kamp uniformly on
    # [boost_min, boost_max) and reuse a kangle from the real pool.
    boost_min: float
    boost_max: float
    p_boost: float
    frac_unstreaked: float
    # Per-port energy-scale jitter (Gaussian sigma in bins). Mimics the
    # alpha-extrapolation spread between calibrated ports.
    per_port_e_jitter_bins: float
    streak_threshold_eV: float
    high_pulse_min: int
    high_pulse_max: int
    # Per-port dead-time (eV). Any hit landing within dead_time_eV of an
    # earlier accepted hit on the same port is dropped. 0.0 -> no gating.
    dead_time_eV: float
    # Signal-localized extra noise. For each accepted hit, draw
    # Poisson(loc_noise_rate) extra hits at real_bin + Normal(0, sigma_bins).
    # 0.0 -> no extra hits (still leaves the flat dark background).
    loc_noise_rate: float
    loc_noise_sigma_bins: float
    # Dark-background window in energy bins. When lo < hi, the sim's flat
    # dark floor is zeroed outside [lo, hi) so uniform noise doesn't leak
    # into KE regions where the real detector sees ~zero counts. lo >= hi
    # disables the gate (dark stays on all nenergies bins).
    dark_roi_bin_lo: int
    dark_roi_bin_hi: int
    # Class-mixture recipe. Deterministically overrides the per-shot
    # (streak_binary, ncenters) draws so the dataset on disk has a chosen
    # balance instead of the natural bucket-driven marginals. Accepted:
    #   "natural"              -> original behavior (dev-mode compatible).
    #   "streak_5050"          -> alternating streaked/unstreaked (seed%2).
    #   "zero_pulse_5050"      -> alternating ncenters=0 / natural bucket.
    #   "pulse_1234plus_2525"  -> ncenters cycles 1/2/3/rand(hi) (seed%4),
    #                             every shot forced streaked. Forces every
    #                             pulse in the shot to the same kick angle.
    #   "pulse_123_1of3"       -> ncenters cycles 1/2/rand(high) (seed%3),
    #                             every shot forced streaked. Pulses draw
    #                             INDEPENDENT random phases; 2-pulse shots
    #                             are rejection-sampled so |arccos(cos Δφ)|
    #                             is at least min_dphi_rad. Target dataset
    #                             for the split_bottleneck_ae_real_data
    #                             fork (3-class collapse: 1 / 2 / >=3).
    recipe: str = "natural"
    # Minimum wrapped phase separation |arccos(cos(phi_i - phi_j))| (rad)
    # for pulses within a shot. Only enforced when the recipe is one that
    # uses independent per-pulse phases (currently: pulse_123_1of3), and
    # only on 2-pulse shots -- 3+-pulse shots collapse into a single ">=3"
    # class, so their internal separation is downstream noise. 0.0 means
    # "no rejection sampling" (accept any independent draw).
    min_dphi_rad: float = 0.0
    # Cap on the number of independent-phase redraws per 2-pulse shot when
    # min_dphi_rad > 0. Set higher for tight thresholds; at pi/4 the
    # rejection rate is ~25%, so 32 attempts is safely oversized.
    dphi_reject_max_attempts: int = 32


def _load_river_kicks(river_path: str, min_kamp: float, drop_nonfinite: bool = True):
    """Return (kamp, kangle) arrays from a river file's duck_* datasets.

    Filters non-finite values and shots below `min_kamp` (typically 0).
    Kangle is stored in radians in the river file; we keep it that way
    because build_XY expects sasephases in radians too.
    """
    with h5py.File(river_path, "r") as rf:
        kamp = np.asarray(rf["duck_kamp"][:], dtype=np.float64)
        kangle = np.asarray(rf["duck_kangle"][:], dtype=np.float64)
    if drop_nonfinite:
        m = np.isfinite(kamp) & np.isfinite(kangle)
        kamp = kamp[m]
        kangle = kangle[m]
    m = kamp >= min_kamp
    return kamp[m], kangle[m]


def _draw_kick(rng: np.random.Generator, cfg: GenConfig):
    """Return (kamp, kangle) for one shot.

    * With probability frac_unstreaked, (0.0, 0.0) -> spectroscopy mode.
    * Otherwise, with probability p_boost, kamp uniform on
      [boost_min, boost_max) and kangle re-sampled from the real pool.
    * Otherwise, sample (kamp, kangle) jointly from the real pool.
    """
    if float(rng.random()) < cfg.frac_unstreaked:
        return 0.0, 0.0

    pool_n = cfg.kamp_pool.shape[0]
    if pool_n == 0:
        raise RuntimeError("empty kamp/kangle pool")

    if float(rng.random()) < cfg.p_boost:
        kamp = float(rng.uniform(cfg.boost_min, cfg.boost_max))
        idx = int(rng.integers(0, pool_n))
        return kamp, float(cfg.kangle_pool[idx])

    idx = int(rng.integers(0, pool_n))
    return float(cfg.kamp_pool[idx]), float(cfg.kangle_pool[idx])


def _draw_kick_streaked_only(rng: np.random.Generator, cfg: GenConfig):
    """Force a streaked (kamp >= streak_threshold_eV) draw regardless of
    cfg.frac_unstreaked. Falls through the real pool first, then the boost
    band; whichever branch fires, kamp is clamped up to streak_threshold_eV
    so the streaked half of a mixture is unambiguously above threshold.

    boost_min may be below streak_threshold_eV for the natural recipe
    (which wants a broader kick distribution). This helper widens the low
    edge of the boost band up to the threshold so recipe callers don't
    need to hand-tune boost_min per dataset.
    """
    pool_n = cfg.kamp_pool.shape[0]
    if pool_n == 0:
        raise RuntimeError("empty kamp/kangle pool")
    thr = float(cfg.streak_threshold_eV)
    lo = max(cfg.boost_min, thr)
    hi = max(cfg.boost_max, lo + 1e-3)  # guarantee lo < hi

    if float(rng.random()) < cfg.p_boost:
        kamp = float(rng.uniform(lo, hi))
        idx = int(rng.integers(0, pool_n))
        return kamp, float(cfg.kangle_pool[idx])
    for _ in range(8):
        idx = int(rng.integers(0, pool_n))
        kamp = float(cfg.kamp_pool[idx])
        if kamp >= thr:
            return kamp, float(cfg.kangle_pool[idx])
    kamp = float(rng.uniform(lo, hi))
    idx = int(rng.integers(0, pool_n))
    return kamp, float(cfg.kangle_pool[idx])


def _recipe_choice(seed: int, rng: np.random.Generator, cfg: GenConfig):
    """Return (streak_forced, ncenters_forced, kick_pair_forced) applied
    before the normal draws in sample_shot.

    Each field is Optional[...]:
      * streak_forced        - bool or None. If True the shot must be
                               streaked; if False it must be unstreaked
                               (kamp=0). None -> honor the normal draw.
      * ncenters_forced      - int or None. If set, overrides the 4-bucket
                               pulse-count draw.
      * kick_pair_forced     - (kamp, kangle) or None. If set, used
                               verbatim; else fall through to _draw_kick
                               or _draw_kick_streaked_only per streak_forced.

    The `seed` argument (which is the global shot index in sample_shot)
    drives the modular alternation so mixtures are exact on any prefix,
    not just in expectation.
    """
    r = cfg.recipe
    if r == "natural":
        return None, None, None
    if r == "streak_5050":
        # seed % 2: 0 -> streaked, 1 -> unstreaked. Pulse count from bucket.
        streaked = (seed % 2 == 0)
        if streaked:
            return True, None, _draw_kick_streaked_only(rng, cfg)
        return False, None, (0.0, 0.0)
    if r == "zero_pulse_5050":
        # seed % 2: 0 -> ncenters=0 (dark only), 1 -> normal bucket + kick.
        zero_pulse = (seed % 2 == 0)
        if zero_pulse:
            # Zero pulses => nothing to streak; force unstreaked so the
            # attrs record kamp=0 and don't misleadingly flag "streak=1".
            return False, 0, (0.0, 0.0)
        return None, None, None  # natural bucket + natural kick
    if r == "pulse_1234plus_2525":
        # seed % 4 cycles 1 / 2 / 3 / rand(high). All shots streaked.
        m = seed % 4
        if m < 3:
            nc = m + 1
        else:
            nc = int(rng.integers(cfg.high_pulse_min, cfg.high_pulse_max + 1))
        return True, nc, _draw_kick_streaked_only(rng, cfg)
    if r == "pulse_123_1of3":
        # seed % 3 cycles 1 / 2 / rand(high). All shots streaked.
        # ncenters == 3 is drawn uniformly from [high_pulse_min,
        # high_pulse_max]; the downstream classifier collapses n>=3 into
        # one class, so the exact multiplicity in that bucket doesn't
        # need its own slot in the mixture. Per-pulse phases stay random
        # (see the phase-forcing block in sample_shot); the recipe here
        # is only responsible for the count draw.
        m = seed % 3
        if m == 0:
            nc = 1
        elif m == 1:
            nc = 2
        else:
            nc = int(rng.integers(cfg.high_pulse_min, cfg.high_pulse_max + 1))
        return True, nc, _draw_kick_streaked_only(rng, cfg)
    raise ValueError(f"unknown recipe {r!r}")


def _apply_dead_time(hits, dead_time_bins):
    """Non-paralyzable dead-time gate in bin-space. Sort each port's hits
    by energy bin; keep the first hit; keep subsequent hits only if they
    fall at least `dead_time_bins` bins past the last accepted one.
    Returns a new list-of-arrays (leaves inputs untouched).
    """
    if dead_time_bins <= 0:
        return hits
    out = []
    for row in hits:
        if len(row) == 0:
            out.append(row)
            continue
        arr = np.sort(np.asarray(row, dtype=np.float64))
        kept = [arr[0]]
        last = arr[0]
        for v in arr[1:]:
            if v - last >= dead_time_bins:
                kept.append(v)
                last = v
        out.append(np.asarray(kept, dtype=np.float64))
    return out


def _apply_signal_localized_noise(hits, rate, sigma_bins, nenergies, rng):
    """For each accepted hit, spawn Poisson(rate) extra hits placed at
    real_bin + Normal(0, sigma_bins). Returns a new list-of-arrays with
    the ghost hits appended per port.

    Concentrates extra counts near the signal band (unlike the flat dark
    background). Simulates whatever narrow-core-plus-more-noise mechanism
    is visible in real duck shots (space-charge / crosstalk / secondaries).
    """
    if rate <= 0.0 or sigma_bins < 0.0:
        return hits
    out = []
    lo, hi = 0.0, float(nenergies)
    for row in hits:
        if len(row) == 0:
            out.append(row)
            continue
        real = np.asarray(row, dtype=np.float64)
        extras = []
        n_per = rng.poisson(rate, size=real.size)
        for center, n in zip(real, n_per):
            if n <= 0:
                continue
            g = rng.normal(center, sigma_bins, size=int(n))
            # Drop ghosts that fall outside the energy axis so they don't
            # get absorbed by the histogram's under/overflow edges.
            g = g[(g >= lo) & (g < hi)]
            if g.size:
                extras.append(g)
        if extras:
            out.append(np.concatenate([real, *extras]))
        else:
            out.append(real)
    return out


def _histogram_with_jitter(hits, nangles, nenergies, per_port_shift, rng):
    """Same as v4's histogram loop but adds a per-port Gaussian energy
    shift `per_port_shift[a]` (bins) to each hit before binning. Mirrors
    the per-port alpha spread observed in the real calibration."""
    img = np.zeros((nangles, nenergies), dtype=np.uint16)
    edges = np.arange(nenergies + 1)
    for a, row in enumerate(hits):
        if len(row) == 0:
            continue
        shifted = np.asarray(row, dtype=np.float64) + per_port_shift[a]
        img[a, :] += np.histogram(shifted, edges)[0].astype(np.uint16)
    return img


def sample_shot(seed: int, cfg: GenConfig, return_ypdf: bool = False):
    rng = np.random.default_rng(seed)

    # Recipe hook. Non-"natural" recipes deterministically override the
    # normal pulse-count / kick draws to enforce a chosen class mixture.
    streak_forced, ncenters_forced, kick_forced = _recipe_choice(seed, rng, cfg)

    # Pulse-count bucket (uniform over 4 buckets, same as v4).
    if ncenters_forced is not None:
        ncenters = int(ncenters_forced)
    else:
        bucket = int(rng.integers(0, 4))
        if bucket < 3:
            ncenters = bucket + 1
        else:
            ncenters = int(rng.integers(cfg.high_pulse_min, cfg.high_pulse_max + 1))

    if kick_forced is not None:
        kamp, kangle = float(kick_forced[0]), float(kick_forced[1])
    elif streak_forced is True:
        kamp, kangle = _draw_kick_streaked_only(rng, cfg)
    elif streak_forced is False:
        kamp, kangle = 0.0, 0.0
    else:
        kamp, kangle = _draw_kick(rng, cfg)
    streak_binary = int(kamp >= cfg.streak_threshold_eV)

    ce = float(rng.uniform(cfg.e_center_min, cfg.e_center_max))
    sw = float(rng.uniform(cfg.sase_width_min, cfg.sase_width_max))
    dark = float(np.exp(rng.uniform(np.log(cfg.dark_min), np.log(cfg.dark_max))))

    p = Params(".", "streak_realistic", 1)
    p.setnangles(cfg.nangles)
    p.setnenergies(cfg.nenergies)
    p.setdrawscale(cfg.drawscale)
    p.setdarkscale(dark)
    # Gate the flat dark background to an energy-bin window. build_XY does
    # `bgmat = params.darkscale * np.ones((nangles, nenergies))`, so a
    # length-nenergies vector broadcasts to zero out bgmat outside [lo, hi)
    # while keeping the scalar value inside. Real MRCO data has ~0 counts
    # outside a narrow band, so the previous flat-across-all-512-bins
    # background was leaking noise into KE bins where the detector sees
    # nothing.
    lo = int(cfg.dark_roi_bin_lo)
    hi = int(cfg.dark_roi_bin_hi)
    if 0 <= lo < hi <= cfg.nenergies:
        dark_row = np.zeros(cfg.nenergies, dtype=float)
        dark_row[lo:hi] = dark
        p.darkscale = dark_row
    p.setsecondaryscale(cfg.secondaryscale)
    p.setcentralenergy(ce)
    p.centralenergyvar = cfg.ce_var  # setcentralenergyvar has a known bug
    p.setcentralenergywidth(sw)
    p.setsasescale(ncenters)         # placeholder; _ExactSpikeRng is authoritative
    p.setsasewidth(sw)
    p.setkickstrength(kamp)
    p.setkickstrengthvar(0.0)
    p.setfixedlinear()

    if kamp > 0.0:
        p.setstreaking()
    else:
        p.setspectroscopy()

    p.rng = _ExactSpikeRng(rng, ncenters)

    # Phase-forcing hook. Two disjoint modes, chosen by the recipe:
    #
    #   1. Old behavior ("pulse_1234plus_2525" and everything except
    #      "pulse_123_1of3"): force every pulse in the shot to sasephase
    #      = kangle so the streak arc peaks at port 0 and there is a
    #      single, deterministic streak direction. This is what the old
    #      realistic dataset produced -- pulses only differ in spectral
    #      center, and 2-pulse Δφ is exactly 0 shot-to-shot.
    #
    #   2. Independent-phase recipes ("pulse_123_1of3"): keep build_XY's
    #      random per-pulse phase draw so 2-pulse shots have real Δφ
    #      variation, then rejection-sample 2-pulse draws to enforce a
    #      minimum wrapped separation |arccos(cos Δφ)| >= cfg.min_dphi_rad.
    #      3+-pulse shots pass through with no rejection (they collapse
    #      into a single ">=3" class downstream).
    #
    # Both modes work by monkey-patching setphases -- build_XY calls
    # params.setphases(...) with a fresh random draw AFTER setup, so
    # we can't just pre-set p.sasephases here (build_XY would overwrite it).
    _orig_setphases = p.setphases
    independent_phases = (cfg.recipe == "pulse_123_1of3")

    if not independent_phases:
        forced_phases = [kangle] * max(1, ncenters)

        def _forced_setphases(_phases_from_buildxy, _forced=forced_phases,
                              _orig=_orig_setphases):
            _orig(_forced)
        p.setphases = _forced_setphases
    else:
        min_dphi = float(cfg.min_dphi_rad)
        max_attempts = max(1, int(cfg.dphi_reject_max_attempts))
        # Only 2-pulse shots need the Δφ gate; skip the whole loop
        # otherwise so the fast path stays fast.
        gate_active = (ncenters == 2 and min_dphi > 0.0)

        def _interceptor(phases_from_buildxy, _orig=_orig_setphases,
                         _gate=gate_active, _min_dphi=min_dphi,
                         _rng=rng, _n=int(ncenters),
                         _max_attempts=max_attempts):
            if not _gate:
                _orig(list(phases_from_buildxy))
                return
            # Rejection sample until |arccos(cos(Δφ))| >= threshold. Cap
            # the attempts so a pathological threshold can't deadlock;
            # after the cap, take whatever the last draw was and let the
            # loader decide whether to keep the shot.
            phi = list(phases_from_buildxy)
            for _ in range(_max_attempts):
                dphi = float(np.arccos(np.clip(
                    np.cos(phi[0] - phi[1]), -1.0, 1.0,
                )))
                if dphi >= _min_dphi:
                    _orig(phi)
                    return
                phi = list(_rng.random(_n) * 2.0 * np.pi)
            _orig(phi)
        p.setphases = _interceptor

    hits, ymat = build_XY(p)

    # Restore setphases so nothing else on this Params object misbehaves.
    p.setphases = _orig_setphases

    # Dead-time gate runs BEFORE the localized-noise stage so ghost hits
    # can't be dropped by a real hit that would have been suppressed
    # anyway. Both stages act on `hits` (bin-space); the histogram sees
    # the final list.
    dead_bins = cfg.dead_time_eV / 0.25 if cfg.dead_time_eV > 0 else 0.0
    if dead_bins > 0.0:
        hits = _apply_dead_time(hits, dead_bins)
    if cfg.loc_noise_rate > 0.0:
        hits = _apply_signal_localized_noise(
            hits, cfg.loc_noise_rate, cfg.loc_noise_sigma_bins,
            cfg.nenergies, rng,
        )

    if cfg.per_port_e_jitter_bins > 0.0:
        per_port_shift = rng.normal(
            0.0, cfg.per_port_e_jitter_bins, size=cfg.nangles,
        )
        img = _histogram_with_jitter(
            hits, cfg.nangles, cfg.nenergies, per_port_shift, rng,
        )
    else:
        img = np.zeros((cfg.nangles, cfg.nenergies), dtype=np.uint16)
        edges = np.arange(cfg.nenergies + 1)
        for a, row in enumerate(hits):
            if len(row) == 0:
                continue
            img[a, :] += np.histogram(row, edges)[0].astype(np.uint16)

    attrs = {
        "streak_amplitude": np.float32(kamp),
        "streak": np.uint8(streak_binary),
        "streak_threshold_eV": np.float32(cfg.streak_threshold_eV),
        "kickstrength": np.float32(kamp),
        "kickangle": np.float32(kangle),
        "centralenergy": np.float32(ce),
        "sasewidth": np.float32(sw),
        "darkscale": np.float32(dark),
        "secondaryscale": np.float32(cfg.secondaryscale),
        "drawscale": np.float32(cfg.drawscale),
        "npulses": np.uint8(ncenters),
        "phases": np.asarray(p.sasephases, dtype=np.float32),
        "sasecenters": np.asarray(p.sasecenters, dtype=np.float32),
        "polstrengths": np.asarray(p.polstrengths, dtype=np.float32),
        "poldirections": np.asarray(p.poldirections, dtype=np.float32),
        "polmode": np.bytes_(b"linear"),
        "seed": np.uint64(seed),
        "dead_time_eV": np.float32(cfg.dead_time_eV),
        "loc_noise_rate": np.float32(cfg.loc_noise_rate),
        "loc_noise_sigma_bins": np.float32(cfg.loc_noise_sigma_bins),
    }
    if return_ypdf:
        # ymat has the same (nangles, nenergies) shape as img. It's the
        # noise-free 2D PDF the streak/dark/hit-draw stages sample from,
        # so it's the clean-truth target for the AE's recon head. Real
        # data has no equivalent; this is sim-only.
        return img, attrs, _normalize_ypdf(ymat)
    return img, attrs


def calibrate_drawscale(cfg: GenConfig, target_hits: float, base_seed: int,
                        n_calib: int = 32) -> float:
    """Draw a few shots with cfg.drawscale=1, measure mean total hits,
    then scale to hit the target. Cheap fixed-point -- one pass is enough
    because sum(Ximg) is linear in drawscale.
    """
    probe_cfg = GenConfig(**{**cfg.__dict__, "drawscale": 1.0})
    totals = []
    for i in range(n_calib):
        img, _ = sample_shot(base_seed + 1_000_000 + i, probe_cfg)
        totals.append(int(img.sum()))
    mean_hits = max(1.0, float(np.mean(totals)))
    scale = target_hits / mean_hits
    print("  drawscale calibration: probe mean=%.1f hits/shot at "
          "drawscale=1.0 -> using drawscale=%.4f for target %.1f"
          % (mean_hits, scale, target_hits))
    return float(scale)


def _write_shard(task):
    # write_ypdf is threaded through the (shard_idx, seeds, out_path, cfg,
    # write_ypdf) tuple so worker pickling picks it up. Keep it defaulted so
    # older callers with a 4-tuple task still work.
    if len(task) == 5:
        shard_idx, seeds, out_path, cfg, write_ypdf = task
    else:
        shard_idx, seeds, out_path, cfg = task
        write_ypdf = False
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        for k, seed in enumerate(seeds):
            if write_ypdf:
                img, attrs, ypdf = sample_shot(seed, cfg, return_ypdf=True)
            else:
                img, attrs = sample_shot(seed, cfg)
                ypdf = None
            grp = f.create_group("shot_%06d" % k)
            grp.create_dataset("Ximg", data=img, dtype=np.uint16, compression=None)
            if ypdf is not None:
                grp.create_dataset(
                    "Ypdf", data=ypdf, dtype=np.float32, compression=None,
                )
            for name, val in attrs.items():
                grp.attrs.create(name, val)
    os.replace(tmp_path, out_path)
    return shard_idx, len(seeds)


def _plan_shards(n_total: int, shots_per_shard: int, base_seed: int):
    tasks = []
    for shard_idx, start in enumerate(range(0, n_total, shots_per_shard)):
        end = min(start + shots_per_shard, n_total)
        seeds = list(range(base_seed + start, base_seed + end))
        tasks.append((shard_idx, seeds))
    return tasks


def _run_split(split_name, n_shots, base_seed, outdir, cfg,
               n_workers, shots_per_shard, flat=False, write_ypdf=False):
    # flat=True writes shards directly in outdir (no <split_name>/ prefix).
    # Used by dev-mode so we can drop the file into an arbitrary directory
    # like copied_ipynbs/MRCO_streaking_rawData/ without adding a nested
    # 'dev/' folder underneath.
    split_dir = outdir if flat else os.path.join(outdir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    plan = _plan_shards(n_shots, shots_per_shard, base_seed)
    tasks = []
    for shard_idx, seeds in plan:
        out_path = os.path.join(
            split_dir,
            "streak_realistic_%s_%05d.h5" % (split_name, shard_idx),
        )
        tasks.append((shard_idx, seeds, out_path, cfg, write_ypdf))

    t0 = time.time()
    if n_workers <= 1:
        for t in tasks:
            _write_shard(t)
    else:
        with mp.Pool(processes=n_workers) as pool:
            done = 0
            for _shard_idx, _n in pool.imap_unordered(_write_shard, tasks, chunksize=1):
                done += 1
                if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                    dt = time.time() - t0
                    rate = (done * shots_per_shard) / max(dt, 1e-6)
                    print("  [%s] %d/%d shards  (%.0f shots/s)"
                          % (split_name, done, len(tasks), rate), flush=True)
    dt = time.time() - t0
    print("  [%s] wrote %d shots in %.1f s  (%.0f shots/s)"
          % (split_name, n_shots, dt, n_shots / max(dt, 1e-6)), flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=str,
                    default="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/"
                            "miaed_mnis_data/streak_finder_realistic")
    ap.add_argument("--river_path", type=str,
                    default="/sdf/data/lcls/ds/tmo/tmol1043723/results/"
                            "streaking_results/run[99]_vjtw_river.h5",
                    help="River h5 to sample (duck_kamp, duck_kangle) from.")
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--n_test", type=int, default=20_000)
    ap.add_argument("--nangles", type=int, default=16)
    ap.add_argument("--nenergies", type=int, default=512)
    ap.add_argument("--shots_per_shard", type=int, default=100)
    ap.add_argument("--n_workers", type=int, default=24)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--secondaryscale", type=float, default=0.0)
    # Real duck shots (run99) cluster the pulse energy at bin ~204 on the
    # 512-bin axis with a shot-to-shot std of only ~3.7 bins. Sim was
    # previously using [50, 200) which smeared the streak across the whole
    # detector -- p5-p95 hit spread of ~450 bins vs real's ~115. Keeping
    # the flags around so callers can widen it back on request.
    ap.add_argument("--e_center_min", type=float, default=200.0)
    ap.add_argument("--e_center_max", type=float, default=210.0)
    ap.add_argument("--sase_width_min", type=float, default=3.0)
    ap.add_argument("--sase_width_max", type=float, default=5.0)
    # Per-pulse center jitter (bins). Multi-pulse shots draw each pulse's
    # center from Normal(centralenergy, ce_var). Previously 8.0 which put
    # pulse centers up to ~30 bins apart on either side; combined with
    # kickstrength-driven streak modulation and sase_width, that produced
    # a per-shot p5-p95 spread nearly 3x what the real data shows. 3.0 is
    # tighter but still lets multi-pulse shots be visibly resolved.
    ap.add_argument("--ce_var", type=float, default=3.0)
    # Dark background is uniform across ALL 512 energy bins per port. Old
    # defaults (0.007-0.03) meant 23-56% of every shot's hits were uniform
    # noise, smeared over the full axis -- real duck shots concentrate
    # ~99% of hits in a 200-bin window. Dropped ~250x so dark accounts for
    # ~1-3% of the total hit count and doesn't dominate the tails.
    ap.add_argument("--dark_min", type=float, default=0.00003)
    ap.add_argument("--dark_max", type=float, default=0.0001)
    ap.add_argument("--boost_min", type=float, default=1.0)
    ap.add_argument("--boost_max", type=float, default=2.0)
    ap.add_argument("--p_boost", type=float, default=0.3,
                    help="Fraction of streaked shots drawn from the boosted "
                         "kamp band. Set 0 to sample straight from the real "
                         "(kamp, kangle) pool.")
    ap.add_argument("--frac_unstreaked", type=float, default=0.25,
                    help="Fraction of shots emitted with kamp=0 (spectroscopy "
                         "mode). Balances the streak/no-streak label at the "
                         "chosen threshold.")
    ap.add_argument("--streak_threshold_eV", type=float, default=2.0)
    ap.add_argument("--high_pulse_min", type=int, default=4)
    ap.add_argument("--high_pulse_max", type=int, default=8)
    ap.add_argument("--target_hits", type=float, default=250.0,
                    help="Target mean sum(Ximg) per shot. drawscale is "
                         "calibrated to hit this. Real run 99 duck shots "
                         "have ~16 hits/port x 16 ports = ~256.")
    ap.add_argument("--per_port_e_jitter_bins", type=float, default=1.0,
                    help="Gaussian sigma (bins) added to each port's energy "
                         "axis at histogram time. 1.0 bin = 0.25 eV at the "
                         "sim's default 0.25 eV/bin. Set 0 to disable.")
    ap.add_argument("--dead_time_eV", type=float, default=0.0,
                    help="Per-port dead-time window in eV. Consecutive hits "
                         "on the same port closer than this in energy are "
                         "dropped (non-paralyzable). 0 -> disabled.")
    ap.add_argument("--loc_noise_rate", type=float, default=0.0,
                    help="Extra 'ghost' hits per real hit, per port, added "
                         "with Gaussian offset around the real hit. 0 -> "
                         "no localized noise (dark background is still on).")
    ap.add_argument("--loc_noise_sigma_bins", type=float, default=1.5,
                    help="Sigma (bins) of the Gaussian used to place the "
                         "ghost hits from --loc_noise_rate around each real "
                         "hit. Tighter values concentrate the extra noise "
                         "on the signal band.")
    # Real MRCO run 99 duck shots (V_RET = 280 eV, 0.25 eV/bin) show roughly
    # uniform background counts across KE ~[0, 35] eV. On the 512-bin axis
    # that maps to bins ~[148, 288). Sim notebook cell 8 uses the same
    # EBIN_EDGES layout, so we gate the dark background to those bins by
    # default. Set --dark_roi_bin_hi <= lo to disable and keep dark flat
    # across all --nenergies bins.
    ap.add_argument("--dark_roi_bin_lo", type=int, default=148,
                    help="Low edge (inclusive, in energy bins) of the dark "
                         "background window. Bins below this get zero dark "
                         "counts. Set hi <= lo to disable and fall back to "
                         "a flat dark across all --nenergies bins.")
    ap.add_argument("--dark_roi_bin_hi", type=int, default=288,
                    help="High edge (exclusive, in energy bins) of the dark "
                         "background window. Bins at or above this get zero "
                         "dark counts.")
    ap.add_argument("--pool_max", type=int, default=50_000,
                    help="Cap on how many (kamp, kangle) samples to load "
                         "from the river file. Sampled with replacement.")
    ap.add_argument("--n_dev", type=int, default=0,
                    help="If >0, write exactly this many shots to "
                         "<outdir>/dev/streak_realistic_dev_00000.h5 and "
                         "skip the train/val/test splits. Biases sampling "
                         "toward high-kamp shots so streaked examples "
                         "actually appear (see --dev_kamp_min).")
    ap.add_argument("--dev_kamp_min", type=float, default=1.5,
                    help="When --n_dev is set, force every shot's kamp to "
                         "be drawn uniform on [dev_kamp_min, boost_max) sim "
                         "eV. Kangle still comes from the real river pool. "
                         "Default 1.5 eV, i.e. above the streak threshold.")
    ap.add_argument("--dev_outdir", type=str, default=None,
                    help="Override the dev-mode output directory. Default "
                         "is <outdir>/dev/.")
    ap.add_argument("--recipe", type=str, default="natural",
                    choices=["natural", "streak_5050",
                             "zero_pulse_5050", "pulse_1234plus_2525",
                             "pulse_123_1of3"],
                    help="Class-mixture recipe. Overrides per-shot draws "
                         "deterministically so the dataset on disk has a "
                         "chosen streak/npulses balance. 'natural' keeps "
                         "the pre-recipe behavior. 'pulse_123_1of3' emits "
                         "1/2/>=3 pulses at exact 33/33/33, with "
                         "independent per-pulse phases; 2-pulse shots are "
                         "rejection-sampled to guarantee |Δφ| >= "
                         "--min_dphi_rad.")
    ap.add_argument("--min_dphi_rad", type=float, default=0.0,
                    help="Minimum wrapped phase separation |arccos(cos Δφ)| "
                         "(rad) for 2-pulse shots when the recipe uses "
                         "independent per-pulse phases. 0 disables the "
                         "gate. pi/4 ~= 0.785 rad gives visibly-distinct "
                         "streak arcs; rejection rate ~25 percent.")
    ap.add_argument("--write_ypdf", action="store_true",
                    help="Also write each shot's clean 2D PDF (from build_XY's "
                         "ymat, per-shot MinMax to [0, 1]) as dataset 'Ypdf'. "
                         "Needed for split_bottleneck_ae training where the AE "
                         "decoder reconstructs Ypdf; leave off for the "
                         "streak_finder / 0-or-1 pipelines that only read Ximg.")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    # The river's duck_kamp is in a different unit than sim `kickstrength`
    # (max ~0.05 vs sim eV where 1-2 eV = streaked). Normal-mode uses
    # p_boost=0.3 so a fraction of shots get a synthetic kamp uniform in
    # [boost_min, boost_max] with kangle from the real pool. For dev mode
    # we force p_boost=1.0 and frac_unstreaked=0.0 so every shot is
    # streaked; --dev_kamp_min sets the boost band's lower edge in sim eV.
    kamp_pool, kangle_pool = _load_river_kicks(args.river_path, min_kamp=0.0)
    if kamp_pool.size == 0:
        raise RuntimeError("river pool is empty after filtering non-finite values")
    if kamp_pool.size > args.pool_max:
        rng = np.random.default_rng(args.base_seed)
        idx = rng.choice(kamp_pool.size, size=args.pool_max, replace=False)
        kamp_pool = kamp_pool[idx]
        kangle_pool = kangle_pool[idx]
    print("river pool: %d shots  kamp median=%.3f p95=%.3f p99=%.3f"
          % (kamp_pool.size, np.median(kamp_pool),
             np.percentile(kamp_pool, 95), np.percentile(kamp_pool, 99)))
    print("            kangle range [%.2f, %.2f] rad  (%.1f, %.1f deg)"
          % (kangle_pool.min(), kangle_pool.max(),
             np.degrees(kangle_pool.min()), np.degrees(kangle_pool.max())))

    cfg = GenConfig(
        nangles=args.nangles,
        nenergies=args.nenergies,
        drawscale=1.0,
        secondaryscale=args.secondaryscale,
        e_center_min=args.e_center_min,
        e_center_max=args.e_center_max,
        sase_width_min=args.sase_width_min,
        sase_width_max=args.sase_width_max,
        ce_var=args.ce_var,
        dark_min=args.dark_min,
        dark_max=args.dark_max,
        kamp_pool=kamp_pool,
        kangle_pool=kangle_pool,
        boost_min=args.boost_min,
        boost_max=args.boost_max,
        p_boost=args.p_boost,
        frac_unstreaked=args.frac_unstreaked,
        per_port_e_jitter_bins=args.per_port_e_jitter_bins,
        streak_threshold_eV=args.streak_threshold_eV,
        high_pulse_min=args.high_pulse_min,
        high_pulse_max=args.high_pulse_max,
        dead_time_eV=args.dead_time_eV,
        loc_noise_rate=args.loc_noise_rate,
        loc_noise_sigma_bins=args.loc_noise_sigma_bins,
        dark_roi_bin_lo=args.dark_roi_bin_lo,
        dark_roi_bin_hi=args.dark_roi_bin_hi,
        recipe=args.recipe,
        min_dphi_rad=args.min_dphi_rad,
    )

    ds = calibrate_drawscale(cfg, args.target_hits, args.base_seed)
    cfg = GenConfig(**{**cfg.__dict__, "drawscale": ds})

    if args.dry_run:
        print("dry_run: 8 shots serially, no files written")
        for i in range(8):
            img, attrs = sample_shot(args.base_seed + i, cfg)
            print("  shot %d: kamp=%.3f kang=%.2f rad  npulses=%d  hits=%d"
                  % (i, float(attrs["streak_amplitude"]),
                     float(attrs["kickangle"]),
                     int(attrs["npulses"]), int(img.sum())))
        return

    os.makedirs(args.outdir, exist_ok=True)
    print("outdir       : %s" % args.outdir)
    print("drawscale    : %.4f  (target %.1f hits/shot)"
          % (cfg.drawscale, args.target_hits))
    print("e_jitter     : sigma=%.2f bins" % cfg.per_port_e_jitter_bins)
    print("dead_time    : %.3f eV" % cfg.dead_time_eV)
    print("loc_noise    : rate=%.3f  sigma=%.2f bins"
          % (cfg.loc_noise_rate, cfg.loc_noise_sigma_bins))
    if 0 <= cfg.dark_roi_bin_lo < cfg.dark_roi_bin_hi <= cfg.nenergies:
        print("dark window : bins [%d, %d)  (dark=0 outside)"
              % (cfg.dark_roi_bin_lo, cfg.dark_roi_bin_hi))
    else:
        print("dark window : disabled (flat dark on all %d bins)"
              % cfg.nenergies)
    print("streak_thr   : %.2f eV" % cfg.streak_threshold_eV)
    print("frac_unstrk  : %.2f" % cfg.frac_unstreaked)
    print("boost band   : [%.2f, %.2f) at p_boost=%.2f (rest from real pool)"
          % (cfg.boost_min, cfg.boost_max, cfg.p_boost))
    print("recipe       : %s" % cfg.recipe)
    if cfg.recipe == "pulse_123_1of3":
        print("min_dphi_rad : %.4f rad  (2-pulse rejection gate)"
              % cfg.min_dphi_rad)
    print("write_ypdf   : %s" % ("yes" if args.write_ypdf else "no"))

    if args.n_dev > 0:
        dev_dir = args.dev_outdir or os.path.join(args.outdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        # Force every shot into the boost band and disable the unstreaked
        # fraction so all 20 dev outputs are streaked examples worth
        # eyeballing. dev_kamp_min sets the boost band's lower edge; the
        # upper edge is whatever --boost_max was (default 2.0 eV).
        dev_boost_min = args.dev_kamp_min
        dev_boost_max = max(args.boost_max, args.dev_kamp_min + 0.5)
        dev_cfg = GenConfig(**{**cfg.__dict__,
                               "boost_min": dev_boost_min,
                               "boost_max": dev_boost_max,
                               "p_boost": 1.0,
                               "frac_unstreaked": 0.0})
        print("dev mode     : n=%d shots -> %s"
              % (args.n_dev, dev_dir))
        print("dev boost    : kamp uniform in [%.2f, %.2f) eV, "
              "frac_unstreaked=0, p_boost=1"
              % (dev_boost_min, dev_boost_max))
        # One shard is enough for 20-ish shots; keeps the notebook loader
        # simple (single file to point at).
        _run_split("dev", args.n_dev, args.base_seed, dev_dir, dev_cfg,
                   n_workers=1, shots_per_shard=args.n_dev, flat=True,
                   write_ypdf=args.write_ypdf)
        return

    print("splits       : train=%d val=%d test=%d"
          % (args.n_train, args.n_val, args.n_test))

    seed_train = args.base_seed
    seed_val = seed_train + args.n_train
    seed_test = seed_val + args.n_val
    for split_name, n_shots, split_seed in (
        ("train", args.n_train, seed_train),
        ("val",   args.n_val,   seed_val),
        ("test",  args.n_test,  seed_test),
    ):
        if n_shots <= 0:
            continue
        print("split %s  n=%d  seed=%d" % (split_name, n_shots, split_seed))
        _run_split(split_name, n_shots, split_seed, args.outdir, cfg,
                   args.n_workers, args.shots_per_shard,
                   write_ypdf=args.write_ypdf)


if __name__ == "__main__":
    main()
