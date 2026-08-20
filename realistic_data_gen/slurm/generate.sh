#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:tmox42619
#SBATCH --job-name=streak_gen_real_data
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4g
#SBATCH --time=0-02:00:00

# ----------------------------------------------------------------------------
# split_bottleneck_ae_real_data -- 33/33/33 for npulses = 1/2/>=3, with
# INDEPENDENT per-pulse phases and a minimum wrapped Δφ >= pi/4 on 2-pulse
# shots so the two streak arcs are visibly distinct.
#
# Uses generate_streak_data_realistic.py --recipe pulse_123_1of3
# --min_dphi_rad $(python -c "import math; print(math.pi/4)")
#
# Why this exists (vs the old split_bottleneck_ae_realistic_sharp dataset):
#   * Old recipe forced every pulse's phase to kangle (see
#     generate_streak_data_realistic.py `forced_phases = [kangle]*ncenters`),
#     so 2-pulse Δφ was exactly 0 and 3+ pulse shots had zero angular
#     separation. That killed both count classification (n>=2 all looked
#     like one blob) and the two-pulse phase regression (target degenerate).
#   * New recipe leaves build_XY's random per-pulse phase draw untouched
#     and rejection-samples 2-pulse shots until |arccos(cos Δφ)| >= pi/4.
#   * 3+ pulse shots are all clamped into the same ">=3" class downstream,
#     so their internal separation isn't a training signal here.
#
# Output shape per shot: Ximg (16, 512). Attrs include npulses, phases (in
# RADIANS), streak_amplitude, kickstrength, kickangle. --write_ypdf is on
# so the AE's decoder has a reconstruction target.
#
# Usage / overrides identical to s3df_generate_split_bottleneck_ae_realistic.sh.
# ----------------------------------------------------------------------------

export OUT_DIR="${OUT_DIR:-/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/split_bottleneck_ae_real_data}"

export N_TRAIN="${N_TRAIN:-200000}"
export N_VAL="${N_VAL:-20000}"
export N_TEST="${N_TEST:-20000}"

export N_WORKERS="${N_WORKERS:-24}"
export SHOTS_PER_SHARD="${SHOTS_PER_SHARD:-100}"
export BASE_SEED="${BASE_SEED:-0}"

export NANGLES="${NANGLES:-16}"
export NENERGIES="${NENERGIES:-512}"

export TARGET_HITS="${TARGET_HITS:-130}"
export DEAD_TIME_EV="${DEAD_TIME_EV:-3.0}"
export LOC_NOISE_RATE="${LOC_NOISE_RATE:-1.0}"
export LOC_NOISE_SIGMA_BINS="${LOC_NOISE_SIGMA_BINS:-2}"
export DARK_MIN="${DARK_MIN:-0.02}"
export DARK_MAX="${DARK_MAX:-0.05}"
export DARK_ROI_BIN_LO="${DARK_ROI_BIN_LO:-80}"
export DARK_ROI_BIN_HI="${DARK_ROI_BIN_HI:-400}"
export E_CENTER_MIN="${E_CENTER_MIN:-165}"
export E_CENTER_MAX="${E_CENTER_MAX:-175}"
export SASE_WIDTH_MIN="${SASE_WIDTH_MIN:-7.5}"
export SASE_WIDTH_MAX="${SASE_WIDTH_MAX:-7.5}"
export PER_PORT_E_JITTER_BINS="${PER_PORT_E_JITTER_BINS:-0.3}"
export CE_VAR="${CE_VAR:-3.0}"

# ">=3" bucket range. Uniform on [HIGH_PULSE_MIN, HIGH_PULSE_MAX]. The
# classifier collapses all of these into one class, so the exact range
# is a knob for how much intra-class variety we expose; keep sane
# defaults matching the old recipe.
export HIGH_PULSE_MIN="${HIGH_PULSE_MIN:-3}"
export HIGH_PULSE_MAX="${HIGH_PULSE_MAX:-6}"

export STREAK_THRESHOLD_EV="${STREAK_THRESHOLD_EV:-5.0}"
export BOOST_MIN="${BOOST_MIN:-2.5}"
export BOOST_MAX="${BOOST_MAX:-7.5}"
export P_BOOST="${P_BOOST:-0.3}"

export FRAC_UNSTREAKED="${FRAC_UNSTREAKED:-0.0}"

export RECIPE="pulse_123_1of3"

# Minimum wrapped |arccos(cos Δφ)| (rad) on 2-pulse shots. Default pi/4.
export MIN_DPHI_RAD="${MIN_DPHI_RAD:-0.7853981633974483}"

# ----------------------------------------------------------------------------
# End of user config.
# ----------------------------------------------------------------------------

MODE="${1:-run}"
DRY_FLAG=""

case "$MODE" in
    dry)
        DRY_FLAG="--dry_run"
        ;;
    smoke)
        N_TRAIN=2000
        N_VAL=200
        N_TEST=200
        SHOTS_PER_SHARD=100
        ;;
    run|"")
        ;;
    *)
        echo "unknown mode: $MODE (expected: run | dry | smoke)"
        exit 1
        ;;
esac

mkdir -p /sdf/home/m/miaed/slurm_logs
mkdir -p "$OUT_DIR"

# Resolve realistic_data/ (parent of this slurm/ dir) so the script runs
# from a fresh checkout without needing a hard-coded absolute path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "----------------------------------------------------------------"
echo "split_bottleneck_ae_real_data dataset (recipe=$RECIPE)"
echo "mode           : $MODE"
echo "out_dir        : $OUT_DIR"
echo "n_train/val/test: $N_TRAIN / $N_VAL / $N_TEST"
echo "workers x shard: $N_WORKERS x $SHOTS_PER_SHARD"
echo "image shape    : (${NANGLES}, ${NENERGIES})"
echo "sase width     : uniform in [$SASE_WIDTH_MIN, $SASE_WIDTH_MAX]"
echo "loc_noise      : rate=$LOC_NOISE_RATE  sigma=$LOC_NOISE_SIGMA_BINS bins"
echo "port e_jitter  : $PER_PORT_E_JITTER_BINS bins"
echo "e_center       : [$E_CENTER_MIN, $E_CENTER_MAX] (bin)"
echo "dark_min/max   : $DARK_MIN / $DARK_MAX"
echo "dark_roi       : [$DARK_ROI_BIN_LO, $DARK_ROI_BIN_HI)"
echo "boost band     : [$BOOST_MIN, $BOOST_MAX) eV  streak_thr=$STREAK_THRESHOLD_EV"
echo "high_pulse     : uniform in [$HIGH_PULSE_MIN, $HIGH_PULSE_MAX]"
echo "recipe         : $RECIPE"
echo "min_dphi_rad   : $MIN_DPHI_RAD  (~pi/4 = 0.7854)"
echo "----------------------------------------------------------------"

python3 "$SCRIPT_DIR/generate.py" \
    --outdir "$OUT_DIR" \
    --n_train "$N_TRAIN" \
    --n_val "$N_VAL" \
    --n_test "$N_TEST" \
    --nangles "$NANGLES" \
    --nenergies "$NENERGIES" \
    --shots_per_shard "$SHOTS_PER_SHARD" \
    --n_workers "$N_WORKERS" \
    --base_seed "$BASE_SEED" \
    --target_hits "$TARGET_HITS" \
    --dead_time_eV "$DEAD_TIME_EV" \
    --loc_noise_rate "$LOC_NOISE_RATE" \
    --loc_noise_sigma_bins "$LOC_NOISE_SIGMA_BINS" \
    --dark_min "$DARK_MIN" \
    --dark_max "$DARK_MAX" \
    --dark_roi_bin_lo "$DARK_ROI_BIN_LO" \
    --dark_roi_bin_hi "$DARK_ROI_BIN_HI" \
    --e_center_min "$E_CENTER_MIN" \
    --e_center_max "$E_CENTER_MAX" \
    --sase_width_min "$SASE_WIDTH_MIN" \
    --sase_width_max "$SASE_WIDTH_MAX" \
    --per_port_e_jitter_bins "$PER_PORT_E_JITTER_BINS" \
    --ce_var "$CE_VAR" \
    --high_pulse_min "$HIGH_PULSE_MIN" \
    --high_pulse_max "$HIGH_PULSE_MAX" \
    --boost_min "$BOOST_MIN" \
    --boost_max "$BOOST_MAX" \
    --p_boost "$P_BOOST" \
    --streak_threshold_eV "$STREAK_THRESHOLD_EV" \
    --frac_unstreaked "$FRAC_UNSTREAKED" \
    --recipe "$RECIPE" \
    --min_dphi_rad "$MIN_DPHI_RAD" \
    --write_ypdf \
    $DRY_FLAG
