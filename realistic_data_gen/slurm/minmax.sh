#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=lcls:tmox42619
#SBATCH --job-name=minmax_realistic
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-01:00:00

# ----------------------------------------------------------------------------
# Post-process the realistic-generator dataset (split_bottleneck_ae_real_data)
# with a corpus-column MinMax(0,1) scaler, matching what
# src/data_processing/universal_cookiesimslim_processor.py does for the
# sibling forks.
#
# Three stages:
#   fit       : partial_fit MinMaxScaler on every shard in <SRC_ROOT>/train
#   transform : scale train / val / test shards, writing to <OUT_ROOT>/{split}/
#
# The scaler joblib is saved once at <OUT_ROOT>/min_max_scaler_realistic_ximg.joblib
# and reused for val / test / real inference (no per-split refit).
# ----------------------------------------------------------------------------

SRC_ROOT="${SRC_ROOT:-/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/split_bottleneck_ae_real_data}"
OUT_ROOT="${OUT_ROOT:-/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/split_bottleneck_ae_real_data_minmax}"
SCALER_PATH="${SCALER_PATH:-${OUT_ROOT}/min_max_scaler_realistic_ximg.joblib}"

mkdir -p /sdf/home/m/miaed/slurm_logs
mkdir -p "$OUT_ROOT"

# Resolve realistic_data/ (parent of this slurm/ dir) so the script runs
# from a fresh checkout without needing a hard-coded absolute path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "starting run at: $(date)"
echo "hostname       : $(hostname)"
echo "SRC_ROOT       : $SRC_ROOT"
echo "OUT_ROOT       : $OUT_ROOT"
echo "SCALER_PATH    : $SCALER_PATH"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# ----------------------------------------------------------------------------
# Stage 1: fit
# ----------------------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "fit: MinMax(0,1) per-column across ${SRC_ROOT}/train"
echo "----------------------------------------------------------------"
python3 "$SCRIPT_DIR/postprocess/minmax_shards.py" fit \
    --in_dir "${SRC_ROOT}/train" \
    --scaler_path "$SCALER_PATH" \
    || { echo "fit failed"; exit 1; }

# ----------------------------------------------------------------------------
# Stage 2: transform train / val / test
# Each split's --in_dir/--out_dir is inline on the python call so nothing
# leaks between stages (see feedback-inline-env-vars).
# ----------------------------------------------------------------------------
for SPLIT in train val test; do
    if [ ! -d "${SRC_ROOT}/${SPLIT}" ]; then
        echo "skip: ${SRC_ROOT}/${SPLIT} not present"
        continue
    fi
    echo "----------------------------------------------------------------"
    echo "transform: ${SRC_ROOT}/${SPLIT}  ->  ${OUT_ROOT}/${SPLIT}"
    echo "----------------------------------------------------------------"
    python3 "$SCRIPT_DIR/postprocess/minmax_shards.py" transform \
        --in_dir "${SRC_ROOT}/${SPLIT}" \
        --out_dir "${OUT_ROOT}/${SPLIT}" \
        --scaler_path "$SCALER_PATH" \
        || { echo "transform of $SPLIT failed"; exit 1; }
done

echo "finished at: $(date)"
