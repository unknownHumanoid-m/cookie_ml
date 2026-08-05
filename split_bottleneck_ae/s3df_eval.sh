#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=split_bottleneck_ae_eval
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-04:00:00
#SBATCH --gpus=1

# ----------------------------------------------------------------------------
# Evaluate a trained split-bottleneck AE checkpoint.
#
# By default, eval.py loads $IO['save_dir']/$IO['run_name']/model.pth from
# config.py. Override the checkpoint by passing it as the first arg:
#   sbatch s3df_eval.sh /path/to/other/model.pth
# ----------------------------------------------------------------------------

echo "starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

PROJ_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/split_bottleneck_ae
cd "$PROJ_DIR" || exit 1

CKPT="${1:-}"
if [ -n "$CKPT" ]; then
    python3 eval.py --ckpt "$CKPT"
else
    python3 eval.py
fi

echo "finished at: $(date)"
