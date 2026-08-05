#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=split_bottleneck_ae
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# ----------------------------------------------------------------------------
# Train the split-bottleneck autoencoder + count / phase heads.
#
# All hyperparameters live in config.py. This launcher only wires up the
# SLURM environment and cd's into the project directory. Ampere GPU
# partition; 32 GB RAM to stay clear of prior OOMs on this cluster.
# ----------------------------------------------------------------------------

echo "starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

PROJ_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/split_bottleneck_ae
cd "$PROJ_DIR" || exit 1

python3 train.py

echo "finished at: $(date)"
