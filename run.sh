#!/bin/bash
#SBATCH --job-name=sam3seg
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/worker_%a.out
#SBATCH --error=logs/worker_%a.err
#
# 4 workers en file d'attente partagée (claim atomique, reprenable).
# Resoumettre `sbatch run.sh` pour reprendre après un kill (48 h walltime).
#
# ================= À ADAPTER =================
export DATASET_ROOT=/path/to/processed_files_ALL_frames   # racine dataset (patient/puits/cavite.tif)
CONDA_ENV=sam3                                            # nom de l'env conda
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}       # cache des poids SAM3
# ============================================

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p logs
source ~/.bashrc
conda activate "$CONDA_ENV"
python -u run_segmentation.py --queue --worker-id ${SLURM_ARRAY_TASK_ID:-0}
