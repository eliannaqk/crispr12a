#!/bin/bash -l
#SBATCH -J tttv_noncore_strict2strict_nosnr
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
echo "[slurm] Host: $(hostname)"
echo "[slurm] Start: $(date)"
echo "[slurm] Using conda env: oc-opencrispr"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate oc-opencrispr
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd ~/crispr12/opencrispr-repro-main

export WANDB_API_KEY="${WANDB_API_KEY:-fa3648edd4db4a0bbcf4157a516c81da9940463e}"

BASE=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna
AGG_DIR=${AGG_DIR:-${BASE}/pampredict_aggregate_20250911_131538}
TRAIN_CSV=${TRAIN_CSV:-${AGG_DIR}/pam_noncore_strict_train.csv}
VALID_CSV=${VALID_CSV:-${AGG_DIR}/pam_noncore_strict_valid.csv}
OUT_MODEL=${OUT_MODEL:-${AGG_DIR}/pam_noncore_strict2strict_nosnr_esm2_t33.pt}

echo "[slurm] TRAIN_CSV=${TRAIN_CSV}"
echo "[slurm] VALID_CSV=${VALID_CSV}"
echo "[slurm] OUT_MODEL=${OUT_MODEL}"

python filtering/train_pam_tttv_classifier_esm.py \
  --train_csv "${TRAIN_CSV}" \
  --valid_csv "${VALID_CSV}" \
  --out_model "${OUT_MODEL}" \
  --epochs 8 \
  --batch_size 32 \
  --lr 1e-3 \
  --early-stop-patience 3 \
  --wandb-project crispr_filtering \
  --wandb-entity eqk3 \
  --wandb-run-name noncore_strict2strict_nosnr_bcebal_${SLURM_JOB_ID:-local} \
  --pos-weight-scale 1.0 \
  --aux-snr-weight 0.0 \
  --bce-weighting balanced \
  --bce-neg-mult 1.0 \
  --bce-pos-mult 2.0

echo "[slurm] Done: $(date)"

