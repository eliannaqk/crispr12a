#!/bin/bash -l
#SBATCH -J lbdr_wandb_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Initialize Conda and env (policy)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

# Data paths (pre-built labeled splits)
TRAIN_CSV="lbdr_runs/dr_evidence_splits/train_labeled.csv"
VALID_CSV="lbdr_runs/dr_evidence_splits/valid_labeled.csv"
OUT_DIR="lbdr_runs/esm2_t33_wandb_h200"
mkdir -p "$OUT_DIR"

# If needed, ensure W&B is logged in beforehand, or set env vars prior to submit
echo "[env] $(conda info --envs | grep '*' || true)"; nvidia-smi || true

python filtering/train_lbdr_classifier_esm.py \
  --train_csv "$TRAIN_CSV" \
  --valid_csv "$VALID_CSV" \
  --out_model "$OUT_DIR/lbdr_esm2_t33.pt" \
  --epochs 6 \
  --batch_size 16 \
  --lr 1e-3 \
  --wandb-project crispr_filtering \
  --wandb-entity eqk3 \
  --wandb-run-name "lbdr_esm2_t33_h200_$(date +%Y%m%d_%H%M%S)"

echo "[done] Model at $OUT_DIR/lbdr_esm2_t33.pt"

