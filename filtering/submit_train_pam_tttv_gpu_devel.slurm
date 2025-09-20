#!/bin/bash -l
#SBATCH -J pamtttv_train
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate oc-opencrispr environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate oc-opencrispr
echo "[env] Using conda env: $(conda info --envs | awk '/\*/{print $1}')"

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

# W&B setup (expects prior wandb login; do not hardcode keys)
export WANDB_PROJECT=crispr_filtering
# Optionally set your entity if needed (e.g., eqk3)
export WANDB_ENTITY=${WANDB_ENTITY:-}

# Data paths
TRAIN_CSV=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna/pampredict_aggregate_20250909_103217/pam_tttv_train.csv
VALID_CSV=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna/pampredict_aggregate_20250909_103217/pam_tttv_valid.csv
OUT_MODEL=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna/pampredict_aggregate_20250909_103217/pam_tttv_esm2_t33.pt

python filtering/pam_filtering_esm/train_pam_tttv_classifier_esm.py \
  --train_csv "$TRAIN_CSV" \
  --valid_csv "$VALID_CSV" \
  --out_model "$OUT_MODEL" \
  --epochs 6 \
  --batch_size 32 \
  --lr 1e-3 \
  --wandb-project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
  --wandb-run-name "pam_tttv_esm2_t33_${SLURM_JOB_ID}"
