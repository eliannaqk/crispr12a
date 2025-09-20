#!/bin/bash -l
# Evaluate ba4000 model on lbcas40 valid.csv (and train sample) using oc-opencrispr.

#SBATCH -J eval_ba4000_lbcas40
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate approved environment (GPU-ready)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env (GPU): ${CONDA_DEFAULT_ENV}"

cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# Optional W&B (token already provided in guidelines)
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

# Model used in recent generation + ppl-filter runs
MODEL_PATH="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250828-193254/huggingface/ba4000"
DATA_DIR="$(pwd)/datasets/lbcas40"
OUT_DIR="eval_outputs/ba4000_lbcas40"
RUN_NAME="eval_ba4000_lbcas40"
BATCH=${BATCH:-32}

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[error] DATA_DIR not found: ${DATA_DIR}" >&2
  exit 1
fi

python eval_perplexity.py \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --batch-size ${BATCH} \
  --output-dir "${OUT_DIR}" \
  --wandb-project "crispr12a" \
  --wandb-entity "eqk3" \
  --wandb-run-name "${RUN_NAME}" \
  --only-val-and-train-sample

echo "[done] Eval completed. Output: ${OUT_DIR}"

