#!/bin/bash -l
#SBATCH -J eval_gpu_devel_baseline_lbcas40
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

echo "[env] Using conda env (GPU): $(conda info --json | jq -r '.active_prefix_name')"
cd ~/crispr12/opencrispr-repro-main

echo "[cwd] $(pwd)"
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

# Baseline model (ProGen2 HF export)
BASELINE_MODEL="/home/eqk3/project_pi_mg269/eqk3/crisprData/model/zenodo_15128064"
DATA_DIR="$(pwd)/datasets/lbcas40"  # uses symlinks to train_40.csv + val_40.csv
WANDP="crispr12a"
WANDE="eqk3"
BATCH=${BATCH:-16}
OUT_DIR="eval_outputs_gpu_aug21/baseline_lbcas40"
RUN_NAME="eval_gpu_devel_baseline_lbcas40"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[error] DATA_DIR not found: ${DATA_DIR}" >&2
  exit 1
fi

python eval_perplexity.py \
  --model-path "${BASELINE_MODEL}" \
  --data-dir "${DATA_DIR}" \
  --batch-size ${BATCH} \
  --output-dir "${OUT_DIR}" \
  --wandb-project "${WANDP}" \
  --wandb-entity "${WANDE}" \
  --wandb-run-name "${RUN_NAME}" \
  --only-val-and-train-sample

echo "[done] Baseline eval completed. Output: ${OUT_DIR}"
