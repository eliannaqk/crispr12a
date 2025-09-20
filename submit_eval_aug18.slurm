#!/bin/bash -l
#SBATCH -J eval_ppl_aug18
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate approved environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env: $(python -c 'import sys;print(sys.prefix)')"

# Move to project directory
cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# W&B setup (prefer prior login; export here if needed)
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

# Inputs
MODEL_ROOT="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug18"
DATA_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv/"
WANDP="crispr12a"
WANDE="eqk3"
BATCH=8

echo "[info] Scanning models under: ${MODEL_ROOT}"

shopt -s nullglob
for RUN_DIR in "${MODEL_ROOT}"/run-*; do
  [[ -d "${RUN_DIR}" ]] || continue
  RUN_BASE=$(basename "${RUN_DIR}")
  HF_DIR=$(ls -1d "${RUN_DIR}"/huggingface/ba* 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -z "${HF_DIR}" ]]; then
    echo "[skip] No HF export found in ${RUN_DIR}"
    continue
  fi
  STEP=$(basename "${HF_DIR}")
  OUT_DIR="eval_outputs/${RUN_BASE}_${STEP}"
  RUN_NAME="eval_${RUN_BASE}_${STEP}"
  echo "[eval] ${RUN_BASE} -> ${HF_DIR} (run: ${RUN_NAME})"

  python eval_perplexity.py \
    --model-path "${HF_DIR}" \
    --data-dir "${DATA_DIR}" \
    --batch-size ${BATCH} \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDP}" \
    --wandb-entity "${WANDE}" \
    --wandb-run-name "${RUN_NAME}" || {
      echo "[error] Evaluation failed for ${RUN_DIR}"
    }
done

echo "[done] All evaluations attempted."

