#!/bin/bash -l
#SBATCH -J eval_ppl_aug18_cpu
#SBATCH -p day
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CUDA_VISIBLE_DEVICES=""

# Activate approved environment (CPU-only)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env (CPU): $(python -c 'import sys;print(sys.prefix)')"

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
BATCH=4

echo "[info] CPU eval: scanning models under ${MODEL_ROOT}"

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
  OUT_DIR="eval_outputs_cpu/${RUN_BASE}_${STEP}"
  RUN_NAME="eval_cpu_${RUN_BASE}_${STEP}"
  echo "[eval][CPU] ${RUN_BASE} -> ${HF_DIR} (run: ${RUN_NAME})"

  python eval_perplexity.py \
    --model-path "${HF_DIR}" \
    --data-dir "${DATA_DIR}" \
    --batch-size ${BATCH} \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDP}" \
    --wandb-entity "${WANDE}" \
    --wandb-run-name "${RUN_NAME}" || {
      echo "[error] CPU evaluation failed for ${RUN_DIR}"
    }
done

echo "[done] CPU evaluations attempted."

