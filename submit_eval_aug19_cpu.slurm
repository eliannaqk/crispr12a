#!/bin/bash -l
#SBATCH -J eval_ppl_aug19_cpu
#SBATCH -p day
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
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

# W&B setup
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

# Inputs
MODEL_ROOT="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug19"
DATA_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv/"
WANDP="crispr12a"
WANDE="eqk3"
BATCH=4

echo "[info] CPU eval: scanning models under ${MODEL_ROOT} (pick first/middle/last HF exports)"

shopt -s nullglob
for RUN_DIR in "${MODEL_ROOT}"/run-*; do
  [[ -d "${RUN_DIR}" ]] || continue
  RUN_BASE=$(basename "${RUN_DIR}")
  mapfile -t STEPS < <(find "${RUN_DIR}/huggingface" -maxdepth 1 -mindepth 1 -type d -name 'ba*' 2>/dev/null | sort -V)
  N=${#STEPS[@]}
  if (( N == 0 )); then
    echo "[skip] No HF export found in ${RUN_DIR}"
    continue
  fi
  # Indices to evaluate: first, middle, last (dedup automatically by associative array)
  declare -A PICK=()
  PICK[0]=1
  if (( N >= 3 )); then
    MID=$(( N / 2 ))
    (( MID >= N )) && MID=$(( N - 1 ))
    PICK[${MID}]=1
  fi
  PICK[$(( N - 1 ))]=1

  for IDX in "${!PICK[@]}"; do
    HF_DIR=${STEPS[$IDX]}
    STEP=$(basename "${HF_DIR}")
    OUT_DIR="eval_outputs_cpu_aug19/${RUN_BASE}_${STEP}"
    RUN_NAME="eval_cpu_aug19_${RUN_BASE}_${STEP}"
    echo "[eval][CPU] ${RUN_BASE} -> ${HF_DIR} (run: ${RUN_NAME})"

    python eval_perplexity.py \
      --model-path "${HF_DIR}" \
      --data-dir "${DATA_DIR}" \
      --batch-size ${BATCH} \
      --output-dir "${OUT_DIR}" \
      --wandb-project "${WANDP}" \
      --wandb-entity "${WANDE}" \
      --wandb-run-name "${RUN_NAME}" || {
        echo "[error] CPU evaluation failed for ${RUN_DIR} @ ${STEP}"
      }
  done
done

echo "[done] CPU evaluations attempted for aug19."

