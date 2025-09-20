#!/bin/bash -l
#SBATCH -J eval_gpu_run_20250822_144424
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate approved environment (GPU-ready)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env (GPU): $(python -c 'import sys;print(sys.prefix)')"

# Move to project directory
cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# W&B (token assumed cached; export here to be safe for non-interactive)
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

# Config
MODEL_ROOT="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21"
RUN_ID="run-20250822-144424"
DATA_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv2"
WANDP="crispr12a"
WANDE="eqk3"
BATCH=${BATCH:-16}
OUT_BASE="eval_outputs_gpu_aug21"

RUN_SHORT="${RUN_ID#run-}"
RUN_DIR="${MODEL_ROOT}/${RUN_ID}"
if [[ ! -d "${RUN_DIR}/huggingface" ]]; then
  echo "[error] Missing huggingface exports in ${RUN_DIR}" >&2
  exit 1
fi

STEPS=(1000 2000 3000 4000 6000)
echo "[info] Evaluating on GPU for ${RUN_ID}: steps ${STEPS[*]}"

for STEP in "${STEPS[@]}"; do
  HF_DIR="${RUN_DIR}/huggingface/ba${STEP}"
  if [[ ! -d "${HF_DIR}" ]]; then
    echo "[warn] Skip step ${STEP}: ${HF_DIR} not found"
    continue
  fi
  OUT_DIR="${OUT_BASE}/${RUN_SHORT}_${STEP}"
  RUN_NAME="eval_gpu_aug21_${RUN_SHORT}_${STEP}"
  echo "[eval][GPU] ${RUN_ID} step ${STEP} -> ${HF_DIR}"

  python eval_perplexity.py \
    --model-path "${HF_DIR}" \
    --data-dir "${DATA_DIR}" \
    --batch-size ${BATCH} \
    --output-dir "${OUT_DIR}" \
    --wandb-project "${WANDP}" \
    --wandb-entity "${WANDE}" \
    --wandb-run-name "${RUN_NAME}" \
    --only-val-and-train-sample || {
      echo "[error] Eval failed for step ${STEP} (${HF_DIR})" >&2
    }
done

echo "[done] GPU evals completed for ${RUN_ID}"

