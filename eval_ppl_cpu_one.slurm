#!/bin/bash -l
#SBATCH -J eval_ppl_cpu_one
#SBATCH -p day
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CUDA_VISIBLE_DEVICES=""

# Expected env vars from sbatch --export
: "${HF_DIR:?HF_DIR is required}"
: "${RUN_BASE:?RUN_BASE is required}"
: "${STEP:?STEP is required}"
DATA_DIR=${DATA_DIR:-"/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv/"}
WANDP=${WANDP:-crispr12a}
WANDE=${WANDE:-eqk3}
BATCH=${BATCH:-4}
BASELINE_MODEL=${BASELINE_MODEL:-}

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env (CPU): $(python -c 'import sys;print(sys.prefix)')"

# Move to project
cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# W&B API key (assumes prior login; env available for non-interactive)
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

OUT_DIR=${OUT_DIR:-"eval_outputs_cpu_aug19/${RUN_BASE}_${STEP}"}
RUN_NAME=${RUN_NAME:-"eval_cpu_aug19_${RUN_BASE}_${STEP}"}

echo "[eval][CPU] RUN_BASE=${RUN_BASE} STEP=${STEP} HF_DIR=${HF_DIR} BASELINE=${BASELINE_MODEL:-<none>}"
python eval_perplexity.py \
  --model-path "${HF_DIR}" \
  --data-dir "${DATA_DIR}" \
  --batch-size ${BATCH} \
  --output-dir "${OUT_DIR}" \
  --wandb-project "${WANDP}" \
  --wandb-entity "${WANDE}" \
  --wandb-run-name "${RUN_NAME}" \
  ${BASELINE_MODEL:+--baseline-model-path} ${BASELINE_MODEL:+"${BASELINE_MODEL}"} \
  ${EXTRA_ARGS:-}

echo "[done] ${RUN_NAME}"
