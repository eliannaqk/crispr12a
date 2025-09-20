#!/bin/bash -l
# GPU eval for run 20250822-202945 with baseline comparisons; adds ba8000 step
# Uses oc-opencrispr env per repo policy
#SBATCH -J eval_gpu_run_20250822_202945
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] Initializing Conda and activating oc-opencrispr (GPU)"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Active env: $(python -c 'import sys;print(sys.prefix)')"

cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

# Config
MODEL_ROOT="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21"
RUN_ID="run-20250822-202945"
DATA_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv2"
BASELINE_MODEL="/home/eqk3/project_pi_mg269/eqk3/crisprData/model/zenodo_15128064/"
WANDP="crispr12a"
WANDE="eqk3"
BATCH=${BATCH:-16}
OUT_BASE="eval_outputs_gpu_aug21"

RUN_SHORT="${RUN_ID#run-}"
RUN_DIR="${MODEL_ROOT}/${RUN_ID}"
if [[ ! -d "${RUN_DIR}/huggingface" ]]; then
  echo "[error] Missing huggingface exports in ${RUN_DIR}" >&2
  echo "[hint] Available runs under ${MODEL_ROOT}:"
  ls -1 "${MODEL_ROOT}" | sed -n '1,50p' || true
  exit 1
fi

if [[ ! -d "${BASELINE_MODEL}" ]]; then
  echo "[error] Baseline model not found at ${BASELINE_MODEL}" >&2
  exit 1
fi

# Add ba8000 to the existing sweep
STEPS=(1000 2000 3000 4000 6000 8000)
echo "[info] Evaluating ${RUN_ID} on GPU; steps: ${STEPS[*]}"

for STEP in "${STEPS[@]}"; do
  HF_DIR="${RUN_DIR}/huggingface/ba${STEP}"
  if [[ ! -d "${HF_DIR}" ]]; then
    echo "[warn] Skip step ${STEP}: ${HF_DIR} not found"
    continue
  fi
  OUT_DIR="${OUT_BASE}/${RUN_SHORT}_${STEP}"
  RUN_NAME="eval_gpu_aug22_${RUN_SHORT}_${STEP}_with_base"
  echo "[eval][GPU] ${RUN_ID} step ${STEP} -> ${HF_DIR} (baseline included)"

  python eval_perplexity.py \
    --model-path "${HF_DIR}" \
    --baseline-model-path "${BASELINE_MODEL}" \
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

echo "[done] GPU evals with baseline completed for ${RUN_ID}"

