#!/bin/bash -l
# Stepwise training probe on H200: log per-batch Composer, HF, and manual CE losses
#SBATCH -J train-step-probe-h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] Initializing Conda and activating oc-opencrispr"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Active env: $(python -c 'import sys;print(sys.prefix)')"

cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

CFG=${CFG:-cas12a_ft_finetuneapi.yaml}
STEPS=${STEPS:-200}
EXTRA_ARGS=${EXTRA_ARGS:---preserve-config}
# If RUN_ID is provided, auto-pick latest HF checkpoint under that run
if [[ -n "${RUN_ID:-}" ]]; then
  BASE=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21
  HF_DIR=$(ls -1d ${BASE}/${RUN_ID}/huggingface/ba* 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "$HF_DIR" ]]; then
    echo "[ckpt] Using HF checkpoint: $HF_DIR"
    EXTRA_ARGS+=" --hf-path $HF_DIR"
  else
    echo "[ckpt] No HF checkpoints found for run $RUN_ID under $BASE" >&2
  fi
fi

echo "[run] TrainStepwiseProbe ${STEPS} batches (config=$CFG) ${EXTRA_ARGS}"
python scripts/run_train_stepwise_probe.py --config "$CFG" --steps "$STEPS" ${EXTRA_ARGS}

echo "[done] Train stepwise probe completed"
