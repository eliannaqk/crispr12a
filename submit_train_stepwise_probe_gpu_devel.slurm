#!/bin/bash -l
# Stepwise training probe: log per-batch Composer, HF, and manual CE losses
#SBATCH -J train-step-probe
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
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
STEPS=${STEPS:-10}
echo "[run] TrainStepwiseProbe ${STEPS} batches (config=$CFG) ${EXTRA_ARGS:-}"
python scripts/run_train_stepwise_probe.py --config "$CFG" --steps "$STEPS" ${EXTRA_ARGS:-}

echo "[done] Train stepwise probe completed"
