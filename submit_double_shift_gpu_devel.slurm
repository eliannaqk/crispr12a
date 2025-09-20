#!/bin/bash -l
# Run eval-only with DoubleShiftDetector callback to detect label shifting
#SBATCH -J double-shift-detector
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
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

# Use the same FinetuneAPI YAML you train with
CFG=${CFG:-cas12a_ft_finetuneapi.yaml}

echo "[run] Evaluating with DoubleShiftDetector (config=$CFG)"
python scripts/run_eval_with_double_shift.py --config "$CFG"

echo "[done] Double shift detection eval completed"

