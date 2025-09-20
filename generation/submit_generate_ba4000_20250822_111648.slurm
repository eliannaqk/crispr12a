#!/bin/bash -l
#SBATCH --job-name=gen-ba4000-1k-20250822_111648
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o slurm-generate-ba4000-1k-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate env
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

export TOKENIZERS_PARALLELISM=false

MODEL_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-111648/huggingface/ba4000"
SAVE_ROOT="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-111648/"

echo "[Run] Generating 1000 sequences for ba4000 on $(hostname)"
python -u generate.py \
  --model-path "$MODEL_DIR" \
  --config generation/generate_1k.yml \
  --save-folder "$SAVE_ROOT"

echo "[Done] Generation job finished. Output under: ${SAVE_ROOT}/generations/ba4000/"
