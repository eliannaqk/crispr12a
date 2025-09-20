#!/bin/bash -l
#SBATCH --job-name=gen-ba6000
#SBATCH -A pi_mg269                        # Account
#SBATCH --partition=gpu_h200               # Bouchet H200 partition
#SBATCH --gres=gpu:h200:1                # Default: 1x H200 (override with: sbatch --gres=gpu:h200:N ...)
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=16
#SBATCH -t 16:00:00
#SBATCH -o slurm-generate-ba6000-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

# Model and config
MODEL_PATH=${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/huggingface/ba6000}
CFG=${CFG:-generation/generate_10k.yml}

# New storage layout under atlas/generated_sequences/<run-tag>-<model-tag>-<seed-tag>
GEN_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences
mkdir -p "$GEN_ROOT"

# Derive tags from MODEL_PATH
MP_NOSLASH=${MODEL_PATH%/}
MODEL_TAG=${MP_NOSLASH##*/}
RUN_TAG=$(echo "$MODEL_PATH" | sed -E 's|.*/(run-[^/]+)/huggingface/.*|\1|')
# Seed tag inferred from config filename, default seed0
SEED_TAG=$(basename "$CFG" | sed -n 's/.*seed\([0-9]\+\).*/seed\1/p')
SEED_TAG=${SEED_TAG:-seed0}

SAVE_FOLDER="$GEN_ROOT/${RUN_TAG}-${MODEL_TAG}-${SEED_TAG}"
mkdir -p "$SAVE_FOLDER"

echo "[Run] $(date) host=$(hostname) model=${MODEL_TAG} run=${RUN_NUM} cfg=$(basename "$CFG") -> $SAVE_FOLDER"
python -u generate.py \
  --model-path "$MODEL_PATH" \
  --config "$CFG" \
  --save-folder "$SAVE_FOLDER"

echo "[Done] Outputs under $SAVE_FOLDER/generations/${MODEL_TAG}/"
