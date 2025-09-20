#!/bin/bash -l
#SBATCH --job-name=gen-ba4000-500
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o slurm-generate-ba4000-%j.out

# Set reasonable threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# Activate environment
conda activate oc-opencrispr

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug19/run-20250819-220243/huggingface/ba4000}
CFG=${CFG:-generation/generate_500_bs36.yml}

# New storage layout: atlas/generated_sequences/<run-tag>-<ba>-<seed>
GEN_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences
mkdir -p "$GEN_ROOT"
MP_NOSLASH=${MODEL_PATH%/}
MODEL_TAG=${MP_NOSLASH##*/}
RUN_TAG=$(echo "$MODEL_PATH" | sed -E 's|.*/(run-[^/]+)/huggingface/.*|\1|')
SEED_TAG=$(basename "$CFG" | sed -n 's/.*seed\([0-9]\+\).*/seed\1/p')
SEED_TAG=${SEED_TAG:-seed0}
SAVE_FOLDER="$GEN_ROOT/${RUN_TAG}-${MODEL_TAG}-${SEED_TAG}"
mkdir -p "$SAVE_FOLDER"

echo "[Run] Generating (cfg=$(basename "$CFG")) for ${MODEL_TAG} -> $SAVE_FOLDER on $(hostname)"
python -u generate.py \
  --model-path "$MODEL_PATH" \
  --config "$CFG" \
  --save-folder "$SAVE_FOLDER"

echo "[Done] Output CSV under $SAVE_FOLDER/generations/${MODEL_TAG}/$(basename "${CFG%.*}").csv"
