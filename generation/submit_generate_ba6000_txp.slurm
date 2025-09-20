#!/bin/bash -l
#SBATCH --job-name=gen-ba6000-txp
#SBATCH -A pi_mg269                        # Account
#SBATCH --partition=gpu_h200               # H200 partition
#SBATCH --gres=gpu:h200:1                  # 1x H200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH -o slurm-generate-ba6000-txp-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ENV NOTE: using approved oc-opencrispr conda environment
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

# Adjust these paths as needed
MODEL_PATH=${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-111648/huggingface/ba8000/}
CFG=${CFG:-generation/generate_50k_txp.yml}

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

echo "[Run] Generating TXP combos for ${MODEL_TAG} -> $SAVE_FOLDER on $(hostname)"
python -u generate.py \
  --model-path "$MODEL_PATH" \
  --config "$CFG" \
  --save-folder "$SAVE_FOLDER"

echo "[Done] See CSV under $SAVE_FOLDER/generations/${MODEL_TAG}/$(basename "${CFG%.*}").csv"
