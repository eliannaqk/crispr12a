#!/bin/bash -l
#SBATCH --job-name=gen-ba6000-seed3
#SBATCH -A pi_mg269
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -o slurm-generate-ba6000-seed3-20250828_193252-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate oc-opencrispr
echo "[Env] Using conda env: oc-opencrispr"

cd ~/crispr12/opencrispr-repro-main

# Default model for this run tag (updated to existing run-20250828-193254 ba4000)
MODEL_PATH=${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250828-193254/huggingface/ba4000/}
CFG=${CFG:-generation/generate_50k_txp_seed3.yml}

GEN_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences
mkdir -p "$GEN_ROOT"
MP_NOSLASH=${MODEL_PATH%/}
MODEL_TAG=${MP_NOSLASH##*/}
RUN_TAG=$(echo "$MODEL_PATH" | sed -E 's|.*/(run-[^/]+)/huggingface/.*|\1|')
SEED_TAG=seed3
SAVE_FOLDER="$GEN_ROOT/${RUN_TAG}-${MODEL_TAG}-${SEED_TAG}"
mkdir -p "$SAVE_FOLDER"

echo "[Run] Generating TXP combos for ${MODEL_TAG} ${SEED_TAG} -> $SAVE_FOLDER (bf16)"
python -u generate.py \
  --model-path "$MODEL_PATH" \
  --config "$CFG" \
  --save-folder "$SAVE_FOLDER" 

echo "[Done] Outputs in: $SAVE_FOLDER/generations/${MODEL_TAG}/"
