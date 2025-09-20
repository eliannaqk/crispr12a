#!/bin/bash -l
#SBATCH -J gen-smoke
#SBATCH -A pi_mg269
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o slurm-generate-smoke-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

cd ~/crispr12/opencrispr-repro-main

# Required: set MODEL_PATH to an HF folder like .../huggingface/ba8000/
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "[error] Please export MODEL_PATH to your HF model folder (e.g., .../huggingface/ba8000/)" >&2
  exit 2
fi

GEN_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences
mkdir -p "$GEN_ROOT"
MP_NOSLASH=${MODEL_PATH%/}
MODEL_TAG=${MP_NOSLASH##*/}
RUN_TAG=$(echo "$MODEL_PATH" | sed -E 's|.*/(run-[^/]+)/huggingface/.*|\1|')
SEED_TAG=seed0
SAVE_FOLDER="$GEN_ROOT/${RUN_TAG}-${MODEL_TAG}-${SEED_TAG}-smoke"
mkdir -p "$SAVE_FOLDER"

echo "[Run] Smoke test -> $SAVE_FOLDER (precision=bf16)"
python -u generate.py \
  --model-path "$MODEL_PATH" \
  --config generation/generate_smoke.yml \
  --save-folder "$SAVE_FOLDER" \
  --precision bf16

echo "[Done] Check: $SAVE_FOLDER/generations/${MODEL_TAG}/generate_smoke_{raw,filtered}.csv"

