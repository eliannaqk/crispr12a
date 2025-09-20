#!/bin/bash -l
# Per-seed perplexity filtering on GPU for the LB runs.
# Uses oc-opencrispr env and runs one seed per array task on a GPU node.

#SBATCH -J ppl_filter_lb_gpu
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out
# One task per seed; adjust as needed
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job uses oc-opencrispr for perplexity filtering (GPU)"

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
# MODEL_DIR can be a local HF model directory or a HuggingFace Hub id (e.g., Salesforce/progen2-medium)
MODEL_DIR="${MODEL_DIR:-}"
PPL_THRESHOLD="${PPL_THRESHOLD:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Auto-detect MODEL_DIR if not provided: infer <model> from run-<date>-<time>-<model>[-suffix]
if [[ -z "$MODEL_DIR" ]]; then
  RUN_ROOT="$(dirname "$(dirname "$RUN_DIR")")" # .../generated_sequences/run-...-<model>-...
  BASE="$(basename "$RUN_ROOT")"
  IFS='-' read -r TAG DATESTR TIMESTR REST <<< "$BASE" || true
  MODEL_NAME="${REST%%-*}"
  if [[ -z "$MODEL_NAME" || "$MODEL_NAME" == "$BASE" ]]; then
    MODEL_NAME="$(basename "$RUN_DIR")" # fallback: generations/<model>
  fi
  RUN_ID="${TAG}-${DATESTR}-${TIMESTR}"
  echo "[detect] Parsed run_id=${RUN_ID} model=${MODEL_NAME} from ${BASE}"
  # Probe common local HF save locations (prefer model_saves_aug21/<run_id>/huggingface/<model>)
  CANDIDATES=(
    "/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/${RUN_ID}/huggingface/${MODEL_NAME}"
    "/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves/huggingface/${MODEL_NAME}"
    "/home/eqk3/project_pi_mg269/eqk3/crisprData/huggingface/${MODEL_NAME}"
    "/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves/${MODEL_NAME}"
  )
  for c in "${CANDIDATES[@]}"; do
    if [[ -d "$c" && -f "$c/config.json" ]]; then
      MODEL_DIR="$c"; echo "[detect] Using local model dir: $MODEL_DIR"; break
    fi
  done
  if [[ -z "$MODEL_DIR" ]]; then
    echo "[warn] Local model dir not found for ${MODEL_NAME}. Will pass '${MODEL_NAME}' to HF loader (hub id or env-resolved)."
    MODEL_DIR="$MODEL_NAME"
  fi
fi

# === Setup ===
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

SEED="${SLURM_ARRAY_TASK_ID}"
SEED_DIR="$RUN_DIR/by-seed-lb/seed${SEED}"
if [[ ! -d "$SEED_DIR" ]]; then
  echo "[error] Seed directory not found: $SEED_DIR" >&2
  exit 1
fi

IN_FASTA="$SEED_DIR/seed${SEED}.step2_hmm_pass.fasta"
if [[ ! -f "$IN_FASTA" ]]; then
  echo "[error] Input FASTA (HMM survivors) not found: $IN_FASTA" >&2
  echo "[hint] Wait for HMM step to finish or adjust IN_FASTA to a different input."
  exit 2
fi

OUT_FASTA="$SEED_DIR/seed${SEED}.step3_ppl_pass.fasta"
OUT_RPT="$SEED_DIR/seed${SEED}.step3_ppl_report.tsv"

echo "[seed] Perplexity filtering (GPU) seed${SEED}"
echo "[in]  $IN_FASTA"
echo "[out] $OUT_FASTA"
echo "[model] $MODEL_DIR"

python filtering/perplexity_filter.py \
  --model "$MODEL_DIR" \
  --in_fasta "$IN_FASTA" \
  --out_fasta "$OUT_FASTA" \
  --out_tsv "$OUT_RPT" \
  --ppl_threshold "$PPL_THRESHOLD" \
  --batch_size "$BATCH_SIZE" \
  --device cuda

echo "[ok] Seed ${SEED} done. Outputs under: $SEED_DIR"
