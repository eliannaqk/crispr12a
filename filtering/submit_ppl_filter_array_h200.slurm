#!/bin/bash -l
# Per-seed perplexity filtering on H200 GPU nodes. One task per seed.
# Uses oc-opencrispr env; aggregates per-seq results across seeds.

#SBATCH -J ppl_filter_lb_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job uses oc-opencrispr for perplexity filtering (H200, array)"

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
MODEL_DIR="${MODEL_DIR:-}"
PPL_THRESHOLD="${PPL_THRESHOLD:-4.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Auto-detect MODEL_DIR if not provided
if [[ -z "$MODEL_DIR" ]]; then
  RUN_ROOT="$(dirname "$(dirname "$RUN_DIR")")"
  BASE="$(basename "$RUN_ROOT")"
  IFS='-' read -r TAG DATESTR TIMESTR REST <<< "$BASE" || true
  MODEL_NAME="${REST%%-*}"
  if [[ -z "$MODEL_NAME" || "$MODEL_NAME" == "$BASE" ]]; then
    MODEL_NAME="$(basename "$RUN_DIR")"
  fi
  RUN_ID="${TAG}-${DATESTR}-${TIMESTR}"
  echo "[detect] Parsed run_id=${RUN_ID} model=${MODEL_NAME} from ${BASE}"
  CANDIDATES=(
    "/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/${RUN_ID}/huggingface/${MODEL_NAME}"
  )
  for c in "${CANDIDATES[@]}"; do
    if [[ -d "$c" && -f "$c/config.json" ]]; then
      MODEL_DIR="$c"; echo "[detect] Using local model dir: $MODEL_DIR"; break
    fi
  done
  if [[ -z "$MODEL_DIR" ]]; then
    echo "[warn] Local model dir not found for ${MODEL_NAME}. Using '${MODEL_NAME}' with HF loader."
    MODEL_DIR="$MODEL_NAME"
  fi
fi

# === Setup ===
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

SEED="${SLURM_ARRAY_TASK_ID}"
SEED_DIR="$RUN_DIR/by-seed/seed${SEED}"
if [[ ! -d "$SEED_DIR" ]]; then
  echo "[error] Seed directory not found: $SEED_DIR" >&2
  exit 1
fi

IN_FASTA="$SEED_DIR/seed${SEED}.step2_hmm_pass.fasta"
if [[ ! -f "$IN_FASTA" ]]; then
  echo "[error] Input FASTA (HMM survivors) not found: $IN_FASTA" >&2
  exit 2
fi

OUT_FASTA="$SEED_DIR/seed${SEED}.step3_ppl_pass.fasta"
OUT_RPT="$SEED_DIR/seed${SEED}.step3_ppl_report.tsv"
AGG_DIR="$RUN_DIR/by-seed"
AGG_TSV="$AGG_DIR/ppl_all.tsv"

echo "[seed] Perplexity filtering (H200) seed${SEED}"
echo "[in]  $IN_FASTA"
echo "[out] $OUT_FASTA"
echo "[model] $MODEL_DIR  thr=$PPL_THRESHOLD  bs=$BATCH_SIZE"

python filtering/perplexity_filter.py \
  --model "$MODEL_DIR" \
  --in_fasta "$IN_FASTA" \
  --out_fasta "$OUT_FASTA" \
  --out_tsv "$OUT_RPT" \
  --ppl_threshold "$PPL_THRESHOLD" \
  --align_training_collator \
  --batch_size "$BATCH_SIZE" \
  --device cuda

echo "[ok] Seed ${SEED} step3 outputs under: $SEED_DIR"

# Aggregate per-seq rows (with locking)
mkdir -p "$AGG_DIR"
LOCK_FD=210
LOCK_FILE="${AGG_TSV}.lock"
exec {LOCK_FD}>>"$LOCK_FILE"
flock -x $LOCK_FD
if [[ ! -s "$AGG_TSV" ]]; then
  printf "seed\tid\tn_tokens\tnll\tppl\tkeep\n" > "$AGG_TSV"
fi
awk -v s="seed${SEED}" 'BEGIN{FS="\t"; OFS="\t"} NR>1 {print s,$0}' "$OUT_RPT" >> "$AGG_TSV"
flock -u $LOCK_FD

# Print combined-so-far stats
if [[ -f "$AGG_TSV" ]]; then
  TOTAL=$(awk 'NR>1 {n++} END{print n+0}' "$AGG_TSV")
  KEPT=$(awk -F"\t" 'NR>1 {if ($6==1) k++} END{print k+0}' "$AGG_TSV")
  AVG_PPL=$(awk -F"\t" 'NR>1 {if ($5+0==$5){sum+=$5; n++}} END{if(n>0) printf "%.6f\n", sum/n; else print "nan"}' "$AGG_TSV")
  if [[ "$TOTAL" -gt 0 ]]; then PASS_RATE=$(awk "BEGIN{printf \"%.2f\", ($KEPT/$TOTAL)*100}"); else PASS_RATE="0.00"; fi
  echo "[agg] Combined-so-far: rows=${TOTAL} kept=${KEPT} pass_rate=${PASS_RATE}% avg_ppl=${AVG_PPL}"
fi
