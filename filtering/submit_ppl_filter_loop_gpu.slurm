#!/bin/bash -l
# Sequential PPL filtering over all seeds in by-seed using one GPU job.
# Uses oc-opencrispr env; writes per-seed outputs and an aggregated ppl_all.tsv.

#SBATCH -J ppl_filter_lb_loop
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job uses oc-opencrispr for perplexity filtering (GPU, loop)"

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
MODEL_DIR="${MODEL_DIR:-}"
PPL_THRESHOLD="${PPL_THRESHOLD:-4.0}"
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
  CANDIDATES=(
    "/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/${RUN_ID}/huggingface/${MODEL_NAME}"
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

SEED_BASE="$RUN_DIR/by-seed"
if [[ ! -d "$SEED_BASE" ]]; then
  echo "[error] Seed base not found: $SEED_BASE" >&2
  exit 1
fi

AGG_DIR="$SEED_BASE"
AGG_TSV="$AGG_DIR/ppl_all.tsv"
echo -n > /tmp/ppl_all.$$ # temp aggregate we will move atomically at end
printf "seed\tid\tn_tokens\tnll\tppl\tkeep\n" > /tmp/ppl_all.$$

# Iterate seeds in numeric order
for d in "$SEED_BASE"/seed*; do
  [[ -d "$d" ]] || continue
  s=$(basename "$d" | sed 's/^seed//')
  IN_FASTA="$d/seed${s}.step2_hmm_pass.fasta"
  OUT_FASTA="$d/seed${s}.step3_ppl_pass.fasta"
  OUT_RPT="$d/seed${s}.step3_ppl_report.tsv"
  echo "[seed] Perplexity filtering (GPU) seed${s}"
  if [[ ! -f "$IN_FASTA" ]]; then
    echo "[skip] Missing HMM survivors: $IN_FASTA"; continue
  fi
  python filtering/perplexity_filter.py \
    --model "$MODEL_DIR" \
    --in_fasta "$IN_FASTA" \
    --out_fasta "$OUT_FASTA" \
    --out_tsv "$OUT_RPT" \
    --ppl_threshold "$PPL_THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --device cuda
  # Append to temp aggregate (skip header)
  awk -v s="seed${s}" 'BEGIN{FS="\t"; OFS="\t"} NR>1 {print s,$0}' "$OUT_RPT" >> /tmp/ppl_all.$$
done

mkdir -p "$AGG_DIR"
mv /tmp/ppl_all.$$ "$AGG_TSV"

# Print final summary
if [[ -f "$AGG_TSV" ]]; then
  TOTAL=$(awk 'NR>1 {n++} END{print n+0}' "$AGG_TSV")
  KEPT=$(awk -F"\t" 'NR>1 {if ($6==1) k++} END{print k+0}' "$AGG_TSV")
  AVG_PPL=$(awk -F"\t" 'NR>1 {if ($5+0==$5){sum+=$5; n++}} END{if(n>0) printf "%.6f\n", sum/n; else print "nan"}' "$AGG_TSV")
  if [[ "$TOTAL" -gt 0 ]]; then
    PASS_RATE=$(awk "BEGIN{printf \"%.2f\", ($KEPT/$TOTAL)*100}")
  else
    PASS_RATE="0.00"
  fi
  echo "[agg] Final: total_rows=${TOTAL} kept=${KEPT} pass_rate=${PASS_RATE}% avg_ppl=${AVG_PPL}"
  echo "[agg] Aggregated TSV: $AGG_TSV"
fi

