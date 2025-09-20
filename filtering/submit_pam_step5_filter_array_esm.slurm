#!/bin/bash -l
# Step 5 (ESM): PAM filter using core‑aware strict→strict classifier on Step 4 motif-pass FASTAs.
#
# - Seeds 1, 2, 3, 4: run the ESM-based PAM filter on full-length designs.
# - Before running seeds, print seq1.csv predicted label at threshold 0.5.
#
# Uses oc-opencrispr env. One seed per array task.

#SBATCH -J pam_step5_esm_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
# Use jobID_arrayTaskID in log filename to avoid clobbering between tasks
#SBATCH -o slurm-%A_%a.out
# Four seeds total
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

echo "[env] Using oc-opencrispr (GPU H200) for ESM PAM filtering."

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
MODEL_PATH="${MODEL_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna/pampredict_aggregate_20250911_131538/pam_coreaware_strict2strict_esm2_t33.pt}"
THRESH="${THRESH:-0.45}"

# === Setup ===
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

# One-time check: predict seq1.csv label
if [[ -f "seq1.csv" ]]; then
  echo "[check] seq1.csv found; predicting PAM-binding with model: $MODEL_PATH"
  export TMPD="/tmp/pam_check_${SLURM_JOB_ID:-local}"
  mkdir -p "$TMPD"
  python - <<'PY'
import csv, os
rows = list(csv.DictReader(open('seq1.csv')))
seq = rows[0].get('sequence','').strip() if rows else ''
out = os.environ.get('TMPD') or '/tmp'
fa = os.path.join(out, 'seq1.faa')
open(fa,'w').write('>seq1\n'+seq+'\n')
print(f"[check] wrote temporary FASTA: {fa}")
PY
  python filtering/pam_filtering_esm/pam_filter_from_esm.py \
    --designs "$TMPD/seq1.faa" \
    --outdir "$TMPD/out" \
    --model "$MODEL_PATH" \
    --threshold "$THRESH" \
    --device cuda || true
  if [[ -f "$TMPD/out/report.tsv" ]]; then
    P=$(awk 'NR==2{print $2}' "$TMPD/out/report.tsv" 2>/dev/null || echo "")
    LBL=$(awk 'NR==2{print $3}' "$TMPD/out/report.tsv" 2>/dev/null || echo "")
    echo "[check] seq1.csv predicted: ${LBL:-NA} (prob=${P:-NA}, thr=${THRESH})"
  fi
else
  echo "[check] seq1.csv not found; skipping label check"
fi

SEED="${SLURM_ARRAY_TASK_ID}"
SEED_DIR="$RUN_DIR/by-seed/seed${SEED}"
if [[ ! -d "$SEED_DIR" ]]; then
  echo "[error] Seed directory not found: $SEED_DIR" >&2
  exit 1
fi

IN_STEP4_FASTA="$SEED_DIR/seed${SEED}.step4_motif_pass.fasta"
if [[ ! -f "$IN_STEP4_FASTA" ]]; then
  echo "[error] Step4 motif FASTA not found: $IN_STEP4_FASTA" >&2
  echo "[hint] Make sure step 4 completed successfully."
  exit 2
fi

OUT_STEP5_FASTA="$SEED_DIR/seed${SEED}.step5_pam_filter_pass.fasta"
OUT_DIR="$SEED_DIR/step5_pam_filter"
mkdir -p "$OUT_DIR"

echo "[seed${SEED}] Running ESM PAM filter on step4_motif FASTA (thr=$THRESH)"
python filtering/pam_filtering_esm/pam_filter_from_esm.py \
  --designs "$IN_STEP4_FASTA" \
  --outdir "$OUT_DIR" \
  --model "$MODEL_PATH" \
  --threshold "$THRESH" \
  --device cuda

# Copy pass.faa to canonical step5 filename for monitoring
if [[ -f "$OUT_DIR/pass.faa" ]]; then
  cp -f "$OUT_DIR/pass.faa" "$OUT_STEP5_FASTA"
  echo "[seed${SEED}] Wrote step5 pam_filter pass FASTA: $OUT_STEP5_FASTA"
else
  echo "[seed${SEED}] ERROR: Expected $OUT_DIR/pass.faa not found" >&2
  exit 3
fi

# Simple summary per seed
N_IN=$(awk '/^>/{n++} END{print n+0}' "$IN_STEP4_FASTA")
N_OUT=$(awk '/^>/{n++} END{print n+0}' "$OUT_STEP5_FASTA" || echo 0)
PCT=$(awk -v a="$N_OUT" -v b="$N_IN" 'BEGIN{ if(b>0) printf "%.2f", (a/b)*100; else print "0.00" }')
echo "[seed${SEED}] step5 summary: in=${N_IN} out=${N_OUT} (pass_rate=${PCT}%)"

echo "[ok] Seed ${SEED} step5 outputs under: $SEED_DIR"
