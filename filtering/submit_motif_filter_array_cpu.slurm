#!/bin/bash -l
# Step 4: Per-seed Cas12a motif-only HMMER filter (array job)
# - Runs after Step 3 (PPL). Input: seed${SEED}.step3_ppl_pass.fasta
# - Uses oc-opencrispr for Python, hmmalign from bio-utils via PATH prepend.
# - Produces: seed${SEED}.step4_motif_pass.fasta, seed${SEED}.step4_motif_report.tsv, stats file, and a per-run aggregate TSV.

#SBATCH -J motif_filter_lb
#SBATCH -p day
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o slurm-%A_%a.out
# One task per seed; run 1-4 but branch so 2/4 do copy-only
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] Using oc-opencrispr (Python) + bio-utils hmmalign on CPU day partition (16 cores, 64G, 24h)"

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
HMM_PATH="${HMM_PATH:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/hmm_filter_data/cas12a.hmm}"
CONFIG_PATH="${CONFIG_PATH:-filtering/cas12a_motif_filter_package/cas12a_motifs_lb_hmm.json}"
ANCHOR_CSV="${ANCHOR_CSV:-seq1.csv}"

# === Setup ===
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
# Prepend bio-utils bin so hmmalign is found by the Python tool
export PATH="/home/eqk3/.conda/envs/bio-utils/bin:${PATH}"

cd ~/crispr12/opencrispr-repro-main

SEED="${SLURM_ARRAY_TASK_ID}"
SEED_DIR="$RUN_DIR/by-seed/seed${SEED}"
if [[ ! -d "$SEED_DIR" ]]; then
  echo "[error] Seed directory not found: $SEED_DIR" >&2
  exit 1
fi

IN_FASTA="$SEED_DIR/seed${SEED}.step3_ppl_pass.fasta"
if [[ ! -s "$IN_FASTA" ]]; then
  echo "[skip] Missing Step 4 survivors: $IN_FASTA"; exit 0
fi

# For seeds 2 and 4: copy-only (no HMM compute). Make step4 FASTA for monitoring.
if [[ "$SEED" == "2" || "$SEED" == "4" ]]; then
  echo "[copy-only] Seed ${SEED}: creating step4_motif_pass.fasta from step3 survivors; skipping HMM computation."
  OUT_FASTA="$SEED_DIR/seed${SEED}.step4_motif_pass.fasta"
  cp -f "$IN_FASTA" "$OUT_FASTA"
  echo "[copy-only] Wrote: $OUT_FASTA (identical to step3)"
  # Optional: touch a minimal stats file for visibility
  echo -e "seed=seed${SEED}\ncopy_only=true\nsource=$IN_FASTA\nout_fasta=$OUT_FASTA\n" > "$SEED_DIR/seed${SEED}.step4_motif_stats.txt"
  exit 0
fi

echo "[seed] Motif filter Step 4 for seed${SEED}"
echo "[in]  $IN_FASTA"
echo "[hmm] $HMM_PATH"
echo "[cfg] $CONFIG_PATH"
echo "[anc] $ANCHOR_CSV (CSV with column 'sequence')"

python filtering/run_motif_filter_seed.py \
  --seed-dir "$SEED_DIR" \
  --hmm "$HMM_PATH" \
  --config "$CONFIG_PATH" \
  --anchor-csv "$ANCHOR_CSV"

echo "[ok] Seed ${SEED} step4 outputs under: $SEED_DIR"

# === Aggregate per-seed report into one TSV ===
AGG_DIR="$RUN_DIR/by-seed"
AGG_TSV="$AGG_DIR/motif_all.tsv"
mkdir -p "$AGG_DIR"
LOCK_FD=208
LOCK_FILE="${AGG_TSV}.lock"
exec {LOCK_FD}>>"$LOCK_FILE"
flock -x $LOCK_FD
if [[ ! -s "$AGG_TSV" ]]; then
  printf "seed\tseq_id\tWED_pass\tPI_pass\tNUC_pass\tBH_pass\tLID_pass\toverall_pass\tseed_masked\tfail_reasons\n" > "$AGG_TSV"
fi
RPT="$SEED_DIR/seed${SEED}.step4_motif_report.tsv"
if [[ -s "$RPT" ]]; then
  awk -v s="seed${SEED}" 'BEGIN{FS="\t"; OFS="\t"} NR>1 {print s,$0}' "$RPT" >> "$AGG_TSV"
fi
flock -u $LOCK_FD

# === Print combined summary so far ===
if [[ -s "$AGG_TSV" ]]; then
  TOTAL=$(awk 'NR>1 {n++} END{print n+0}' "$AGG_TSV")
  KEPT=$(awk -F"\t" 'NR>1 {v=tolower($8); if (v=="true" || v=="1") k++} END{print k+0}' "$AGG_TSV")
  if [[ "$TOTAL" -gt 0 ]]; then
    PASS_RATE=$(awk "BEGIN{printf \"%.2f\", ($KEPT/$TOTAL)*100}")
  else
    PASS_RATE="0.00"
  fi
  echo "[agg] Motif combined-so-far: total=${TOTAL} kept=${KEPT} pass_rate=${PASS_RATE}%"
  echo "[agg] Aggregated TSV: $AGG_TSV"
fi

echo "[done] Step 4 motif filter complete for seed${SEED}"
