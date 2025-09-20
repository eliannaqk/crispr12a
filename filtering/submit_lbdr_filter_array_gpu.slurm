#!/bin/bash -l
# Per-seed DR classifier filtering on GPU for the LB runs.
# Uses oc-opencrispr env and runs one seed per array task on a GPU node.

#SBATCH -J dr_filter_lb_gpu
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out
# One task per seed; adjust as needed
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job uses oc-opencrispr-esm for DR classifier filtering (GPU)"

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
CLASSIFIER_PT="${CLASSIFIER_PT:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_dr_datasets/lbdr_cls.pt}"
DR_THRESHOLD="${DR_THRESHOLD:-0.5}"

# === Setup ===
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm
cd ~/crispr12/opencrispr-repro-main

SEED="${SLURM_ARRAY_TASK_ID}"
SEED_DIR="$RUN_DIR/by-seed/seed${SEED}"
if [[ ! -d "$SEED_DIR" ]]; then
  echo "[error] Seed directory not found: $SEED_DIR" >&2
  exit 1
fi

IN_FASTA="$SEED_DIR/seed${SEED}.step3_ppl_pass.fasta"
if [[ ! -f "$IN_FASTA" ]]; then
  echo "[error] Input FASTA (PPL survivors) not found: $IN_FASTA" >&2
  echo "[hint] Make sure step 3 completed successfully."
  exit 2
fi

OUT_FASTA="$SEED_DIR/seed${SEED}.step4_dr_pass.fasta"
OUT_RPT="$SEED_DIR/seed${SEED}.step4_dr_report.tsv"
AGG_DIR="$RUN_DIR/by-seed"
AGG_TSV="$AGG_DIR/dr_all.tsv"

echo "[seed] DR filtering (GPU) seed${SEED}"
echo "[in]  $IN_FASTA"
echo "[out] $OUT_FASTA"
echo "[clf] $CLASSIFIER_PT  thr=$DR_THRESHOLD"

python -m filtering.dr_filter_fasta \
  --classifier_pt "$CLASSIFIER_PT" \
  --in_fasta "$IN_FASTA" \
  --out_fasta "$OUT_FASTA" \
  --out_tsv "$OUT_RPT" \
  --threshold "$DR_THRESHOLD" \
  --device cuda

echo "[ok] Seed ${SEED} step4 outputs under: $SEED_DIR"

# === Append per-seed report into a single aggregated TSV ===
mkdir -p "$AGG_DIR"
LOCK_FD=201
LOCK_FILE="${AGG_TSV}.lock"
exec {LOCK_FD}>>"$LOCK_FILE"
flock -x $LOCK_FD
if [[ ! -s "$AGG_TSV" ]]; then
  printf "seed\tid\tn_aa\tlogit\tprob\tkeep\n" > "$AGG_TSV"
fi
awk -v s="seed${SEED}" 'BEGIN{FS="\t"; OFS="\t"} NR>1 {print s,$0}' "$OUT_RPT" >> "$AGG_TSV"
flock -u $LOCK_FD

# === Print simple summary statistics (combined so far) ===
if [[ -f "$AGG_TSV" ]]; then
  TOTAL=$(awk 'NR>1 {n++} END{print n+0}' "$AGG_TSV")
  KEPT=$(awk -F"\t" 'NR>1 {if ($6==1) k++} END{print k+0}' "$AGG_TSV")
  AVG_PROB=$(awk -F"\t" 'NR>1 {if ($5+0==$5){sum+=$5; n++}} END{if(n>0) printf "%.6f\n", sum/n; else print "nan"}' "$AGG_TSV")
  if [[ "$TOTAL" -gt 0 ]]; then
    PASS_RATE=$(awk "BEGIN{printf \"%.2f\", ($KEPT/$TOTAL)*100}")
  else
    PASS_RATE="0.00"
  fi
  echo "[agg] Combined-so-far: total_rows=${TOTAL} kept=${KEPT} pass_rate=${PASS_RATE}% avg_prob=${AVG_PROB}"
  echo "[agg] Aggregated TSV: $AGG_TSV"
fi
