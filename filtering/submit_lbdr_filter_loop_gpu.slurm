#!/bin/bash -l
# Sequential DR classifier filtering over all seeds using one GPU job.

#SBATCH -J dr_filter_lb_loop_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job uses oc-opencrispr for DR classifier filtering (H200, loop)"

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
CLASSIFIER_PT="${CLASSIFIER_PT:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_dr_datasets/lbdr_cls.pt}"
DR_THRESHOLD="${DR_THRESHOLD:-0.5}"

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

SEED_BASE="$RUN_DIR/by-seed"
if [[ ! -d "$SEED_BASE" ]]; then
  echo "[error] Seed base not found: $SEED_BASE" >&2
  exit 1
fi

AGG_TSV="$SEED_BASE/dr_all.tsv"
printf "seed\tid\tn_aa\tlogit\tprob\tkeep\n" > /tmp/dr_all.$$

for d in "$SEED_BASE"/seed*; do
  [[ -d "$d" ]] || continue
  s=$(basename "$d" | sed 's/^seed//')
  IN_FASTA="$d/seed${s}.step3_ppl_pass.fasta"
  OUT_FASTA="$d/seed${s}.step4_dr_pass.fasta"
  OUT_RPT="$d/seed${s}.step4_dr_report.tsv"
  echo "[seed] DR filtering (GPU) seed${s}"
  if [[ ! -f "$IN_FASTA" ]]; then
    echo "[skip] Missing PPL survivors: $IN_FASTA"; continue
  fi
  python filtering/dr_filter_fasta.py \
    --classifier_pt "$CLASSIFIER_PT" \
    --in_fasta "$IN_FASTA" \
    --out_fasta "$OUT_FASTA" \
    --out_tsv "$OUT_RPT" \
    --threshold "$DR_THRESHOLD" \
    --device cuda
  awk -v s="seed${s}" 'BEGIN{FS="\t"; OFS="\t"} NR>1 {print s,$0}' "$OUT_RPT" >> /tmp/dr_all.$$
done

mv /tmp/dr_all.$$ "$AGG_TSV"
if [[ -f "$AGG_TSV" ]]; then
  TOTAL=$(awk 'NR>1 {n++} END{print n+0}' "$AGG_TSV")
  KEPT=$(awk -F"\t" 'NR>1 {if ($6==1) k++} END{print k+0}' "$AGG_TSV")
  AVG_PROB=$(awk -F"\t" 'NR>1 {if ($5+0==$5){sum+=$5; n++}} END{if(n>0) printf "%.6f\n", sum/n; else print "nan"}' "$AGG_TSV")
  if [[ "$TOTAL" -gt 0 ]]; then
    PASS_RATE=$(awk "BEGIN{printf \"%.2f\", ($KEPT/$TOTAL)*100}")
  else
    PASS_RATE="0.00"
  fi
  echo "[agg] Final: total_rows=${TOTAL} kept=${KEPT} pass_rate=${PASS_RATE}% avg_prob=${AVG_PROB}"
  echo "[agg] Aggregated TSV: $AGG_TSV"
fi
