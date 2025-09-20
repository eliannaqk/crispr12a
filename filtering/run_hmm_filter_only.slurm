#!/bin/bash -l
# Re-run only HMM coverage filtering for specified seeds, using existing prefilter CSVs.
# Uses bio-utils for hmmscan and oc-opencrispr for Python steps.

#SBATCH -J hmm_filter_only
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Config
RUN_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250822-111648-ba4000-sweep/generations/ba4000"
HMM_PROFILE="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/hmm_filter_data/cas12a.hmm"
SEEDS=(1 2)

echo "[env] Will use oc-opencrispr (Python) and bio-utils (HMMER)"

# Init conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
cd ~/crispr12/opencrispr-repro-main

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[error] RUN_DIR not found: $RUN_DIR" >&2
  exit 2
fi
if [[ ! -s "$HMM_PROFILE" ]]; then
  echo "[error] HMM profile missing: $HMM_PROFILE" >&2
  exit 3
fi

for SEED in "${SEEDS[@]}"; do
  echo "[seed] seed${SEED}: HMM filtering"
  SEED_DIR="$RUN_DIR/by-seed/seed${SEED}"
  PREF="$SEED_DIR/seed${SEED}.step1_prefilter.csv"
  if [[ ! -s "$PREF" ]]; then
    echo "[error] Prefilter CSV missing: $PREF" >&2
    exit 4
  fi

  # Step 3: hmmscan coverage gate
  conda activate bio-utils
  echo "[env] Using env: $CONDA_DEFAULT_ENV for hmmscan"
  S2_FASTA="$SEED_DIR/seed${SEED}.step2_hmm_pass.fasta"
  S2_RPT="$SEED_DIR/seed${SEED}.step2_hmm_report.tsv"
  python filtering/filter_coverage.py \
    --input "$PREF" \
    --out_fasta "$S2_FASTA" \
    --out_tsv "$S2_RPT" \
    --hmm "$HMM_PROFILE" \
    --min_qcov 0.8 \
    --min_tcov 0.8 \
    --min_bitscore 50 \
    --threads "$SLURM_CPUS_PER_TASK"

  # Step 4: select pass CSV
  conda activate oc-opencrispr
  echo "[env] Using env: $CONDA_DEFAULT_ENV to write step2 CSV"
  OUT_CSV="$SEED_DIR/seed${SEED}.step2_hmm_pass.csv"
  python filtering/select_by_tsv_keep.py \
    --in_csv "$PREF" \
    --coverage_tsv "$S2_RPT" \
    --out_csv "$OUT_CSV"

  echo "[seed] Done seed${SEED} -> $OUT_CSV"
done

echo "[done] HMM-only filtering complete under $RUN_DIR/by-seed"

