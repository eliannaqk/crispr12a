#!/bin/bash -l
# Collects filtering stats across seeds after the array finishes.

#SBATCH -J collect_filter_stats
#SBATCH -p devel
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "[env] Activating oc-opencrispr for stats collection"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using env: $CONDA_DEFAULT_ENV"

cd ~/crispr12/opencrispr-repro-main

RUN_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000"
BY_SEED="$RUN_DIR/by-seed"
OUT_TSV="$BY_SEED/summary_all.tsv"

python filtering/collect_filter_stats.py \
  --base-dir "$BY_SEED" \
  --out-tsv "$OUT_TSV"

echo "[done] Stats collected at $OUT_TSV"

