#!/bin/bash -l
#SBATCH -J cas12a_filter
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- User paths (edit as needed) ---
REF_FASTA="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_ref_rep_seq.fasta"
MODEL_DIR="/home/eqk3/crispr12/opencrispr-repro-main/huggingface/huggingface/ba8000"  # local HF model dir
# Optional HMM profile (leave empty to skip HMM step)
CAS12A_HMM=""   # e.g., /home/eqk3/path/to/cas12a.hmm



set -euo pipefail

# Initialize conda and move to project directory
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
cd ~/crispr12/opencrispr-repro-main

GEN_DIR="${INPUTS[$SLURM_ARRAY_TASK_ID]:-$(pwd)}"
OUT_ROOT="${GEN_DIR%/}/_filtered"
mkdir -p "$OUT_ROOT" logs

echo "[info] Working on: $GEN_DIR"
mapfile -t FASTAS < <(find "$GEN_DIR" -maxdepth 2 -type f \( -iname "*.fa" -o -iname "*.faa" -o -iname "*.fasta" \) | sort)
if [[ ${#FASTAS[@]} -eq 0 ]]; then
  echo "[warn] No FASTA files found under $GEN_DIR"
  exit 0
fi

for Q in "${FASTAS[@]}"; do
  base=$(basename "$Q")
  stem="${base%.*}"

  # Step 1: Coverage filter (bio-utils)
  conda activate bio-utils
  echo "[env] Using env: $CONDA_DEFAULT_ENV for HMM coverage"
  S1_FASTA="$OUT_ROOT/${stem}.step1_coverage_pass.fasta"
  S1_RPT="$OUT_ROOT/${stem}.step1_coverage_report.tsv"
  if [[ -n "$CAS12A_HMM" ]]; then
    python filtering/filter_coverage.py \
      --input "$Q" \
      --out_fasta "$S1_FASTA" \
      --out_tsv "$S1_RPT" \
      --hmm "$CAS12A_HMM" \
      --min_coverage 0.5 \
      --threads "$SLURM_CPUS_PER_TASK"
  else
    python filtering/filter_coverage.py \
      --input "$Q" \
      --out_fasta "$S1_FASTA" \
      --out_tsv "$S1_RPT" \
      --threads "$SLURM_CPUS_PER_TASK"
  fi

  # Step 2: Perplexity filter (oc-opencrispr)
  conda activate oc-opencrispr
  echo "[env] Using env: $CONDA_DEFAULT_ENV for perplexity filtering"
  S2_FASTA="$OUT_ROOT/${stem}.step2_ppl_pass.fasta"
  S2_RPT="$OUT_ROOT/${stem}.step2_ppl_report.tsv"
  python filtering/perplexity_filter.py \
    --model "$MODEL_DIR" \
    --in_fasta "$S1_FASTA" \
    --out_fasta "$S2_FASTA" \
    --out_tsv "$S2_RPT" \
    --ppl_threshold 3.0 \
    --batch_size 8

done

echo "[done] Pipeline finished for $GEN_DIR"
