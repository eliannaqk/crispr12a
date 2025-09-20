#!/bin/bash -l
#SBATCH -J build_pam_noncore
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "[slurm] Host: $(hostname)"
echo "[slurm] Start: $(date)"
echo "[slurm] Using conda env: oc-opencrispr"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate oc-opencrispr
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd ~/crispr12/opencrispr-repro-main

BASE=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas
AGG_DIR=${AGG_DIR:-${BASE}/viral_dna/pampredict_aggregate_20250911_131538}
ATLAS_JSON=${ATLAS_JSON:-${BASE}/crispr-cas-atlas-v1.0.json}
PAM_TSV=${PAM_TSV:-${AGG_DIR}/pam_crispr.per_protein.tsv}

echo "[build] AGG_DIR=${AGG_DIR}"
echo "[build] ATLAS_JSON=${ATLAS_JSON}"
echo "[build] PAM_TSV=${PAM_TSV}"

# Build non-core-aware datasets (inclusive and strict scan across motif)
INC_ALL=${AGG_DIR}/pam_noncore_inclusive_all.csv
STR_ALL=${AGG_DIR}/pam_noncore_strict_all.csv

python filtering/build_pam_tttv_dataset.py \
  --atlas-json "${ATLAS_JSON}" \
  --pam-per-protein "${PAM_TSV}" \
  --out-csv "${INC_ALL}" \
  --confidence high_only \
  --tttv-mode inclusive

python filtering/build_pam_tttv_dataset.py \
  --atlas-json "${ATLAS_JSON}" \
  --pam-per-protein "${PAM_TSV}" \
  --out-csv "${STR_ALL}" \
  --confidence high_only \
  --tttv-mode strict

# Split 90/10 stratified
python scripts/split_csv_stratified.py --in "${INC_ALL}" \
  --out-train "${AGG_DIR}/pam_noncore_inclusive_train.csv" \
  --out-valid "${AGG_DIR}/pam_noncore_inclusive_valid.csv" \
  --valid-frac 0.1 --seed 42

python scripts/split_csv_stratified.py --in "${STR_ALL}" \
  --out-train "${AGG_DIR}/pam_noncore_strict_train.csv" \
  --out-valid "${AGG_DIR}/pam_noncore_strict_valid.csv" \
  --valid-frac 0.1 --seed 42

echo "[slurm] Done: $(date)"

