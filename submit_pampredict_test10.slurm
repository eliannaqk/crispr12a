#!/bin/bash -l
#SBATCH -J pampredict10
#SBATCH -p devel
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate bio-utils (BLAST/MSA tools)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils

echo "[env] Using conda env: ${CONDA_DEFAULT_ENV}"
python -V
blastn -version | sed -n '1,1p' || true
muscle -version 2>/dev/null | sed -n '1,2p' || echo muscle-not-found

# Data/code paths
ATLAS_DIR=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas
BASE_DIR=$ATLAS_DIR/viral_dna
PP_DIR=$BASE_DIR/PAMpredict/PAMpredict
DB_DIR=$BASE_DIR/blast_db
SUBSET_DIR=$(ls -d $BASE_DIR/pampredict_subset_* | sort | tail -n 1)
SPDIR=$SUBSET_DIR/spacers

# Select target FASTA for this array index
mapfile -t FILES < <(ls -1 "$SPDIR"/*.fna)
IDX=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=${#FILES[@]}
if (( IDX >= TOTAL )); then
  echo "[skip] IDX $IDX >= total $TOTAL"; exit 0
fi
IN_FA=${FILES[$IDX]}
BASE=$(basename "$IN_FA")
SAFE=$(printf '%s' "$BASE" | sed 's/[^A-Za-z0-9._-]/_/g')

# Outdir (absolute, sanitized)
OUTBASE=$HOME/crispr12/opencrispr-repro-main/eval_outputs/pampredict_runs/test10_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
OUTDIR="$OUTBASE/${SAFE%.fna}"
mkdir -p "$OUTDIR"

# Log
echo "[job] JOB_ID=${SLURM_JOB_ID} ARRAY_ID=${SLURM_ARRAY_TASK_ID:-na} CPUs=${SLURM_CPUS_PER_TASK}"
echo "[in ] $IN_FA"
echo "[out] $OUTDIR"

# Run PAMpredict
cd "$PP_DIR"
python PAMpredict.py "$IN_FA" "$DB_DIR" "$OUTDIR" -p UPSTREAM -l 10 -t ${SLURM_CPUS_PER_TASK} --keep_tmp --force

# Quick status
if [ -f "$OUTDIR/PAM_prediction.txt" ]; then
  echo "--- PAM_prediction.txt"; sed -n '1,40p' "$OUTDIR/PAM_prediction.txt"
else
  echo "[no_call] No PAM_prediction.txt (likely low mapped hits)."
fi
