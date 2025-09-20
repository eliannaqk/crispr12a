#!/bin/bash -l
#SBATCH -J pampredict10
#SBATCH -p devel
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils

echo "[env] Using conda env: ${CONDA_DEFAULT_ENV}"
python -V

ATLAS_DIR=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas
BASE_DIR=$ATLAS_DIR/viral_dna
PP_DIR=$BASE_DIR/PAMpredict/PAMpredict
DB_DIR=$BASE_DIR/blast_db
SUBSET_DIR=$(ls -d $BASE_DIR/pampredict_subset_* | sort | tail -n 1)
SPDIR=$SUBSET_DIR/spacers

OUTBASE=$HOME/crispr12/opencrispr-repro-main/eval_outputs/pampredict_runs/test10_${SLURM_JOB_ID}
mkdir -p "$OUTBASE"

i=0
for IN_FA in $(ls -1 "$SPDIR"/*.fna | head -n 10); do
  BASE=$(basename "$IN_FA")
  SAFE=$(printf '%s' "$BASE" | sed 's/[^A-Za-z0-9._-]/_/g')
  OUTDIR="$OUTBASE/${SAFE%.fna}"
  mkdir -p "$OUTDIR"
  echo "[run] ($((++i)) / 10) $IN_FA -> $OUTDIR"
  (cd "$PP_DIR" && python PAMpredict.py "$IN_FA" "$DB_DIR" "$OUTDIR" -p UPSTREAM -l 10 -t ${SLURM_CPUS_PER_TASK} --keep_tmp --force)
  if [ -f "$OUTDIR/PAM_prediction.txt" ]; then
    echo "--- PAM_prediction.txt ($SAFE)"; sed -n '1,20p' "$OUTDIR/PAM_prediction.txt"; fi
  echo "------"
  sleep 1
done

# Write a stamp with OUTBASE for downstream summarizer
echo "$OUTBASE" > $OUTBASE/.outbase
