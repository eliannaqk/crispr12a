#!/bin/bash -l
#SBATCH -J pampredict32
#SBATCH -p devel
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Env
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils

echo "[env] Using conda env: ${CONDA_DEFAULT_ENV}"
python -V

# Inputs
IN_FA=${IN_FA:?must set IN_FA path}
ATLAS_DIR=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas
BASE_DIR=$ATLAS_DIR/viral_dna
PP_DIR=$BASE_DIR/PAMpredict/PAMpredict
DB_DIR=$BASE_DIR/blast_db

# Output (absolute, sanitized)
BASE=$(basename "$IN_FA")
SAFE=$(printf '%s' "$BASE" | sed 's/[^A-Za-z0-9._-]/_/g')
OUTBASE=$HOME/crispr12/opencrispr-repro-main/eval_outputs/pampredict_runs/parallel_${SLURM_JOB_ID}
OUTDIR="$OUTBASE/${SAFE%.fna}"
mkdir -p "$OUTDIR"

echo "[job] JOB_ID=${SLURM_JOB_ID} CPUs=${SLURM_CPUS_PER_TASK}"
echo "[in ] $IN_FA"
echo "[out] $OUTDIR"

cd "$PP_DIR"
python PAMpredict.py "$IN_FA" "$DB_DIR" "$OUTDIR" -p UPSTREAM -l 10 -t ${SLURM_CPUS_PER_TASK} --keep_tmp --force

if [ -f "$OUTDIR/PAM_prediction.txt" ]; then
  echo "--- PAM_prediction.txt"; sed -n '1,40p' "$OUTDIR/PAM_prediction.txt"; fi
