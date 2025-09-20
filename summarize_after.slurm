#!/bin/bash -l
#SBATCH -J pam_summarize
#SBATCH -p devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH -o slurm-%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils

REPO_DIR=$HOME/crispr12/opencrispr-repro-main
cd "$REPO_DIR"
# discover OUTBASE from job ID-specific folder (created above)
RUN_ROOT="$REPO_DIR/eval_outputs/pampredict_runs"
OUTBASE=$(ls -d "$RUN_ROOT"/test10_* 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$OUTBASE" ]; then echo "[err] OUTBASE not found"; exit 1; fi
python scripts/summarize_pampredict.py "$OUTBASE"
