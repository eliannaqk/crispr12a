#!/bin/bash -l
#SBATCH -J select_diverse
#SBATCH -p cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out

# Environment policy: use bio-utils for MMseqs2 and bioinformatics tooling
echo "[Env] Initializing conda and activating bio-utils (bioinformatics tools)."
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate bio-utils || { echo "[Error] Failed to activate bio-utils env"; exit 1; }
echo "[Env] Using conda env: bio-utils"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Move to project directory (adjust if repo path differs)
cd ~/crispr12/opencrispr-repro-main || { echo "Repo path not found"; exit 1; }

set -euo pipefail

usage() {
  cat << USAGE
Usage: sbatch filtering/submit_select_diverse.slurm \
  --generated /path/to/generated.faa \
  --known /path/to/known.faa \
  --metrics /path/to/metrics.tsv \
  --outdir /path/to/output_dir \
  [--seed-map /path/to/seeds.tsv] \
  [--threads 16] [--cluster-id 0.90] [--novelty-cutoff 80]

Notes:
  - This job uses bio-utils and leverages MMseqs2 (native or via apptainer/singularity).
  - No GPU is required.
USAGE
}

GEN=""; KN=""; MET=""; OUT=""; SEEDS=""; THREADS=${SLURM_CPUS_PER_TASK:-16}
CL_ID=0.90; NOV=80; COV=0.8; SEG_COV=0.8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generated) GEN="$2"; shift 2;;
    --known)     KN="$2"; shift 2;;
    --metrics)   MET="$2"; shift 2;;
    --outdir)    OUT="$2"; shift 2;;
    --seed-map)  SEEDS="$2"; shift 2;;
    --threads)   THREADS="$2"; shift 2;;
    --cluster-id) CL_ID="$2"; shift 2;;
    --novelty-cutoff) NOV="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -z "$GEN" || -z "$KN" || -z "$MET" || -z "$OUT" ]] && { usage; exit 1; }

echo "[Args] generated=$GEN"
echo "[Args] known=$KN"
echo "[Args] metrics=$MET"
echo "[Args] outdir=$OUT"
echo "[Args] seed_map=$SEEDS"

python filtering/select_diverse_novel.py \
  --generated "$GEN" \
  --known "$KN" \
  --metrics "$MET" \
  --outdir "$OUT" \
  ${SEEDS:+--seed-map "$SEEDS"} \
  --threads "$THREADS" \
  --cluster-id "$CL_ID" \
  --novelty-cutoff "$NOV" \
  --cov "$COV" \
  --seg-cov "$SEG_COV"

echo "[Done] Selection pipeline finished. Outputs under: $OUT"

