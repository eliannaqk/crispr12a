#!/bin/bash -l
#SBATCH -J pampredict_subset
#SBATCH -p devel
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "[slurm] Host: $(hostname)"
echo "[slurm] Start: $(date)"
echo "[slurm] Using conda env: bio-utils"

# Conda (bio-utils per policy for bioinformatics tooling)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Move to project directory per policy
cd ~/crispr12/opencrispr-repro-main

# Inputs
ATLAS_JSON=${ATLAS_JSON:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/crispr-cas-atlas-v1.0.json}
BASE_DIR=${BASE_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna}
PAMPREDICT_DIR=${PAMPREDICT_DIR:-${BASE_DIR}/PAMpredict}
BLAST_DB_DIR=${BLAST_DB_DIR:-${BASE_DIR}/blast_db}
OPERON_IDS_FILE=${OPERON_IDS_FILE:-}
MAX_OPERONS=${MAX_OPERONS:-500}
THREADS=${SLURM_CPUS_PER_TASK:-8}
OUTDIR=${OUTDIR:-${BASE_DIR}/pampredict_subset_$(date +%Y%m%d_%H%M%S)}
MIN_SPACERS=${MIN_SPACERS:-5}

echo "[slurm] ATLAS_JSON: ${ATLAS_JSON}"
echo "[slurm] OPERON_IDS_FILE: ${OPERON_IDS_FILE:-<none>}"
echo "[slurm] MAX_OPERONS: ${MAX_OPERONS}"
echo "[slurm] BLAST_DB_DIR: ${BLAST_DB_DIR}"
echo "[slurm] OUTDIR: ${OUTDIR}"
echo "[slurm] THREADS: ${THREADS}"
echo "[slurm] MIN_SPACERS: ${MIN_SPACERS}"

python -V
which blastn && blastn -version | head -n1 || { echo "[slurm] ERROR: blastn missing" >&2; exit 2; }
test -f "${BLAST_DB_DIR}/viruses_plasmids.fna" || { echo "[slurm] ERROR: BLAST DB missing" >&2; exit 2; }
mkdir -p "${OUTDIR}"

CMD=(python atlas_make_pam_crispr.py \
  --atlas-json "${ATLAS_JSON}" \
  --pampredict-dir "${PAMPREDICT_DIR}" \
  --dbdir "${BLAST_DB_DIR}" \
  --outdir "${OUTDIR}" \
  --threads "${THREADS}" \
  --min_spacers_per_array "${MIN_SPACERS}" \
  --max-operons "${MAX_OPERONS}")

if [[ -n "${OPERON_IDS_FILE}" ]]; then
  CMD+=(--include-operon-ids "${OPERON_IDS_FILE}")
fi

echo "[slurm] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[slurm] Done: $(date)"
echo "[slurm] Outputs in: ${OUTDIR}"
