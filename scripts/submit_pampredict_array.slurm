#!/bin/bash -l
#SBATCH -J pampredict_array
#SBATCH -p devel
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%A_%a.out

set -euo pipefail

echo "[slurm] Host: $(hostname)"
echo "[slurm] Start: $(date)"
echo "[slurm] Using conda env: bio-utils"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Move to project root
cd ~/crispr12/opencrispr-repro-main

# Inputs
ATLAS_JSON=${ATLAS_JSON:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/crispr-cas-atlas-v1.0.json}
BASE_DIR=${BASE_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna}
PAMPREDICT_DIR=${PAMPREDICT_DIR:-${BASE_DIR}/PAMpredict}
BLAST_DB_DIR=${BLAST_DB_DIR:-${BASE_DIR}/blast_db}
OPERON_IDS_DIR=${OPERON_IDS_DIR:?Set OPERON_IDS_DIR to a directory of chunk_*.txt}
CHUNK_GLOB=${CHUNK_GLOB:-chunk_*.txt}
INDEX_BASE=${INDEX_BASE:-0}
MIN_SPACERS=${MIN_SPACERS:-5}
THREADS=${SLURM_CPUS_PER_TASK:-8}
OUT_BASE=${OUT_BASE:-${BASE_DIR}/pampredict_array_$(date +%Y%m%d_%H%M%S)}

mkdir -p "${OUT_BASE}"

mapfile -t FILES < <(ls -1v "${OPERON_IDS_DIR}"/${CHUNK_GLOB})
N=${#FILES[@]}
IDX=$((SLURM_ARRAY_TASK_ID - INDEX_BASE))
if (( IDX < 0 || IDX >= N )); then
  echo "[slurm] ERROR: Task index ${SLURM_ARRAY_TASK_ID} (base ${INDEX_BASE}) out of range 0..$((N-1))" >&2
  exit 2
fi
OPERON_IDS_FILE=${FILES[$IDX]}

OUTDIR="${OUT_BASE}/arr${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

echo "[slurm] ATLAS_JSON: ${ATLAS_JSON}"
echo "[slurm] OPERON_IDS_FILE: ${OPERON_IDS_FILE}"
echo "[slurm] OUTDIR: ${OUTDIR}"
echo "[slurm] THREADS: ${THREADS}"
echo "[slurm] MIN_SPACERS: ${MIN_SPACERS}"

python -V
which blastn && blastn -version | head -n1 || { echo "[slurm] ERROR: blastn missing" >&2; exit 2; }
test -f "${BLAST_DB_DIR}/viruses_plasmids.fna" || { echo "[slurm] ERROR: BLAST DB missing" >&2; exit 2; }

CMD=(python atlas_make_pam_crispr.py \
  --atlas-json "${ATLAS_JSON}" \
  --pampredict-dir "${PAMPREDICT_DIR}" \
  --dbdir "${BLAST_DB_DIR}" \
  --outdir "${OUTDIR}" \
  --threads "${THREADS}" \
  --min_spacers_per_array "${MIN_SPACERS}" \
  --include-operon-ids "${OPERON_IDS_FILE}")

echo "[slurm] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[slurm] Done: $(date)"
echo "[slurm] Outputs in: ${OUTDIR}"

