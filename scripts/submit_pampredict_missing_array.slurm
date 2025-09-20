#!/bin/bash -l
#SBATCH -J pampredict_missing
#SBATCH -p day
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o slurm-pampredict-missing-%A_%a.out

set -euo pipefail

echo "[slurm] Hostname: $(hostname)"
echo "[slurm] Start time: $(date)"
echo "[slurm] Using conda env: bio-utils"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate bio-utils
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Required env
: "${PAMPREDICT_DIR:?need PAMPREDICT_DIR}"
: "${BLAST_DB_DIR:?need BLAST_DB_DIR}"
: "${CHUNK_DIR:?need CHUNK_DIR}"

THREADS=${THREADS:-32}
IDX_BASE=${INDEX_BASE:-0}
IDX=${SLURM_ARRAY_TASK_ID}
printf -v CHUNK_NAME "chunk_%04d.tsv" $((IDX - IDX_BASE))
CHUNK_FILE="${CHUNK_DIR}/${CHUNK_NAME}"

echo "[slurm] THREADS=${THREADS}"
echo "[slurm] BLAST_DB_DIR=${BLAST_DB_DIR}"
echo "[slurm] PAMPREDICT_DIR=${PAMPREDICT_DIR}"
echo "[slurm] CHUNK_FILE=${CHUNK_FILE}"

python -V
which blastn && blastn -version | head -n1
which samtools && samtools --version | head -n1

test -f "${CHUNK_FILE}" || { echo "[err] Missing chunk file: ${CHUNK_FILE}" >&2; exit 2; }
test -f "${BLAST_DB_DIR}/viruses_plasmids.fna" || { echo "[err] Missing BLAST DB fasta" >&2; exit 2; }
if [[ ! -f "${BLAST_DB_DIR}/viruses_plasmids.fna.fai" ]]; then
  echo "[slurm] Building samtools faidx index for viruses_plasmids.fna ..."
  samtools faidx "${BLAST_DB_DIR}/viruses_plasmids.fna"
fi

PAM_PY="${PAMPREDICT_DIR}/PAMpredict.py"
if [[ ! -f "$PAM_PY" ]]; then PAM_PY="${PAMPREDICT_DIR}/PAMpredict/PAMpredict.py"; fi
test -f "$PAM_PY" || { echo "[err] Cannot find PAMpredict.py in ${PAMPREDICT_DIR}" >&2; exit 2; }

echo "[slurm] Processing arrays in ${CHUNK_FILE}"
N=0; DONE=0; RUN=0; FAIL=0
while IFS=$'\t' read -r OUTDIR NAME; do
  [[ -z "$OUTDIR" || -z "$NAME" ]] && continue
  let N+=1
  SPACER_FA="${OUTDIR}/spacers/${NAME}.fna"
  RUN_DIR="${OUTDIR}/pampredict_runs/${NAME}"
  UP="${RUN_DIR}/upstream_flanking_sequence_info.tsv"
  DN="${RUN_DIR}/downstream_flanking_sequence_info.tsv"
  if [[ -f "$UP" && -f "$DN" ]]; then
    echo "[skip] ${NAME} already has info tables"
    let DONE+=1; continue
  fi
  mkdir -p "$RUN_DIR"
  echo "[run] $(date +%H:%M:%S) ${NAME}"
  set +e
  /usr/bin/env python "$PAM_PY" "$SPACER_FA" "$BLAST_DB_DIR" "$RUN_DIR" -p UPSTREAM -l 10 -t "$THREADS" --keep_tmp --force
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    let RUN+=1
  else
    let FAIL+=1
  fi
done < "$CHUNK_FILE"

echo "[slurm] N=$N DONE=$DONE RUN=$RUN FAIL=$FAIL"
echo "[slurm] Finished at $(date)"

