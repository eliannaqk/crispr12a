#!/bin/bash -l
# Cluster Cas12a proteins at 95% ID and split by clusters into train/test.
# - Prefers bio-utils env for bioinformatics tools.
# - Uses mmseqs2 (via local install or Apptainer Docker image) to cluster.
# - Ensures all members of a cluster stay together in the same split.

set -euo pipefail

# Config
FASTA_IN=${FASTA_IN:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_atlas_raw.faa}
OUT_DIR=${OUT_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a}
PREFIX=${PREFIX:-raw95}
TMPDIR=${TMPDIR:-${OUT_DIR}/mm_tmp_${PREFIX}}
TEST_FRAC=${TEST_FRAC:-0.10}   # fraction of clusters to place in test
THREADS=${THREADS:-32}

# Env
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate bio-utils
echo "[env] Using conda environment: ${CONDA_DEFAULT_ENV}" 1>&2

mkdir -p "${OUT_DIR}" "${TMPDIR}"
cd "${OUT_DIR}"

PFX_PATH="${OUT_DIR}/${PREFIX}"
CLUST_TSV="${PFX_PATH}_cluster.tsv"

echo "[info] Input FASTA: ${FASTA_IN}" 1>&2
echo "[info] Output prefix: ${PFX_PATH}" 1>&2
echo "[info] Temp dir: ${TMPDIR}" 1>&2

# Detect mmseqs2 runner
RUN_MM=()
if command -v mmseqs >/dev/null 2>&1; then
  RUN_MM=(mmseqs)
else
  if command -v apptainer >/dev/null 2>&1; then
    RUN_MM=(apptainer exec docker://soedinglab/mmseqs2:latest mmseqs)
  elif command -v singularity >/dev/null 2>&1; then
    RUN_MM=(singularity exec docker://soedinglab/mmseqs2:latest mmseqs)
  else
    echo "[error] mmseqs2 not found, and no apptainer/singularity available." 1>&2
    exit 1
  fi
fi

# Cluster if needed
if [[ ! -s "${CLUST_TSV}" ]]; then
  echo "[mmseqs] Clustering at 95% identity..." 1>&2
  "${RUN_MM[@]}" easy-cluster \
    "${FASTA_IN}" "${PFX_PATH}" "${TMPDIR}" \
    --min-seq-id 0.95 --cov-mode 0 -c 0.8 --cluster-mode 0 --threads "${THREADS}"
else
  echo "[mmseqs] Reusing existing clusters: ${CLUST_TSV}" 1>&2
fi

echo "[count] clusters lines: $(wc -l < "${CLUST_TSV}")" 1>&2

# Python split by clusters (train/test), keep all members of a cluster together
python - << 'PY'
import random,sys,os
from collections import defaultdict
random.seed(42)
out_dir = os.environ.get('OUT_DIR')
pfx = os.environ.get('PREFIX')
test_frac = float(os.environ.get('TEST_FRAC','0.10'))
fasta_in = os.environ.get('FASTA_IN')
clust_tsv = f"{out_dir}/{pfx}_cluster.tsv"

rep2mem = defaultdict(set)
with open(clust_tsv) as f:
    for line in f:
        rep, mem = line.rstrip().split('\t')
        rep2mem[rep].add(mem)
clusters = list(rep2mem.items())
num_clusters = len(clusters)
test_clusters = max(1, round(num_clusters * test_frac))
random.shuffle(clusters)
test_reps = set(rep for rep,_ in clusters[:test_clusters])

# Load sequences
seqs = {}
h=None; buf=[]
with open(fasta_in) as f:
    for line in f:
        if line.startswith('>'):
            if h is not None:
                seqs[h] = ''.join(buf)
            h = line[1:].strip().split()[0]
            buf=[]
        else:
            buf.append(line.strip())
if h is not None:
    seqs[h] = ''.join(buf)

test_set, train_set = set(), set()
for rep, mems in clusters:
    (test_set if rep in test_reps else train_set).update(mems)

with open(f"{out_dir}/train_{pfx}.faa",'w') as ft, open(f"{out_dir}/test_{pfx}.faa",'w') as fv:
    for hid, s in seqs.items():
        if hid in test_set:
            fv.write(f'>{hid}\n{s}\n')
        else:
            ft.write(f'>{hid}\n{s}\n')
print(f"[split95] clusters={num_clusters} test_clusters={len(test_reps)} train_seqs={sum(1 for h in seqs if h in train_set)} test_seqs={sum(1 for h in seqs if h in test_set)}", file=sys.stderr)
PY

echo -n "[count] train_${PREFIX}.faa: "; grep -c '^>' "${OUT_DIR}/train_${PREFIX}.faa" || true
echo -n "[count] test_${PREFIX}.faa: "; grep -c '^>' "${OUT_DIR}/test_${PREFIX}.faa" || true

# Convert to CSV with the repo helper (requires oc-opencrispr env)
conda activate oc-opencrispr
python /home/eqk3/crispr12/opencrispr-repro-main/fasta_to_csv.py "${OUT_DIR}/train_${PREFIX}.faa" "${OUT_DIR}/train_${PREFIX}.csv"
python /home/eqk3/crispr12/opencrispr-repro-main/fasta_to_csv.py "${OUT_DIR}/test_${PREFIX}.faa" "${OUT_DIR}/test_${PREFIX}.csv"

ls -lh "${OUT_DIR}/train_${PREFIX}."* "${OUT_DIR}/test_${PREFIX}."* | sed -n '1,200p'

