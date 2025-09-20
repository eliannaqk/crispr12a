#!/bin/bash -l
#SBATCH -J cas12a_mmseqs95
#SBATCH -p day
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Load apptainer for mmseqs2 container
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

# Bio tools env for helpers
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate bio-utils

cd /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a
IN=cas12a_atlas_raw.faa
PFX=raw95
TMP=mm_tmp_95

echo "[count] input seqs: $(grep -c '^>' "$IN")"

# Cluster at 95% identity with mmseqs2 in container
if [[ ! -s ${PFX}_cluster.tsv ]]; then
  rm -rf "$TMP" || true; mkdir -p "$TMP"
  if command -v apptainer >/dev/null 2>&1; then
    apptainer exec docker://soedinglab/mmseqs2:latest \
      mmseqs easy-cluster "$IN" "$PFX" "$TMP" \
      --min-seq-id 0.95 --cov-mode 0 -c 0.8 --cluster-mode 0 --threads ${SLURM_CPUS_PER_TASK}
  else
    singularity exec docker://soedinglab/mmseqs2:latest \
      mmseqs easy-cluster "$IN" "$PFX" "$TMP" \
      --min-seq-id 0.95 --cov-mode 0 -c 0.8 --cluster-mode 0 --threads ${SLURM_CPUS_PER_TASK}
  fi
else
  echo "[mmseqs] Reusing existing ${PFX}_cluster.tsv"
fi

echo "[count] ${PFX}_cluster.tsv lines: $(wc -l < ${PFX}_cluster.tsv)"

# Split clusters 90/10 and write FASTA
python - << 'PY'
import random,sys
random.seed(42)
rep2mem = {}
for line in open('raw95_cluster.tsv'):
    rep, mem = line.rstrip().split('\t')
    rep2mem.setdefault(rep, set()).add(mem)
clusters = list(rep2mem.items())
num_clusters = len(clusters)
# 10% clusters to test
test_clusters = max(1, round(num_clusters * 0.10))
random.shuffle(clusters)
test_reps = set(rep for rep,_ in clusters[:test_clusters])
# load sequences
seqs = {}
h=None; buf=[]
for line in open('cas12a_atlas_raw.faa'):
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
with open('train_raw95.faa','w') as ft, open('test_raw95.faa','w') as fv:
    for hid, s in seqs.items():
        if hid in test_set:
            fv.write(f'>{hid}\n{s}\n')
        else:
            ft.write(f'>{hid}\n{s}\n')
print(f"[split95] clusters={num_clusters} test_clusters={len(test_reps)} train_seqs={sum(1 for h in seqs if h in train_set)} test_seqs={sum(1 for h in seqs if h in test_set)}", file=sys.stderr)
PY

echo "[count] train_raw95.faa: $(grep -c '^>' train_raw95.faa)"
echo "[count] test_raw95.faa:  $(grep -c '^>' test_raw95.faa)"

# Convert to CSV
conda activate oc-opencrispr
python /home/eqk3/crispr12/opencrispr-repro-main/fasta_to_csv.py train_raw95.faa train_raw95.csv
python /home/eqk3/crispr12/opencrispr-repro-main/fasta_to_csv.py test_raw95.faa test_raw95.csv

ls -lh train_raw95.* test_raw95.* | sed -n '1,200p'
