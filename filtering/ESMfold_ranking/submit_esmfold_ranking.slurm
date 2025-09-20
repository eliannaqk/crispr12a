#!/bin/bash -l
#SBATCH -J esmfold_rank
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Init conda and activate ESMFold-ready env
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm

# Ensure conda's libstdc++ is preferred over system to avoid GLIBCXX issues
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "[run] LD_LIBRARY_PATH prefixes ${CONDA_PREFIX}/lib" 1>&2

# Move to project root
cd ~/crispr12/opencrispr-repro-main

# Optional: set W&B if you want logging via env
# export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Inputs (override via: sbatch ... --export=ALL,FASTA=...,REF=...,DOMAINS=...,OUT=...,PDB_DIR=...)
FASTA=${FASTA:-/path/to/cas12a_candidates.fasta}
REF=${REF:-/path/to/lbcas12a_ref.fasta}
DOMAINS=${DOMAINS:-filtering/ESMfold_ranking/lb_domains.yaml}
OUT=${OUT:-cas12a_esmfold_scores.csv}
PDB_DIR=${PDB_DIR:-esm_pdbs}

echo "[run] env=$(conda info --envs | sed -n 's/^\* //p' | awk '{print $1}')" 1>&2
echo "[run] fasta=$FASTA" 1>&2
echo "[run] ref=$REF" 1>&2
echo "[run] domains=$DOMAINS" 1>&2

mkdir -p "$PDB_DIR"

python esmfold_domain_assessor.py \
  --fasta "$FASTA" \
  --ref_fasta "$REF" \
  --domains "$DOMAINS" \
  --pdb_dir "$PDB_DIR" \
  --out_csv "$OUT"
