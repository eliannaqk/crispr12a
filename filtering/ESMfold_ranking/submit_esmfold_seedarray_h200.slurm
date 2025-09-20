#!/bin/bash -l
#SBATCH -J esmfold_seedarray
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --array=1-4
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Initialize Conda and activate the approved env
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

# Log environment for transparency
echo "[env] Using conda env: oc-opencrispr-esm" 1>&2

# Inputs
# Export these when submitting, e.g.:
# sbatch filtering/ESMfold_ranking/submit_esmfold_seedarray_h200.slurm \
#   --export=ALL,SEED_ROOT=/path/to/by-seed,REF=/path/to/lbcas12a_ref.fasta

SEED_ROOT=${SEED_ROOT:-/nfs/roberts/project/pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-YYYYMMDD-XXXX/by-seed}
REF=${REF:-/path/to/lbcas12a_ref.fasta}
DOMAINS=${DOMAINS:-filtering/ESMfold_ranking/lb_domains.yaml}
OUT_DIR=${OUT_DIR:-esmfold_seed_outputs}

# Optional memory-control knobs for ESMFold
CHUNK_SIZE=${CHUNK_SIZE:-128}
RECYCLES=${RECYCLES:-1}
ESMFOLD_VARIANT=${ESMFOLD_VARIANT:-esmfold_v1}

SEED=seed${SLURM_ARRAY_TASK_ID}
FASTA="${SEED_ROOT}/${SEED}/${SEED}.filtered_step7_pass.fasta"

mkdir -p "$OUT_DIR" "$OUT_DIR/esm_pdbs_${SEED}"

echo "[run] seed=$SEED" 1>&2
echo "[run] fasta=$FASTA" 1>&2
echo "[run] ref=$REF" 1>&2
echo "[run] domains=$DOMAINS" 1>&2

python esmfold_domain_assessor.py \
  --fasta "$FASTA" \
  --ref_fasta "$REF" \
  --domains "$DOMAINS" \
  --pdb_dir "$OUT_DIR/esm_pdbs_${SEED}" \
  --out_csv "$OUT_DIR/esmfold_scores_${SEED}.csv" \
  --chunk-size "$CHUNK_SIZE" \
  --recycles "$RECYCLES" \
  --esmfold-variant "$ESMFOLD_VARIANT"
