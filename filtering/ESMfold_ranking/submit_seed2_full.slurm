#!/bin/bash -l
#SBATCH -J seed2_full_esmfold
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[env] Using conda env: oc-opencrispr-esm" 1>&2
echo "[env] LD_LIBRARY_PATH=$LD_LIBRARY_PATH" 1>&2

cd ~/crispr12/opencrispr-repro-main

export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

FASTA=${FASTA:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/by-seed/seed2/seed2.filtered_step7_pass.fasta}
REF=${REF:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/by-seed/seed1/step4_motif/anchor.faa}
DOMAINS=${DOMAINS:-filtering/ESMfold_ranking/lb_domains.yaml}
OUT_DIR=${OUT_DIR:-esmfold_seed_outputs/full_seed2}
CHUNK_SIZE=${CHUNK_SIZE:-128}
RECYCLES=${RECYCLES:-2}
ESMFOLD_VARIANT=${ESMFOLD_VARIANT:-esmfold_v1}

mkdir -p "$OUT_DIR" "$OUT_DIR/esm_pdbs_seed2_full"

echo "[run] fasta=$FASTA" 1>&2
echo "[run] ref=$REF" 1>&2
echo "[run] domains=$DOMAINS" 1>&2
echo "[run] out_dir=$OUT_DIR" 1>&2
echo "[run] chunk_size=$CHUNK_SIZE recycles=$RECYCLES variant=$ESMFOLD_VARIANT" 1>&2

python esmfold_domain_assessor.py \
  --fasta "$FASTA" \
  --ref_fasta "$REF" \
  --domains "$DOMAINS" \
  --pdb_dir "$OUT_DIR/esm_pdbs_seed2_full" \
  --out_csv "$OUT_DIR/esmfold_scores_seed2_full.csv" \
  --chunk-size "$CHUNK_SIZE" \
  --recycles "$RECYCLES" \
  --esmfold-variant "$ESMFOLD_VARIANT"
