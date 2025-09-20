#!/bin/bash -l
#SBATCH -J score_seed14
#SBATCH -p devel
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CUDA_VISIBLE_DEVICES=

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

echo "[env] Using conda env: oc-opencrispr" 1>&2
echo "[env] Host=$(hostname) | CPUs=${SLURM_CPUS_PER_TASK:-1}" 1>&2

cd ~/crispr12/opencrispr-repro-main

export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

BY_SEED_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/by-seed"
NOVELTY_ROOT="${NOVELTY_ROOT:-eval_outputs/novelty_cluster_run-20250828-193254-ba4000-sweep_ba4000}"
ESMFOLD_ROOT="${ESMFOLD_ROOT:-esmfold_seed_outputs}"
ESMFOLD_BASELINE_ROOT="${ESMFOLD_BASELINE_ROOT:-esmfold_baseline/full_model_single}"
OUTPUT_DIR="${OUTPUT_DIR:-metrics}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ba4000_composite_seed14}"
SEEDS="${SEEDS:-seed1 seed4}"

echo "[run] seeds=$SEEDS" 1>&2
echo "[run] by_seed_dir=$BY_SEED_DIR" 1>&2
echo "[run] novelty_root=$NOVELTY_ROOT" 1>&2
echo "[run] esmfold_root=$ESMFOLD_ROOT" 1>&2
echo "[run] esmfold_baseline_root=$ESMFOLD_BASELINE_ROOT" 1>&2
echo "[run] output_dir=$OUTPUT_DIR" 1>&2
echo "[run] output_prefix=$OUTPUT_PREFIX" 1>&2

python -m filtering.composite_scoring.score_pipeline \
  --by-seed-dir "$BY_SEED_DIR" \
  --novelty-root "$NOVELTY_ROOT" \
  --esmfold-root "$ESMFOLD_ROOT" \
  --esmfold-baseline-root "$ESMFOLD_BASELINE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --seeds $SEEDS
