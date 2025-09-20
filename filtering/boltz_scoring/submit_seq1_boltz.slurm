#!/bin/bash -l
#SBATCH -J boltz_seq1
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm

cd ~/crispr12/opencrispr-repro-main

CSV_PATH="filtering/boltz_scoring/inputs/seq1_reference.csv"
OUTDIR="filtering/boltz_scoring/boltz_runs/seq1_reference"
SCORES_OUT="metrics/seq1_reference_boltz_scores.csv"
WED_RES=""

mkdir -p "${OUTDIR}"
mkdir -p metrics

echo "Running in env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(python --version)"

python filtering/boltz_scoring/run_boltz_prerna.py \
  --csv "${CSV_PATH}" \
  --outdir "${OUTDIR}" \
  --use_msa_server \
  --accelerator gpu \
  --devices "${SLURM_GPUS_ON_NODE:-1}" \
  --model boltz1 \
  --override

python filtering/boltz_scoring/score_wed_prerna.py \
  --csv "${CSV_PATH}" \
  --runs "${OUTDIR}" \
  --wed_residues "${WED_RES}" \
  --out_csv "${SCORES_OUT}"
