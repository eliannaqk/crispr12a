#!/bin/bash -l
#SBATCH -J boltz_prerna
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate approved environment (ESM/GPU-ready)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

# Optional: set W&B if you want logging (avoid committing keys in code)
# export WANDB_API_KEY="<your_key>"

echo "Running in env: ${CONDA_DEFAULT_ENV} (expected: oc-opencrispr-esm)"
echo "Python: $(python --version)" 
echo "Boltz: $(boltz --version 2>/dev/null || echo 'not found')"

# Inputs (edit as needed)
CSV_PATH="inputs.csv"  # path to your CSV (id, protein_seq|protein_fasta, rna_seq, kkh_residues)
OUTDIR="filtering/boltz_scoring/boltz_runs"
WED_RES=""  # e.g., "500,501,502,503,504,505,506,507,508,509"

# 1) Prepare and run Boltz predictions (MSA server recommended)
python filtering/boltz_scoring/run_boltz_prerna.py \
  --csv "${CSV_PATH}" \
  --outdir "${OUTDIR}" \
  --use_msa_server

# 2) Score results
python filtering/boltz_scoring/score_wed_prerna.py \
  --csv "${CSV_PATH}" \
  --runs "${OUTDIR}" \
  --wed_residues "${WED_RES}" \
  --out_csv filtering/boltz_scoring/boltz_screen_scores.csv

# For multi-GPU (if available), run multiple jobs or increase concurrency upstream.

