#!/bin/bash -l
#SBATCH -J pid8A
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate the approved environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

# If using Weights & Biases logging elsewhere in the pipeline, ensure login is cached.
# export WANDB_API_KEY="<your_key>"

# Example run (edit paths/IDs as needed):
# torch is not required; this script runs on CPU by default unless dependencies use GPU.
python filtering/pid_8A_filter_fullseq.py \
  --pdb /path/to/LbCas12a_DNA_complex.pdb \
  --protein-chain A \
  --pid-range 900-1068 \
  --designs /path/to/designs_full.faa \
  --outdir filtering/out_pid_example \
  --cutoff 8.0 \
  --save-contacts filtering/out_pid_example/lb_pid_contacts.json

