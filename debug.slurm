#!/bin/bash

#SBATCH --job-name=progen2-debug     # A name for your job
#SBATCH --partition=gpu_devel        # Partition (queue) to run on
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # Request 8 CPUs per task
#SBATCH --time=01:00:00              # Request 10 minutes of runtime
#SBATCH --mem=16G                    # Request 16 GB of memory
#SBATCH --output=slurm-debug-%j.out  # Standard output and error log

# --- Your actual commands start here ---
CONDA_BASE=$(conda info --base)         # or hard-code /home/eqk3/miniconda3
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate opencrispr 

# 2. Navigate to the directory where your script and config are
cd ~/crispr12/opencrispr-repro-main

# 3. Run the debug script
echo "Starting debug script..."
python debug_run.py \
    --config cas12a_ft_finetuneapi.yaml \
    --ckpt /home/eqk3/crispr12/opencrispr-repro-main/huggingface/huggingface/ba46000/


echo "Debug script finished."
