#!/bin/bash -l

#SBATCH --job-name=opencrispr-train-h200   # Job name
#SBATCH -A pi_mg269                        # Account
#SBATCH --partition=gpu_h200               # Bouchet H200 partition
#SBATCH --gres=gpu:h200:1                # Default: 1x H200 (override with: sbatch --gres=gpu:h200:N ...)
#SBATCH --ntasks=1                         # Single task
#SBATCH --cpus-per-task=8                  # CPU cores per task
#SBATCH --time=8:00:00                    # Walltime
#SBATCH --output=./slurm-%j.out            # Log file

# Set reasonable threading
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate environment
conda activate oc-opencrispr

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

# Optional: W&B
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

echo "Starting training on $(hostname) with $SLURM_GPUS_ON_NODE GPUs"

# Launch training
torchrun --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE} main.py --config cas12a_ft_finetuneapi.yaml

echo "Training finished."


