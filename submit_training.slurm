#!/bin/bash

#SBATCH --job-name=progen2-finetune  # A name for your job
#SBATCH --partition=gpu          # The GPU partition
#SBATCH --gres=gpu:a100:1              # Request 4 A100 GPUs
#SBATCH --cpus-per-task=8          # Request 32 CPUs (8 per GPU)
#SBATCH --time=08:00:00              # Request 4 hours of runtime
#SBATCH --mem=160G                   # Request 160 GB of memory (40GB per GPU)
#SBATCH --output=./slurm-%j.out        # Standard output and error log

# --- Your actual commands start here ---

# 1. Load necessary modules (if any, e.g., for conda or a specific cuda version)
# Example: module load miniconda

# 2. Activate your conda environment
source activate opencrispr

# 3. Navigate to the directory where your script and config are
cd ~/crispr12/opencrispr-repro-main

# 4. Set your W&B API key
export WANDB_API_KEY='fa3648edd4db4a0bbcf4157a516c81da9940463e'

# 5. Run the training command
echo "Starting training..."
#python main.py --config cas12a_ft_finetuneapi.yaml
python main.py --config cas12a_lora.yaml

echo "Training finished."
