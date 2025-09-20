#!/bin/bash -l
#SBATCH -J eval_ba2000_cpu36
#SBATCH --cpus-per-task=36
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
# Ensure CPU-only
export CUDA_VISIBLE_DEVICES=

# Activate approved environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

# Move to project directory
cd ~/crispr12/opencrispr-repro-main

# Optional: enable W&B in non-interactive job if token configured
# export WANDB_API_KEY=""

# Log context
echo "[job] Env=oc-opencrispr | Host=$(hostname) | CPUs= | Step=2000"

python eval_perplexity.py \
  --ckpt \
    /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250824-222758/huggingface/ba2000 \
  --model-path \
    /home/eqk3/project_pi_mg269/eqk3/crisprData/model/zenodo_15128064 \
  --baseline-model-path \
    /home/eqk3/project_pi_mg269/eqk3/crisprData/model/zenodo_15128064 \
  --data-dir \
    /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv2 \
  --sequence-col sequence \
  --batch-size 8 \
  --only-val-and-train-sample \
  --output-dir eval_outputs/submit_eval_run-20250824-222758/ba2000
