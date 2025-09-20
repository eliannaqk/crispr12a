#!/bin/bash -l
#SBATCH -J lbdr-eval-h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-%j.out
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate correct environment for ESM models
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr-esm

cd ~/crispr12/opencrispr-repro-main
echo "[env] Using oc-opencrispr-esm; running eval on H200"

# If needed, you can export a WANDB key (prefer cached login)
# export WANDB_API_KEY="${WANDB_API_KEY}"

python -m filtering.eval_lbdr_classifier_esm \
  --train_csv /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_dr_datasets/train_labeled_pos.csv \
  --valid_csv /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_dr_datasets/valid_labeled_pos.csv \
  --model_path /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_dr_datasets/lbdr_cls.pt \
  --threshold 0.5 \
  --wandb-project crispr_filtering \
  --wandb-entity eqk3 \
  --wandb-run-id 9e73j1ij

