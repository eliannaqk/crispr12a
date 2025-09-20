#!/bin/bash -l
#SBATCH -J lbdr_tiny
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

PREP_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/prep_20250828_141345_dr_aatttctactaagtgtagat"
TRAIN_PREP="$PREP_DIR/cas12a_training_prepped.csv"
mkdir -p tmp_lbdr
head -n 201 "$TRAIN_PREP" > tmp_lbdr/train_small.csv
(head -n 1 "$TRAIN_PREP" && tail -n +202 "$TRAIN_PREP" | head -n 100) > tmp_lbdr/valid_small.csv

echo "[env] $(conda info --envs | grep '*' || true)"; nvidia-smi || true

python filtering/train_lbdr_classifier_esm.py \
  --train_csv tmp_lbdr/train_small.csv \
  --valid_csv tmp_lbdr/valid_small.csv \
  --out_model tmp_lbdr/lbdr_cls_tiny.pt \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-3

ls -lah tmp_lbdr || true

