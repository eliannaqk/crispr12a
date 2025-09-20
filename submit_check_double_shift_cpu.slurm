#!/bin/bash -l
#SBATCH -J check-double-shift
#SBATCH -p day
#SBATCH --cpus-per-task=36
#SBATCH --time=01:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CUDA_VISIBLE_DEVICES=""

# Activate approved environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env (CPU): $(python -c 'import sys;print(sys.prefix)')"

# Move to project
cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# Inputs
MODEL_PATH=${MODEL_PATH:-"/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-111648/huggingface/ba4000"}
DATA_DIR=${DATA_DIR:-"/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv2"}
OUT=${OUT:-"double_shift_check.json"}

echo "[run] Checking double-shift hypothesis on CPU"
echo "      MODEL_PATH=$MODEL_PATH"
echo "      DATA_DIR=$DATA_DIR"

python -u check_double_shift.py \
  --model-path "$MODEL_PATH" \
  --data-dir "$DATA_DIR" \
  --sequence-col sequence \
  --max-rows 128 \
  --output "$OUT"

echo "[done] Check complete. See $OUT and log above for metrics."

