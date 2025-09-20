#!/bin/bash -l
#SBATCH -J eval-ppl-first-last
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "[env] Initializing Conda and activating oc-opencrispr"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Using conda env: $(conda info --json | jq -r .active_prefix 2>/dev/null || echo oc-opencrispr)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TOKENIZERS_PARALLELISM=false

# Optional: set W&B key if not already present (non-interactive job)
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# Paths and data prep
RUNS_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21
VALID_CSV=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv2/valid.csv

if [[ ! -f "$VALID_CSV" ]]; then
  echo "[error] valid.csv not found at $VALID_CSV" >&2
  exit 1
fi

# Prepare a data directory with test.csv from valid.csv as requested
DATA_DIR=${SLURM_TMPDIR:-/tmp}/eval_data_$SLURM_JOB_ID
mkdir -p "$DATA_DIR"
cp -f "$VALID_CSV" "$DATA_DIR/test.csv"
echo "[data] Using $DATA_DIR/test.csv (copied from valid.csv)"

# Discover first and last run directories (lexicographic order)
mapfile -t RUN_DIRS < <(ls -1d "$RUNS_ROOT"/*/ | sed 's@/$@@' | sort)
if (( ${#RUN_DIRS[@]} < 1 )); then
  echo "[error] No runs found under $RUNS_ROOT" >&2
  exit 1
fi
FIRST_RUN=${RUN_DIRS[0]}
LAST_RUN=${RUN_DIRS[${#RUN_DIRS[@]}-1]}
echo "[runs] First: $FIRST_RUN"
echo "[runs] Last : $LAST_RUN"

# Steps strictly before 6000
STEPS=(2000 4000)

echo "[gpu] CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

run_eval() {
  local run_dir="$1"
  local step="$2"
  local ckpt_hf_dir="$run_dir/huggingface/ba${step}"
  local wandb_name="eval_ppl:$(basename "$run_dir")_ba${step}"
  if [[ ! -d "$ckpt_hf_dir" ]]; then
    echo "[warn] Missing HF export for step ${step} at $ckpt_hf_dir; skipping"
    return 0
  fi
  echo "[eval] Run=$(basename "$run_dir"), step=$step, ckpt=$ckpt_hf_dir"
  python eval_perplexity.py \
    --ckpt "$ckpt_hf_dir" \
    --config cas12a_ft_finetuneapi.yaml \
    --data-dir "$DATA_DIR" \
    --batch-size 8 \
    --wandb-project crispr12a \
    --wandb-entity eqk3 \
    --wandb-run-name "$wandb_name"
}

for RUN in "$FIRST_RUN" "$LAST_RUN"; do
  for STEP in "${STEPS[@]}"; do
    run_eval "$RUN" "$STEP"
  done
done

echo "[done] Submitted evaluations for first/last runs at steps < 6000"

