#!/bin/bash -l
#SBATCH -J eval-ppl-aug21-remaining
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH -o slurm-%j.out

set -euo pipefail

echo "[env] Initializing Conda and activating oc-opencrispr"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
echo "[env] Active env: $(conda info --json | jq -r .active_prefix 2>/dev/null || echo oc-opencrispr)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="fa3648edd4db4a0bbcf4157a516c81da9940463e"

cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

RUNS_ROOT=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21
BASELINE_MODEL=/home/eqk3/project_pi_mg269/eqk3/crisprData/model/zenodo_15128064/
VALID_CSV=/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20/csv2/valid.csv

if [[ ! -d "$BASELINE_MODEL" ]]; then
  echo "[error] Baseline model path not found: $BASELINE_MODEL" >&2
  exit 1
fi
if [[ ! -f "$VALID_CSV" ]]; then
  echo "[error] valid.csv not found at $VALID_CSV" >&2
  exit 1
fi

# Prepare test.csv from valid.csv as before
DATA_DIR=${SLURM_TMPDIR:-/tmp}/eval_data_${SLURM_JOB_ID}
mkdir -p "$DATA_DIR"
cp -f "$VALID_CSV" "$DATA_DIR/test.csv"
echo "[data] Using $DATA_DIR/test.csv (copied from valid.csv)"

echo "[gpu] CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Remaining runs to evaluate (excluding run-20250822-094416 which was already evaluated separately)
RUNS=(
  "$RUNS_ROOT/run-20250821-205122"
  "$RUNS_ROOT/run-20250821-211403"
)

STEPS=(2000 4000)

run_eval() {
  local run_dir="$1"; shift
  local step="$1"; shift
  local ckpt_hf_dir="$run_dir/huggingface/ba${step}"
  local wandb_name="eval_ppl:$(basename "$run_dir")_ba${step}_with_base"
  if [[ ! -d "$ckpt_hf_dir" ]]; then
    echo "[warn] Missing HF export for step ${step} at $ckpt_hf_dir; skipping"
    return 0
  fi
  echo "[eval] run=$(basename "$run_dir"), step=$step"
  python eval_perplexity.py \
    --ckpt "$ckpt_hf_dir" \
    --config cas12a_ft_finetuneapi.yaml \
    --baseline-model-path "$BASELINE_MODEL" \
    --data-dir "$DATA_DIR" \
    --batch-size 8 \
    --wandb-project crispr12a \
    --wandb-entity eqk3 \
    --wandb-run-name "$wandb_name"
}

for RUN in "${RUNS[@]}"; do
  if [[ ! -d "$RUN" ]]; then
    echo "[warn] Run dir not found: $RUN; skipping"; continue
  fi
  for STEP in "${STEPS[@]}"; do
    run_eval "$RUN" "$STEP"
  done
done

echo "[done] Completed evaluations for remaining Aug21 runs with baseline"

