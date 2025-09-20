#!/bin/bash -l
#SBATCH -J eval-ppl-aug21-first
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o slurm-eval-aug21-first-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] Initializing conda and activating oc-opencrispr..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

cd ~/crispr12/opencrispr-repro-main
echo "[cwd] $(pwd)"

# WANDB (uses cached login; can be overridden by exporting API key)
export WANDB_PROJECT=${WANDB_PROJECT:-crispr12a}
export WANDB_ENTITY=${WANDB_ENTITY:-eqk3}
export WANDB_API_KEY=${WANDB_API_KEY:-"fa3648edd4db4a0bbcf4157a516c81da9940463e"}

RUNS_BASE="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21"
FIRST_RUN=$(ls -1 "$RUNS_BASE" | sort | head -n1)
RUN_DIR="$RUNS_BASE/$FIRST_RUN"
HF_DIR="$RUN_DIR/huggingface"
BASELINE_HF="/home/eqk3/project_pi_mg269/eqk3/crisprData/model/zenodo_15128064/"

echo "[run] First run: $FIRST_RUN"

# Data directory: prefer csv (has test.csv); fallback to csv2 with a shim
DATA_DIR_BASE="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/final_data_training_aug20"
PREF="$DATA_DIR_BASE/csv"
ALT="$DATA_DIR_BASE/csv2"

if [ -d "$PREF" ]; then
  DATA_DIR="$PREF"
else
  DATA_DIR="$ALT"
fi

# If no test.csv exists, create a temp eval dir where valid.csv is duplicated to test.csv
if [ ! -f "$DATA_DIR/test.csv" ]; then
  echo "[data] No test.csv in $DATA_DIR; creating shim with valid as test..."
  TMP_EVAL_DIR="eval_data_shim_${SLURM_JOB_ID:-$$}"
  rm -rf "$TMP_EVAL_DIR" && mkdir -p "$TMP_EVAL_DIR"
  for f in train.csv valid.csv; do
    if [ -f "$DATA_DIR/$f" ]; then ln -sf "$DATA_DIR/$f" "$TMP_EVAL_DIR/$f"; fi
  done
  if [ -f "$DATA_DIR/valid.csv" ]; then ln -sf "$DATA_DIR/valid.csv" "$TMP_EVAL_DIR/test.csv"; fi
  DATA_DIR="$TMP_EVAL_DIR"
fi

# Verify torch CUDA
python - << 'PY'
import torch
print('[env] torch', torch.__version__)
print('[env] cuda available', torch.cuda.is_available())
print('[env] device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY

# Collect available steps
mapfile -t STEPS < <(ls -1 "$HF_DIR" 2>/dev/null | sed -n 's/^ba\([0-9][0-9]*\)$/\1/p' | sort -n)
if [ ${#STEPS[@]} -eq 0 ]; then
  echo "[error] No HF steps under $HF_DIR" >&2
  exit 1
fi

# Select 3 checkpoints: 6000 (if present), a middle, and the last
SEL=()
if printf '%s\n' "${STEPS[@]}" | grep -qx "6000"; then
  SEL+=("6000")
else
  SEL+=("${STEPS[0]}")
fi
MID_IDX=$((${#STEPS[@]} / 2))
SEL+=("${STEPS[$MID_IDX]}")
SEL+=("${STEPS[$((${#STEPS[@]}-1))]}")

echo "[info] Evaluating $FIRST_RUN at steps: ${SEL[*]}"

for STEP in "${SEL[@]}"; do
  MODEL_PATH="$HF_DIR/ba$STEP"
  if [ ! -d "$MODEL_PATH" ]; then
    echo "[warn] Missing $MODEL_PATH; skipping" >&2
    continue
  fi
  OUT_DIR="eval_outputs/aug21/$FIRST_RUN/ba$STEP"
  mkdir -p "$OUT_DIR"
  echo "[eval] $FIRST_RUN ba$STEP -> $OUT_DIR" >&2
  python eval_perplexity.py \
    --model-path "$MODEL_PATH" \
    --baseline-model-path "$BASELINE_HF" \
    --data-dir "$DATA_DIR" \
    --batch-size 8 \
    --output-dir "$OUT_DIR" \
    --wandb-run-name "ppl-$FIRST_RUN-ba$STEP" | tee "$OUT_DIR/run.log"
done

echo "[done] Results under eval_outputs/aug21/$FIRST_RUN/"

