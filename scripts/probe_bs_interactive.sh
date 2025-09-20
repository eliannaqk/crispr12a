#!/bin/bash -l
# Probe the largest generate batch size that fits on a single H200
# Run this INSIDE an interactive GPU shell so you hold the GPU:
#   srun -p gpu_h200 --gres=gpu:h200:1 -t 2:00:00 --pty bash -l
#   source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
#   conda activate oc-opencrispr
#   cd ~/crispr12/opencrispr-repro-main
#   bash scripts/probe_bs_interactive.sh -m <model_path> -c generate_10k.yml

set -euo pipefail

MODEL_PATH=""
CFG_PATH=""
SIZES=""
SAMPLES=64

usage() {
  echo "Usage: $0 -m <model_path> -c <config.yml> [-n <num_samples>] [-s \"32 40 48 ...\"]" >&2
  exit 1
}

while getopts ":m:c:n:s:" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG" ;;
    c) CFG_PATH="$OPTARG" ;;
    n) SAMPLES="$OPTARG" ;;
    s) SIZES="$OPTARG" ;;
    *) usage ;;
  esac
done

[[ -z "$MODEL_PATH" || -z "$CFG_PATH" ]] && usage

if [[ -z "$SIZES" ]]; then
  SIZES="32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160"
fi

echo "[Env] Expecting conda env 'oc-opencrispr'. Active env: $(python -c 'import sys;import os;print(os.environ.get("CONDA_DEFAULT_ENV","<none>"))')"
echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"

TMP_DIR="${SLURM_TMPDIR:-/tmp}/probe_gen_bs_$$"
mkdir -p "$TMP_DIR"
echo "[Tmp] $TMP_DIR"

last_ok=0

for bs in $SIZES; do
  OUT_CFG="$TMP_DIR/gen_bs_${bs}.yml"
  # Override num_samples and batch_size in a copy of the YAML
  sed -E "s/^([[:space:]]*)batch_size:[[:space:]]*[0-9]+/\1batch_size: ${bs}/; s/^([[:space:]]*)num_samples:[[:space:]]*[0-9]+/\1num_samples: ${SAMPLES}/" "$CFG_PATH" > "$OUT_CFG"
  echo "\n[Run] batch_size=${bs} num_samples=${SAMPLES}"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
  LOG="$TMP_DIR/run_bs_${bs}.log"
  set +e
  python -u generate.py --model-path "$MODEL_PATH" --config "$OUT_CFG" --save-folder "$TMP_DIR" --job-idx "$bs" >"$LOG" 2>&1
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "[OK] bs=${bs} succeeded"
    last_ok=$bs
  else
    if grep -qi "cuda out of memory" "$LOG"; then
      echo "[OOM] bs=${bs} failed due to CUDA OOM"
    else
      echo "[FAIL] bs=${bs} failed (exit $status). Last lines:"; tail -n 20 "$LOG" || true
    fi
    break
  fi
done

echo "\n[Result] Largest successful batch_size: ${last_ok}"
if [[ $last_ok -gt 0 ]]; then
  SUGGESTED_CFG="$TMP_DIR/$(basename "$CFG_PATH" .yml)_bs${last_ok}.yml"
  sed -E "s/^([[:space:]]*)batch_size:[[:space:]]*[0-9]+/\1batch_size: ${last_ok}/" "$CFG_PATH" > "$SUGGESTED_CFG"
  echo "[Saved] Suggested config: $SUGGESTED_CFG"
fi

echo "[Done] Inspect logs in $TMP_DIR"

