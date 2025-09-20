#!/bin/bash
set -euo pipefail

# Usage: ./generation/submit_generate_ba_sweep_parallel.sh <MODEL_PATH>
# Submits four parallel jobs (seed1..seed4) on gpu_h200 using oc-opencrispr.

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <MODEL_PATH>" >&2
  exit 1
fi

MODEL_PATH="$1"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[error] MODEL_PATH not a directory: $MODEL_PATH" >&2
  exit 1
fi

echo "[submit] Submitting sweep jobs for model: $MODEL_PATH"
for seed in seed1 seed2 seed3 seed4; do
  echo "[submit] -> $seed"
  sbatch --export=ALL,MODEL_PATH="$MODEL_PATH",SEED_SET="$seed" generation/submit_generate_ba_sweep.slurm
done

echo "[submit] All jobs submitted. Use squeue to monitor."

