#!/bin/bash -l
# Purpose: Find the largest generation batch size that fits on H200 for ba6000
# It runs quick probes (small num_samples) and reports the largest successful bs.

#SBATCH -J gen-bs-sweep-ba6000
#SBATCH -A pi_mg269
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o slurm-gen-bs-sweep-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Ensure Conda is initialized and activate approved env
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

cd ~/crispr12/opencrispr-repro-main

# Optional: W&B key if needed elsewhere (generation itself does not log to W&B)
## If you need W&B for anything in this job, ensure you are already logged in
## interactively (`wandb login`) or export WANDB_API_KEY in your shell before sbatch.

echo "[Env] Host=$(hostname) GPU=H200 CondaEnv=$(conda info --json | python -c 'import sys,json; print(json.load(sys.stdin)["active_prefix_name"])')"

MODEL_PATH="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/huggingface/ba6000"
BASE_CFG="generation/generate_10k.yml"
TMP_DIR="${SLURM_TMPDIR:-/tmp}/gen_bs_sweep_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMP_DIR"

echo "[Info] Using model: $MODEL_PATH"
echo "[Info] Base config: $BASE_CFG"
echo "[Info] Temp dir: $TMP_DIR"

# Candidate batch sizes to probe (ascending)
SIZES=(64 72 80 88 96 104 112 120 128 136 144 152 160)

last_ok=0

probe_cfg() {
  local bs="$1"
  # Reduce per-probe runtime: allow override via PROBE_SAMPLES, default 8
  local samples=${PROBE_SAMPLES:-8}
  local out_cfg="$TMP_DIR/gen_bs_${bs}.yml"
  # Rewrite batch_size and num_samples in the YAML (keep everything else identical)
  sed -E "s/^([[:space:]]*)batch_size:[[:space:]]*[0-9]+/\1batch_size: ${bs}/; s/^([[:space:]]*)num_samples:[[:space:]]*[0-9]+/\1num_samples: ${samples}/" "$BASE_CFG" > "$out_cfg"
  echo "$out_cfg"
}

for bs in "${SIZES[@]}"; do
  CFG_PATH=$(probe_cfg "$bs")
  echo "\n[Run] Probing batch_size=$bs with num_samples=${PROBE_SAMPLES:-8}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
  set +e
  python -u generate.py \
    --model-path "$MODEL_PATH" \
    --config "$CFG_PATH" \
    --save-folder "$TMP_DIR" \
    --job-idx "$bs"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo "[OK] batch_size=$bs succeeded"
    last_ok=$bs
  else
    echo "[FAIL] batch_size=$bs failed with exit code $status"
    break
  fi
done

echo "\n[Result] Largest successful batch_size: ${last_ok}"
if [[ $last_ok -gt 0 ]]; then
  # Optionally write out a suggested config with the discovered bs
  SUGGESTED_CFG="$TMP_DIR/generate_10k_bs${last_ok}.yml"
  sed -E "s/^([[:space:]]*)batch_size:[[:space:]]*[0-9]+/\1batch_size: ${last_ok}/" "$BASE_CFG" > "$SUGGESTED_CFG"
  echo "[Saved] Suggested config with batch_size=${last_ok}: $SUGGESTED_CFG"
fi

echo "[Done] Sweep complete. Inspect $TMP_DIR and slurm log for details."
