#!/bin/bash -l
#SBATCH -J clone-oc-opencrispr
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
echo "Running on node: $(hostname)"
echo "Requested CPUs: ${SLURM_CPUS_PER_TASK:-unknown}  MEM: ${SLURM_MEM_PER_NODE:-unknown}"

# Initialize conda in non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

ORIG_ENV="oc-opencrispr"
NEW_ENV="oc-opencrispr-esm"

echo "Cloning conda env: ${ORIG_ENV} -> ${NEW_ENV} (if not exists)"
if conda env list | awk '{print $1}' | grep -qx "${NEW_ENV}"; then
  echo "Environment ${NEW_ENV} already exists; skipping clone."
else
  conda create --yes --name "${NEW_ENV}" --clone "${ORIG_ENV}"
fi

echo "Activating ${NEW_ENV}"
conda activate "${NEW_ENV}"
echo "Active conda env: $(conda info --json | python -c "import sys,json;print(json.load(sys.stdin)['active_prefix_name'])")"
python -V
pip --version

echo "Installing fair-esm into ${NEW_ENV}..."
pip install --no-input fair-esm | cat

echo "Verifying esm and torch imports..."
python - <<'PY'
import esm, sys
print("esm import OK; version:", getattr(esm, "__version__", "n/a"))
try:
    import torch
    print("torch import OK; version:", torch.__version__)
except Exception as e:
    print("torch import failed:", repr(e))
    sys.exit(1)
PY

echo "Conda environments:"
conda env list

echo "Done."



