#!/bin/bash -l
#SBATCH -J install-fair-esm
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
echo "Running on node: $(hostname)"
echo "Requested CPUs: ${SLURM_CPUS_PER_TASK:-unknown}  MEM: ${SLURM_MEM_PER_NODE:-unknown}"

# Activate approved environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Active conda env: $(conda info --json | python -c "import sys,json;print(json.load(sys.stdin)['active_prefix_name'])")"
python -V
pip --version

cd ~/crispr12/opencrispr-repro-main

echo "Installing fair-esm into oc-opencrispr..."
pip install --no-input fair-esm | cat

echo "Verifying esm import..."
python - <<'PY'
import esm
print("esm import OK; version:", getattr(esm, "__version__", "n/a"))
PY

echo "Done."



