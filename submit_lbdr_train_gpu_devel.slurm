#!/bin/bash -l
#SBATCH -J lbdr_train
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Activate environment per policy
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

echo "[env] $(conda info --envs | grep '*' || true)"; nvidia-smi || true

# Inputs
RAW_TRAIN="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/train_raw95.csv"
RAW_VALID="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/test_raw95.csv"
PREPPED="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/prep_20250828_141345_dr_aatttctactaagtgtagat/cas12a_training_prepped.csv"

OUT_DIR="lbdr_runs/esm2_t33"
mkdir -p "$OUT_DIR"

echo "[info] Joining raw CSVs with prepped labels (by exact sequence match)"
python - << 'PY'
import csv, sys
from pathlib import Path

raw_train = Path("/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/train_raw95.csv")
raw_valid = Path("/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/test_raw95.csv")
prepped = Path("/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/prep_20250828_141345_dr_aatttctactaagtgtagat/cas12a_training_prepped.csv")

def load_labels(p):
    lab = {}
    with open(p, newline='') as f:
        r = csv.DictReader(f)
        need = {"protein_seq","dr_exact"}
        if not need.issubset(r.fieldnames):
            raise SystemExit(f"[error] {p} missing columns {need}")
        for row in r:
            v = row.get("dr_exact", "").strip()
            if v not in ("0","1"):
                continue
            lab[row["protein_seq"]] = int(v)
    return lab

labels = load_labels(prepped)
print(f"[label] loaded {len(labels)} labeled sequences from prepped CSV")

def join_and_write(raw_csv, out_csv):
    kept = 0
    with open(raw_csv, newline='') as fin, open(out_csv, 'w', newline='') as fout:
        r = csv.DictReader(fin)
        if not {"id","sequence"}.issubset(r.fieldnames):
            raise SystemExit(f"[error] {raw_csv} missing id,sequence")
        w = csv.DictWriter(fout, fieldnames=["protein_seq","dr_exact"])
        w.writeheader()
        for row in r:
            seq = row["sequence"].strip()
            lab = labels.get(seq)
            if lab is None:
                continue
            w.writerow({"protein_seq": seq, "dr_exact": lab})
            kept += 1
    print(f"[join] {raw_csv} -> {out_csv} (kept {kept})")

join_and_write(raw_train, "lbdr_runs/esm2_t33/train_labeled.csv")
join_and_write(raw_valid, "lbdr_runs/esm2_t33/valid_labeled.csv")
PY

echo "[train] Starting training"
python filtering/train_lbdr_classifier_esm.py \
  --train_csv "${OUT_DIR}/train_labeled.csv" \
  --valid_csv "${OUT_DIR}/valid_labeled.csv" \
  --out_model "${OUT_DIR}/lbdr_esm2_t33.pt" \
  --epochs 6 \
  --batch_size 16 \
  --lr 1e-3

echo "[done] Model and logs under ${OUT_DIR}"

