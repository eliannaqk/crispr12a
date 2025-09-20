#!/bin/bash -l
# Build a Cas12a profile-HMM from filtered95_40_rep_seq.fasta into a separate hmm_lb_run folder.
# Environment: bio-utils (must have HMMER3 + FAMSA/MAFFT + trimAl)

#SBATCH -J cas12a_hmm_lb
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Initialize conda and activate required env per policy
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate bio-utils
echo "[env] Using conda: $CONDA_DEFAULT_ENV"

# Ensure conda libstdc++ is preferred (fixes GLIBCXX mismatch for trimAl)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6.0.34" && ! -e "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  ln -s "$CONDA_PREFIX/lib/libstdc++.so.6.0.34" "$CONDA_PREFIX/lib/libstdc++.so.6"
fi

cd ~/crispr12/opencrispr-repro-main

# Paths
REF_FASTA="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/lbcasfiltering/filtered95_40_rep_seq.fasta"
OUT_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/hmm_filter_data/hmm_lb_run"
mkdir -p "$OUT_DIR"

ALIGN_FA="$OUT_DIR/filtered95_40_rep_seq.aln.fasta"
TRIM_FA="$OUT_DIR/filtered95_40_rep_seq.aln.trim.fasta"
HMM_OUT="$OUT_DIR/cas12a_lb.hmm"

# hmmbuild occupancy threshold (relaxed to retain more columns)
SYMFRAC="0.7"

echo "[info] REF_FASTA: $REF_FASTA"
echo "[info] OUT_DIR  : $OUT_DIR"

if [[ ! -s "$REF_FASTA" ]]; then
  echo "[error] Reference FASTA not found: $REF_FASTA" >&2
  exit 2
fi

# Align (prefer FAMSA)
if command -v famsa >/dev/null 2>&1; then
  echo "[info] Running FAMSA alignment"
  famsa -t "$OMP_NUM_THREADS" "$REF_FASTA" "$ALIGN_FA"
elif command -v mafft >/dev/null 2>&1; then
  echo "[info] Running MAFFT alignment"
  mafft --thread "$OMP_NUM_THREADS" --auto "$REF_FASTA" > "$ALIGN_FA"
else
  echo "[error] Neither famsa nor mafft found in PATH (env=$CONDA_DEFAULT_ENV)." >&2
  exit 3
fi

# Require trimAl; fail if unavailable or fails
if ! command -v trimal >/dev/null 2>&1; then
  echo "[error] trimAl not found in PATH (env=$CONDA_DEFAULT_ENV). Cannot proceed." >&2
  exit 10
fi
echo "[info] Trimming alignment with trimAl (gentler: -gt 0.6; fallback -gappyout)"
if ! trimal -gt 0.6 -in "$ALIGN_FA" -out "$TRIM_FA"; then
  echo "[warn] trimAl -gt 0.6 failed; trying -gappyout"
  if ! trimal -gappyout -in "$ALIGN_FA" -out "$TRIM_FA"; then
    echo "[error] trimAl failed on $ALIGN_FA" >&2
    exit 11
  fi
fi
if [[ ! -s "$TRIM_FA" ]]; then
  echo "[error] trimAl produced empty/zero output: $TRIM_FA" >&2
  exit 12
fi

# Report trimmed alignment columns
COLS=$(awk 'BEGIN{RS=">"} NR==2{for(i=2;i<=NF;i++){gsub(/[\r\n]/, "", $i); gsub(/[^A-Za-z\-]/, "", $i); seq=seq $i} print length(seq); exit}' "$TRIM_FA")
echo "[info] Trimmed alignment columns: $COLS (file: $TRIM_FA)"

# Build HMM and press
if ! command -v hmmbuild >/dev/null 2>&1; then
  echo "[error] hmmbuild not found; ensure HMMER3 is installed in bio-utils." >&2
  exit 4
fi

echo "[info] Building HMM -> $HMM_OUT (symfrac=$SYMFRAC)"
hmmbuild --amino --symfrac "$SYMFRAC" -n Cas12a_LB "$HMM_OUT" "$TRIM_FA"

if command -v hmmpress >/dev/null 2>&1; then
  echo "[info] hmmpress $HMM_OUT"
  hmmpress "$HMM_OUT"
else
  echo "[warn] hmmpress not found; hmmscan will still work but slower"
fi

echo "[done] HMM files in $OUT_DIR"
ls -l "$OUT_DIR" | sed -n '1,200p'

# Report model length
if [[ -s "$HMM_OUT" ]]; then
  echo -n "[info] HMM LENG: "; grep -m1 '^LENG' "$HMM_OUT" || true
  if command -v hmmstat >/dev/null 2>&1; then
    echo "[info] hmmstat summary:"
    hmmstat "$HMM_OUT" | sed -n '1,80p'
  fi
fi
