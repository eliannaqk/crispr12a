#!/bin/bash -l
#SBATCH -J cas12a_hmm
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o slurm-%j.out

# Build a Cas12a profile-HMM from a reference FASTA and place it under
# /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/hmm_filter_data
#
# Requirements (on the node): bio-utils conda env with HMMER3 + MAFFT/FAMSA (+optional trimal)

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Initialize conda and activate approved env (bio-utils per policy)
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate bio-utils
echo "[env] Using conda: $CONDA_DEFAULT_ENV"

# Ensure conda libstdc++ is preferred (fixes GLIBCXX mismatch for trimAl)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6.0.34" && ! -e "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  ln -s "$CONDA_PREFIX/lib/libstdc++.so.6.0.34" "$CONDA_PREFIX/lib/libstdc++.so.6"
fi

# Move to project directory (per repo guidelines)
cd ~/crispr12/opencrispr-repro-main

# Paths
REF_FASTA="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_ref_99_rep_seq.fasta"
OUT_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/hmm_filter_data"
mkdir -p "$OUT_DIR"

# Filenames
ALIGN_FA="$OUT_DIR/cas12a_ref_99_rep_seq.aln.fasta"
TRIM_FA="$OUT_DIR/cas12a_ref_99_rep_seq.aln.trim.fasta"
HMM_OUT="$OUT_DIR/cas12a.hmm"

# hmmbuild occupancy threshold (drop low-occupancy columns)
# gentler than before, to retain more columns
SYMFRAC="0.7"

echo "[info] REF_FASTA: $REF_FASTA"
echo "[info] OUT_DIR  : $OUT_DIR"

if [[ ! -s "$REF_FASTA" ]]; then
  echo "[error] Reference FASTA not found: $REF_FASTA" >&2
  exit 2
fi

# Keep a copy of the reference in the output directory for provenance
cp -f "$REF_FASTA" "$OUT_DIR/"

# Choose aligner: prefer FAMSA, else MAFFT
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

# Trimming: require trimAl success and non-empty output, else halt.
if ! command -v trimal >/dev/null 2>&1; then
  echo "[error] trimAl not found in PATH (env=$CONDA_DEFAULT_ENV). Cannot proceed." >&2
  exit 10
fi
echo "[info] Trimming alignment with trimAl (gentler: -gt 0.6; fallback gappyout)"
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
ALIGN_FOR_HMM="$TRIM_FA"

# Quick sanity: report alignment column count for first sequence in TRIMMED alignment
COLS=$(awk 'BEGIN{RS=">"} NR==2{for(i=2;i<=NF;i++){gsub(/[\r\n]/, "", $i); gsub(/[^A-Za-z\-]/, "", $i); seq=seq $i} print length(seq); exit}' "$ALIGN_FOR_HMM")
echo "[info] Trimmed alignment columns: $COLS (file: $ALIGN_FOR_HMM)"

# Build HMM and press
if ! command -v hmmbuild >/dev/null 2>&1; then
  echo "[error] hmmbuild not found; ensure HMMER3 is installed in bio-utils." >&2
  exit 4
fi

echo "[info] Building HMM -> $HMM_OUT (symfrac=$SYMFRAC)"
hmmbuild --amino --symfrac "$SYMFRAC" -n Cas12a "$HMM_OUT" "$ALIGN_FOR_HMM"

if command -v hmmpress >/dev/null 2>&1; then
  echo "[info] Removing existing HMM index files before hmmpress"
  rm -f "$HMM_OUT.h3f" "$HMM_OUT.h3i" "$HMM_OUT.h3m" "$HMM_OUT.h3p" || true
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
fi
