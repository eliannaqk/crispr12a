#!/bin/bash -l
# CPU array pipeline for filtering generated sequences per seed using full-length Cas12a HMM.
# Environments: oc-opencrispr for Python steps; bio-utils for HMMER steps.

#SBATCH -J seed_filter_cpu
#SBATCH -p devel
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH -o slurm-%j.out
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job will use oc-opencrispr for Python and bio-utils for HMMER (CPU only)"

# Initialize conda per repo policy and cd to project root
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
cd ~/crispr12/opencrispr-repro-main

# Directories and inputs
RUN_DIR="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000"
OUT_ROOT="$RUN_DIR/by-seed"
HMM_PROFILE="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/hmm_filter_data/cas12a.hmm"  # full-length model

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[error] RUN_DIR does not exist: $RUN_DIR" >&2
  exit 1
fi
if [[ ! -f "$HMM_PROFILE" ]]; then
  echo "[error] HMM_PROFILE not found: $HMM_PROFILE" >&2
  exit 2
fi

SEED="${SLURM_ARRAY_TASK_ID}"
echo "[seed] Processing seed${SEED} (array task ${SLURM_ARRAY_TASK_ID})"

SEED_DIR="$OUT_ROOT/seed${SEED}"
mkdir -p "$SEED_DIR"

# Optional cleanup to avoid stale collisions for this seed only
echo "[clean] Removing prior outputs for seed${SEED} under $SEED_DIR"
rm -f "$SEED_DIR/seed${SEED}.aggregated_raw.csv" \
      "$SEED_DIR/seed${SEED}.step1_prefilter.csv" \
      "$SEED_DIR/seed${SEED}.step2_hmm_pass.csv" \
      "$SEED_DIR/seed${SEED}.step2_hmm_pass.fasta" \
      "$SEED_DIR/seed${SEED}.step2_hmm_report.tsv" \
      "$SEED_DIR/seed${SEED}.filter_summary.tsv" || true

# Step 1: aggregate raw CSVs
conda activate oc-opencrispr
echo "[env] Using env: $CONDA_DEFAULT_ENV for aggregation & prefilter"
AGG="$SEED_DIR/seed${SEED}.aggregated_raw.csv"
python filtering/aggregate_seed_csvs.py \
  --base_dir "$RUN_DIR" \
  --seed "$SEED" \
  --out_csv "$AGG"

# Step 2: prefilter
PREF="$SEED_DIR/seed${SEED}.step1_prefilter.csv"
python filtering/prefilter_csv.py \
  --in_csv "$AGG" \
  --out_csv "$PREF" \
  --flip_on_eos1 \
  --kmer_filter \
  --min_len 1000 \
  --max_len 1500

# Step 3: HMM coverage gate with full-length HMM (cas12a.hmm)
conda activate bio-utils
echo "[env] Using env: $CONDA_DEFAULT_ENV for hmmscan"
S2_FASTA="$SEED_DIR/seed${SEED}.step2_hmm_pass.fasta"
S2_RPT="$SEED_DIR/seed${SEED}.step2_hmm_report.tsv"

# Ensure HMM profile is pressed for hmmscan (creates .h3* aux files); handle races across array tasks
LOCK_FILE="${HMM_PROFILE}.lock"
(
  exec 200>"${LOCK_FILE}"
  if command -v flock >/dev/null 2>&1; then
    flock -x 200
  fi
  if [[ ! -f "${HMM_PROFILE}.h3m" || ! -f "${HMM_PROFILE}.h3i" || ! -f "${HMM_PROFILE}.h3f" || ! -f "${HMM_PROFILE}.h3p" ]]; then
    if command -v hmmpress >/dev/null 2>&1; then
      echo "[info] Preparing HMM aux files for ${HMM_PROFILE}"
      rm -f "${HMM_PROFILE}.h3i" "${HMM_PROFILE}.h3m" "${HMM_PROFILE}.h3f" "${HMM_PROFILE}.h3p" || true
      hmmpress "${HMM_PROFILE}"
    else
      echo "[error] hmmpress not found in PATH (env=$CONDA_DEFAULT_ENV). Cannot scan $HMM_PROFILE." >&2
      exit 20
    fi
  fi
) 200>"${LOCK_FILE}"

python filtering/filter_coverage.py \
  --input "$PREF" \
  --out_fasta "$S2_FASTA" \
  --out_tsv "$S2_RPT" \
  --hmm "$HMM_PROFILE" \
  --min_qcov 0.8 \
  --min_tcov 0.8 \
  --min_bitscore 50 \
  --threads "$SLURM_CPUS_PER_TASK"

# Step 4: Write passing CSV and stats
conda activate oc-opencrispr
echo "[env] Using env: $CONDA_DEFAULT_ENV to write step2 CSV and stats"
OUT_CSV="$SEED_DIR/seed${SEED}.step2_hmm_pass.csv"
python filtering/select_by_tsv_keep.py \
  --in_csv "$PREF" \
  --coverage_tsv "$S2_RPT" \
  --out_csv "$OUT_CSV"

# Report stats summary
SUM="$SEED_DIR/seed${SEED}.filter_summary.tsv"
{
  echo -e "metric\tvalue"
  ar=$(($(wc -l < "$AGG") - 1)); echo -e "aggregated_raw\t${ar}"
  pf=$(($(wc -l < "$PREF") - 1)); echo -e "step1_prefilter\t${pf}"
  kp=$(($(wc -l < "$OUT_CSV") - 1)); echo -e "step2_hmm_pass\t${kp}"
  # Failure breakdown from the coverage report
  awk -F"\t" 'NR==1{for(i=1;i<=NF;i++){h[$i]=i}; next} {n++; if($h["qcov"]<0.8) fq++; if($h["tcov"]<0.8) ft++; if($h["bitscore"]<50) fb++} END{printf "fail_qcov\t%d\nfail_tcov\t%d\nfail_bitscore\t%d\nrows_in_report\t%d\n", fq+0, ft+0, fb+0, n+0}' "$S2_RPT"
} > "$SUM"

echo "[summary] seed${SEED} counts: aggregated_raw=${ar} prefilter=${pf} kept=${kp} (see $SUM)"
echo "[seed] Done seed${SEED} -> $OUT_CSV"
echo "[done] Seed ${SEED} finished under $SEED_DIR"

