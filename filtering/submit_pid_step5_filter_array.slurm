#!/bin/bash -l
# Step 5: PID 8Å contact-conservation filter over Step 4 motif-pass FASTAs.
#
# - Seeds 1 and 3: run the contact filter (semi-global alignment on full-length designs).
# - Seeds 2 and 4: do NOT compute; copy step4_motif FASTA to step5 PID_angstrom FASTA for monitoring.
#
# Uses oc-opencrispr env. One seed per array task.

#SBATCH -J pid_step5
#SBATCH -p day
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# Use jobID_arrayTaskID in log filename to avoid clobbering between tasks
#SBATCH -o slurm-%A_%a.out
# Four seeds total
#SBATCH -a 1-4

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "[env] This job uses oc-opencrispr for PID DNA-angstrom contact filtering (CPU-bound)."

# === User config (override via --export=ALL,VAR=value) ===
RUN_DIR="${RUN_DIR:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000}"
# PDB + region specification (8Y04 biological assembly recommended)
PDB_PATH="${PDB_PATH:-$HOME/crispr12/opencrispr-repro-main/filtering/resources/8Y04.pdb1}"
PROTEIN_CHAIN="${PROTEIN_CHAIN:-A}"
DNA_CHAINS="${DNA_CHAINS:-C,D}"
PID_RANGE="${PID_RANGE:-900-1068}"
CUTOFF_A="${CUTOFF_A:-8.0}"
# Optional: reuse a precomputed contact cache
CONTACTS_JSON="${CONTACTS_JSON:-$HOME/crispr12/opencrispr-repro-main/filtering/out_pid_demo/lb_pid_contacts.json}"
# Whether to allow conservative K<->R and D<->E at contacts (0/1)
ALLOW_CONSERVATIVE="${ALLOW_CONSERVATIVE:-0}"

# === Setup ===
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
cd ~/crispr12/opencrispr-repro-main

SEED="${SLURM_ARRAY_TASK_ID}"
SEED_DIR="$RUN_DIR/by-seed/seed${SEED}"
if [[ ! -d "$SEED_DIR" ]]; then
  echo "[error] Seed directory not found: $SEED_DIR" >&2
  exit 1
fi

IN_STEP4_FASTA="$SEED_DIR/seed${SEED}.step4_motif_pass.fasta"
if [[ ! -f "$IN_STEP4_FASTA" ]]; then
  echo "[error] Step4 motif FASTA not found: $IN_STEP4_FASTA" >&2
  echo "[hint] Make sure step 4 completed successfully."
  exit 2
fi

OUT_STEP5_FASTA="$SEED_DIR/seed${SEED}.step5_PID_angstrom_pass.fasta"
OUT_DIR="$SEED_DIR/step5_pid_angstrom"
mkdir -p "$OUT_DIR"

case "$SEED" in
  1|3)
    echo "[seed${SEED}] Running PID contact filter on step4_motif FASTA"
    # If a contacts cache exists and matches your PDB/settings, use it; otherwise compute.
    LOAD_ARGS=()
    if [[ -f "$CONTACTS_JSON" ]]; then
      LOAD_ARGS=(--load-contacts "$CONTACTS_JSON")
      echo "[seed${SEED}] Using cached contacts: $CONTACTS_JSON"
    else
      echo "[seed${SEED}] No contacts cache; computing from PDB: $PDB_PATH"
      LOAD_ARGS=(--pdb "$PDB_PATH")
    fi
    ALLOW_FLAG=()
    if [[ "$ALLOW_CONSERVATIVE" == "1" ]]; then
      ALLOW_FLAG=(--allow-conservative)
    fi

    python filtering/pid_8A_filter_fullseq.py \
      --pdb "$PDB_PATH" \
      "${LOAD_ARGS[@]}" \
      --protein-chain "$PROTEIN_CHAIN" \
      --dna-chains "$DNA_CHAINS" \
      --pid-range "$PID_RANGE" \
      --designs "$IN_STEP4_FASTA" \
      --outdir "$OUT_DIR" \
      --cutoff "$CUTOFF_A" \
      "${ALLOW_FLAG[@]}"

    # Copy pass.faa to canonical step5 filename for monitoring
    if [[ -f "$OUT_DIR/pass.faa" ]]; then
      cp -f "$OUT_DIR/pass.faa" "$OUT_STEP5_FASTA"
      echo "[seed${SEED}] Wrote step5 PID_angstrom pass FASTA: $OUT_STEP5_FASTA"
    else
      echo "[seed${SEED}] ERROR: Expected $OUT_DIR/pass.faa not found" >&2
      exit 3
    fi
    ;;
  2|4)
    echo "[seed${SEED}] Skipping computation; copying step4_motif FASTA to step5 PID_angstrom for monitoring"
    cp -f "$IN_STEP4_FASTA" "$OUT_STEP5_FASTA"
    # Create minimal stub outputs for consistency
    : > "$OUT_DIR/fail.faa"
    printf "design_id\tpasses\tnum_violations\tviolating_sites\n" > "$OUT_DIR/report.tsv"
    echo "[seed${SEED}] Wrote step5 PID_angstrom pass FASTA (copied): $OUT_STEP5_FASTA"
    ;;
  *)
    echo "[error] Unexpected SEED=$SEED (expected 1..4)" >&2
    exit 4
    ;;
 esac

# Simple summary per seed
N_IN=$(awk '/^>/{n++} END{print n+0}' "$IN_STEP4_FASTA")
N_OUT=$(awk '/^>/{n++} END{print n+0}' "$OUT_STEP5_FASTA" || echo 0)
PCT=$(awk -v a="$N_OUT" -v b="$N_IN" 'BEGIN{ if(b>0) printf "%.2f", (a/b)*100; else print "0.00" }')
echo "[seed${SEED}] step5 summary: in=${N_IN} out=${N_OUT} (pass_rate=${PCT}%)"

echo "[ok] Seed ${SEED} step5 outputs under: $SEED_DIR"
