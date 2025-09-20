#!/bin/bash -l

# Filter generated sequences by alignment to a reference MMseqs2 DB
# - Accepts either a CSV from generate.py (columns: context_name,context,sequence)
#   or a FASTA/FAA file of sequences. Produces a filtered FASTA of sequences
#   that have at least one alignment to the reference DB.
#
# Requirements per cluster policy:
# - Use conda env: bio-utils
# - Invoke MMseqs2 via Apptainer/Singularity container each time
#   apptainer exec docker://soedinglab/mmseqs2:latest mmseqs <cmd>
# - Reference DB location (default): /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_refDB
#
# Usage:
#   scripts/mmseqs_filter_generated.sh \
#     --input /path/to/generate_500_bs36.csv \
#     --outdir /path/to/output_dir \
#     [--ref-db /home/.../newCas12a/cas12a_refDB] \
#     [--threads 32] [--min-id 0.25] [--cov 0.8]
#
set -euo pipefail

echo "[Env] Initializing conda and activating bio-utils (bioinformatics tools)."
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate bio-utils || { echo "[Error] Failed to activate bio-utils env"; exit 1; }
echo "[Env] Using conda env: bio-utils"

# Load apptainer/singularity if available (non-fatal if modules not present)
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

INPUT=""
OUTDIR=""
REF_DB="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_refDB"
THREADS=32
MIN_ID=0.25
COV=0.8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"; shift 2;;
    --outdir)
      OUTDIR="$2"; shift 2;;
    --ref-db)
      REF_DB="$2"; shift 2;;
    --threads)
      THREADS="$2"; shift 2;;
    --min-id)
      MIN_ID="$2"; shift 2;;
    --cov)
      COV="$2"; shift 2;;
    -h|--help)
      sed -n '1,80p' "$0"; exit 0;;
    *)
      echo "[Error] Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTDIR" ]]; then
  echo "[Error] --input and --outdir are required"; exit 1
fi

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "[Paths] INPUT=$INPUT"
echo "[Paths] OUTDIR=$OUTDIR"
echo "[Paths] REF_DB=$REF_DB"

# Decide container runner
if command -v apptainer >/dev/null 2>&1; then
  CNTR_RUN=(apptainer exec docker://soedinglab/mmseqs2:latest)
elif command -v singularity >/dev/null 2>&1; then
  CNTR_RUN=(singularity exec docker://soedinglab/mmseqs2:latest)
else
  echo "[Error] Neither apptainer nor singularity found in PATH"; exit 1
fi

GEN_FAA="generated.faa"
if [[ "$INPUT" =~ \.csv$ ]]; then
  echo "[Step] Converting CSV to FASTA with name derived from context"
  GEN_FAA=$(python - "$INPUT" <<'PY'
import sys, csv, re, hashlib, os
src = sys.argv[1]

def slug_from_context(ctx: str) -> str:
    # Expect pattern like 1<seq1>2<seq2>1<seq3>; derive lengths and heads
    parts = re.split(r'([12])', ctx)
    # parts like ['', '1', s1, '2', s2, '1', s3, ...]
    seqs = []
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            tag, seq = parts[i], parts[i+1]
            if tag in {'1','2'} and seq:
                seqs.append((tag, seq))
    # Build concise slug
    items = []
    for tag, s in seqs[:3]:
        head = re.sub('[^A-Z]', '', s)[:6]
        items.append(f"{tag}{len(s)}_{head}")
    base = "_".join(items) if items else "ctx"
    # keep length reasonable; add short hash for uniqueness
    h = hashlib.sha1(ctx.encode()).hexdigest()[:8]
    return f"generated_{base}_{h}.faa"

# Read first non-empty context for naming
first_ctx = None
with open(src, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        c = (row.get('context') or '').strip()
        if c:
            first_ctx = c
            break
name = slug_from_context(first_ctx or 'ctx')

with open(src, newline='') as f, open(name, 'w') as out:
    r = csv.DictReader(f)
    for i, row in enumerate(r, 1):
        ctx_name = (row.get('context_name') or 'ctx').replace(' ', '_')
        seq = (row.get('sequence') or '').strip()
        if not seq:
            continue
        hdr = f">{ctx_name}|{i}"
        out.write(hdr + "\n")
        for j in range(0, len(seq), 80):
            out.write(seq[j:j+80] + "\n")

print(name)
PY
  )
  echo "[Info] FASTA created: $GEN_FAA"
elif [[ "$INPUT" =~ \.(fa|fasta|faa)$ ]]; then
  echo "[Step] Using provided FASTA input -> $GEN_FAA"
  base=$(basename "$INPUT")
  GEN_FAA="generated_${base}"
  cp -f "$INPUT" "$GEN_FAA"
else
  echo "[Error] Unsupported input format: $INPUT"; exit 1
fi

echo "[QC] Counting sequences in $GEN_FAA"
SEQ_BEFORE=$(grep -c '^>' "$GEN_FAA" || echo 0)
echo "[QC] Sequences before: $SEQ_BEFORE"

echo "[Step] Building query DB: genDB"
"${CNTR_RUN[@]}" mmseqs createdb "$GEN_FAA" genDB

echo "[Step] Running search vs reference DB"
mkdir -p tmp
"${CNTR_RUN[@]}" mmseqs search genDB "$REF_DB" gen_vs_ref tmp \
  --min-seq-id "$MIN_ID" --cov-mode 0 -c "$COV" --threads "$THREADS"

echo "[Step] Converting alignments to m8 table: gen_vs_ref.m8"
"${CNTR_RUN[@]}" mmseqs convertalis genDB "$REF_DB" gen_vs_ref gen_vs_ref.m8 \
  --format-output "query,target,pident,alnlen,qcov,tcov,evalue,bits"

echo -n "[QC] Alignments: "
if [[ -s gen_vs_ref.m8 ]]; then
  wc -l < gen_vs_ref.m8
else
  echo 0
fi
echo "[Head] First 5 lines of gen_vs_ref.m8 (if any):"
head -n 5 gen_vs_ref.m8 2>/dev/null || true

echo "[Step] Filtering FASTA to keep only queries with any alignment"
awk '{print $1}' gen_vs_ref.m8 | sort -u > hits.txt || true
if [[ ! -s hits.txt ]]; then
  echo "[Warn] No hits found; producing empty filtered.faa"
  : > filtered.faa
else
  python - "$GEN_FAA" hits.txt filtered.faa <<'PY'
import sys
from pathlib import Path

fa, hits, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
keep = set(x.strip().split()[0] for x in hits.read_text().splitlines() if x.strip())

def write_record(h, s, fh):
    if not h:
        return
    # ID up to first whitespace
    rid = h.split()[0]
    if rid in keep:
        fh.write('>' + h + '\n')
        for i in range(0, len(s), 80):
            fh.write(s[i:i+80] + '\n')

hdr = None
seq = []
with fa.open() as f, out.open('w') as o:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('>'):
            write_record(hdr, ''.join(seq), o)
            hdr = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    write_record(hdr, ''.join(seq), o)
PY
fi

SEQ_AFTER=$(grep -c '^>' filtered.faa || echo 0)
echo "[QC] Sequences after filter: $SEQ_AFTER"

echo "[Done] Outputs in $OUTDIR:"
echo "  - $GEN_FAA (input FASTA)"
echo "  - gen_vs_ref.m8 (tabular alignments)"
echo "  - filtered.faa (kept sequences)"
echo "  - hits.txt (unique query IDs with hits)"
