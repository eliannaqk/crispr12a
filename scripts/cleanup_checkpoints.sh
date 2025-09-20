#!/bin/bash -l
# Cleanup script for CRISPR atlas model saves
#
# Keeps only Hugging Face checkpoints under huggingface/{ba4000,ba6000,ba8000}
# Deletes all other model folders starting with "ba*" under run-* directories.
# Always preserves any path containing "/generations/" or "/generate".
# Optionally protect specific absolute paths via --protect.
#
# Usage:
#   scripts/cleanup_checkpoints.sh <BASE_DIR> [--dry-run] [--protect <ABS_PATH>]...
# Example:
#   scripts/cleanup_checkpoints.sh \
#     /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas \
#     --protect /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/huggingface/ba6000 \
#     --protect /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/gpu/generations/ba6000 \
#     --protect /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/model_saves_aug21/run-20250822-202947/generations/ba6000

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <BASE_DIR> [--dry-run] [--protect <ABS_PATH>]..." >&2
  exit 1
fi

BASE="$1"; shift
DRY_RUN=0
PROTECT_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1; shift ;;
    --protect)
      [[ $# -lt 2 ]] && { echo "--protect requires a path" >&2; exit 1; }
      PROTECT_PATHS+=("$2"); shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$BASE" ]]; then
  echo "Base directory does not exist: $BASE" >&2
  exit 1
fi

echo "[cleanup] Base: $BASE"
date '+[cleanup] Started: %Y-%m-%d %H:%M:%S'

# Keep set
KEEP_RE='^(ba4000|ba6000|ba8000)$'

# Find candidate directories named ba* under run-* trees, excluding generations/generate
mapfile -t CANDS < <(find "$BASE" -maxdepth 6 -type d -regex ".*/run-[^/]*/.*" -name "ba*" 2>/dev/null | \
                    grep -v "/generations/" | grep -v "/generate")

KEEP_LIST=()
DEL_LIST=()

is_protected() {
  local p="$1"
  for prot in "${PROTECT_PATHS[@]:-}"; do
    [[ -n "${prot:-}" ]] || continue
    if [[ "$p" == "$prot" ]]; then
      return 0
    fi
  done
  return 0
}

for d in "${CANDS[@]}"; do
  [[ -z "${d:-}" ]] && continue
  # Explicit protection
  for prot in "${PROTECT_PATHS[@]:-}"; do
    if [[ -n "${prot:-}" && "$d" == "$prot" ]]; then
      KEEP_LIST+=("$d (PROTECTED)")
      continue 2
    fi
  done

  case "$d" in
    *"/huggingface/"ba*)
      leaf=$(basename -- "$d")
      if [[ "$leaf" =~ $KEEP_RE ]]; then
        KEEP_LIST+=("$d")
      else
        DEL_LIST+=("$d")
      fi
      ;;
    *)
      # Non-HF ba* under run-* are training checkpoints -> delete
      DEL_LIST+=("$d")
      ;;
  esac
done

echo "[cleanup] Candidates: ${#CANDS[@]}"
echo "[cleanup] Keep: ${#KEEP_LIST[@]}"
echo "[cleanup] Delete: ${#DEL_LIST[@]}"

if (( DRY_RUN == 1 )); then
  echo "[cleanup] Dry run - no deletions performed"
  printf '%s\n' "${DEL_LIST[@]}" | sed 's/^/[delete]/' | head -n 50
  exit 0
fi

DELETED=0
for d in "${DEL_LIST[@]}"; do
  [[ -z "${d:-}" ]] && continue
  # Safety checks
  if [[ "$d" == $BASE/* && $(basename -- "$d") == ba* ]]; then
    rm -rf --one-file-system -- "$d"
    DELETED=$((DELETED+1))
    if (( DELETED % 200 == 0 )); then
      echo "[cleanup] Deleted $DELETED"
    fi
  else
    echo "[cleanup][SKIP unsafe] $d" >&2
  fi
done

echo "[cleanup] Deleted total: $DELETED"

# Prune empty directories under model_saves* trees
find "$BASE"/model_saves* -type d -empty -delete 2>/dev/null || true

date '+[cleanup] Finished: %Y-%m-%d %H:%M:%S'

