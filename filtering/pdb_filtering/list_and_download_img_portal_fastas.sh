#!/usr/bin/env bash
set -euo pipefail

# Fallback downloader: lists IMG_VR / IMG_PR files and downloads only FASTA-like files.
# Uses the ext-api get-directory → get_tape_file endpoints.

# === EDIT THESE LINES IF NEEDED ===
EMAIL="Elizabeth.knight@yale.edu"
DEST_BASE="/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna"
# ==================================

VR_DIR="${DEST_BASE}/IMG_VR"
PR_DIR="${DEST_BASE}/IMG_PR"
mkdir -p "$VR_DIR" "$PR_DIR"

read -s -p "JGI password for ${EMAIL}: " PASSWORD; echo

COOKIES="$(mktemp)"
cleanup() { rm -f "$COOKIES"; }
trap cleanup EXIT

echo "[login] JGI Single Sign-On"
curl 'https://signon.jgi.doe.gov/signon/create' \
  --data-urlencode "login=${EMAIL}" \
  --data-urlencode "password=${PASSWORD}" \
  -c "$COOKIES" > /dev/null

echo "[list] Querying directory listings (IMG_VR, IMG_PR)"
curl -s 'https://genome-downloads.jgi.doe.gov/portal/ext-api/downloads/get-directory?organism=IMG_VR' -b "$COOKIES" > "$VR_DIR/files.xml"
curl -s 'https://genome-downloads.jgi.doe.gov/portal/ext-api/downloads/get-directory?organism=IMG_PR' -b "$COOKIES" > "$PR_DIR/files.xml"

download_urls() {
  local XML="$1"; local DEST="$2"; mkdir -p "$DEST"
  echo "[dl] Parsing $(basename "$XML") for FASTA URLs → $DEST"
  grep -Eo 'url="[^"]+\.(fa|fna|fasta)(\.gz)?[^"]*"' "$XML" \
    | sed -E 's/^url="|"$//g; s/&amp;/\&/g' \
    | while read -r REL; do
        local FN="$(basename "$REL")"
        local OUT="${DEST}/${FN}"
        echo "  -> $FN"
        curl -L -b "$COOKIES" \
          "https://genome-downloads.jgi.doe.gov/portal/ext-api/downloads/get_tape_file?blocking=true&url=${REL}" \
          -o "$OUT"
      done
}

download_urls "$VR_DIR/files.xml" "$VR_DIR"
download_urls "$PR_DIR/files.xml" "$PR_DIR"

echo "[done] Files under:"
echo "  $VR_DIR"
echo "  $PR_DIR"

