#!/usr/bin/env bash
set -euo pipefail

# JGI IMG/VR + IMG/PR downloader (Sequence FASTAs)
# - Logs into JGI SSO, stages "Sequence" files for IMG_VR and IMG_PR via ext-api,
#   downloads the staged ZIP(s), and unpacks to your Atlas folder.
# - Prompts for your JGI password and only stores a temporary cookie (deleted on exit).

# === EDIT THESE TWO LINES IF NEEDED ===
# Can also be provided via env: JGI_EMAIL, JGI_PASSWORD (non-interactive)
EMAIL="${JGI_EMAIL:-Elizabeth.knight@yale.edu}"
ATLAS_DEST_BASE="${ATLAS_DEST_BASE:-/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/viral_dna}"
# ======================================

BASE="${ATLAS_DEST_BASE}"
VR_DIR="${BASE}/IMG_VR"
PR_DIR="${BASE}/IMG_PR"
mkdir -p "$VR_DIR" "$PR_DIR"

echo "[env] Recommended conda env: oc-opencrispr (curl, unzip available)."

if [[ -z "${JGI_PASSWORD:-}" ]]; then
  read -s -p "JGI password for ${EMAIL}: " PASSWORD; echo
else
  PASSWORD="$JGI_PASSWORD"
fi

COOKIES="$(mktemp)"
cleanup() { rm -f "$COOKIES"; }
trap cleanup EXIT

echo "[1/5] Logging in to JGI Single Sign-On..."
curl 'https://signon.jgi.doe.gov/signon/create' \
  --data-urlencode "login=${EMAIL}" \
  --data-urlencode "password=${PASSWORD}" \
  -c "$COOKIES" > /dev/null

MAX_POLLS="${MAX_POLLS:-360}"       # default: ~60 minutes with SLEEP_SECS=10
SLEEP_SECS="${SLEEP_SECS:-10}"

stage_and_download() {
  local PORTAL="$1"   # IMG_VR or IMG_PR
  local DEST="$2"

  echo "[2/5] Requesting staged ZIP for portal=${PORTAL} (Sequence FASTA files)..."
  REQ=$(curl -s 'https://genome-downloads.jgi.doe.gov/portal/ext-api/downloads/bulk/request' \
      -b "$COOKIES" \
      --data-urlencode "portals=${PORTAL}" \
      --data-urlencode "fileTypes=Sequence" \
      --data-urlencode "filePattern=.*\\.(fa|fna|fasta)(\\.gz)?$" \
      --data-urlencode "organizedByFileType=true")

  STATUS_URL=$(echo "$REQ" | grep -Eo 'https?://[^ ]+/portal/ext-api/downloads/(bulk|globus)/[^ ]+/status' || true)
  if [[ -z "$STATUS_URL" ]]; then
    echo "[error] Could not parse status URL from response; raw reply:" >&2
    echo "$REQ" >&2
    exit 1
  fi
  echo "[3/5] Polling status: ${STATUS_URL}"

  DATA_URL=""
  for ((i=1;i<=MAX_POLLS;i++)); do
    STATUS=$(curl -s "$STATUS_URL" -b "$COOKIES")
    echo "  -> $(echo "$STATUS" | tr -s ' ' | cut -c1-120)"
    DATA_URL=$(echo "$STATUS" | sed -n 's/.*Data URL: *\([^ ]*\).*/\1/p')
    if [[ -n "$DATA_URL" ]]; then break; fi
    sleep "$SLEEP_SECS"
  done
  if [[ -z "$DATA_URL" ]]; then
    echo "[error] Timed out waiting for staging. Open this URL in a browser for details:" >&2
    echo "        $STATUS_URL" >&2
    exit 1
  fi

  if [[ "$DATA_URL" =~ genome-downloads.jgi.doe.gov ]]; then
    echo "[4/5] Downloading staged ZIP..."
    TMPZIP="$(mktemp --suffix .zip)"
    curl -L -b "$COOKIES" -o "$TMPZIP" "$DATA_URL"
    echo "[5/5] Unzipping into: $DEST"
    unzip -o "$TMPZIP" -d "$DEST" >/dev/null
    rm -f "$TMPZIP"
    echo "[done] $(find "$DEST" -type f | wc -l) files under $DEST"
  else
    echo "[globus] Your request was staged to Globus:"
    echo "          $DATA_URL"
    echo "          Use Globus File Manager to pull to: $DEST"
  fi
}

stage_and_download "IMG_VR" "$VR_DIR"
stage_and_download "IMG_PR" "$PR_DIR"

echo "All done. Files under:"
echo "  $VR_DIR"
echo "  $PR_DIR"
