#!/usr/bin/env bash
set -euo pipefail

# Example: build HMM from references (must include anchor)
python cas12a_motif_filter_hmm.py \
  --designs my_designs.faa \
  --config cas12a_motifs_lb_hmm.yaml \
  --refs_fasta cas12a_refs.faa \
  --anchor_fasta lb_anchor.faa \
  --anchor_id_substring LbCas12a \
  --outdir out_hmm --threads 8

echo "Done. See out_hmm/results/motif_summary.tsv"
