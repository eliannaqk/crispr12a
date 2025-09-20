# Cas12a motif-only filter — HMMER (LbCas12a anchor)

This tool **does not use raw positions in your designs**. Instead, it:
1) Aligns sequences with **HMMER (hmmalign)** to an **LbCas12a profile HMM** (provided or built).
2) Converts **LbCas12a motif positions** → **HMM match-state indices** using an anchor-vs-HMM alignment.
3) Reads the residue your design places at those **match states** and enforces strict motif checks.

Because it uses HMM match states, it is robust to **insertions/deletions** and to **divergent orthologs**.

## Install
```bash
# Python
pip install biopython pyyaml

# CLI tools
# - HMMER (hmmalign required; hmmbuild if you build the HMM)
# - MAFFT (only if you want the script to build the HMM from a reference FASTA)
```
Ensure `hmmalign` (and `hmmbuild` if needed) are on your PATH.

## Run

**A) Use a prebuilt HMM:**
```bash
python cas12a_motif_filter_hmm.py \
  --designs my_designs.faa \
  --config cas12a_motifs_lb_hmm.yaml \
  --hmm lbcas12a.hmm \
  --anchor_fasta lb_anchor.faa \
  --anchor_id_substring LbCas12a \
  --outdir out_hmm
```

**B) Build the HMM from a references FASTA (must include your Lb anchor):**
```bash
python cas12a_motif_filter_hmm.py \
  --designs my_designs.faa \
  --config cas12a_motifs_lb_hmm.yaml \
  --refs_fasta cas12a_refs.faa \
  --anchor_fasta lb_anchor.faa \
  --anchor_id_substring LbCas12a \
  --outdir out_hmm --threads 8
```

## What is checked (Lb numbering → HMM states)
- **WED triad**: H759, K768, K785 → must be **H/K/K**.
- **PI Lysines**: K538 and K595 → must be **K** (set `require_exact_lys: false` in config to allow **K/R**).
- **NUC routing Arg**: R1138 → must be **R**.
- **Bridge helix aromatic**: W890 → must be **W** unless you broaden `allowed`.
- **LID basic patch**: optional; add positions + `min_basic` if you want it enforced.

> All coordinates live in the **Lb anchor** you provide. The script aligns the anchor to the HMM to find the **match states** for those positions, then checks each design on those **states**.

## Seeds
Put your variable seed blocks under `seeds.list`. If a seed is found in a design, any motif whose **query residue**
maps inside the seed span is **masked** (ignored) for that sequence.

## Outputs
- `out_hmm/results/motif_summary.tsv` — per-design pass/fail per motif and overall.
- `out_hmm/results/details/<id>.json` — residue calls and mask info.
- `out_hmm/sets/motif_pass.faa`, `motif_fail.faa` — all-pass vs any-fail splits.
- `out_hmm/sets/by_motif/*_pass.faa` — convenience files per motif.

## Notes
- This pipeline uses **HMMER** for alignment (not raw indices in designs), as requested.
- If you later want to use **MMseqs2** instead of HMMER, the logic is similar: compute a profile (or pick your anchor),
  run `mmseqs align` or `search` to generate a pairwise alignment (anchor vs design), then read residues at the
  **anchor-numbered** motif sites from the pairwise MSA. I can add an MMseqs mode if you prefer that toolchain.

## Troubleshooting
- *“hmmalign not found”*: install HMMER and make sure `hmmalign` is on PATH.
- *Anchor not found*: pass `--anchor_id_substring` that matches a header in your `--anchor_fasta` or `--refs_fasta`.
- *No HMM*: pass `--hmm` or `--refs_fasta` to build one.
