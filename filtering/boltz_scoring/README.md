Boltz-1 Screening for LbCas12a pre-crRNA Processing
===================================================

This folder contains two scripts to predict and score LbCas12a : pre-crRNA complexes using Boltz-1. The goal is to triage variants that retain WED-mediated RNase activity (K–K–H triad and DR docking).

Scripts
- run_boltz_prerna.py: Writes Boltz YAMLs from a CSV and runs predictions.
- score_wed_prerna.py: Parses outputs, computes ipTM/pLDDT, WED↔RNA contacts, and K/K/H-to-phosphate distances; writes a CSV for ranking.

Environment
- Use conda env: `oc-opencrispr-esm` (GPU-ready; ESM tooling).
- Initialize Conda in non-interactive shells:
  source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
  conda activate oc-opencrispr-esm

Dependencies
- Boltz-1 CLI: `pip install boltz[cuda] -U` (or CPU if needed)
- Python libs: Biopython, NumPy, pandas (should already be present in the env)

Inputs
Create a CSV (e.g., `inputs.csv`) with columns:
- id: unique target name
- protein_fasta or protein_seq: FASTA file path or sequence string
- rna_seq: RNA sequence (A,C,G,U) for pre-crRNA or DR-only with ~6–10 nt 5′ flank
- (optional) kkh_residues: 1-based indices for K/K/H catalytic triad (e.g., `492,517,539`)

Run: Local GPU
1) Generate YAMLs and run Boltz
   python filtering/boltz_scoring/run_boltz_prerna.py \
       --csv inputs.csv \
       --outdir filtering/boltz_scoring/boltz_runs \
       --use_msa_server

2) Score results
   python filtering/boltz_scoring/score_wed_prerna.py \
       --csv inputs.csv \
       --runs filtering/boltz_scoring/boltz_runs \
       --wed_residues "500,501,502,503,504,505,506,507,508,509" \
       --out_csv filtering/boltz_scoring/boltz_screen_scores.csv

Run: SLURM (GPU)
- Submit `submit_boltz_prerna.slurm` (edit partitions, WED residues, paths):
  sbatch filtering/boltz_scoring/submit_boltz_prerna.slurm

Choosing WED residues and K/K/H
- Use an LbCas12a reference (e.g., PDB 5XUS or 8Y04) to map residues that rim the DR pocket (WED) and annotate the K/K/H triad in your numbering.
- If `kkh_residues` is omitted in the CSV, the scorer finds the most RNA-proximal K–K–H triplet as a fallback.

Scoring summary (defaults)
- ipTM/interface confidence: higher is better (≥0.65–0.70 is binding-like).
- Mean pLDDT: protein, RNA, and WED region (want ≥70–80 for confident local geometry).
- WED↔RNA contact count: heavy-atom pairs ≤4.0 Å.
- Triad distances: nearest distance from K/K/H heavy atoms to RNA phosphates (P/OP1/OP2); processing-like if min ≤5–7 Å and ≥2 residues within ≤8 Å.

Tips
- Input RNA should include the DR hairpin plus ~6–10 nt 5′ flank to capture the cleavage determinant upstream of the stem.
- You can increase `--num_samples` in the run script (e.g., 5–8) and check reproducibility (RNA RMSD around WED among top-confidence samples).

