ESMFold Domain Ranking
======================

What this does
- Runs ESMFold on each input sequence
- Reads per-residue pLDDT from the PDB B-factor (CA atoms)
- Maps reference-domain windows to each query by global alignment
- Reports per-domain mean pLDDT and fraction of residues with pLDDT ≥80/≥90
- Always emits pass_WED, pass_RuvC, pass_NUC, pass_PI by aggregating matching domains

Environment
- Use `oc-opencrispr-esm` (GPU preferred for speed; CPU works):
  - `source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh`
  - `conda activate oc-opencrispr-esm`
- Deps (already in env, otherwise): `pip install fair-esm torch biopython pandas numpy pyyaml`

Files
- `lb_domains.yaml`: domain spans on the LbCas12a reference. Set `reference.id` to the exact header of your `--ref_fasta` (or leave as `*` to ignore id mismatch warnings).
- `../../esmfold_domain_assessor.py`: scorer script at repo root.

Example usage
```
python esmfold_domain_assessor.py \
  --fasta cas12a_candidates.fasta \
  --ref_fasta lbcas12a_ref.fasta \
  --domains filtering/ESMfold_ranking/lb_domains.yaml \
  --pdb_dir esm_pdbs \
  --out_csv cas12a_esmfold_scores.csv
```

SLURM (H200 example)
```
sbatch filtering/ESMfold_ranking/submit_esmfold_ranking.slurm \
  --fasta /path/to/cas12a_candidates.fasta \
  --ref   /path/to/lbcas12a_ref.fasta \
  --out   esmfold_scores.csv \
  --pdb   esm_pdbs
```

SLURM array (H200, seeds 1–4)
```
sbatch filtering/ESMfold_ranking/submit_esmfold_seedarray_h200.slurm \
  --export=ALL,SEED_ROOT=/nfs/.../generations/<run>/by-seed,REF=/path/to/lbcas12a_ref.fasta
```
This looks for FASTAs at: /.../by-seed/seedN/seedN.filtered_step7_pass.fasta

Notes
- Keep WED and RuvC windows broad initially; refine after overlaying on a trusted structure. The scorer only needs coarse spans.
- Columns use domain names verbatim (e.g., `RUV-I/BH_mean_pLDDT`). The pass flags are computed over grouped domains: WED = all names starting with WED; RuvC = names containing RuvC/RUV; NUC = NUC; PI = PI.
