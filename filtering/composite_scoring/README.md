# Composite Candidate Scoring

This module assembles per-seed metrics (LM perplexity, ESMFold confidence, DR compatibility, and PAM compatibility) and produces composite rankings for the novelty-filtered CRISPR candidates.

## Inputs
- `--by-seed-dir`: directory that contains `ppl_all.tsv`, `dr_all.tsv`, `motif_all.tsv`, and the `seed*/` subdirectories with Step 5 PAM reports.
- `--novelty-root`: directory holding `seed*/seed*_novel80_full.faa` from the novelty filter.
- `--esmfold-root`: root folder for seed ESMFold runs (defaults to `esmfold_seed_outputs`).
- `--esmfold-baseline-root`: optional baseline ESMFold scores (defaults to `esmfold_baseline/full_model_single`).
- `--output-dir`: destination for the CSV rankings (defaults to `metrics`).

## Usage
```bash
conda activate oc-opencrispr
python -m filtering.composite_scoring.score_pipeline \
  --by-seed-dir /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/by-seed \
  --novelty-root /home/eqk3/crispr12/opencrispr-repro-main/eval_outputs/novelty_cluster_run-20250828-193254-ba4000-sweep_ba4000 \
  --esmfold-root /home/eqk3/crispr12/opencrispr-repro-main/esmfold_seed_outputs \
  --esmfold-baseline-root /home/eqk3/crispr12/opencrispr-repro-main/esmfold_baseline/full_model_single \
  --output-dir /home/eqk3/crispr12/opencrispr-repro-main/metrics \
  --output-prefix ba4000_composite \
  --seeds seed2 seed3
```

The script writes a seed-specific CSV and an `*_all_seeds.csv` aggregate inside `--output-dir`. Columns include the composite score, rank within seed, z-scores, and classifier probabilities. Hard gates are intentionally omitted so you can filter later.

Running the command produces two artifacts per invocation:
- `<prefix>_raw_metrics.csv` – exact LM/DR/PAM/pLDDT values per sequence.
- `<prefix>_all_seeds.csv` plus per-seed splits – composite scores and z-scores (no hard gates applied; filter downstream as needed).

## Scoring formula
```
score = weight_plddt * z(pLDDT)
        + weight_lm * z(-PPL)
        + weight_dr * DR_prob
        + weight_pam * PAM_prob
        + optional novelty bonus
```
Defaults mirror the paper’s emphasis on LM signal (λ=0.7) while using pLDDT (weight 1.0) as a viability gate and DR/PAM probabilities as interoperability priors.
