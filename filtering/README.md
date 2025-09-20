# Filtering Pipeline (ba4000 run)

This folder holds the tooling that generates the per-seed filtered FASTA files under
`/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/by-seed/seedN/`.
Each stage is driven by a helper script plus an `sbatch` wrapper. GPU steps run in
`oc-opencrispr` (or `oc-opencrispr-esm` when the classifier needs ESM), while HMM/PID
utilities use `bio-utils` as noted in the job scripts.

## Stage overview

| Stage | Outputs | Driver script | Launcher |
|-------|---------|---------------|----------|
|1. Aggregate & prefilter | `seedN.aggregated_raw.csv`, `seedN.step1_prefilter.csv` | `filtering/aggregate_seed_csvs.py`, `filtering/prefilter_csv.py` | `filtering/seed_filtering_array_cpu.slurm`|
|2. Cas12a HMM coverage | `seedN.step2_hmm_pass.{csv,fasta}`, `seedN.step2_hmm_report.tsv` | `filtering/filter_coverage.py` + `filtering/select_by_tsv_keep.py` | `filtering/seed_filtering_array_cpu.slurm`|
|3. LM perplexity gate | `seedN.step3_ppl_pass.fasta`, `seedN.step3_ppl_report.tsv`, `ppl_all.tsv` | `filtering/perplexity_filter.py` | `filtering/submit_ppl_filter_array_h200.slurm`|
|4a. DR classifier | `seedN.step4_dr_pass.fasta`, `seedN.step4_dr_report.tsv`, `dr_all.tsv` | `filtering/dr_filter_fasta.py` | `filtering/submit_lbdr_filter_array_gpu.slurm`|
|4b. DR re-threshold (0.2) | `seedN.step4_dr_threshold2_pass.fasta`, `seedN.step4_dr_threshold2_report.tsv` | `filtering/rethreshold_dr_reports.py` | manual CLI |
|4c. Motif HMM | `seedN.step4_motif_pass.fasta`, `motif_all.tsv` | `filtering/run_motif_filter_seed.py` | `filtering/submit_motif_filter_array_cpu.slurm`|
|5. PAM classifier (0.45 / 0.50) | `seedN.step5_pam_filter_pass.fasta`, `seedN.step5_pam_filter_pass_thr0p50.fasta`, `step5_pam_filter/{pass,pass_thr0p50}.faa` | `filtering/pam_filtering_esm/pam_filter_from_esm.py` | `filtering/submit_pam_step5_filter_array_esm.slurm`|
|5b. PID 8Å contacts (monitoring) | `seedN.step5_PID_angstrom_pass.fasta` | `filtering/pid_8A_filter_fullseq.py` | `filtering/submit_pid_step5_filter_array.slurm`|
|7. Final intersection | `filtered_step7/pass.faa`, `seedN.filtered_step7_pass.fasta` | `filtering/build_filtered_step7.py` | manual CLI |

## Step 7: building the canonical `filtered_step7` FASTAs

The final “step7” survivors are the intersection of three earlier filters:

* Motif pass list (`seedN.step4_motif_pass.fasta`)
* DR classifier pass list at probability ≥0.2 (`seedN.step4_dr_threshold2_pass.fasta`)
* PAM classifier pass list at probability ≥0.5 (`seedN.step5_pam_filter_pass_thr0p50.fasta`)

Use the helper below (requires `oc-opencrispr`) to regenerate the canonical FASTA files
and record counts. Existing files are left in place unless `--force-write` is provided.

```bash
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate oc-opencrispr
python filtering/build_filtered_step7.py \
  --by-seed-dir /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/by-seed
```

The script writes a summary to `metrics/filtered_step7_summary.csv`. The current counts are:

| seed | motif_pass | dr_threshold_pass | pam_pass | filtered_step7 |
|------|------------|-------------------|----------|----------------|
|seed1 | 85467 | 2599 | 83982 | 2078 |
|seed2 | 79445 | 387 | 62978 | 385 |
|seed3 | 105665 | 1550 | 28215 | 272 |
|seed4 | 119400 | 39595 | 119400 | 39595 |

These numbers match the files already checked into the shared data directory (derived
on 2025‑09‑15). Re-run the utility whenever the upstream thresholds change so the
CSV stays in sync with `/metrics`.

## Notes

* Monitor submitted jobs with `squeue -u eqk3`.
* All scripts assume the ba4000 generation CSVs live under
  `/home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000`.
  Override paths via the corresponding environment variables/CLI flags in the `sbatch` wrappers.
* Keep Weights & Biases credentials out of source; the SLURM launchers export `WANDB_API_KEY` at runtime when needed.
