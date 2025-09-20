#!/usr/bin/env python3
"""
Run Cas12a motif-only HMMER filter for a single seed directory (Step 5).

Inputs (from Step 3):
- seed_dir: contains seed{N}.step3_ppl_pass.fasta
- anchor_csv: CSV with a header 'sequence' and the LbCas12a anchor sequence (e.g., seq1.csv at repo root)
- hmm: full-length Cas12a HMM (pressed or not)
- config: filtering/cas12a_motif_filter_package/cas12a_motifs_lb_hmm.yaml

Outputs (into seed_dir):
- seed{N}.step4_motif_pass.fasta (copy of motif_pass.faa)
- seed{N}.step4_motif_report.tsv (copy of motif_summary.tsv)
- seed{N}.step4_motif_stats.txt (counts and pass rate)
- Step artifacts under seed_dir/step4_motif/{results,sets,...}

This wrapper calls the package script cas12a_motif_filter_package/cas12a_motif_filter_hmm.py.
It assumes hmmalign is available on PATH (we prepend bio-utils/bin in SLURM script).
"""
import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path


def read_anchor_csv(anchor_csv: Path) -> str:
    with open(anchor_csv, newline="") as f:
        r = csv.DictReader(f)
        if 'sequence' not in r.fieldnames:
            raise ValueError(f"Anchor CSV missing 'sequence' column: {anchor_csv}")
        rows = list(r)
        if not rows:
            raise ValueError(f"Anchor CSV empty: {anchor_csv}")
        seq = (rows[0].get('sequence') or '').strip().upper()
        if not seq:
            raise ValueError(f"Anchor CSV has empty sequence in first row: {anchor_csv}")
        # Strip any non-letter just in case
        seq = ''.join(ch for ch in seq if ch.isalpha())
        return seq


def write_fasta(seq_id: str, seq: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write(f'>{seq_id}\n')
        for i in range(0, len(seq), 80):
            fh.write(seq[i:i+80] + '\n')


def run_motif_filter(seed_dir: Path, in_fasta: Path, hmm: Path, config: Path, anchor_faa: Path, outdir: Path) -> None:
    # Delegate to the package script; ensure paths exist
    outdir.mkdir(parents=True, exist_ok=True)
    pkg_script = Path('filtering/cas12a_motif_filter_package/cas12a_motif_filter_hmm.py').resolve()
    if not pkg_script.exists():
        raise FileNotFoundError(f"Script not found: {pkg_script}")
    cmd = [
        'python', str(pkg_script),
        '--designs', str(in_fasta),
        '--config', str(config),
        '--hmm', str(hmm),
        '--anchor_fasta', str(anchor_faa),
        '--anchor_id_substring', 'LbCas12a',
        '--outdir', str(outdir),
    ]
    print('[motif-step4] RUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def summarize(seed_dir: Path, outdir: Path) -> None:
    # Copy outputs to seed-level files and compute simple stats
    summary_tsv = outdir / 'results' / 'motif_summary.tsv'
    pass_fa = outdir / 'sets' / 'motif_pass.faa'
    if not summary_tsv.exists():
        raise FileNotFoundError(f"Missing motif_summary.tsv: {summary_tsv}")
    kept = 0
    total = 0
    with open(summary_tsv, 'r') as fh:
        header = fh.readline()
        for line in fh:
            total += 1
            parts = line.rstrip('\n').split('\t')
            # columns: seq_id WED_pass PI_pass NUC_pass BH_pass LID_pass overall_pass seed_masked fail_reasons
            if len(parts) >= 7:
                overall = parts[6].strip()
                if overall.lower() in ('true', '1', 'yes'):
                    kept += 1
    rate = (kept / total) if total > 0 else 0.0

    seed_tag = seed_dir.name
    out_report = seed_dir / f'{seed_tag}.step4_motif_report.tsv'
    out_pass = seed_dir / f'{seed_tag}.step4_motif_pass.fasta'
    out_stats = seed_dir / f'{seed_tag}.step4_motif_stats.txt'
    shutil.copyfile(summary_tsv, out_report)
    if pass_fa.exists():
        shutil.copyfile(pass_fa, out_pass)
    with open(out_stats, 'w') as fh:
        fh.write(f'seed={seed_tag}\n')
        fh.write(f'total={total}\n')
        fh.write(f'kept={kept}\n')
        fh.write(f'filtered_out={total - kept}\n')
        fh.write(f'pass_rate={rate:.4f}\n')
    print(f"[motif-step4] Summary: kept {kept}/{total} (pass_rate={rate:.2%}) -> {out_pass}")


def main():
    ap = argparse.ArgumentParser(description='Run motif-only HMM filter (Step 4) for a seed directory')
    ap.add_argument('--seed-dir', required=True, help='Path to seed directory (contains step4 outputs)')
    ap.add_argument('--hmm', required=True, help='Path to full-length Cas12a HMM')
    ap.add_argument('--config', required=True, help='Path to cas12a_motifs_lb_hmm.yaml')
    ap.add_argument('--anchor-csv', required=True, help='CSV with header "sequence" containing the Lb anchor')
    args = ap.parse_args()

    seed_dir = Path(args.seed_dir).resolve()
    in_fasta = seed_dir / f'{seed_dir.name}.step3_ppl_pass.fasta'
    if not in_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found (Step 4 survivors): {in_fasta}")

    hmm = Path(args.hmm).resolve()
    if not hmm.exists():
        raise FileNotFoundError(f"HMM not found: {hmm}")
    config = Path(args.config).resolve()
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    anchor_csv = Path(args.anchor_csv).resolve()
    if not anchor_csv.exists():
        raise FileNotFoundError(f"Anchor CSV not found: {anchor_csv}")

    outdir = seed_dir / 'step4_motif'
    anchor_seq = read_anchor_csv(anchor_csv)
    anchor_faa = outdir / 'anchor.faa'
    write_fasta('LbCas12a_anchor', anchor_seq, anchor_faa)

    run_motif_filter(seed_dir, in_fasta, hmm, config, anchor_faa, outdir)
    summarize(seed_dir, outdir)


if __name__ == '__main__':
    main()
