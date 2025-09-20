#!/usr/bin/env python3
"""
Create alternative Step 4 (DR) outputs using a different probability threshold
without re-running the ESM model. Uses existing per-sequence DR report TSVs.

For each seed directory under --by-seed-dir, this script reads:
  - <seed>/<seed>.step4_dr_report.tsv  (contains id, n_aa, logit, prob, keep)
  - <seed>/<seed>.step3_ppl_pass.fasta (to get sequences)

It writes:
  - <seed>/<seed>.<label>_report.tsv       (keep recomputed with --threshold)
  - <seed>/<seed>.<label>_pass.fasta       (sequences with prob >= threshold)

Example:
  python filtering/rethreshold_dr_reports.py \
    --by-seed-dir /path/to/by-seed \
    --threshold 0.2 \
    --label step4_dr_threshold2
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Recompute DR keep flags at a new threshold.")
    ap.add_argument('--by-seed-dir', required=True, type=Path)
    ap.add_argument('--threshold', required=True, type=float)
    ap.add_argument('--label', default=None, help='Output label prefix (e.g., step4_dr_threshold2). Default uses threshold value.')
    return ap.parse_args()


def load_dr_rows(dr_tsv: Path) -> Tuple[list, list]:
    ids = []
    rows = []
    with dr_tsv.open() as fh:
        header = fh.readline().rstrip('\n').split('\t')
        # Expect: id, n_aa, logit, prob, keep
        idx_id = header.index('id')
        idx_n = header.index('n_aa')
        idx_logit = header.index('logit')
        idx_prob = header.index('prob')
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= idx_prob:
                continue
            ids.append(parts[idx_id])
            rows.append((parts[idx_id], parts[idx_n], parts[idx_logit], parts[idx_prob]))
    return ids, rows


def load_fasta_index(fasta: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    if not fasta.exists():
        return seqs
    rid = None
    buf = []
    with fasta.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if rid is not None:
                    seqs[rid] = ''.join(buf)
                rid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if rid is not None:
            seqs[rid] = ''.join(buf)
    return seqs


def main() -> int:
    args = parse_args()
    base: Path = args.by_seed_dir
    label = args.label or f"step4_dr_thr{str(args.threshold).replace('.', '_')}"

    seeds = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith('seed')])
    if not seeds:
        print(f"No seed* directories found under {base}")
        return 1

    print(f"[rethr] by-seed: {base}")
    print(f"[rethr] threshold: {args.threshold} label: {label}")

    for sd in seeds:
        seed = sd.name
        prefix = seed
        dr_tsv = sd / f"{prefix}.step4_dr_report.tsv"
        ppl_fa = sd / f"{prefix}.step3_ppl_pass.fasta"
        out_tsv = sd / f"{prefix}.{label}_report.tsv"
        out_fa = sd / f"{prefix}.{label}_pass.fasta"

        if not dr_tsv.exists():
            print(f"[skip] {seed}: missing {dr_tsv}")
            continue
        if not ppl_fa.exists():
            print(f"[skip] {seed}: missing {ppl_fa}")
            continue

        ids, rows = load_dr_rows(dr_tsv)
        fasta_index = load_fasta_index(ppl_fa)

        kept = 0
        total = 0
        with out_tsv.open('w') as tfh, out_fa.open('w') as ffh:
            tfh.write('\t'.join(['id', 'n_aa', 'logit', 'prob', 'keep']) + '\n')
            for rid, n_aa, logit, prob in rows:
                total += 1
                try:
                    p = float(prob)
                except Exception:
                    p = float('nan')
                k = 1 if (p == p and p >= args.threshold) else 0
                kept += k
                tfh.write(f"{rid}\t{n_aa}\t{logit}\t{prob}\t{k}\n")
                if k == 1:
                    seq = fasta_index.get(rid)
                    if seq:
                        ffh.write(f">{rid}\n{seq}\n")

        rate = (kept / total) if total else 0.0
        print(f"[seed] {seed}: kept {kept}/{total} (pass_rate={rate:.4f}) -> {out_fa.name}, {out_tsv.name}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

