#!/usr/bin/env python3
"""
Investigate overlap between two step-4 filters (DR vs motif) per seed.

Inputs: a by-seed directory containing subfolders seed1/, seed2/, ...
Each seed folder may contain:
  - <seed>.step4_dr_pass.fasta
  - <seed>.step4_motif_pass.fasta
  - <seed>.step4_dr_report.tsv (fallback: use keep==1)
  - <seed>.step4_motif_report.tsv (fallback: use overall_pass==True)

Output: prints per-seed counts and intersection, plus an optional TSV.

Usage:
  python investigate_step4_overlap.py --by-seed-dir /path/to/by-seed [--out overlap.tsv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def read_fasta_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return ids
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids


def read_dr_report_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            id_idx = header.index("id")
            keep_idx = header.index("keep")
        except ValueError:
            return ids
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, keep_idx):
                continue
            try:
                keep = int(parts[keep_idx])
            except ValueError:
                continue
            if keep == 1:
                ids.add(parts[id_idx])
    return ids


def read_motif_report_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        # columns: seq_id, WED_pass, PI_pass, NUC_pass, BH_pass, LID_pass, overall_pass, ...
        try:
            id_idx = header.index("seq_id")
            overall_idx = header.index("overall_pass")
        except ValueError:
            return ids
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, overall_idx):
                continue
            overall = parts[overall_idx]
            if overall == "True":
                ids.add(parts[id_idx])
    return ids


def summarize_seed(seed_dir: Path) -> Tuple[str, int, int, int, float]:
    seed = seed_dir.name
    prefix = seed  # files are named like seedX.seedX.step4_...

    dr_fa = seed_dir / f"{prefix}.step4_dr_pass.fasta"
    motif_fa = seed_dir / f"{prefix}.step4_motif_pass.fasta"
    dr_ids = read_fasta_ids(dr_fa)
    motif_ids = read_fasta_ids(motif_fa)

    # Fallbacks to reports if FASTA not present or empty
    if not dr_ids:
        dr_ids = read_dr_report_ids(seed_dir / f"{prefix}.step4_dr_report.tsv")
    if not motif_ids:
        motif_ids = read_motif_report_ids(seed_dir / f"{prefix}.step4_motif_report.tsv")

    inter = dr_ids & motif_ids
    union = dr_ids | motif_ids
    jaccard = (len(inter) / len(union)) if union else 0.0
    return seed, len(dr_ids), len(motif_ids), len(inter), jaccard


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-seed-dir", required=True, type=Path, help="Path to by-seed directory")
    ap.add_argument("--out", type=Path, default=None, help="Optional TSV output path")
    args = ap.parse_args(argv)

    base = args.by_seed_dir
    if not base.exists():
        print(f"ERROR: by-seed directory not found: {base}", file=sys.stderr)
        return 2

    seeds = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed")])
    if not seeds:
        print(f"ERROR: no seed* subdirectories found in {base}", file=sys.stderr)
        return 2

    rows: List[Tuple[str, int, int, int, float]] = []
    print("seed\tdr_pass\tmotif_pass\tintersection\tjaccard")
    for sd in seeds:
        seed, n_dr, n_motif, n_inter, j = summarize_seed(sd)
        rows.append((seed, n_dr, n_motif, n_inter, j))
        print(f"{seed}\t{n_dr}\t{n_motif}\t{n_inter}\t{j:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as out:
            out.write("seed\tdr_pass\tmotif_pass\tintersection\tjaccard\n")
            for seed, n_dr, n_motif, n_inter, j in rows:
                out.write(f"{seed}\t{n_dr}\t{n_motif}\t{n_inter}\t{j:.6f}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

