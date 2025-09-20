#!/usr/bin/env python3
"""Assemble Step 7 filtered FASTAs by intersecting earlier filter outputs.

This utility walks a ``by-seed`` directory produced by the filtering pipeline and
creates the canonical ``seedN.filtered_step7_pass.fasta`` files together with the
``filtered_step7/pass.faa`` subdirectories. The intersection is defined as the
set of sequence IDs that survive all of:

* Step 4 motif filter (``seedN.step4_motif_pass.fasta``)
* Step 4 DR thresholded filter (default: ``seedN.step4_dr_threshold2_pass.fasta``)
* Step 5 PAM filter at the stricter probability threshold (default: ``seedN.step5_pam_filter_pass_thr0p50.fasta``)

The script reuses the sequences stored in the DR threshold FASTA so the header
order follows the DR classifier output. A per-seed summary CSV is also written
(to ``metrics/filtered_step7_summary.csv`` by default).
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class SeedSummary:
    name: str
    motif_count: int
    dr_count: int
    pam_count: int
    intersection_count: int
    wrote_outputs: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by-seed-dir",
        required=True,
        type=Path,
        help="Directory containing seed*/ subfolders (e.g., generations/.../by-seed)",
    )
    parser.add_argument(
        "--motif-suffix",
        default="step4_motif_pass.fasta",
        help="Filename (within each seed dir) holding motif survivors.",
    )
    parser.add_argument(
        "--dr-suffix",
        default="step4_dr_threshold2_pass.fasta",
        help="Filename for the DR thresholded survivors used as sequence source.",
    )
    parser.add_argument(
        "--pam-suffix",
        default="step5_pam_filter_pass_thr0p50.fasta",
        help="Filename for the PAM survivors at the stricter threshold.",
    )
    parser.add_argument(
        "--out-subdir",
        default="filtered_step7",
        help="Subdirectory name to hold pass.faa outputs inside each seed dir.",
    )
    parser.add_argument(
        "--canonical-template",
        default="{seed}.filtered_step7_pass.fasta",
        help="Filename template (inside each seed dir) for the canonical FASTA.",
    )
    parser.add_argument(
        "--summary-csv",
        default="metrics/filtered_step7_summary.csv",
        help="Path to write per-seed counts (CSV). Relative paths resolve from CWD.",
    )
    parser.add_argument(
        "--force-write",
        action="store_true",
        help="Overwrite existing pass.faa / canonical FASTA files.",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> OrderedDict[str, str]:
    records: OrderedDict[str, str] = OrderedDict()
    if not path.exists():
        return records
    header: str | None = None
    seq_chunks: List[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            records[header] = "".join(seq_chunks)
    return records


def write_fasta(pairs: Iterable[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for header, seq in pairs:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def main() -> int:
    args = parse_args()
    base = args.by_seed_dir.expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"by-seed directory not found: {base}")

    seed_dirs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed"))
    if not seed_dirs:
        raise RuntimeError(f"No seed*/ directories found under {base}")

    summaries: List[SeedSummary] = []

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        motif_path = seed_dir / f"{seed_name}.{args.motif_suffix}" if "{" not in args.motif_suffix else Path(
            args.motif_suffix.format(seed=seed_name)
        )
        if not motif_path.exists():
            motif_path = seed_dir / args.motif_suffix
        dr_path = seed_dir / f"{seed_name}.{args.dr_suffix}" if "{" not in args.dr_suffix else Path(
            args.dr_suffix.format(seed=seed_name)
        )
        if not dr_path.exists():
            dr_path = seed_dir / args.dr_suffix
        pam_path = seed_dir / f"{seed_name}.{args.pam_suffix}" if "{" not in args.pam_suffix else Path(
            args.pam_suffix.format(seed=seed_name)
        )
        if not pam_path.exists():
            pam_path = seed_dir / args.pam_suffix

        motif_ids = set(read_fasta(motif_path).keys())
        dr_records = read_fasta(dr_path)
        pam_ids = set(read_fasta(pam_path).keys())

        if not motif_ids or not dr_records or not pam_ids:
            print(
                f"[warn] Skipping {seed_name}: missing inputs (motif={motif_path.exists()} dr={dr_path.exists()} pam={pam_path.exists()})"
            )
            summaries.append(
                SeedSummary(
                    name=seed_name,
                    motif_count=len(motif_ids),
                    dr_count=len(dr_records),
                    pam_count=len(pam_ids),
                    intersection_count=0,
                    wrote_outputs=False,
                )
            )
            continue

        intersection_ids = [sid for sid in dr_records.keys() if sid in motif_ids and sid in pam_ids]
        print(
            f"[info] {seed_name}: motif={len(motif_ids)} dr={len(dr_records)} pam={len(pam_ids)} intersection={len(intersection_ids)}"
        )

        pairs = [(sid, dr_records[sid]) for sid in intersection_ids]
        out_dir = seed_dir / args.out_subdir
        pass_path = out_dir / "pass.faa"
        canonical_name = args.canonical_template.format(seed=seed_name)
        canonical_path = seed_dir / canonical_name

        wrote = False
        if pairs:
            if args.force_write or not pass_path.exists():
                write_fasta(pairs, pass_path)
                wrote = True
                print(f"  -> wrote {pass_path}")
            else:
                print(f"  -> existing {pass_path} (skip; use --force-write to overwrite)")

            if args.force_write or not canonical_path.exists():
                write_fasta(pairs, canonical_path)
                wrote = True or wrote
                print(f"  -> wrote {canonical_path}")
            else:
                print(f"  -> existing {canonical_path} (skip; use --force-write to overwrite)")
        else:
            print(f"  -> no intersection survivors for {seed_name}; outputs not written")

        summaries.append(
            SeedSummary(
                name=seed_name,
                motif_count=len(motif_ids),
                dr_count=len(dr_records),
                pam_count=len(pam_ids),
                intersection_count=len(intersection_ids),
                wrote_outputs=wrote,
            )
        )

    summary_path = Path(args.summary_csv)
    if not summary_path.is_absolute():
        summary_path = Path.cwd() / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "seed",
            "motif_pass",
            "dr_threshold_pass",
            "pam_pass",
            "filtered_step7",
            "wrote_outputs",
        ])
        for row in summaries:
            writer.writerow(
                [
                    row.name,
                    row.motif_count,
                    row.dr_count,
                    row.pam_count,
                    row.intersection_count,
                    int(row.wrote_outputs),
                ]
            )
    print(f"[info] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
