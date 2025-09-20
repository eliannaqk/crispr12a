#!/usr/bin/env python
"""
fasta_to_csv.py

Utility to convert one or many FASTA files to 2-column CSV files (id, sequence).

Examples
--------
# Single file
python fasta_to_csv.py input.fasta output.csv

# Convert all *.fasta in a directory (outputs side-by-side *.csv)
python fasta_to_csv.py --dir /data/fasta_files

# Custom suffix and separate output directory
python fasta_to_csv.py --dir /data/fasta_files --suffix .fa --out-dir /data/csv_files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import csv
from Bio import SeqIO


def convert_one(fasta_path: Path, csv_path: Path):
    """Convert *fasta_path* to *csv_path* with columns: id, sequence."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        # Match reference format: two columns 'id,sequence'
        writer.writerow(["id", "sequence"])
        for rec in SeqIO.parse(fasta_path, "fasta"):
            writer.writerow([str(rec.id), str(rec.seq)])
            count += 1
    if count == 0:
        raise SystemExit(f"No FASTA records found in {fasta_path}")
    print(f"[✓] {fasta_path} -> {csv_path} ({count} rows)")


def convert_many(in_dir: Path, suffix: str, out_dir: Path | None):
    """Convert every *suffix* file inside *in_dir* to CSV inside *out_dir* (defaults to *in_dir*)."""
    out_dir = out_dir or in_dir
    fasta_files = sorted(in_dir.glob(f"*{suffix}"))
    if not fasta_files:
        print(f"No files ending with '{suffix}' found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    for fp in fasta_files:
        csv_name = fp.stem + ".csv"
        convert_one(fp, out_dir / csv_name)


def main():
    parser = argparse.ArgumentParser(description="Convert FASTA files to CSV (id,sequence).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Directory containing FASTA files to convert in bulk.")
    group.add_argument("fasta_file", nargs="?", type=Path, help="Path to a single FASTA file.")

    parser.add_argument("csv_file", nargs="?", type=Path, help="Output CSV path for single-file mode.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for CSVs when using --dir.")
    parser.add_argument("--suffix", default=".fasta", help="FASTA file suffix to match in --dir mode (default: .fasta)")
    args = parser.parse_args()

    if args.dir is not None:
        convert_many(args.dir, args.suffix, args.out_dir)
    else:
        if args.csv_file is None:
            parser.error("csv_file required when converting a single file")
        convert_one(args.fasta_file, args.csv_file)


if __name__ == "__main__":
    main()
