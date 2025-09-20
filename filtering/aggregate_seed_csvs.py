#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from pathlib import Path

def find_seed_files(base_dir: str, seed: int):
    pat = os.path.join(base_dir, f"generate_*_seed{seed}_*_raw.csv")
    return sorted(glob.glob(pat))

def main():
    ap = argparse.ArgumentParser(description="Aggregate generated raw CSVs for a seed into one CSV and add 'id' column.")
    ap.add_argument("--base_dir", required=True, help="Directory containing generate_* CSVs (e.g., .../generations/ba4000)")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seq_col", default="sequence")
    args = ap.parse_args()

    files = find_seed_files(args.base_dir, args.seed)
    if not files:
        raise SystemExit(f"[error] No raw CSVs found for seed{args.seed} under {args.base_dir}")

    n = 0
    header_written = False
    fieldnames = None
    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as fout:
        w = None
        for f in files:
            with open(f, newline='') as fin:
                r = csv.DictReader(fin)
                if args.seq_col not in r.fieldnames:
                    raise SystemExit(f"[error] missing column '{args.seq_col}' in {f}")
                # Prepare writer
                if not header_written:
                    fieldnames = list(r.fieldnames)
                    if 'id' not in fieldnames:
                        fieldnames = ['id'] + fieldnames
                    w = csv.DictWriter(fout, fieldnames=fieldnames)
                    w.writeheader()
                    header_written = True
                for row in r:
                    n += 1
                    row.setdefault('id', f"seed{args.seed}_row{n}")
                    w.writerow(row)
    print(f"[aggregate] seed{args.seed}: wrote {n} rows to {args.out_csv}")

if __name__ == "__main__":
    main()

