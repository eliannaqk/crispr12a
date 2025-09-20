#!/usr/bin/env python3
import argparse
import csv

def main():
    ap = argparse.ArgumentParser(description="Select rows from CSV where coverage TSV has keep==1 (match by 'id').")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--coverage_tsv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col", default="id")
    args = ap.parse_args()

    keep_ids = set()
    with open(args.coverage_tsv, newline="") as fin:
        r = csv.DictReader(fin, delimiter='\t')
        # domtbl writer used explicit header order; ensure 'id' and 'keep'
        for row in r:
            try:
                if row.get('keep') in (1, '1', 'True', 'true'):
                    keep_ids.add(str(row.get('id')))
            except Exception:
                continue

    n_in = 0
    n_out = 0
    with open(args.in_csv, newline="") as fin, open(args.out_csv, 'w', newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            n_in += 1
            if str(row.get(args.id_col)) in keep_ids:
                w.writerow(row)
                n_out += 1
    print(f"[select] kept {n_out} / {n_in} -> {args.out_csv}")

if __name__ == "__main__":
    main()

