#!/usr/bin/env python3
"""
Stratified split of a labeled CSV into train/valid.

Requires a binary label column named 'pam_tttv'. Keeps header and all columns.

Usage:
  python scripts/split_csv_stratified.py \
    --in <input.csv> --out-train <train.csv> --out-valid <valid.csv> \
    --valid-frac 0.1 --seed 42
"""
from __future__ import annotations
import argparse, csv, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_csv', required=True)
    ap.add_argument('--out-train', required=True)
    ap.add_argument('--out-valid', required=True)
    ap.add_argument('--valid-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.in_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if 'pam_tttv' not in reader.fieldnames:
            raise SystemExit("[split] input missing 'pam_tttv' column")

    pos = [r for r in rows if str(r['pam_tttv']) == '1']
    neg = [r for r in rows if str(r['pam_tttv']) == '0']
    rng.shuffle(pos); rng.shuffle(neg)

    nvp = max(1, int(len(pos) * args.valid_frac))
    nvn = max(1, int(len(neg) * args.valid_frac))
    valid = pos[:nvp] + neg[:nvn]
    train = pos[nvp:] + neg[nvn:]
    rng.shuffle(train); rng.shuffle(valid)

    fields = list(rows[0].keys()) if rows else []
    for path, subset in [(args.out_train, train), (args.out_valid, valid)]:
        with open(path, 'w', newline='') as out:
            w = csv.DictWriter(out, fieldnames=fields)
            w.writeheader()
            for r in subset:
                w.writerow(r)
    print(f"[split] wrote train={len(train)} valid={len(valid)}  pos_train={sum(1 for r in train if str(r['pam_tttv'])=='1')} pos_valid={sum(1 for r in valid if str(r['pam_tttv'])=='1')}")

if __name__ == '__main__':
    main()

