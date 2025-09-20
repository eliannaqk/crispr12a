#!/usr/bin/env python3
import argparse
import csv
from typing import Dict

AA_SET = set(list("ACDEFGHIKLMNPQRSTVWYXBZJUO"))

def _kmer_n_repeat(k: int, n: int) -> str:
    return "(.)" * k + "".join([f"\\{i+1}" for i in range(k)]) * (n - 1)

import re
_KMER_THRESHOLDS = [6, 4, 3, 3, 3, 3, 2]  # k=1..7
_KMER_RES = [re.compile(_kmer_n_repeat(k, n)) for k, n in enumerate(_KMER_THRESHOLDS, start=1)]

def has_kmer_repeats(seq: str) -> bool:
    return any(r.search(seq) for r in _KMER_RES)

def clean_seq(seq: str) -> str:
    return "".join(ch for ch in str(seq).strip().upper() if ch not in {"*", " ", "\n", "\r"})

def normalize_eos_and_orientation(raw: str, flip_on_eos1: bool = True):
    s = clean_seq(raw)
    if s and s[0] in ("1", "2"):
        s = s[1:]
    eos = None
    flipped = False
    if s and s[-1] in ("1", "2"):
        eos = s[-1]
        s = s[:-1]
        if flip_on_eos1 and eos == "1":
            s = s[::-1]
            flipped = True
    s = "".join(ch for ch in s if ch.isalpha())
    return s, eos, flipped

def seq_ok(seq: str) -> bool:
    return bool(seq) and all((ch in AA_SET) for ch in seq)

def main():
    ap = argparse.ArgumentParser(description="Prefilter CSV by EOS handling, length, and k-mer repeats.")
    ap.add_argument("--in_csv", required=True, help="Input aggregated CSV with a sequence column.")
    ap.add_argument("--out_csv", required=True, help="Output CSV after prefiltering (same columns, cleaned sequences).")
    ap.add_argument("--seq_col", default="sequence", help="Sequence column name (default 'sequence').")
    ap.add_argument("--flip_on_eos1", action="store_true", help="Flip N<->C if trailing EOS '1'.")
    ap.add_argument("--kmer_filter", action="store_true", help="Drop sequences with repeat k-mers (1..7-mer thresholds).")
    ap.add_argument("--min_len", type=int, default=0, help="Min AA length after cleaning (e.g., 1000).")
    ap.add_argument("--max_len", type=int, default=0, help="Max AA length after cleaning (e.g., 1500).")
    ap.add_argument("--id_col", default="id", help="ID column name to preserve if present.")
    args = ap.parse_args()

    with open(args.in_csv, newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        # Ensure id column exists for downstream linking
        if args.id_col not in fieldnames:
            fieldnames = [args.id_col] + fieldnames
        # Ensure seq_col exists
        if args.seq_col not in fieldnames:
            raise SystemExit(f"[error] CSV missing sequence column '{args.seq_col}'")

        with open(args.out_csv, "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=fieldnames)
            w.writeheader()
            idx = 0
            kept = 0
            for row in reader:
                idx += 1
                if not row.get(args.id_col):
                    row[args.id_col] = f"row{idx}"
                raw = row.get(args.seq_col, "")
                seq, eos, flipped = normalize_eos_and_orientation(raw, flip_on_eos1=args.flip_on_eos1)
                if not seq or not seq_ok(seq):
                    continue
                if args.min_len and len(seq) < args.min_len:
                    continue
                if args.max_len and len(seq) > args.max_len:
                    continue
                if args.kmer_filter and has_kmer_repeats(seq):
                    continue
                row[args.seq_col] = seq
                w.writerow(row)
                kept += 1
    print(f"[prefilter] wrote {kept} sequences to {args.out_csv}")

if __name__ == "__main__":
    main()

