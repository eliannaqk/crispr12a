#!/usr/bin/env python3
"""
attach_dr_to_coverage.py — join DR map onto coverage/filter results.

Takes a TSV from your coverage step that includes the reference header (e.g., BLAST sseqid),
and attaches columns from the DR map produced by atlas_to_cas12a_dr_index.py.

Usage:
  python filtering/attach_dr_to_coverage.py \
    --coverage-tsv step1_coverage.tsv \
    --dr-map-tsv dr_map.ref_to_dr.tsv \
    --ref-id-col sseqid \
    --out-tsv step1_coverage_with_dr.tsv
"""
import argparse
import csv
import sys
from typing import Tuple, List


def read_tsv(path: str) -> Tuple[List[str], list[dict]]:
    with open(path, newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        rows = list(r)
        return r.fieldnames or [], rows


def load_dr_map(path: str) -> dict[str, dict]:
    d: dict[str, dict] = {}
    with open(path) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            d[row["ref_id"]] = row
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach DR info to coverage/filter TSV")
    ap.add_argument("--coverage-tsv", required=True, help="Output of your step-1 coverage filter")
    ap.add_argument("--dr-map-tsv", required=True, help="dr_map.ref_to_dr.tsv from atlas_to_cas12a_dr_index.py")
    ap.add_argument("--ref-id-col", default="sseqid", help="Column in coverage TSV holding the reference header")
    ap.add_argument("--out-tsv", required=True, help="Output TSV with DR columns appended (or '-' for stdout)")
    args = ap.parse_args()

    cov_cols, cov_rows = read_tsv(args.coverage_tsv)
    dr_map = load_dr_map(args.dr_map_tsv)

    add_cols = ["operon_id", "dr", "dr_len", "gene_name", "hmm_name", "subtype"]
    # include optional boolean if present in DR map
    extra = []
    if dr_map:
        sample = next(iter(dr_map.values()))
        if "dr_exact_to_lbcas12a" in sample:
            extra.append("dr_exact_to_lbcas12a")
    add_cols += extra

    out_cols = cov_cols + [c for c in add_cols if c not in cov_cols]
    out_fh = sys.stdout if args.out_tsv == "-" else open(args.out_tsv, "w")
    with out_fh:
        w = csv.DictWriter(out_fh, delimiter="\t", lineterminator="\n", fieldnames=out_cols)
        w.writeheader()
        miss = 0
        for row in cov_rows:
            ref_id = row.get(args.ref_id_col, "")
            dr_row = dr_map.get(ref_id)
            if dr_row:
                for c in add_cols:
                    row[c] = dr_row.get(c, "")
            else:
                miss += 1
                for c in add_cols:
                    row.setdefault(c, "")
            w.writerow(row)
    sys.stderr.write(f"[attach_dr_to_coverage] Annotated {len(cov_rows)-miss} rows, missing DR for {miss} rows\n")


if __name__ == "__main__":
    main()

