#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

def read_seed_summary(seed_dir: Path):
    summary = seed_dir / f"{seed_dir.name}.filter_summary.tsv"
    if not summary.exists():
        return None
    metrics = {}
    with open(summary, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r, None)
        for row in r:
            if len(row) >= 2:
                metrics[row[0]] = row[1]
    # standard keys
    def geti(k):
        try:
            return int(metrics.get(k, "0"))
        except Exception:
            return 0
    return {
        "aggregated_raw": geti("aggregated_raw"),
        "step1_prefilter": geti("step1_prefilter"),
        "step2_hmm_pass": geti("step2_hmm_pass"),
        "fail_qcov": geti("fail_qcov"),
        "fail_tcov": geti("fail_tcov"),
        "fail_bitscore": geti("fail_bitscore"),
        "rows_in_report": geti("rows_in_report"),
    }

def main():
    ap = argparse.ArgumentParser(description="Collect per-seed filtering stats into one TSV with totals.")
    ap.add_argument("--base-dir", required=True, help="Base directory containing seed*/ subfolders (e.g., .../by-seed)")
    ap.add_argument("--out-tsv", required=True, help="Output TSV path for combined stats")
    args = ap.parse_args()

    base = Path(args.base_dir)
    seeds = sorted([p for p in base.glob("seed*") if p.is_dir()])
    rows = []
    totals = {
        "aggregated_raw": 0,
        "step1_prefilter": 0,
        "step2_hmm_pass": 0,
        "fail_qcov": 0,
        "fail_tcov": 0,
        "fail_bitscore": 0,
        "rows_in_report": 0,
    }
    for sd in seeds:
        stats = read_seed_summary(sd)
        if not stats:
            continue
        row = {
            "seed": sd.name,
            **stats
        }
        rows.append(row)
        for k in totals:
            totals[k] += stats.get(k, 0)

    outp = Path(args.out_tsv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed", "aggregated_raw", "step1_prefilter", "step2_hmm_pass", "fail_qcov", "fail_tcov", "fail_bitscore", "rows_in_report"]) 
        for r in rows:
            w.writerow([r["seed"], r["aggregated_raw"], r["step1_prefilter"], r["step2_hmm_pass"], r["fail_qcov"], r["fail_tcov"], r["fail_bitscore"], r["rows_in_report"]])
        w.writerow(["TOTAL", totals["aggregated_raw"], totals["step1_prefilter"], totals["step2_hmm_pass"], totals["fail_qcov"], totals["fail_tcov"], totals["fail_bitscore"], totals["rows_in_report"]])

    print(f"[stats] Wrote combined stats -> {outp}")
    print("[stats] Totals:", totals)

if __name__ == "__main__":
    main()

