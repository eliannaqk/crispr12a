#!/usr/bin/env python3
import argparse, sys, re
from pathlib import Path
from typing import List, Tuple


def list_missing(outdir: Path) -> Tuple[List[Tuple[Path, str]], dict]:
    spacers = sorted((outdir / "spacers").glob("*.fna"))
    total = len(spacers)
    run_base = outdir / "pampredict_runs"
    have_blast = have_info = have_pam = 0
    missing = []
    for fa in spacers:
        name = fa.stem  # <operon_safe>__arrN
        rd = run_base / name
        up = rd / "upstream_flanking_sequence_info.tsv"
        dn = rd / "downstream_flanking_sequence_info.tsv"
        pam = rd / "PAM_prediction.txt"
        filt = rd / "blastn" / "viruses_plasmids_filtered_matches_with_flanking_sequences.tsv"
        if filt.exists():
            have_blast += 1
        if up.exists() and dn.exists():
            have_info += 1
        if pam.exists():
            have_pam += 1
        if not (up.exists() and dn.exists()):
            missing.append((outdir, name))
    stats = {
        "total": total,
        "have_blast": have_blast,
        "have_info": have_info,
        "have_pam": have_pam,
        "missing": len(missing),
    }
    return missing, stats


def write_chunks(items: List[Tuple[Path, str]], chunk_dir: Path, n_chunks: int):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # simple round-robin split for balance
    files = [open(chunk_dir / f"chunk_{i:04d}.tsv", "w") for i in range(n_chunks)]
    try:
        for idx, (outdir, name) in enumerate(items):
            f = files[idx % n_chunks]
            f.write(f"{outdir}\t{name}\n")
    finally:
        for f in files:
            f.close()


def main():
    ap = argparse.ArgumentParser(description="Build chunk files for missing PAMpredict arrays")
    ap.add_argument("--outdirs", nargs="+", required=True, help="One or more output directories (with spacers/ and pampredict_runs/)")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--chunk-dir", required=True)
    args = ap.parse_args()

    all_missing: List[Tuple[Path, str]] = []
    summary = []
    for od in args.outdirs:
        outdir = Path(od).resolve()
        m, st = list_missing(outdir)
        all_missing.extend(m)
        summary.append((outdir, st))

    chunk_dir = Path(args.chunk_dir).resolve()
    write_chunks(all_missing, chunk_dir, args.chunks)

    # Print summary for caller
    total = sum(s["total"] for _, s in summary)
    have_blast = sum(s["have_blast"] for _, s in summary)
    have_info = sum(s["have_info"] for _, s in summary)
    have_pam = sum(s["have_pam"] for _, s in summary)
    missing = sum(s["missing"] for _, s in summary)
    print(f"[summary] outdirs={len(summary)} total_arrays={total} have_blast={have_blast} have_info={have_info} have_pam={have_pam} missing={missing}")
    for od, st in summary:
        print(f"[detail] {od} total={st['total']} blast={st['have_blast']} info={st['have_info']} pam={st['have_pam']} missing={st['missing']}")
    print(f"[chunks] wrote {args.chunks} files under {chunk_dir}")


if __name__ == "__main__":
    main()

