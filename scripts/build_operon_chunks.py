#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Set


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def read_ids(files: List[Path]) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                op = line.strip()
                if not op:
                    continue
                if op in seen:
                    continue
                seen.add(op)
                ids.append(op)
    return ids


def started_safe_ids(outdirs: List[Path]) -> Set[str]:
    started: Set[str] = set()
    for od in outdirs:
        spdir = od / "spacers"
        if not spdir.exists():
            continue
        for fa in spdir.glob("*.fna"):
            name = fa.stem
            op = name.split("__arr")[0]
            started.add(op)
    return started


def write_chunks(ids: List[str], n: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [open(out_dir / f"chunk_{i:04d}.txt", "w") for i in range(n)]
    try:
        for idx, op in enumerate(ids):
            files[idx % n].write(op + "\n")
    finally:
        for f in files:
            f.close()


def main():
    ap = argparse.ArgumentParser(description="Build chunk files of operon IDs not yet started")
    ap.add_argument("--group-files", nargs="+", required=True, help="Paths to group_A/B.txt (operon IDs)")
    ap.add_argument("--outdirs", nargs="+", required=True, help="Existing output dirs to detect started operons (look under spacers/*.fna)")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--chunk-dir", required=True)
    args = ap.parse_args()

    group_files = [Path(p).resolve() for p in args.group_files]
    outdirs = [Path(p).resolve() for p in args.outdirs]
    all_ids = read_ids(group_files)

    started = started_safe_ids(outdirs)
    # Filter to those not yet started (safe-name based)
    pending = [op for op in all_ids if safe_name(op) not in started]

    write_chunks(pending, args.chunks, Path(args.chunk_dir))

    print(f"[summary] total_ids={len(all_ids)} started_ops={len(started)} pending={len(pending)} chunks={args.chunks}")
    print(f"[chunks] dir={args.chunk_dir}")


if __name__ == "__main__":
    main()

