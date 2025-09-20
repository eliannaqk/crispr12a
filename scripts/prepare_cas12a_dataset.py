#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import subprocess
from typing import List, Tuple


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        header = None
        seq_lines: List[str] = []
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_lines).replace("\n", "").replace("\r", "")
                    records.append((header, seq))
                header = line[1:].strip().split()[0]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            seq = "".join(seq_lines).replace("\n", "").replace("\r", "")
            records.append((header, seq))
    return records


def write_fasta(records: List[Tuple[str, str]], path: str) -> None:
    with open(path, "w") as out:
        for rid, seq in records:
            out.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")


def write_csv(records: List[Tuple[str, str]], path: str) -> None:
    with open(path, "w") as out:
        out.write("id,sequence\n")
        for rid, seq in records:
            out.write(f"{rid},{seq}\n")


def word_length_for_identity(c: float) -> int:
    if c >= 0.7:
        return 5
    elif c >= 0.6:
        return 4
    else:
        return 3


def run_cdhit_protein(input_faa: str, output_faa: str, identity: float) -> bool:
    cdhit = shutil.which("cd-hit")
    if not cdhit:
        return False
    n = word_length_for_identity(identity)
    cmd = [
        cdhit,
        "-i", input_faa,
        "-o", output_faa,
        "-c", str(identity),
        "-n", str(n),
        "-T", "0",
        "-M", "0",
    ]
    subprocess.run(cmd, check=True)
    return True


def split_records(records: List[Tuple[str, str]], train: float, val: float, test: float, seed: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    assert abs(train + val + test - 1.0) < 1e-6, "train+val+test must equal 1.0"
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n = len(records)
    n_train = int(n * train)
    n_val = int(n * val)
    def take(idxs_slice: List[int]) -> List[Tuple[str, str]]:
        return [records[i] for i in idxs_slice]
    train_records = take(idxs[:n_train])
    val_records = take(idxs[n_train:n_train+n_val])
    test_records = take(idxs[n_train+n_val:])
    return train_records, val_records, test_records


def main():
    p = argparse.ArgumentParser(description="Cluster (cd-hit), split, and export FASTA/CSV for Cas12a dataset")
    p.add_argument("--input", required=True, help="Input FASTA (.faa)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--cdhit-identity", type=float, default=0.95, help="cd-hit identity threshold (protein)")
    p.add_argument("--train", type=float, default=0.8, help="Train fraction")
    p.add_argument("--val", type=float, default=0.1, help="Val fraction")
    p.add_argument("--test", type=float, default=0.1, help="Test fraction")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
    p.add_argument("--skip-cdhit", action="store_true", help="Skip cd-hit clustering")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clustered_faa = os.path.join(args.out_dir, "clustered.faa")
    used_faa = args.input
    ran_cdhit = False
    if not args.skip_cdhit:
        try:
            ran_cdhit = run_cdhit_protein(args.input, clustered_faa, args.cdhit_identity)
            if ran_cdhit:
                used_faa = clustered_faa
        except subprocess.CalledProcessError as e:
            print(f"cd-hit failed: {e}. Proceeding without clustering.")
            ran_cdhit = False

    records = parse_fasta(used_faa)

    train_recs, val_recs, test_recs = split_records(records, args.train, args.val, args.test, args.seed)

    write_fasta(train_recs, os.path.join(args.out_dir, "train.faa"))
    write_fasta(val_recs, os.path.join(args.out_dir, "val.faa"))
    write_fasta(test_recs, os.path.join(args.out_dir, "test.faa"))

    write_csv(train_recs, os.path.join(args.out_dir, "train.csv"))
    write_csv(val_recs, os.path.join(args.out_dir, "val.csv"))
    write_csv(test_recs, os.path.join(args.out_dir, "test.csv"))

    stats_path = os.path.join(args.out_dir, "stats.txt")
    with open(stats_path, "w") as s:
        s.write(f"input_fasta: {args.input}\n")
        s.write(f"cdhit_run: {ran_cdhit}\n")
        if ran_cdhit:
            s.write(f"cdhit_identity: {args.cdhit_identity}\n")
            s.write(f"clustered_fasta: {used_faa}\n")
        s.write(f"total_sequences: {len(records)}\n")
        s.write(f"train: {len(train_recs)}\n")
        s.write(f"val: {len(val_recs)}\n")
        s.write(f"test: {len(test_recs)}\n")

    print("Done.")
    print(f"Total: {len(records)} | Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")
    if not ran_cdhit:
        print("Note: cd-hit not run (not found or skipped). Output uses all input sequences.")


if __name__ == "__main__":
    main()

