#!/usr/bin/env python
"""
analyze_data.py

Analyze sequence lengths and simulated minibatch padded lengths for CSV datasets.

Usage:
  python analyze_data.py --data-dir /path/to/csvs --batch-size 8

Notes:
  - Expects CSVs with header containing 'sequence' (and optional 'id').
  - For minibatch analysis, uses the ProGen2 tokenizer and the same sequence
    formatting as training: "1" + sequence + "2" before tokenization.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from opencrispr_repro.model import ModelSchema, get_tokenizer


def _load_sequences(csv_path: Path) -> List[str]:
    # Robust CSV read (handles LF/CRLF); expect either ['sequence'] or ['id','sequence']
    df = pd.read_csv(csv_path)
    cols = [c.strip().lower() for c in df.columns]
    if "sequence" in cols:
        seq_col = df.columns[cols.index("sequence")]
        seqs = df[seq_col].astype(str).tolist()
        return seqs
    raise KeyError(
        f"Column 'sequence' not found in {csv_path}. Available columns: {list(df.columns)}"
    )


def _describe(values: List[int]) -> dict:
    arr = np.asarray(values, dtype=np.int64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()) if arr.size else 0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "max": int(arr.max()) if arr.size else 0,
    }


def _print_stats(title: str, stats: dict) -> None:
    print(title)
    print(f"  count  : {stats['count']}")
    print(f"  min    : {stats['min']}")
    print(f"  mean   : {stats['mean']:.2f}")
    print(f"  median : {stats['median']:.2f}")
    print(f"  p90    : {stats['p90']:.2f}")
    print(f"  p95    : {stats['p95']:.2f}")
    print(f"  max    : {stats['max']}")


def _token_lengths(tokenizer, sequences: List[str]) -> List[int]:
    # Match training collate: wrap with sentinels "1" and "2"
    wrapped = ["1" + s + "2" for s in sequences]
    # Batch tokenize in chunks to avoid memory spikes
    lengths: List[int] = []
    chunk = 2048
    for i in range(0, len(wrapped), chunk):
        batch = wrapped[i : i + chunk]
        toks = tokenizer(batch, return_tensors=None, padding=False)
        # transformers Tokenizer returns list-of-lists under 'input_ids'
        ids = toks["input_ids"]
        lengths.extend(int(len(x)) for x in ids)
    return lengths


def _batch_padded_lengths(token_lengths: List[int], batch_size: int) -> List[int]:
    padded: List[int] = []
    for i in range(0, len(token_lengths), batch_size):
        padded.append(int(max(token_lengths[i : i + batch_size])))
    return padded


def analyze_split(name: str, csv_path: Path, batch_size: int, tokenizer) -> None:
    if not csv_path.exists():
        return
    print(f"=== Split: {name} ({csv_path}) ===")
    seqs = _load_sequences(csv_path)
    char_lengths = [len(s) for s in seqs]
    _print_stats("Character lengths", _describe(char_lengths))

    tok_lengths = _token_lengths(tokenizer, seqs)
    _print_stats("Token lengths (pre-padding)", _describe(tok_lengths))

    padded = _batch_padded_lengths(tok_lengths, batch_size)
    _print_stats(
        f"Batch-padded lengths (batch_size={batch_size})",
        _describe(padded),
    )
    print()


def main():
    ap = argparse.ArgumentParser(description="Analyze sequence and minibatch lengths for CSV datasets.")
    ap.add_argument("--data-dir", required=True, type=Path, help="Directory containing train/valid/test CSV files")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size to simulate for padded lengths")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    batch_size: int = args.batch_size

    # Resolve split files (support both 'valid.csv' and 'val.csv')
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    if not valid_csv.exists():
        valid_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    # Initialize ProGen2 tokenizer (same as training)
    tokenizer = get_tokenizer(ModelSchema(name="progen2"))

    for name, p in ("train", train_csv), ("valid", valid_csv), ("test", test_csv):
        analyze_split(name, p, batch_size, tokenizer)


if __name__ == "__main__":
    main()



