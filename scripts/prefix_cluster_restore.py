#!/usr/bin/env python3
"""Cluster sequence prefixes with MMseqs2 and restore full-length representatives.

This helper trims each input sequence to its variable prefix (length = len(seq) - suffix_length),
clusters the prefixes using mmseqs easy-cluster, and then re-attaches the conserved suffix so the
cluster representative FASTA contains full-length proteins again.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO

_LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster sequence prefixes and restore full length reps")
    parser.add_argument("full_fasta", type=Path, help="FASTA file with full-length sequences")
    parser.add_argument("output_dir", type=Path, help="Directory to place prefix intermediate and cluster outputs")
    parser.add_argument("suffix_length", type=int, help="Number of trailing residues treated as conserved suffix")
    parser.add_argument("--min-seq-id", dest="min_seq_id", type=float, default=0.9, help="Minimum sequence identity for clustering (default: 0.9)")
    parser.add_argument("--coverage", dest="coverage", type=float, default=0.8, help="Alignment coverage threshold passed to mmseqs -c (default: 0.8)")
    parser.add_argument("--cov-mode", dest="cov_mode", type=int, default=0, help="MMseqs coverage mode (default: 0)")
    parser.add_argument("--output-prefix", dest="output_prefix", default="prefix_clusters", help="Prefix for MMseqs output files (default: prefix_clusters)")
    return parser.parse_args()


def _load_sequences(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with fasta_path.open() as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences


def _write_prefix_fasta(sequences: dict[str, str], suffix_length: int, output_path: Path) -> list[str]:
    skipped: list[str] = []
    with output_path.open("w") as handle:
        for seq_id, sequence in sequences.items():
            prefix_end = max(0, len(sequence) - suffix_length)
            if prefix_end <= 0:
                skipped.append(seq_id)
                continue
            prefix_seq = sequence[:prefix_end]
            if not prefix_seq:
                skipped.append(seq_id)
                continue
            handle.write(f">{seq_id}\n{prefix_seq}\n")
    return skipped


def _run_mmseqs(prefix_fasta: Path, output_prefix: Path, min_seq_id: float, coverage: float, cov_mode: int) -> None:
    with tempfile.TemporaryDirectory(prefix="mmseqs_tmp_") as tmp_dir:
        cmd = [
            "mmseqs",
            "easy-cluster",
            str(prefix_fasta),
            str(output_prefix),
            tmp_dir,
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
        ]
        _LOGGER.info("Running mmseqs: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


def _restore_representatives(rep_fasta: Path, sequences: dict[str, str], output_path: Path) -> None:
    with rep_fasta.open() as rep_handle, output_path.open("w") as out_handle:
        for record in SeqIO.parse(rep_handle, "fasta"):
            seq_id = record.id
            if seq_id not in sequences:
                raise KeyError(f"Representative {seq_id} missing from original sequences")
            out_handle.write(f">{seq_id}\n{sequences[seq_id]}\n")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _load_sequences(args.full_fasta)
    _LOGGER.info("Loaded %d sequences", len(sequences))

    prefix_fasta = output_dir / f"{args.output_prefix}_prefix.faa"
    skipped = _write_prefix_fasta(sequences, args.suffix_length, prefix_fasta)
    _LOGGER.info("Wrote prefixes to %s", prefix_fasta)
    if skipped:
        _LOGGER.warning("Skipping %d sequences whose prefix length is <= 0", len(skipped))

    cluster_prefix = output_dir / args.output_prefix
    _run_mmseqs(prefix_fasta, cluster_prefix, args.min_seq_id, args.coverage, args.cov_mode)

    rep_prefix_fasta = output_dir / f"{args.output_prefix}_rep_seq.fasta"
    rep_full_fasta = output_dir / f"{args.output_prefix}_rep_full.faa"
    _LOGGER.info("Restoring full-length representatives to %s", rep_full_fasta)
    _restore_representatives(rep_prefix_fasta, sequences, rep_full_fasta)

    _LOGGER.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
