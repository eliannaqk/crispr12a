#!/usr/bin/env python3
"""Run the final AF3 protein+pre-crRNA pipeline for a batch of sequences.

This wrapper reads a manifest JSON (list of jobs) and dispatches
``assess_precrrna_final.py`` for each entry, ensuring we reuse the exact
pipeline used for the seq1 baselines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "alphafold3_work"
PIPELINE = SCRIPT_DIR / "assess_precrrna_final.py"


def load_jobs(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list, got {type(data)}")
    required = {"sequence_id", "protein_fasta"}
    jobs: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entries must be dicts, got {type(item)}")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Entry missing fields {missing}: {item}")
        jobs.append(item)
    return jobs


def build_command(args: argparse.Namespace, job: Dict[str, str]) -> List[str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_id = job["sequence_id"]
    job_name = job.get("job_name") or f"{sequence_id}_precrrna_{timestamp}"
    cmd = [
        sys.executable,
        str(PIPELINE),
        f"--rna-seq={args.rna_seq}",
        f"--rna-chain-id={args.rna_chain_id}",
        f"--protein-chain-id={args.protein_chain_id}",
        f"--seeds={args.seeds}",
        f"--output-root={args.output_root}",
        f"--input-dir={args.input_dir}",
        f"--model-dir={args.model_dir}",
        f"--db-dir={args.db_dir}",
        f"--jax-cache={args.jax_cache}",
        f"--mem-fraction={args.mem_fraction}",
        f"--job-name={job_name}",
        f"--protein-fasta={job['protein_fasta']}",
    ]
    for dna_entry in args.dna_seq:
        cmd.append(f"--dna-seq={dna_entry}")
    if job.get("protein_id"):
        cmd.append(f"--protein-id={job['protein_id']}")
    if job.get("protein_label"):
        cmd.append(f"--protein-label={job['protein_label']}")
    if args.no_data_pipeline:
        cmd.append("--no-data-pipeline")
    if args.no_inference:
        cmd.append("--no-inference")
    return cmd


def run_jobs(args: argparse.Namespace) -> None:
    jobs = load_jobs(Path(args.jobs_json))
    if not PIPELINE.exists():
        raise FileNotFoundError(f"Pipeline script missing: {PIPELINE}")
    for job in jobs:
        cmd = build_command(args, job)
        print(f"[batch] Running {job['sequence_id']} via assess_precrrna_final.py")
        print(f"[batch] Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(
                f"Pipeline failed for {job['sequence_id']} (exit code {result.returncode})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-json",
        type=str,
        default="alphafold3_work/input/precrrna_final_jobs.json",
        help="Manifest describing the sequences to fold",
    )
    parser.add_argument("--rna-seq", required=True, help="Pre-crRNA sequence (5'->3')")
    parser.add_argument(
        "--rna-chain-id",
        default="R",
        help="Chain identifier assigned to the RNA component (default: R)",
    )
    parser.add_argument(
        "--protein-chain-id",
        default="A",
        help="Chain identifier assigned to proteins (default: A)",
    )
    parser.add_argument(
        "--dna-seq",
        action="append",
        default=[],
        help=(
            "DNA sequence(s) to forward to assess_precrrna_final.py. Specify multiple times; "
            "use CHAIN:SEQUENCE syntax to control chain ids."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3,4",
        help="Comma-separated AF3 seeds passed through to the pipeline",
    )
    parser.add_argument(
        "--output-root",
        default=str((Path("alphafold3_work") / "output_h200" / "final_precrrna")),
        help="Directory where per-job outputs will be written",
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path("alphafold3_work") / "input"),
        help="Directory for intermediate AF3 JSON files",
    )
    parser.add_argument(
        "--model-dir",
        default=str(Path("/home/eqk3/project_pi_mg269/eqk3/alphafold3")),
        help="Location of AF3 weights/config",
    )
    parser.add_argument(
        "--db-dir",
        default=str(Path("/home/eqk3/project_pi_mg269/eqk3/public_databases")),
        help="Location of AF3 databases",
    )
    parser.add_argument(
        "--jax-cache",
        default=str(Path("alphafold3_work") / "jax_cache"),
        help="JAX compilation cache directory",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.95,
        help="XLA_PYTHON_CLIENT_MEM_FRACTION passed through to the pipeline",
    )
    parser.add_argument(
        "--no-data-pipeline",
        action="store_true",
        help="Disable AF3 data pipeline stage (passed through)",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Disable AF3 inference stage (passed through)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_jobs(args)


if __name__ == "__main__":
    main()
