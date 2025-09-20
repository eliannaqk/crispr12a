#!/usr/bin/env python3
"""
Prepare Boltz-1 inputs and run predictions for Cas12a : pre-crRNA complexes.

Usage:
  python filtering/boltz_scoring/run_boltz_prerna.py \
      --csv inputs.csv \
      --outdir filtering/boltz_scoring/boltz_runs \
      --use_msa_server

CSV columns (header required):
  - id: unique identifier per protein
  - protein_fasta or protein_seq: FASTA filepath or raw amino acid sequence
  - rna_seq: RNA sequence (A,C,G,U) for pre-crRNA/DR fragment
  - (optional) kkh_residues: comma-separated 1-based residue numbers (e.g., 492,517,539)

Notes:
  - This script writes one YAML per row under --outdir and then calls:
      boltz predict <outdir> [--use_msa_server]
  - If you want to softly bias docking into WED, pass --wed_residues "r1,r2,...".
  - Ensure you are in the approved conda environment: oc-opencrispr-esm.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import textwrap
from typing import List


YAML_TMPL = """\
version: 1
sequences:
  - protein:
      id: "PROT"
      sequence: "{protein_seq}"
  - rna:
      id: "RNA"
      sequence: "{rna_seq}"
options:
  num_samples: {num_samples}
  num_recycles: {num_recycles}
  use_msa_server: {use_msa}
  # Optional soft pocket constraint to encourage docking near WED:
  # pocket_constraints:
  #   - chain_id: PROT
  #     residues: [{wed_residues}]
"""


def read_fasta(path: str) -> str:
    seq: List[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.append(line.strip())
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Boltz YAMLs and run predictions for Cas12a:pre-crRNA.")
    ap.add_argument("--csv", required=True, help="inputs.csv with columns: id, protein_fasta|protein_seq, rna_seq, (optional) kkh_residues")
    ap.add_argument("--outdir", default="filtering/boltz_scoring/boltz_runs", help="Output directory for YAMLs and predictions")
    ap.add_argument("--num_samples", type=int, default=3, help="Boltz samples per target")
    ap.add_argument("--num_recycles", type=int, default=3, help="Model recycles")
    ap.add_argument("--use_msa_server", action="store_true", help="Enable MSA server (recommended for proteins)")
    ap.add_argument("--wed_residues", default="", help="Comma-separated WED residue numbers (1-based) to bias docking (optional)")
    ap.add_argument("--boltz_bin", default="boltz", help="Path to 'boltz' CLI if not on PATH")
    ap.add_argument(
        "--accelerator",
        default="gpu",
        choices=["cpu", "gpu", "tpu"],
        help="Accelerator flag forwarded to 'boltz predict' (default: gpu)",
    )
    ap.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices forwarded to 'boltz predict' (default: 1)",
    )
    ap.add_argument(
        "--model",
        default="boltz2",
        choices=["boltz1", "boltz2"],
        help="Boltz model variant (default: boltz2)",
    )
    ap.add_argument(
        "--override",
        action="store_true",
        help="Force Boltz to recompute predictions even if cached outputs are present",
    )
    ap.add_argument(
        "--max_parallel_samples",
        type=int,
        default=2,
        help="Limit concurrent Boltz samples to control memory usage (default: 2)",
    )
    args = ap.parse_args()

    out_root = pathlib.Path(args.outdir)
    yaml_dir = out_root / "yamls"
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(yaml_dir, exist_ok=True)
    yamls = []

    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row["id"].strip()
            protein_seq = row.get("protein_seq", "").strip()
            protein_fasta = row.get("protein_fasta", "").strip()
            if not protein_seq and protein_fasta:
                protein_seq = read_fasta(protein_fasta)
            if not protein_seq:
                raise ValueError(f"{pid}: need protein_seq or protein_fasta")

            rna_seq = row["rna_seq"].replace("T", "U").strip().upper()
            if not set(rna_seq) <= set("ACGU"):
                raise ValueError(f"{pid}: rna_seq must be RNA (A,C,G,U)")

            ypath = yaml_dir / f"{pid}.yaml"
            yaml_str = YAML_TMPL.format(
                protein_seq=protein_seq,
                rna_seq=rna_seq,
                num_samples=args.num_samples,
                num_recycles=args.num_recycles,
                use_msa=str(bool(args.use_msa_server)).lower(),
                wed_residues=args.wed_residues,
            )
            ypath.write_text(yaml_str)
            yamls.append(str(ypath))

    # Log active environment for reproducibility
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    print(f"[run_boltz_prerna] Active conda env: {conda_env or 'UNKNOWN'} (expected: oc-opencrispr-esm)")
    print(f"[run_boltz_prerna] Prepared {len(yamls)} YAMLs in {yaml_dir}")

    # Batch run `boltz predict` on the directory of YAMLs
    cmd = [
        args.boltz_bin,
        "predict",
        str(yaml_dir),
        "--out_dir",
        str(out_root),
        "--accelerator",
        args.accelerator,
        "--devices",
        str(args.devices),
        "--model",
        args.model,
    ]
    if args.use_msa_server:
        cmd.append("--use_msa_server")
    if args.override:
        cmd.append("--override")
    if args.max_parallel_samples:
        cmd.extend(["--max_parallel_samples", str(args.max_parallel_samples)])
    print("[run_boltz_prerna] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
