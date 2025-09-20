#!/usr/bin/env python3
"""Collect AF3 confidence metrics for protein:pre-crRNA complexes.

For each job listed in the JSON manifest (default: alphafold3_work/input/seed_variants_jobs.json)
this script reads the AF3 summary output, computes WED domain pLDDT means and
K/K/H triad statistics, and writes three CSVs per job under metrics/:
  - <job_name>_summary.csv
  - <job_name>_wed_plddt.csv
  - <job_name>_triad_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gemmi

WED_DOMAINS: List[Tuple[str, int, int]] = [
    ("WED-I", 1, 25),
    ("WED-II", 514, 582),
    ("WED-III", 679, 808),
]
TRIAD_RESIDUES: List[Tuple[str, int]] = [
    ("H759", 759),
    ("K768", 768),
    ("K785", 785),
]
PHOSPHATE_NAMES = {"P", "OP1", "OP2"}


def load_jobs(manifest: Path) -> List[Dict[str, str]]:
    data = json.loads(manifest.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {manifest}, got {type(data)}")
    required = {"sequence_id", "json_path", "output_dir", "name"}
    for item in data:
        if not required.issubset(item):
            missing = required.difference(item)
            raise ValueError(f"Job entry missing fields {missing}: {item}")
    return data


def mean_heavy_plddt(residue: gemmi.Residue) -> float:
    values: List[float] = []
    for atom in residue:
        if atom.element == gemmi.Element("H"):
            continue
        values.append(atom.b_iso)
    return float(sum(values) / len(values)) if values else float("nan")


def mean_chain_plddt(chain: gemmi.Chain) -> float:
    values: List[float] = []
    for residue in chain:
        for atom in residue:
            if atom.element == gemmi.Element("H"):
                continue
            values.append(atom.b_iso)
    return float(sum(values) / len(values)) if values else float("nan")


def residue_by_seqnum(chain: gemmi.Chain, seqnum: int) -> gemmi.Residue | None:
    for residue in chain:
        if residue.seqid.num == seqnum:
            return residue
    return None


def domain_mean(chain: gemmi.Chain, start: int, end: int) -> float:
    values: List[float] = []
    for residue in chain:
        seqnum = residue.seqid.num
        if start <= seqnum <= end:
            for atom in residue:
                if atom.element == gemmi.Element("H"):
                    continue
                values.append(atom.b_iso)
    return float(sum(values) / len(values)) if values else float("nan")


def collect_phosphates(chain: gemmi.Chain) -> List[gemmi.Atom]:
    atoms: List[gemmi.Atom] = []
    for residue in chain:
        for atom in residue:
            if atom.name.strip() in PHOSPHATE_NAMES:
                atoms.append(atom)
    return atoms


def triad_metrics(protein: gemmi.Chain, rna: gemmi.Chain) -> List[Dict[str, float | str]]:
    phosphates = collect_phosphates(rna)
    rows: List[Dict[str, float | str]] = []
    triad_plddt: List[float] = []
    min_distances: List[float] = []
    for label, seqnum in TRIAD_RESIDUES:
        residue = residue_by_seqnum(protein, seqnum)
        if residue is None:
            mean_plddt = float("nan")
            min_dist = float("nan")
        else:
            mean_plddt = mean_heavy_plddt(residue)
            min_dist = math.inf
            for atom in residue:
                if atom.element == gemmi.Element("H"):
                    continue
                for pa in phosphates:
                    dist = atom.pos.dist(pa.pos)
                    if dist < min_dist:
                        min_dist = dist
            if math.isinf(min_dist):
                min_dist = float("nan")
        rows.append(
            {
                "residue": label,
                "mean_plddt": f"{mean_plddt:.2f}" if not math.isnan(mean_plddt) else "nan",
                "min_distance_to_phosphate_A": f"{min_dist:.2f}" if not math.isnan(min_dist) else "nan",
            }
        )
        if not math.isnan(mean_plddt):
            triad_plddt.append(mean_plddt)
        if not math.isnan(min_dist):
            min_distances.append(min_dist)
    rows.append(
        {
            "residue": "Triad mean",
            "mean_plddt": f"{(sum(triad_plddt) / len(triad_plddt)):.2f}" if triad_plddt else "nan",
            "min_distance_to_phosphate_A": f"{min(min_distances):.2f}" if min_distances else "nan",
        }
    )
    return rows


def wed_metrics(protein: gemmi.Chain) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    overall_values: List[float] = []
    for name, start, end in WED_DOMAINS:
        mean = domain_mean(protein, start, end)
        rows.append(
            {
                "domain": name,
                "start": str(start),
                "end": str(end),
                "mean_plddt": f"{mean:.2f}" if not math.isnan(mean) else "nan",
            }
        )
        for residue in protein:
            seqnum = residue.seqid.num
            if start <= seqnum <= end:
                for atom in residue:
                    if atom.element != gemmi.Element("H"):
                        overall_values.append(atom.b_iso)
    overall = float(sum(overall_values) / len(overall_values)) if overall_values else float("nan")
    rows.append({"domain": "WED overall", "start": "", "end": "", "mean_plddt": f"{overall:.2f}" if not math.isnan(overall) else "nan"})
    return rows


def ensure_metrics_dir() -> Path:
    path = Path("metrics")
    path.mkdir(exist_ok=True)
    return path


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_job(job: Dict[str, str], metrics_dir: Path) -> None:
    if "output_dir" in job:
        out_dir = Path(job["output_dir"]).resolve()
    else:
        if "output_root" not in job or "job_name" not in job:
            raise ValueError(
                "Job entry must include either 'output_dir' or both 'output_root' and 'job_name'"
            )
        out_dir = (Path(job["output_root"]) / job["job_name"]).resolve()
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json for {job['sequence_id']} at {summary_path}")
    summary = json.loads(summary_path.read_text())

    model_cif = Path(summary.get("model_cif", ""))
    if not model_cif.is_absolute():
        candidate = out_dir / model_cif
        if candidate.exists():
            model_cif = candidate
        else:
            nested = out_dir / job["name"] / model_cif
            if nested.exists():
                model_cif = nested
            else:
                raise FileNotFoundError(f"Missing model CIF for {job['sequence_id']} (looked in {candidate} and {nested})")

    structure = gemmi.read_structure(str(model_cif))
    model = structure[0]
    protein_chain = model.find_chain("A")
    rna_chain = model.find_chain("R")
    if protein_chain is None or rna_chain is None:
        raise ValueError(f"Expected chains A and R in {model_cif}")

    chain_a_mean = mean_chain_plddt(protein_chain)
    chain_r_mean = mean_chain_plddt(rna_chain)

    summary_rows = [
        {"metric": "ptm", "value": f"{summary.get('ptm', float('nan')):.4f}"},
        {"metric": "iptm", "value": f"{summary.get('iptm', float('nan')):.4f}"},
        {"metric": "ranking_score", "value": f"{summary.get('ranking_score', float('nan')):.4f}"},
        {"metric": "global_plddt", "value": f"{summary.get('global_plddt', float('nan')):.4f}"},
        {"metric": "chain_A_avg_plddt", "value": f"{chain_a_mean:.4f}" if not math.isnan(chain_a_mean) else "nan"},
        {"metric": "chain_R_avg_plddt", "value": f"{chain_r_mean:.4f}" if not math.isnan(chain_r_mean) else "nan"},
    ]

    job_name = job["name"]
    summary_csv = metrics_dir / f"{job_name}_summary.csv"
    wed_csv = metrics_dir / f"{job_name}_wed_plddt.csv"
    triad_csv = metrics_dir / f"{job_name}_triad_metrics.csv"

    write_csv(summary_csv, summary_rows, ["metric", "value"])
    write_csv(wed_csv, wed_metrics(protein_chain), ["domain", "start", "end", "mean_plddt"])
    write_csv(triad_csv, triad_metrics(protein_chain, rna_chain), ["residue", "mean_plddt", "min_distance_to_phosphate_A"])

    print(f"[metrics] Wrote {summary_csv}")
    print(f"[metrics] Wrote {wed_csv}")
    print(f"[metrics] Wrote {triad_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-json",
        type=Path,
        default=Path("alphafold3_work/input/seed_variants_jobs.json"),
        help="Manifest generated alongside AF3 JSON inputs",
    )
    args = parser.parse_args()

    jobs = load_jobs(args.jobs_json)
    metrics_dir = ensure_metrics_dir()
    for job in jobs:
        try:
            summarize_job(job, metrics_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {job['sequence_id']}: {exc}")
        except RuntimeError as exc:
            print(f"[skip] {job['sequence_id']}: {exc}")


if __name__ == "__main__":
    main()
