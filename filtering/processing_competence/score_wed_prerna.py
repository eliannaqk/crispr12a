#!/usr/bin/env python3
"""Score Boltz protein:RNA poses for pre-crRNA processing competence.

This script mirrors the dataset used for ESMFold inference and aggregates the
Boltz outputs together with domain annotations (WED residues and K/K/H triads)
into screening metrics that highlight processing-competent conformations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import (
    MMCIFParser,
    NeighborSearch,
    PDBParser,
)
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue

# RNA phosphate atoms that anchor the triad distances.
PA_ATOMS = {"P", "OP1", "OP2"}


def is_heavy(atom: Atom) -> bool:
    """Return True for non-hydrogen atoms."""
    element = atom.element.strip() if atom.element else ""
    if element:
        return element.upper() != "H"
    return not atom.get_name().startswith("H")


def iter_atoms(chain: Chain, selector: Optional[callable] = None) -> List[Atom]:
    atoms: List[Atom] = []
    for residue in chain:
        for atom in residue:
            if selector and not selector(atom):
                continue
            atoms.append(atom)
    return atoms


def load_confidence(json_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return ipTM/interface confidence and pTM values if available."""
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    for json_path in json_dir.glob("*.json"):
        try:
            with json_path.open("r") as handle:
                data = json.load(handle)
        except Exception:
            continue
        candidates = [
            data,
            *(data.get(key, {}) for key in ("metrics", "confidence", "predicted_aligned_error")),
        ]
        for payload in candidates:
            if not isinstance(payload, dict):
                continue
            for key in ("iptm", "ipTM", "interface_pred_tm", "interface_confidence", "interface_ptm"):
                if key in payload:
                    value = payload[key]
                    if isinstance(value, (int, float)):
                        iptm = float(value)
            for key in ("ptm", "pTM", "pred_tm", "tm"):
                if key in payload:
                    value = payload[key]
                    if isinstance(value, (int, float)):
                        ptm = float(value)
    return iptm, ptm


def parse_structure(pdb_path: Path) -> Tuple[Model, List[Chain]]:
    """Load a PDB/mmCIF structure and return the first model with its chains."""
    parser = MMCIFParser(QUIET=True) if pdb_path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("pred", str(pdb_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    return model, chains


def residue_by_number(chain: Chain, resnum: int) -> Optional[Residue]:
    for residue in chain:
        het, seq_id, _ = residue.id
        if seq_id == resnum and ("CA" in residue or "P" in residue):
            return residue
    return None


def mean_plddt(chain: Chain) -> float:
    values: List[float] = []
    for residue in chain:
        for atom in residue:
            if not is_heavy(atom):
                continue
            bfactor = atom.get_bfactor()
            if bfactor > 0:
                values.append(bfactor)
    return float(np.mean(values)) if values else float("nan")


def phosphate_atoms(chain: Chain) -> List[Atom]:
    return [atom for atom in iter_atoms(chain) if atom.get_name() in PA_ATOMS]


def wed_atoms(chain: Chain, residues: Sequence[int]) -> List[Atom]:
    atoms: List[Atom] = []
    for resnum in residues:
        residue = residue_by_number(chain, resnum)
        if not residue:
            continue
        for atom in residue:
            if is_heavy(atom):
                atoms.append(atom)
    return atoms


def wed_rna_contacts(wed_atoms: Sequence[Atom], rna_chain: Chain, cutoff: float = 4.0) -> int:
    if not wed_atoms:
        return 0
    rna_atoms = [atom for atom in iter_atoms(rna_chain) if is_heavy(atom)]
    if not rna_atoms:
        return 0
    search = NeighborSearch(wed_atoms + rna_atoms)
    contact_count = 0
    for atom in wed_atoms:
        matches = search.search(atom.get_coord(), cutoff, level="A")
        for partner in matches:
            if partner.get_parent().get_parent().id != atom.get_parent().get_parent().id:
                contact_count += 1
    return contact_count


def guess_triads(protein_chain: Chain, rna_chain: Chain, window: int = 15) -> Optional[Tuple[int, int, int]]:
    lysines = [residue for residue in protein_chain if residue.get_resname() == "LYS"]
    histidines = [residue for residue in protein_chain if residue.get_resname() == "HIS"]
    phosphates = phosphate_atoms(rna_chain)
    if not lysines or not histidines or not phosphates:
        return None

    def heavy_centroid(residue: Residue) -> Optional[np.ndarray]:
        coords = [atom.get_coord() for atom in residue if is_heavy(atom)]
        if not coords:
            return None
        return np.mean(np.asarray(coords), axis=0)

    lys_centroids = [(residue, heavy_centroid(residue)) for residue in lysines]
    his_centroids = [(residue, heavy_centroid(residue)) for residue in histidines]

    best_tri: Optional[Tuple[int, int, int]] = None
    best_distance = float("inf")
    for idx, (res_a, centroid_a) in enumerate(lys_centroids):
        for res_b, centroid_b in lys_centroids[idx + 1 :]:
            if not centroid_a or not centroid_b:
                continue
            if abs(res_a.id[1] - res_b.id[1]) > window:
                continue
            for res_h, centroid_h in his_centroids:
                if not centroid_h:
                    continue
                triad_centroid = np.mean(np.asarray([centroid_a, centroid_b, centroid_h]), axis=0)
                distances = [np.linalg.norm(triad_centroid - atom.get_coord()) for atom in phosphates]
                if not distances:
                    continue
                dmin = float(min(distances))
                if dmin < best_distance:
                    best_distance = dmin
                    best_tri = (res_a.id[1], res_b.id[1], res_h.id[1])
    return best_tri


def triad_distances(protein_chain: Chain, rna_chain: Chain, residues: Sequence[int]) -> List[float]:
    phosphates = phosphate_atoms(rna_chain)
    if not phosphates:
        return []
    distances: List[float] = []
    for resnum in residues:
        residue = residue_by_number(protein_chain, resnum)
        if not residue:
            continue
        for atom in residue:
            if not is_heavy(atom):
                continue
            dmin = min((atom - phosphate) for phosphate in phosphates)
            distances.append(float(dmin))
    return distances


def discover_predictions(run_dir: Path, prefix: str) -> List[Path]:
    patterns = ["*.cif", "*.mmcif", "*.pdb"]
    hits: List[Path] = []
    for pattern in patterns:
        hits.extend(run_dir.glob(f"{prefix}*/**/{pattern}"))
    return sorted(set(hits))


def select_best_prediction(predictions: Sequence[Path]) -> Tuple[Optional[Path], Optional[float]]:
    best_path: Optional[Path] = None
    best_iptm: Optional[float] = None
    for candidate in predictions:
        iptm, _ = load_confidence(candidate.parent)
        if iptm is None:
            if best_path is None:
                best_path = candidate
            continue
        if best_iptm is None or iptm > best_iptm:
            best_path = candidate
            best_iptm = iptm
    return best_path, best_iptm


def parse_wed_residues(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(item) for item in re.split(r"[;,:\s]+", raw.strip()) if item]


def load_input_rows(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["id"].strip(): row for row in reader if row.get("id")}


def score_prediction(
    pid: str,
    csv_row: Dict[str, str],
    run_root: Path,
    wed_residues: Sequence[int],
) -> Optional[Dict[str, object]]:
    predictions = discover_predictions(run_root, pid)
    if not predictions:
        return None

    best_path, best_iptm = select_best_prediction(predictions)
    if not best_path:
        best_path = predictions[0]

    _, chains = parse_structure(best_path)
    if len(chains) < 2:
        return None

    protein_chain, rna_chain = chains[0], chains[1]
    iptm, ptm = load_confidence(best_path.parent)

    wed_atoms_subset = wed_atoms(protein_chain, wed_residues) if wed_residues else []
    wed_plddt = float(np.mean([atom.get_bfactor() for atom in wed_atoms_subset if atom.get_bfactor() > 0])) if wed_atoms_subset else float("nan")
    wed_contacts = wed_rna_contacts(wed_atoms_subset, rna_chain) if wed_atoms_subset else math.nan

    triad_spec = csv_row.get("kkh_residues", "").strip()
    if triad_spec:
        triad = [int(tok) for tok in re.split(r"[;,:\s]+", triad_spec) if tok]
    else:
        guess = guess_triads(protein_chain, rna_chain)
        triad = list(guess) if guess else []

    distances = triad_distances(protein_chain, rna_chain, triad)

    result: Dict[str, object] = {
        "id": pid,
        "best_model_path": str(best_path),
        "ipTM": best_iptm if best_iptm is not None else iptm,
        "pTM": ptm,
        "protein_pLDDT_mean": mean_plddt(protein_chain),
        "rna_pLDDT_mean": mean_plddt(rna_chain),
        "wed_pLDDT_mean": wed_plddt,
        "wed_rna_contact_count": wed_contacts,
        "triad_annotation": ",".join(str(x) for x in triad) if triad else "",
        "triad_min_distance_to_RNA_P": float(min(distances)) if distances else float("nan"),
        "triad_mean_distance_to_RNA_P": float(np.mean(distances)) if distances else float("nan"),
    }
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Boltz LbCas12a:pre-crRNA complexes.")
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV used for Boltz inference (id, protein_fasta, rna_seq, optional kkh_residues).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("boltz_runs"),
        help="Root directory containing Boltz output subdirectories.",
    )
    parser.add_argument(
        "--wed_residues",
        default="",
        help="Comma/space separated WED residue numbers for per-domain scoring.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("boltz_screen_scores.csv"),
        help="Output CSV path for aggregated scores.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    wed_residue_ids = parse_wed_residues(args.wed_residues)
    csv_rows = load_input_rows(args.csv)

    scores: List[Dict[str, object]] = []
    for pid, row in csv_rows.items():
        record = score_prediction(pid, row, args.runs, wed_residue_ids)
        if record is not None:
            scores.append(record)

    if not scores:
        print("No predictions scored; check input directories.")
        return

    df = pd.DataFrame(scores)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(df)} rows.")


if __name__ == "__main__":
    main()
