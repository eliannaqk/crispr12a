#!/usr/bin/env python3
"""
Score Boltz-1 predictions for LbCas12a : pre-crRNA complexes.

For each target, select the best-confidence model (by ipTM if available) and
compute:
  - ipTM/interface confidence
  - mean pLDDT (protein, RNA)
  - WED mean pLDDT and WED↔RNA contact count (if WED residues provided)
  - Distances from K/K/H catalytic triad to RNA phosphate atoms (P/OP1/OP2)

Writes a CSV (default: boltz_screen_scores.csv) summarizing the metrics.

Usage:
  python filtering/boltz_scoring/score_wed_prerna.py \
      --csv inputs.csv \
      --runs filtering/boltz_scoring/boltz_runs \
      --wed_residues "500,501,502" \
      --out_csv filtering/boltz_scoring/boltz_screen_scores.csv

Ensure the oc-opencrispr-esm conda environment is active.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser, NeighborSearch


PA_ATOMS = {"P", "OP1", "OP2"}  # RNA phosphate atoms


def heavy(atom) -> bool:
    return getattr(atom, "element", "") != "H"


def load_confidence(json_dir: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (iptm, ptm) from any JSON found under json_dir; else (None, None)."""
    iptm = ptm = None
    for j in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            data = json.load(open(j))
        except Exception:
            continue
        for k in ("iptm", "ipTM", "interface_pred_tm", "interface_confidence", "interface_ptm"):
            if k in data:
                iptm = data[k]
        for k in ("ptm", "pTM", "pred_tm", "tm"):
            if k in data:
                ptm = data[k]
    return iptm, ptm


def parse_structure(path: str):
    parser = MMCIFParser(QUIET=True) if path.endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    structure = parser.get_structure("pred", path)
    model = next(structure.get_models())
    chains = list(model.get_chains())
    return model, chains


def residue_by_number(chain, resnum: int):
    for r in chain:
        if ("CA" in r) or ("P" in r):  # protein or nucleic
            if r.id[1] == resnum:
                return r
    return None


def mean_plddt(chain) -> float:
    vals: List[float] = []
    for r in chain:
        for a in r:
            if not heavy(a):
                continue
            b = a.get_bfactor()  # Boltz uses B-factor slot for pLDDT-like
            if b > 0:
                vals.append(b)
    return float(np.mean(vals)) if vals else float("nan")


def atomlist(chain) -> List:
    atoms = []
    for r in chain:
        for a in r:
            atoms.append(a)
    return atoms


def phosphate_atoms(chain) -> List:
    return [a for a in atomlist(chain) if a.get_name() in PA_ATOMS]


def contact_count(chainA, chainB, cutoff: float = 4.0) -> int:
    atomsA = [a for a in atomlist(chainA) if heavy(a)]
    atomsB = [a for a in atomlist(chainB) if heavy(a)]
    nb = NeighborSearch(atomsA + atomsB)
    n = 0
    for a in atomsA:
        close = nb.search(a.get_coord(), cutoff, level="A")
        for b in close:
            if b.get_parent().get_parent().id != a.get_parent().get_parent().id:
                n += 1
    return n


def guess_triads(protein_chain, rna_chain, window: int = 15):
    """Fallback: find K-K-H triads whose centroid is nearest to RNA phosphates."""
    ks = [r for r in protein_chain if r.get_resname() == "LYS"]
    hs = [r for r in protein_chain if r.get_resname() == "HIS"]
    phosph = phosphate_atoms(rna_chain)
    if not ks or not hs or not phosph:
        return None

    def centroid(res):
        pts = np.array([a.get_coord() for a in res if heavy(a)])
        return np.mean(pts, axis=0) if len(pts) else None

    kcent = [(r, centroid(r)) for r in ks]
    hcent = [(r, centroid(r)) for r in hs]

    best = None
    best_d = 1e9
    for i, (r1, c1) in enumerate(kcent):
        for r2, c2 in kcent[i + 1 :]:
            if abs(r1.id[1] - r2.id[1]) > window:
                continue
            for rh, ch in hcent:
                if c1 is None or c2 is None or ch is None:
                    continue
                tri_centroid = np.mean([c1, c2, ch], axis=0)
                dmin = min(np.linalg.norm(tri_centroid - a.get_coord()) for a in phosph)
                if dmin < best_d:
                    best_d = dmin
                    best = (r1.id[1], r2.id[1], rh.id[1])
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Boltz LbCas12a:pre-crRNA complexes.")
    ap.add_argument("--csv", required=True, help="Same CSV used for run script (id, protein_[...], rna_seq, optional kkh_residues)")
    ap.add_argument("--runs", default="filtering/boltz_scoring/boltz_runs", help="Boltz run directory")
    ap.add_argument("--wed_residues", default="", help="Comma-separated WED residues to compute WED pLDDT/contact metrics (optional)")
    ap.add_argument("--out_csv", default="filtering/boltz_scoring/boltz_screen_scores.csv")
    args = ap.parse_args()

    print(f"[score_wed_prerna] Active conda env: {os.environ.get('CONDA_DEFAULT_ENV','UNKNOWN')} (expected: oc-opencrispr-esm)")

    wed_list: List[int] = []
    if args.wed_residues:
        wed_list = [int(x) for x in re.split(r"[\s,]+", args.wed_residues.strip()) if x]

    rows: List[Dict[str, object]] = []
    csv_rows = {r["id"].strip(): r for r in csv.DictReader(open(args.csv))}

    for pid in csv_rows:
        # Collect predicted structures (cif preferred)
        pred_files = sorted(
            glob.glob(os.path.join(args.runs, f"{pid}*", "**", "*.cif"), recursive=True)
        )
        if not pred_files:
            pred_files = sorted(
                glob.glob(os.path.join(args.runs, f"{pid}*", "**", "*.pdb"), recursive=True)
            )
        if not pred_files:
            pred_files = sorted(
                glob.glob(
                    os.path.join(
                        args.runs,
                        "boltz_results_*",
                        "predictions",
                        f"{pid}",
                        "*.cif",
                    )
                )
            )
        if not pred_files:
            pred_files = sorted(
                glob.glob(
                    os.path.join(
                        args.runs,
                        "boltz_results_*",
                        "predictions",
                        f"{pid}",
                        "*.pdb",
                    )
                )
            )
        if not pred_files:
            continue

        # Pick best by ipTM if confidence JSON is available
        best_path = pred_files[0]
        best_iptm = None
        for path in pred_files:
            iptm, _ = load_confidence(os.path.dirname(path))
            if iptm is not None and (best_iptm is None or iptm > best_iptm):
                best_iptm = iptm
                best_path = path

        model, chains = parse_structure(best_path)
        if len(chains) < 2:
            continue
        prot, rna = chains[0], chains[1]  # Script A order

        iptm, ptm = load_confidence(os.path.dirname(best_path))
        prot_plddt = mean_plddt(prot)
        rna_plddt = mean_plddt(rna)

        # WED metrics
        wed_plddt = float("nan")
        wed_contacts = float("nan")
        if wed_list:
            wed_atoms = []
            for resnum in wed_list:
                res = residue_by_number(prot, resnum)
                if res is None:
                    continue
                for a in res:
                    if heavy(a):
                        wed_atoms.append(a)
            if wed_atoms:
                vals = [a.get_bfactor() for a in wed_atoms if a.get_bfactor() > 0]
                wed_plddt = float(np.mean(vals)) if vals else float("nan")
                rna_atoms = [a for a in atomlist(rna) if heavy(a)]
                cc = 0
                for a in wed_atoms:
                    for b in rna_atoms:
                        if (a - b) <= 4.0:
                            cc += 1
                wed_contacts = cc

        # Triad distances
        kkh_field = (csv_rows[pid].get("kkh_residues", "") or "").strip()
        if kkh_field:
            triad = [int(x) for x in kkh_field.split(",") if x]
        else:
            triad = guess_triads(prot, rna) or []

        triad_dists: List[float] = []
        if triad:
            tri_res = [residue_by_number(prot, n) for n in triad]
            tri_atoms = []
            for r in tri_res:
                if r is None:
                    continue
                for a in r:
                    if heavy(a):
                        tri_atoms.append(a)
            phosph = phosphate_atoms(rna)
            if tri_atoms and phosph:
                for a in tri_atoms:
                    dmin = min((a - p) for p in phosph)
                    triad_dists.append(dmin)

        rows.append(
            {
                "id": pid,
                "best_model_path": best_path,
                "ipTM": best_iptm if best_iptm is not None else iptm,
                "protein_pLDDT_mean": prot_plddt,
                "rna_pLDDT_mean": rna_plddt,
                "wed_pLDDT_mean": wed_plddt,
                "wed_rna_contact_count": wed_contacts,
                "triad_annotation": ",".join(map(str, triad)) if triad else "",
                "triad_min_distance_to_RNA_P": (min(triad_dists) if triad_dists else float("nan")),
                "triad_mean_distance_to_RNA_P": (float(np.mean(triad_dists)) if triad_dists else float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[score_wed_prerna] Wrote {args.out_csv} with {len(df)} rows.")


if __name__ == "__main__":
    main()
