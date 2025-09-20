#!/usr/bin/env python3
"""
Filter LbCas12a designs by conserving DNA-proximal residues (≤ cutoff Å) in the PID/PI region.

Works when your designs are FULL-LENGTH proteins:
- We align the template PID (from your PDB, by PDB residue range) to each full-length design
  using semi-global alignment (global on template, local on query).
- Any mutation or deletion at a template PID residue that lies ≤ cutoff Å from DNA -> FAIL.

Outputs:
- out/pass.faa (full-length sequences that pass)
- out/fail.faa (full-length sequences that fail)
- out/report.tsv (per-design summary, listing which positions violated)
- optional: cache of contact residues to skip recomputing distances

Usage:
python filtering/pid_8A_filter_fullseq.py \
  --pdb LbCas12a_DNA_complex.pdb \
  --protein-chain A \
  --pid-range 900-1068 \
  --designs designs_full.faa \
  --outdir out_pid \
  --cutoff 8.0
"""

import os
import sys
import argparse
import math
import json
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

try:
    import numpy as np  # optional but speeds up distance calc
except Exception:
    np = None

from Bio import SeqIO, pairwise2
# Biopython compatibility: MatrixInfo was removed in newer versions.
try:
    from Bio.SubsMat.MatrixInfo import blosum62  # type: ignore
except Exception:
    # Fallback to the new substitution_matrices API and build a dict for pairwise2
    from Bio.Align import substitution_matrices as _sm  # type: ignore
    _mat = _sm.load("BLOSUM62")
    try:
        _alphabet = list(_mat.alphabet)
    except Exception:
        _alphabet = list("ARNDCQEGHILKMFPSTWYVBZX*")
    blosum62 = {(a, b): int(_mat[a, b]) for a in _alphabet for b in _alphabet}
from Bio.PDB import PDBParser, is_aa

# Minimal three-letter to one-letter mapping for standard amino acids
_AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y', 'SEC': 'U', 'PYL': 'O',
}

DNA_NAMES = set([
    "DA", "DT", "DG", "DC", "DI",  # deoxy bases
    "A", "T", "G", "C", "U",          # sometimes PDBs store as RNA/mono
    "DU", "DG5", "DC5", "DT5", "DG3", "DC3", "DT3",  # termini variants
])


def log(m: str) -> None:
    print(f"[pid-8A] {m}", flush=True)


def load_pdb(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("template", pdb_path)


def get_chain(structure, chain_id: str):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return chain
    raise ValueError(f"Chain '{chain_id}' not found.")


def is_dna_res(res) -> bool:
    return res.get_resname().strip() in DNA_NAMES


def parse_range(r: str) -> Optional[Tuple[int, int]]:
    if not r:
        return None
    a, b = r.replace(" ", "").split("-")
    return int(a), int(b)


def chain_seq_and_residues(chain, rng: Optional[Tuple[int, int]]) -> Tuple[str, List]:
    seq = []
    reslist = []
    for res in chain.get_residues():
        if not is_aa(res, standard=True) and res.get_resname() != "MSE":
            continue
        rn = res.id[1]
        if rng and not (rng[0] <= rn <= rng[1]):
            continue
        name = res.get_resname()
        if name == "MSE":
            one = "M"
        else:
            one = _AA3_TO_1.get(name.upper())
            if one is None:
                raise ValueError(f"Unknown residue name '{name}' for conversion to 1-letter code")
        seq.append(one)
        reslist.append(res)
    if not seq:
        raise ValueError("No amino acids found in the selected region.")
    return "".join(seq), reslist


def collect_dna_atoms(structure, dna_chains: Optional[List[str]]):
    atoms = []
    for model in structure:
        for chain in model:
            if dna_chains is not None and chain.id not in dna_chains:
                continue
            use = True if dna_chains is not None else any(is_dna_res(r) for r in chain.get_residues())
            if not use:
                continue
            for res in chain.get_residues():
                if not is_dna_res(res):
                    continue
                for atom in res.get_atoms():
                    atoms.append(atom)
    if not atoms:
        raise ValueError("No DNA atoms found; specify --dna-chains if autodetect fails.")
    return atoms


def residue_min_dist(res, atoms) -> float:
    min_d = float("inf")
    coords = [a.get_coord() for a in res.get_atoms() if a.element != 'H']
    if not coords:
        return min_d
    if np is not None:
        R = np.array(coords, dtype=float)
        D = np.array([a.get_coord() for a in atoms], dtype=float)
        dif = R[:, None, :] - D[None, :, :]
        d2 = (dif * dif).sum(axis=2)
        return float((d2.min()) ** 0.5)
    else:
        for rc in coords:
            for a in atoms:
                dc = a.get_coord()
                dx, dy, dz = rc[0] - dc[0], rc[1] - dc[1], rc[2] - dc[2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d < min_d:
                    min_d = d
        return min_d


def compute_contacts(reslist: List, dna_atoms, cutoff: float) -> Dict[int, float]:
    contacts: Dict[int, float] = {}
    for i, res in enumerate(reslist, start=1):
        dmin = residue_min_dist(res, dna_atoms)
        if dmin <= cutoff:
            contacts[i] = dmin
    return contacts


def read_fasta(path: str) -> OrderedDict:
    recs: "OrderedDict[str, str]" = OrderedDict()
    with open(path, "r") as h:
        for rec in SeqIO.parse(h, "fasta"):
            recs[rec.id] = str(rec.seq).upper().replace("*", "")
    return recs


def write_fasta(recs: Dict[str, str], path: str) -> None:
    with open(path, "w") as f:
        for rid, seq in recs.items():
            f.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def semi_global_align(template_seq: str, query_seq: str) -> Tuple[str, str]:
    """
    Semi-global: global on TEMPLATE, local on QUERY.
    Implemented by globalds with no end-gap penalties on QUERY ends.
    Fallback to local alignment if Biopython doesn't support tuple penalize_end_gaps.
    """
    try:
        alns = pairwise2.align.globalds(
            template_seq,
            query_seq,
            blosum62,
            -10.0,
            -0.5,
            one_alignment_only=True,
            penalize_end_gaps=(True, False),  # penalize template ends, not query ends
        )
    except TypeError:
        # Fallback: local alignment; we will treat unaligned template residues as deletions
        alns = pairwise2.align.localds(template_seq, query_seq, blosum62, -10.0, -0.5, one_alignment_only=True)
    if not alns:
        raise ValueError("Alignment failed.")
    a_t, a_q, score, start, end = alns[0]
    return a_t, a_q


def analyze_design(
    template_seq: str,
    template_res: List,
    contacts: Dict[int, float],
    design_seq: str,
    allow_conservative: bool = False,
) -> Tuple[bool, Dict]:
    """
    Align template PID (short) to full-length design (long). Enforce:
    - if template position is DNA-proximal: design must have same AA at that aligned column
      (or allowed conservative), and may NOT be a gap (deletion).
    """
    a_t, a_q = semi_global_align(template_seq, design_seq)
    t_idx = 0
    violations = []
    for col in range(len(a_t)):
        tc = a_t[col]
        qc = a_q[col]
        if tc != "-":
            t_idx += 1
            if t_idx in contacts:
                res = template_res[t_idx - 1]
                chain = res.get_parent().id
                resnum = res.id[1]
                icode = res.id[2].strip() if res.id[2] != " " else ""
                if qc == "-":
                    # deletion at DNA-proximal position
                    violations.append(
                        {
                            "template_pos": t_idx,
                            "template_aa": tc,
                            "design_aa": "-",
                            "pdb_chain": chain,
                            "pdb_resnum": resnum,
                            "pdb_icode": icode,
                            "min_dist_to_DNA": contacts[t_idx],
                            "reason": "deletion",
                        }
                    )
                else:
                    same = tc == qc
                    cons = allow_conservative and ((tc, qc) in {("K", "R"), ("R", "K"), ("D", "E"), ("E", "D")})
                    if not (same or cons):
                        violations.append(
                            {
                                "template_pos": t_idx,
                                "template_aa": tc,
                                "design_aa": qc,
                                "pdb_chain": chain,
                                "pdb_resnum": resnum,
                                "pdb_icode": icode,
                                "min_dist_to_DNA": contacts[t_idx],
                                "reason": "mutation",
                            }
                        )
    return (len(violations) == 0), {"violations": violations, "alignment": {"template": a_t, "design": a_q}}


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pdb", required=True, help="LbCas12a·DNA PDB")
    ap.add_argument("--protein-chain", required=True, help="Protein chain id (LbCas12a)")
    ap.add_argument("--dna-chains", default="", help="Comma-separated DNA chain ids; leave blank to autodetect")
    ap.add_argument("--pid-range", default="", help="PDB numbering, e.g. 900-1068 for PI/PID")
    ap.add_argument("--designs", required=True, help="FASTA of FULL-LENGTH protein designs")
    ap.add_argument("--cutoff", type=float, default=8.0, help="Distance cutoff (Å) to define DNA-proximal residues")
    ap.add_argument("--outdir", default="out_pid", help="Output directory")
    ap.add_argument("--allow-conservative", action="store_true", help="Allow K↔R and D↔E at contacts")
    ap.add_argument("--save-contacts", default="", help="Path to save computed contact map as JSON")
    ap.add_argument("--load-contacts", default="", help="Path to precomputed contacts JSON to reuse")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    report = os.path.join(args.outdir, "report.tsv")
    pass_faa = os.path.join(args.outdir, "pass.faa")
    fail_faa = os.path.join(args.outdir, "fail.faa")

    # Load PDB and sequences
    structure = load_pdb(args.pdb)
    prot_chain = get_chain(structure, args.protein_chain)
    pid_rng = parse_range(args.pid_range) if args.pid_range else None
    template_seq, template_res = chain_seq_and_residues(prot_chain, pid_rng)

    # Contacts: load or compute
    if args.load_contacts:
        with open(args.load_contacts, "r") as fh:
            data = json.load(fh)
        contacts = {int(k): float(v) for k, v in data["contacts"].items()}
        log(f"Loaded {len(contacts)} contact residues from cache.")
    else:
        dna_chains = [c.strip() for c in args.dna_chains.split(",") if c.strip()] if args.dna_chains else None
        dna_atoms = collect_dna_atoms(structure, dna_chains)
        contacts = compute_contacts(template_res, dna_atoms, args.cutoff)
        log(f"Computed {len(contacts)} DNA-proximal residues (≤ {args.cutoff} Å).")
        if args.save_contacts:
            with open(args.save_contacts, "w") as fh:
                json.dump(
                    {
                        "pdb": os.path.basename(args.pdb),
                        "chain": args.protein_chain,
                        "pid_range": pid_rng,
                        "cutoff": args.cutoff,
                        "contacts": contacts,
                    },
                    fh,
                    indent=2,
                )
            log(f"Saved contacts -> {args.save_contacts}")

    # Read full-length designs
    designs = read_fasta(args.designs)
    kept: Dict[str, str] = {}
    dropped: Dict[str, str] = {}

    with open(report, "w") as rep:
        rep.write("\t".join(["design_id", "passes", "num_violations", "violating_sites"]) + "\n")
        for did, dseq in designs.items():
            ok, info = analyze_design(
                template_seq, template_res, contacts, dseq, allow_conservative=args.allow_conservative
            )
            if ok:
                kept[did] = dseq
            else:
                dropped[did] = dseq

            sites = []
            for v in info["violations"]:
                sites.append(
                    f"{v['pdb_chain']}:{v['pdb_resnum']}{v['pdb_icode']}({v['template_aa']}→{v['design_aa']};{v['min_dist_to_DNA']:.2f}Å)"
                )
            rep.write("\t".join([did, str(ok), str(len(sites)), ",".join(sites)]) + "\n")

    write_fasta(kept, pass_faa)
    write_fasta(dropped, fail_faa)
    log(f"Wrote {len(kept)} pass -> {pass_faa}")
    log(f"Wrote {len(dropped)} fail -> {fail_faa}")
    log(f"Report -> {report}")


if __name__ == "__main__":
    main()
