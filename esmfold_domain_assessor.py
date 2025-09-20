#!/usr/bin/env python3
"""
ESMFold domain confidence scorer for Cas12a-like proteins.

Inputs:
  --fasta       FASTA of sequences to score
  --ref_fasta   FASTA with ONE reference (e.g., LbCas12a)
  --domains     YAML/JSON with domain coordinates on the reference
  --out_csv     Output CSV (default: cas12a_esmfold_scores.csv)
  --pdb_dir     Optional directory to write ESMFold PDBs
  --device      cuda|cpu (auto if omitted)

Outputs (CSV):
  id, len, <domain>_mean_pLDDT, <domain>_frac_ge80, <domain>_frac_ge90,
  pass_WED, pass_RuvC, pass_NUC, pass_PI

Notes:
  - pLDDT is pulled from the PDB B-factor (CA atoms).
  - Domain spans are mapped from reference to each query by global alignment.

"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd

# Optional Biopython (SeqIO + pairwise2). Provide fallbacks if unavailable.
try:
    from Bio import SeqIO, pairwise2  # type: ignore
    _HAVE_BIO = True
except Exception:
    _HAVE_BIO = False

# ---- ESMFold ---------------------------------------------------------------
try:
    import torch
    from esm import pretrained
    ESM_OK = True
except Exception as e:
    ESM_OK = False
    ESM_ERR = str(e)

try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


@dataclass
class Domain:
    name: str
    ref_start: int
    ref_end: int


def load_domains(path: str) -> Tuple[str, Dict[str, Domain]]:
    if path.endswith((".yaml", ".yml")):
        if not _HAVE_YAML:
            sys.exit("Install pyyaml for YAML domain files (pip install pyyaml).")
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            cfg = json.load(f)
    ref_id = cfg["reference"]["id"]
    out: Dict[str, Domain] = {}
    for d in cfg["domains"]:
        out[d["name"]] = Domain(d["name"], int(d["ref_start"]), int(d["ref_end"]))
    return ref_id, out


def _iter_fasta(path: str) -> Iterable[Tuple[str, str]]:
    """Yield (id, seq) from a FASTA file. Fallback if Biopython is unavailable."""
    if _HAVE_BIO:
        for rec in SeqIO.parse(path, "fasta"):
            yield (rec.id, str(rec.seq))
    else:
        sid = None
        buf: List[str] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if sid is not None:
                        yield (sid, "".join(buf))
                    sid = line[1:].split()[0]
                    buf = []
                else:
                    buf.append(line)
            if sid is not None:
                yield (sid, "".join(buf))


def _align_ref_to_query_affine(ref: str, q: str) -> Dict[int, int]:
    """Global alignment with affine gaps (Gotoh) returning 1-based ref->query map.

    Scoring matches Biopython call above: match=2, mismatch=-1, gap_open=-10, gap_extend=-1.
    """
    n, m = len(ref), len(q)
    if n == 0 or m == 0:
        return {}
    match, mismatch = 2.0, -1.0
    go, ge = -10.0, -1.0
    NEG_INF = -1e9

    # DP matrices
    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.float32)
    X = np.full((n + 1, m + 1), NEG_INF, dtype=np.float32)  # gap in query (ref advances)
    Y = np.full((n + 1, m + 1), NEG_INF, dtype=np.float32)  # gap in ref (query advances)

    # Pointers: store 0/1/2 indicating source state (M/X/Y) for backtrace
    pM = np.full((n + 1, m + 1), -1, dtype=np.int8)
    pX = np.full((n + 1, m + 1), -1, dtype=np.int8)
    pY = np.full((n + 1, m + 1), -1, dtype=np.int8)

    M[0, 0] = 0.0
    # Initialize leading gaps (global alignment)
    for i in range(1, n + 1):
        X[i, 0] = go + ge * (i - 1)
        pX[i, 0] = 1  # from X (extend)
    for j in range(1, m + 1):
        Y[0, j] = go + ge * (j - 1)
        pY[0, j] = 1  # from Y (extend)

    # Fill DP
    for i in range(1, n + 1):
        ai = ref[i - 1]
        for j in range(1, m + 1):
            bj = q[j - 1]
            s = match if ai == bj else mismatch

            # M: come from best of (M, X, Y) diagonally + score
            m_cand = M[i - 1, j - 1]
            x_cand = X[i - 1, j - 1]
            y_cand = Y[i - 1, j - 1]
            if m_cand >= x_cand and m_cand >= y_cand:
                M[i, j] = m_cand + s
                pM[i, j] = 0
            elif x_cand >= y_cand:
                M[i, j] = x_cand + s
                pM[i, j] = 1
            else:
                M[i, j] = y_cand + s
                pM[i, j] = 2

            # X: gap in query (advance ref)
            open_x = M[i - 1, j] + go + ge
            ext_x = X[i - 1, j] + ge
            if open_x >= ext_x:
                X[i, j] = open_x
                pX[i, j] = 0
            else:
                X[i, j] = ext_x
                pX[i, j] = 1

            # Y: gap in ref (advance query)
            open_y = M[i, j - 1] + go + ge
            ext_y = Y[i, j - 1] + ge
            if open_y >= ext_y:
                Y[i, j] = open_y
                pY[i, j] = 0
            else:
                Y[i, j] = ext_y
                pY[i, j] = 1

    # Choose best end state
    end_state = 0
    best = M[n, m]
    if X[n, m] > best:
        best = X[n, m]
        end_state = 1
    if Y[n, m] > best:
        best = Y[n, m]
        end_state = 2

    # Backtrace to build mapping
    i, j, st = n, m, end_state
    mapping: Dict[int, int] = {}
    while i > 0 or j > 0:
        if st == 0:  # M
            # aligned residues => record mapping (1-based)
            if i > 0 and j > 0:
                mapping[i] = j
            prev = pM[i, j]
            i -= 1
            j -= 1
            st = int(prev)
        elif st == 1:  # X: gap in query
            prev = pX[i, j]
            i -= 1
            st = int(prev)
        else:  # st == 2, Y: gap in ref
            prev = pY[i, j]
            j -= 1
            st = int(prev)
        if st < 0:
            break
    return mapping


def align_ref_to_query(ref: str, q: str) -> Dict[int, int]:
    """Global alignment and 1-based ref->query mapping with optional Biopython.

    If Biopython is present, use pairwise2 with match=2, mismatch=-1,
    gap_open=-10, gap_extend=-1. Otherwise, use a built-in Gotoh fallback.
    """
    if _HAVE_BIO:
        aln = pairwise2.align.globalms(ref, q, 2, -1, -10, -1, one_alignment_only=True)[0]
        A, B = aln.seqA, aln.seqB
        rpos = 0
        qpos = 0
        mapping: Dict[int, int] = {}
        for a, b in zip(A, B):
            if a != "-":
                rpos += 1
            if b != "-":
                qpos += 1
            if a != "-" and b != "-":
                mapping[rpos] = qpos
        return mapping
    else:
        return _align_ref_to_query_affine(ref, q)


def infer_esmfold(seq: str, model, device: str, *, chunk_size: int | None = 128, recycles: int | None = 1) -> Tuple[str, List[float]]:
    """Run ESMFold to get PDB string and per-residue pLDDT list (1..N).

    chunk_size and recycles are tuned down by default to reduce GPU memory.
    """
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    # Reduce memory footprint for long sequences
    try:
        if chunk_size is not None:
            model.set_chunk_size(chunk_size)
    except Exception:
        pass
    with torch.no_grad():
        try:
            out = model.infer(seq, num_recycles=recycles)
            pdb_str = model.output_to_pdb(out)[0]
        except Exception:
            # Fallback to convenience API if present
            pdb_str = model.infer_pdb(seq)
    # parse pLDDT per-residue from B-factor (CA atoms)
    plddt_by_pos: Dict[int, float] = {}
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            resi = int(line[22:26])
            plddt = float(line[60:66])
        except Exception:
            continue
        plddt_by_pos[resi] = plddt
    if not plddt_by_pos:
        return pdb_str, []
    N = max(plddt_by_pos.keys())
    arr = [plddt_by_pos.get(i, np.nan) for i in range(1, N + 1)]
    return pdb_str, arr


def stats_for_indices(values: List[float], idx: List[int]) -> Dict[str, float]:
    if not values or not idx:
        return {"mean": np.nan, "frac_ge80": np.nan, "frac_ge90": np.nan}
    vec = np.array([values[i - 1] for i in idx if 1 <= i <= len(values)], dtype=float)
    if vec.size == 0:
        return {"mean": np.nan, "frac_ge80": np.nan, "frac_ge90": np.nan}
    return {
        "mean": float(np.nanmean(vec)),
        "frac_ge80": float(np.nanmean(vec >= 80)),
        "frac_ge90": float(np.nanmean(vec >= 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--domains", required=True)
    ap.add_argument("--out_csv", default="cas12a_esmfold_scores.csv")
    ap.add_argument("--pdb_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk-size", type=int, default=128, help="Axial attention chunk size (memory control)")
    ap.add_argument("--recycles", type=int, default=1, help="Number of recycles (memory control)")
    ap.add_argument(
        "--esmfold-variant",
        default="esmfold_v1",
        choices=[
            "esmfold_v1",
            "esmfold_v0",
            "esmfold_structure_module_only_8M",
            "esmfold_structure_module_only_35M",
            "esmfold_structure_module_only_150M",
            "esmfold_structure_module_only_650M",
        ],
        help="Select a smaller ESMFold variant to reduce memory",
    )
    ap.add_argument("--wed_min", type=float, default=85.0)
    # Adjusted defaults per request: RuvC=70, NUC=80
    ap.add_argument("--ruvc_min", type=float, default=70.0)
    ap.add_argument("--nuc_min", type=float, default=80.0)
    ap.add_argument("--pi_min", type=float, default=80.0)
    args = ap.parse_args()

    if not ESM_OK:
        sys.exit(f"ESMFold not available: {ESM_ERR}\nInstall with: pip install fair-esm torch")

    # load reference
    ref_records = list(_iter_fasta(args.ref_fasta))
    if len(ref_records) != 1:
        sys.exit("ref_fasta must contain exactly one sequence.")
    ref_id_from_file, ref_seq = ref_records[0]

    cfg_ref_id, doms = load_domains(args.domains)
    if cfg_ref_id not in (ref_id_from_file, "*"):
        print(
            f"[warn] domain map expects reference id '{cfg_ref_id}', but ref_fasta has '{ref_id_from_file}'.",
            file=sys.stderr,
        )

    # set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Choose ESMFold variant (smaller models ease memory pressure for smoke tests)
    model_ctor = getattr(pretrained, args.esmfold_variant)
    model = model_ctor()
    model = model.eval()
    print(f"[info] device={device} | ESMFold loaded", flush=True)

    if args.pdb_dir:
        os.makedirs(args.pdb_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for sid, seq in _iter_fasta(args.fasta):
        mapping = align_ref_to_query(ref_seq, seq)

        pdb_str, plddt = infer_esmfold(seq, model, device, chunk_size=args.chunk_size, recycles=args.recycles)
        if args.pdb_dir:
            with open(os.path.join(args.pdb_dir, f"{sid}.pdb"), "w") as f:
                f.write(pdb_str)

        # per-domain stats and global scalar
        entry: Dict[str, object] = {"id": sid, "len": len(seq)}
        try:
            entry["total_mean_pLDDT"] = float(np.nanmean(plddt)) if plddt else np.nan
        except Exception:
            entry["total_mean_pLDDT"] = np.nan
        for name, d in doms.items():
            idx = [mapping[p] for p in range(d.ref_start, d.ref_end + 1) if p in mapping]
            st = stats_for_indices(plddt, idx)
            entry[f"{name}_mean_pLDDT"] = st["mean"]
            entry[f"{name}_frac_ge80"] = st["frac_ge80"]
            entry[f"{name}_frac_ge90"] = st["frac_ge90"]

        # aggregated pass flags required by downstream filters
        # build group indices by matching domain names
        lower_names = {n: n.lower() for n in doms.keys()}

        def group_indices(pred):
            idxs: List[int] = []
            for n, d in doms.items():
                if pred(lower_names[n]):
                    idxs.extend([mapping[p] for p in range(d.ref_start, d.ref_end + 1) if p in mapping])
            return idxs

        wed_idx = group_indices(lambda s: s.startswith("wed"))
        ruvc_idx = group_indices(lambda s: ("ruvc" in s) or ("ruv" in s))
        nuc_idx = group_indices(lambda s: s == "nuc")
        pi_idx = group_indices(lambda s: s == "pi")

        wed_mean = stats_for_indices(plddt, wed_idx)["mean"] if wed_idx else np.nan
        ruvc_mean = stats_for_indices(plddt, ruvc_idx)["mean"] if ruvc_idx else np.nan
        nuc_mean = stats_for_indices(plddt, nuc_idx)["mean"] if nuc_idx else np.nan
        pi_mean = stats_for_indices(plddt, pi_idx)["mean"] if pi_idx else np.nan

        def pass_flag(val, thr):
            try:
                return float(val) >= thr
            except Exception:
                return False

        entry["pass_WED"] = pass_flag(wed_mean, args.wed_min)
        entry["pass_RuvC"] = pass_flag(ruvc_mean, args.ruvc_min)
        entry["pass_NUC"] = pass_flag(nuc_mean, args.nuc_min)
        entry["pass_PI"] = pass_flag(pi_mean, args.pi_min)
        rows.append(entry)

    df = pd.DataFrame(rows)

    # pass/fail flags for common domains if present
    def ok(x, th):
        try:
            return float(x) >= th
        except Exception:
            return False

    if "WED_mean_pLDDT" in df:
        df["pass_WED"] = df["WED_mean_pLDDT"].apply(lambda v: ok(v, args.wed_min))
    if "RuvC_mean_pLDDT" in df:
        df["pass_RuvC"] = df["RuvC_mean_pLDDT"].apply(lambda v: ok(v, args.ruvc_min))
    if "NUC_mean_pLDDT" in df:
        df["pass_NUC"] = df["NUC_mean_pLDDT"].apply(lambda v: ok(v, args.nuc_min))
    if "PI_mean_pLDDT" in df:
        df["pass_PI"] = df["PI_mean_pLDDT"].apply(lambda v: ok(v, args.pi_min))

    df.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
