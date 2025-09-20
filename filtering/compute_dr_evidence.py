#!/usr/bin/env python3
"""
compute_dr_evidence.py — Aggregate Atlas DR evidence per protein (exact AA) or per 90%-ID cluster.

Key ideas:
- Operon-level arrays include direct repeats in `crispr[*].crispr_repeat`.
- Cas12a/Cpf1 proteins live in `cas` with fields `gene_name`, `hmm_name`, and AA `protein`.
- We aggregate all arrays from all operons that map to a given protein (by exact AA SHA1),
  optionally summing across 90%-ID clusters provided via a mapping file.
- We normalize DRs (uppercase, U->T) and fix strand by canonicalizing to min(rep, revcomp(rep)).
- We build histograms of exact DR strings per key and assign high-confidence/ambiguous labels
  for the LbCas12a DR string.

Usage examples:
  # Per-protein evidence and labels
  python filtering/compute_dr_evidence.py \
    --atlas-json /path/to/crispr-cas-atlas-v1.0.json \
    --lb-dr AATTTCTACTAAGTGTAGAT \
    --out-per-protein dr_evidence.per_protein.tsv

  # Per-cluster evidence and labels (provide cluster mapping: sha1<TAB>cluster_id)
  python filtering/compute_dr_evidence.py \
    --atlas-json /path/to/crispr-cas-atlas-v1.0.json \
    --lb-dr AATTTCTACTAAGTGTAGAT \
    --cluster-map sha1_to_cluster.tsv \
    --out-per-cluster dr_evidence.per_cluster.tsv
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable
import hashlib


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def iter_atlas_records(path: str) -> Iterable[dict]:
    with open_text(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
            for rec in data:
                yield rec
        else:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def is_cas12a(gene_name: str, hmm_name: str) -> bool:
    s = f"{gene_name} {hmm_name}".lower()
    return ("cas12a" in s) or ("cpf1" in s)


_RC = str.maketrans("ACGT", "TGCA")


def revcomp(rep: str) -> str:
    return rep.translate(_RC)[::-1]


def normalize_repeat(rep: str | None) -> str | None:
    if rep is None:
        return None
    rep = rep.strip().upper().replace("U", "T")
    rep = re.sub(r"[^ACGT]", "", rep)
    return rep or None


def canonical_repeat(rep: str | None) -> str | None:
    """Legacy helper (unused in new scoring). Keep for compatibility."""
    rep = normalize_repeat(rep)
    if not rep:
        return None
    rc = revcomp(rep)
    return rep if rep <= rc else rc


def hamming(a: str, b: str) -> int | None:
    if len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if x != y)


def slide_min_hamming(seq: str, templates: list[str], anchor_left: int = 6) -> tuple[int, str, int, str]:
    """
    Search only near the 5′ end of the repeat on both orientations.
    Returns (min_distance, best_window, best_template_len, orientation).
    """
    s = normalize_repeat(seq) or ""
    if not s:
        return (10**9, "", 0, "+")
    rc = revcomp(s)
    best = (10**9, "", 0, "+")
    for orient, s_or in (("+", s), ("-", rc)):
        n = len(s_or)
        for tmpl in templates:
            L = len(tmpl)
            if n < L:
                continue
            max_start = min(anchor_left, n - L)
            for i in range(0, max_start + 1):
                win = s_or[i : i + L]
                d = hamming(win, tmpl)
                if d is not None and d < best[0]:
                    best = (d, win, L, orient)
                    if d == 0:
                        return best
    return best


def build_lb_templates(lb_dr: str) -> list[str]:
    """
    Build templates for the Lb handle:
    - If a 20-mer is given, include {20, 21=T+20}.
    - If a 21-mer is given, include {21, 20=trim leading base}.
    - Otherwise, just use the normalized input.
    """
    base = normalize_repeat(lb_dr) or ""
    if not base:
        return []
    T = set()
    if len(base) == 20:
        T.add(base)               # 20-mer (e.g., AATTTCTACTAAGTGTAGAT)
        T.add("T" + base)         # 21-mer canonical DNA (TAATTTCTACTAAGTGTAGAT)
    elif len(base) == 21:
        T.add(base)               # 21-mer
        T.add(base[1:])           # 20-mer trimmed
    else:
        T.add(base)
    return sorted(T, key=len)


def assign_label_from_windows(
    lb_templates: list[str],
    array_repeats: list[str],
    pos_thresh: float = 0.6,
    min_mismatch_neg: int = 2,
):
    """Compute evidence and label using sliding-window logic.

    Returns (modal_win, modal_count, total_arrays, lb_count, lb_frac, modal_frac, min_ham_to_lb, label, reason)
    where modal_win is the most common best-window string across arrays.
    """
    total = len(array_repeats)
    if total == 0:
        return ("", 0, 0, 0, 0.0, 0.0, None, "unknown", "no_repeats")

    win_counter: Counter[str] = Counter()
    lb_count = 0
    min_ham = 10**9
    exact_any = False
    for rep in array_repeats:
        d, win, L, orient = slide_min_hamming(rep, lb_templates)
        if d < min_ham:
            min_ham = d
        if win:
            win_counter[win] += 1
        if d == 0:
            exact_any = True
        if d is not None and d <= 1:
            lb_count += 1

    modal_win, modal_count = ("", 0)
    if win_counter:
        modal_win, modal_count = max(win_counter.items(), key=lambda kv: (kv[1], kv[0]))
    lb_frac = lb_count / total
    modal_frac = (modal_count / total) if total > 0 else 0.0

    if exact_any or lb_frac > pos_thresh:
        return (modal_win, modal_count, total, lb_count, lb_frac, modal_frac, min_ham, "positive", "exact_or_frac")
    if lb_count == 0 and (min_ham is None or min_ham >= min_mismatch_neg):
        return (modal_win, modal_count, total, lb_count, lb_frac, modal_frac, min_ham, "negative", f"min_ham>={min_mismatch_neg}")
    return (modal_win, modal_count, total, lb_count, lb_frac, modal_frac, min_ham, "ambiguous", "between")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Atlas DR evidence per protein or cluster")
    ap.add_argument("--atlas-json", required=True)
    ap.add_argument("--lb-dr", required=True, help="LbCas12a DR string (DNA; case/Us handled)")
    ap.add_argument("--out-per-protein", default=None, help="TSV to write per-protein evidence/labels")
    ap.add_argument("--cluster-map", default=None, help="Optional TSV: sha1\tcluster_id mapping (90%-ID clusters)")
    ap.add_argument("--out-per-cluster", default=None, help="TSV to write per-cluster evidence/labels (requires --cluster-map)")
    ap.add_argument("--ref-fasta", default=None, help="Optional reference FASTA of Cas12a proteins to annotate")
    ap.add_argument("--out-per-ref", default=None, help="TSV to write per-reference-header labels (requires --ref-fasta)")
    ap.add_argument("--positive-threshold", type=float, default=0.5)
    ap.add_argument("--min-mismatch-neg", type=int, default=2)
    args = ap.parse_args()

    # Prepare Lb templates (20- and 21-nt variants) using robust helper
    lb_templates = build_lb_templates(args.lb_dr)
    if not lb_templates:
        raise SystemExit("Invalid --lb-dr provided")

    # Aggregate repeats per protein key (sha1 of AA). Avoid double-counting per operon by
    # updating each unique key in that operon once with all of the operon's arrays.
    # Store raw normalized repeats (not canonicalized) for sliding-window scoring.
    per_protein_reps: dict[str, list[str]] = defaultdict(list)
    n_ops = 0
    for rec in iter_atlas_records(args.atlas_json):
        n_ops += 1
        # Gather this operon's repeats from arrays (normalized, both orientations considered later)
        reps = []
        for arr in (rec.get("crispr") or []):
            c = normalize_repeat(arr.get("crispr_repeat"))
            if c:
                reps.append(c)
        if not reps:
            continue
        # Unique Cas12a protein keys in this operon
        keys = set()
        for cas in (rec.get("cas") or []):
            g = cas.get("gene_name", "")
            h = cas.get("hmm_name", "")
            if not is_cas12a(g, h):
                continue
            aa = (cas.get("protein") or "").strip()
            if aa:
                keys.add(sha1(aa))
        if not keys:
            continue
        for k in keys:
            per_protein_reps[k].extend(reps)

    # Optional: map to clusters and sum
    per_cluster_reps: dict[str, list[str]] | None = None
    sha1_to_cluster: dict[str, str] = {}
    if args.cluster_map:
        per_cluster_reps = defaultdict(list)
        with open(args.cluster_map) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                sha, cid = parts[0], parts[1]
                sha1_to_cluster[sha] = cid
        for sha, reps in per_protein_reps.items():
            cid = sha1_to_cluster.get(sha)
            if cid is None:
                continue
            per_cluster_reps[cid].extend(reps)

    # Write per-protein TSV
    if args.out_per_protein:
        with open(args.out_per_protein, "w") as out:
            out.write("key_type\tkey\ttotal_arrays\tmodal_dr\tmodal_count\tmodal_frac\tlb_count\tlb_frac\tmin_ham_to_lb\tlabel\treason\n")
            for sha, reps in per_protein_reps.items():
                modal_dr, modal_count, total, lb_count, lb_frac, modal_frac, min_h, label, reason = assign_label_from_windows(
                    lb_templates, reps, args.positive_threshold, args.min_mismatch_neg
                )
                out.write("\t".join([
                    "protein_sha1",
                    sha,
                    str(total),
                    modal_dr,
                    str(modal_count),
                    f"{modal_frac:.6f}",
                    str(lb_count),
                    f"{lb_frac:.6f}",
                    "" if min_h is None else str(min_h),
                    label,
                    reason,
                ]) + "\n")

    # Write per-cluster TSV
    if args.out_per_cluster and per_cluster_reps is not None:
        with open(args.out_per_cluster, "w") as out:
            out.write("key_type\tkey\ttotal_arrays\tmodal_dr\tmodal_count\tmodal_frac\tlb_count\tlb_frac\tmin_ham_to_lb\tlabel\treason\n")
            for cid, reps in per_cluster_reps.items():
                modal_dr, modal_count, total, lb_count, lb_frac, modal_frac, min_h, label, reason = assign_label_from_windows(
                    lb_templates, reps, args.positive_threshold, args.min_mismatch_neg
                )
                out.write("\t".join([
                    "cluster_id",
                    cid,
                    str(total),
                    modal_dr,
                    str(modal_count),
                    f"{modal_frac:.6f}",
                    str(lb_count),
                    f"{lb_frac:.6f}",
                    "" if min_h is None else str(min_h),
                    label,
                    reason,
                ]) + "\n")

    # Annotate a provided reference FASTA: produce per-reference-header labels (including unknowns)
    if args.ref_fasta and args.out_per_ref:
        def _read_fasta(fp):
            hdr, buf = None, []
            for line in fp:
                if line.startswith(">"):
                    if hdr is not None:
                        yield hdr, "".join(buf).replace(" ", "").replace("\n", "")
                    hdr = line[1:].strip().split()[0]
                    buf = []
                else:
                    buf.append(line.strip())
            if hdr is not None:
                yield hdr, "".join(buf).replace(" ", "").replace("\n", "")

        with open(args.ref_fasta) as fh, open(args.out_per_ref, "w") as out:
            out.write("ref_id\ttotal_arrays\tmodal_dr\tmodal_count\tmodal_frac\tlb_count\tlb_frac\tmin_ham_to_lb\tlabel\treason\n")
            for ref_id, aa in _read_fasta(fh):
                key = sha1(aa)
                reps = per_protein_reps.get(key)
                if reps:
                    modal_dr, modal_count, total, lb_count, lb_frac, modal_frac, min_h, label, reason = assign_label_from_windows(
                        lb_templates, reps, args.positive_threshold, args.min_mismatch_neg
                    )
                else:
                    modal_dr, modal_count, total, lb_count, lb_frac, modal_frac, min_h, label, reason = (
                        "", 0, 0, 0, 0.0, 0.0, None, "unknown", "no_repeats"
                    )
                out.write("\t".join([
                    ref_id,
                    str(total),
                    modal_dr,
                    str(modal_count),
                    f"{modal_frac:.6f}",
                    str(lb_count),
                    f"{lb_frac:.6f}",
                    "" if min_h is None else str(min_h),
                    label,
                    reason,
                ]) + "\n")

    sys.stderr.write(
        f"[compute_dr_evidence] Processed {n_ops} operons; proteins with evidence: {len(per_protein_reps)}"
        + (f"; clusters: {len(per_cluster_reps)}" if 'per_cluster_reps' in locals() and per_cluster_reps is not None else "")
        + "\n"
    )


if __name__ == "__main__":
    main()
