#!/usr/bin/env python3
"""
atlas_to_cas12a_dr_index.py — map reference FASTA headers to DRs from the Atlas JSON.

- Streams the Atlas (array or JSONL).
- For each operon, picks a single DR from its CRISPR arrays (majority vote, tie → longest).
- For each Cas12a/Cpf1 protein in that operon, indexes by SHA1 of the amino-acid sequence.
- Reads your reference FASTA (e.g., cas12a_ref_rep_seq.fasta) and matches exact AA by SHA1.
- Emits TSV: ref_id, operon_id, dr, dr_len, gene_name, hmm_name, subtype, [dr_exact_to_lbcas12a]

Usage:
  python filtering/atlas_to_cas12a_dr_index.py \
    --atlas-json /path/to/crispr-cas-atlas-v1.0.json \
    --ref-fasta /path/to/cas12a_ref_rep_seq.fasta \
    --out-tsv /path/to/dr_map.ref_to_dr.tsv \
    [--lbcas12a-dr AATTTCTACTAAGTGTAGAT]
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Iterable, Iterator


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_fasta(fp) -> Iterator[tuple[str, str]]:
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


def iter_atlas_records(path: str) -> Iterable[dict]:
    """Stream either a JSON array or JSON-lines file."""
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


def normalize_repeat(rep: str | None) -> str | None:
    if rep is None:
        return None
    rep = rep.strip().upper().replace("U", "T")
    rep = re.sub(r"[^ACGT]", "", rep)
    return rep or None


def pick_operon_dr(
    crispr_list: list[dict] | None,
    preferred_dr: str | None = None,
) -> str | None:
    """Choose a single DR for an operon.

    Selection order:
      1) Majority vote across repeats present in the operon.
      2) If there's a tie among top repeats and a preferred DR (e.g., LbCas12a DR)
         is among those tied top repeats, choose the preferred DR.
      3) Otherwise break ties by longest length.
    """
    reps: list[str] = []
    for arr in (crispr_list or []):
        rep = normalize_repeat(arr.get("crispr_repeat"))
        if rep:
            reps.append(rep)
    if not reps:
        return None
    counts = Counter(reps)
    top = max(counts.values())
    tied = [r for r, c in counts.items() if c == top]
    if preferred_dr:
        preferred_dr = normalize_repeat(preferred_dr)
        if preferred_dr in tied:
            return preferred_dr
    return sorted(tied, key=len, reverse=True)[0]


def is_cas12a(gene_name: str, hmm_name: str) -> bool:
    s = f"{gene_name} {hmm_name}".lower()
    return ("cas12a" in s) or ("cpf1" in s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Map Cas12a reference FASTA headers to DRs from Atlas JSON")
    ap.add_argument("--atlas-json", required=True, help="Path to crispr-cas-atlas-v1.0.json (optionally .gz)")
    ap.add_argument("--ref-fasta", required=True, help="Reference FASTA used in coverage step")
    ap.add_argument("--out-tsv", required=True, help="Output TSV path")
    ap.add_argument(
        "--lbcas12a-dr",
        default=None,
        help="Optional DR string: prefer this on ties and add exact-match boolean column",
    )
    args = ap.parse_args()

    atlas_index: dict[str, list[tuple[str, str, str, str, str, int]]] = defaultdict(list)
    n_ops = n_cas12a = n_with_dr = 0
    for rec in iter_atlas_records(args.atlas_json):
        n_ops += 1
        subtype = (rec.get("summary") or {}).get("subtype", "")
        dr = pick_operon_dr(rec.get("crispr"), args.lbcas12a_dr)
        if not dr:
            continue
        n_with_dr += 1
        opid = rec.get("operon_id", "")
        for cas in (rec.get("cas") or []):
            g = cas.get("gene_name", "")
            h = cas.get("hmm_name", "")
            if not is_cas12a(g, h):
                continue
            aa = (cas.get("protein") or "").strip()
            if not aa:
                continue
            n_cas12a += 1
            atlas_index[sha1(aa)].append((opid, dr, g, h, subtype, len(aa)))

    # Map reference headers by exact AA hash
    header_map: dict[str, tuple[str, str, int, str, str, str]] = {}
    with open_text(args.ref_fasta) as fh:
        for ref_id, aa in read_fasta(fh):
            key = sha1(aa)
            hits = atlas_index.get(key)
            if not hits:
                continue
            opid, dr, g, h, subtype, _ = hits[0]
            header_map[ref_id] = (opid, dr, len(dr), g, h, subtype)

    with open(args.out_tsv, "w") as out:
        cols = ["ref_id", "operon_id", "dr", "dr_len", "gene_name", "hmm_name", "subtype"]
        if args.lbcas12a_dr:
            cols.append("dr_exact_to_lbcas12a")
        out.write("\t".join(cols) + "\n")
        lb = None
        if args.lbcas12a_dr:
            lb = normalize_repeat(args.lbcas12a_dr)
        for ref_id, (opid, dr, dr_len, g, h, subtype) in header_map.items():
            row = [ref_id, opid, dr, str(dr_len), g, h, subtype]
            if lb is not None:
                row.append("1" if dr == lb else "0")
            out.write("\t".join(row) + "\n")

    sys.stderr.write(
        f"[atlas_to_cas12a_dr_index] Parsed {n_ops} operons; Cas12a proteins with DR: {n_cas12a}; "
        f"Operons with DR: {n_with_dr}; Mapped references: {len(header_map)}\n"
    )


if __name__ == "__main__":
    main()
