#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage-based filter for protein sequences with HMMER scan + paper-faithful gates.

- Input: FASTA or CSV (auto-detected) of AA sequences, possibly with EOS tokens ('1'/'2').
- Normalizes sequences: strips EOS; if EOS='1' reverses to N->C; optional k-mer and length filters.
- Runs hmmscan against a single HMM or an HMM library; computes BOTH query(HMM) and target(seq) coverage,
  merges multi-domain hits, and keeps sequences whose BEST hit satisfies:
     bitscore > --min_bitscore AND qcov >= --min_qcov AND tcov >= --min_tcov

Defaults match the paper’s HMMER gate (qcov>=0.8, tcov>=0.8, bitscore>50).  See Methods.  :contentReference[oaicite:4]{index=4}

Requires: HMMER3 in PATH. Recommended: hmmpress the .hmm file for speed.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

from Bio import SeqIO

AA_SET = set(list("ACDEFGHIKLMNPQRSTVWYXBZJUO"))

# ---- k-mer repeat filter (as in the published generate.py) ----
# 1/2/3/4/5/6/7-mers repeated 6/4/3/3/3/3/2 times, respectively. :contentReference[oaicite:5]{index=5}
def _kmer_n_repeat(k: int, n: int) -> str:
    return "(.)" * k + "".join([f"\\{i+1}" for i in range(k)]) * (n - 1)

_KMER_THRESHOLDS = [6, 4, 3, 3, 3, 3, 2]  # k=1..7
_KMER_RES = [re.compile(_kmer_n_repeat(k, n)) for k, n in enumerate(_KMER_THRESHOLDS, start=1)]

def has_kmer_repeats(seq: str) -> bool:
    return any(r.search(seq) for r in _KMER_RES)

# ---- cleaning / EOS normalization ----
def clean_seq(seq: str) -> str:
    # Keep only alphanumeric letters & uppercase; we'll strip EOS digits separately.
    return "".join(ch for ch in str(seq).strip().upper() if ch not in {"*", " ", "\n", "\r"})

@dataclass
class NormResult:
    seq: str
    eos: Optional[str] = None
    flipped: bool = False

def normalize_eos_and_orientation(raw: str, flip_on_eos1: bool = True) -> NormResult:
    s = clean_seq(raw)
    # Strip leading EOS if present
    if s and s[0] in ("1", "2"):
        s = s[1:]
    eos = None
    flipped = False
    if s and s[-1] in ("1", "2"):   # trailing EOS sentinel
        eos = s[-1]
        s = s[:-1]
        if flip_on_eos1 and eos == "1":  # reverse to N->C
            s = s[::-1]
            flipped = True
    # After normalization, drop any stray digits (defensive)
    s = "".join(ch for ch in s if ch.isalpha())
    return NormResult(seq=s, eos=eos, flipped=flipped)

def seq_ok(seq: str) -> bool:
    return bool(seq) and all((ch in AA_SET) for ch in seq)

def iter_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    for rec in SeqIO.parse(str(path), "fasta"):
        yield rec.id, clean_seq(rec.seq)

def csv_to_fasta(tmpdir: Path, csv_path: Path, out_fasta: Path, seq_col: str, id_col: str,
                 flip_on_eos1: bool, use_kmer: bool, min_len: int, max_len: int) -> Tuple[Path, Dict[str, Dict[str, str]]]:
    meta: Dict[str, Dict[str, str]] = {}  # rid -> {eos, flipped}
    with open(csv_path, newline="") as cf, open(out_fasta, "w") as fout:
        reader = csv.DictReader(cf)
        if seq_col not in reader.fieldnames:
            raise ValueError(f"CSV missing sequence column '{seq_col}'")
        idx = 0
        for row in reader:
            rid = str(row.get(id_col, f"row{idx}"))
            raw = row.get(seq_col, "")
            idx += 1
            norm = normalize_eos_and_orientation(raw, flip_on_eos1=flip_on_eos1)
            s = norm.seq
            if not s or not seq_ok(s):
                continue
            if min_len and len(s) < min_len:
                continue
            if max_len and len(s) > max_len:
                continue
            if use_kmer and has_kmer_repeats(s):
                continue
            fout.write(f">{rid}\n{s}\n")
            meta[rid] = {"eos": norm.eos or "", "flipped": "1" if norm.flipped else "0"}
    return out_fasta, meta

# ---- HMMER domtbl parsing & coverage ----

@dataclass
class DomLine:
    tname: str; tlen: int
    qname: str; qlen: int
    full_score: float
    hmmfrom: int; hmmto: int
    alifrom: int; alito: int

@dataclass
class PairAgg:
    # aggregate per (sequence, HMM) across domains
    seq_len: int
    hmm_len: int
    full_score_max: float = -1e9
    hmm_spans: List[Tuple[int, int]] = field(default_factory=list)
    ali_spans: List[Tuple[int, int]] = field(default_factory=list)

def parse_domtblout(domtbl_path: Path) -> List[DomLine]:
    out: List[DomLine] = []
    with open(domtbl_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            # HMMER domtblout columns:
            #  0 tname  1 tacc  2 tlen  3 qname  4 qacc  5 qlen
            #  6 full_E 7 full_score 8 full_bias 9 # 10 of
            # 11 c-E    12 i-E        13 dom_score 14 dom_bias
            # 15 hmmfrom 16 hmmto 17 alifrom 18 alito 19 envfrom 20 envto 21 acc  [desc...]
            if len(parts) < 22:
                continue
            try:
                tname = parts[0]
                tlen  = int(parts[2])
                qname = parts[3]
                qlen  = int(parts[5])
                full_score = float(parts[7])
                hmmfrom = int(parts[15]); hmmto = int(parts[16])
                alifrom = int(parts[17]); alito = int(parts[18])
            except Exception:
                continue
            out.append(DomLine(tname, tlen, qname, qlen, full_score, hmmfrom, hmmto, alifrom, alito))
    return out

def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans: return []
    spans = [(min(a,b), max(a,b)) for a,b in spans]
    spans.sort(key=lambda x: x[0])
    merged = [spans[0]]
    for s,e in spans[1:]:
        ps,pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s,e))
    return merged

def compute_best_hit(domlines: List[DomLine], program: str) -> Dict[str, Dict]:
    """
    Returns: per sequence ID -> dict(hmm, bitscore, qcov, tcov)
    Works with hmmscan (seq queries vs HMM db). If you used hmmsearch instead, set program accordingly.
    """
    # For hmmscan: target is HMM (tlen=hmm), query is sequence (qlen=seq).
    # For hmmsearch: target is SEQUENCE (tlen=seq), query is HMM (qlen=hmm).
    per_pair: Dict[Tuple[str,str], PairAgg] = {}
    for d in domlines:
        if program == "hmmscan":
            seq_id, seq_len = d.qname, d.qlen
            hmm_id, hmm_len = d.tname, d.tlen
            hspan = (d.hmmfrom, d.hmmto)
            aspan = (d.alifrom, d.alito)
        else:  # hmmsearch
            seq_id, seq_len = d.tname, d.tlen
            hmm_id, hmm_len = d.qname, d.qlen
            hspan = (d.hmmfrom, d.hmmto)
            aspan = (d.alifrom, d.alito)
        key = (seq_id, hmm_id)
        agg = per_pair.get(key)
        if agg is None:
            agg = PairAgg(seq_len=seq_len, hmm_len=hmm_len)
            per_pair[key] = agg
        agg.full_score_max = max(agg.full_score_max, d.full_score)
        agg.hmm_spans.append(hspan)
        agg.ali_spans.append(aspan)

    # select best HMM per sequence (max bitscore)
    best_per_seq: Dict[str, Dict] = {}
    for (seq_id, hmm_id), agg in per_pair.items():
        hmm_cov_len = sum(e - s + 1 for s, e in _merge_spans(agg.hmm_spans))
        ali_cov_len = sum(e - s + 1 for s, e in _merge_spans(agg.ali_spans))
        qcov = min(1.0, hmm_cov_len / max(1, agg.hmm_len))   # "query" = HMM model
        tcov = min(1.0, ali_cov_len / max(1, agg.seq_len))   # "target" = sequence
        rec = {"hmm": hmm_id, "bitscore": agg.full_score_max, "qcov": qcov, "tcov": tcov}
        cur = best_per_seq.get(seq_id)
        if cur is None or rec["bitscore"] > cur["bitscore"]:
            best_per_seq[seq_id] = rec
    return best_per_seq

def write_fasta(out_fa: Path, id2seq: Dict[str, str], keep_ids: Iterable[str]) -> None:
    with open(out_fa, "w") as f:
        for rid in keep_ids:
            s = id2seq.get(rid)
            if s:
                f.write(f">{rid}\n{s}\n")

def main():
    ap = argparse.ArgumentParser(description="Coverage filter with HMMER hmmscan (paper-faithful).")
    ap.add_argument("--input", help="Input FASTA or CSV (auto-detect by extension).", required=True)
    ap.add_argument("--out_fasta", required=True, help="Output FASTA with sequences passing filters.")
    ap.add_argument("--out_tsv", required=True, help="Coverage report TSV.")
    ap.add_argument("--hmm", required=True, help="Path to a .hmm (single or concatenated library; pressed or not).")

    # Normalization & prefilters
    ap.add_argument("--flip_on_eos1", action="store_true", help="If a sequence ends with EOS '1', reverse to N->C.")
    ap.add_argument("--kmer_filter", action="store_true", help="Apply k-mer repeat filter (LM degeneracy).")
    ap.add_argument("--min_len", type=int, default=0, help="Minimum AA length AFTER EOS strip/reversal (e.g., 1000).")
    ap.add_argument("--max_len", type=int, default=0, help="Maximum AA length (e.g., 1500).")

    # CSV options
    ap.add_argument("--csv-seq-col", default="sequence", help="CSV column with sequences.")
    ap.add_argument("--csv-id-col", default="id", help="CSV column with IDs (optional).")

    # HMMER thresholds (paper defaults)
    ap.add_argument("--min_qcov", type=float, default=0.8, help="Min HMM/query coverage (default 0.8).")
    ap.add_argument("--min_tcov", type=float, default=0.8, help="Min sequence/target coverage (default 0.8).")
    ap.add_argument("--min_bitscore", type=float, default=50.0, help="Min full-seq bitscore (default 50).")

    ap.add_argument("--threads", type=int, default=4, help="Threads for hmmscan.")
    args = ap.parse_args()

    in_p = Path(args.input)
    tmp_dir = Path(tempfile.mkdtemp(prefix="covfilt_"))
    work_fa = tmp_dir / "input.norm.fasta"

    meta_map: Dict[str, Dict[str, str]] = {}
    try:
        # Build FASTA input with normalization & prefilters
        if in_p.suffix.lower() == ".csv":
            work_fa, meta_map = csv_to_fasta(
                tmpdir=tmp_dir, csv_path=in_p, out_fasta=work_fa,
                seq_col=args.csv_seq_col, id_col=args.csv_id_col,
                flip_on_eos1=args.flip_on_eos1, use_kmer=args.kmer_filter,
                min_len=args.min_len, max_len=args.max_len
            )
        else:
            # FASTA path: read, normalize, and write a cleaned FASTA
            id2seq_in: Dict[str, str] = {rid: s for rid, s in iter_fasta(in_p)}
            with open(work_fa, "w") as fout:
                for rid, raw in id2seq_in.items():
                    norm = normalize_eos_and_orientation(raw, flip_on_eos1=args.flip_on_eos1)
                    s = norm.seq
                    if not s or not seq_ok(s):
                        continue
                    if args.min_len and len(s) < args.min_len:
                        continue
                    if args.max_len and len(s) > args.max_len:
                        continue
                    if args.kmer_filter and has_kmer_repeats(s):
                        continue
                    fout.write(f">{rid}\n{s}\n")
                    meta_map[rid] = {"eos": norm.eos or "", "flipped": "1" if norm.flipped else "0"}

        # Load normalized sequences for downstream writing
        id2seq: Dict[str, str] = {rid: s for rid, s in iter_fasta(work_fa)}

        # Run hmmscan
        if shutil.which("hmmscan") is None:
            print("[error] hmmscan not found in PATH. Activate bio-utils.", file=sys.stderr)
            sys.exit(3)
        domtbl = tmp_dir / "hits.domtblout"
        cmd = [
            "hmmscan",
            "--cpu", str(args.threads),
            "--domtblout", str(domtbl),
            "--noali",
            str(args.hmm),
            str(work_fa),
        ]
        print("[info] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Parse domtblout and compute best hit per sequence
        dlines = parse_domtblout(domtbl)
        best = compute_best_hit(dlines, program="hmmscan")

        # Decide keepers by thresholds on the BEST hit
        keep_ids: List[str] = []
        with open(args.out_tsv, "w", newline="") as ftsv:
            w = csv.writer(ftsv, delimiter="\t")
            w.writerow(["id", "length", "hmm", "bitscore", "qcov", "tcov", "keep", "flipped", "eos"])
            for rid, s in id2seq.items():
                h = best.get(rid)
                if h is None:
                    w.writerow([rid, len(s), "", "", f"{0.0:.3f}", f"{0.0:.3f}", 0,
                                meta_map.get(rid, {}).get("flipped","0"), meta_map.get(rid,{}).get("eos","")])
                    continue
                keep = int(h["bitscore"] > args.min_bitscore and h["qcov"] >= args.min_qcov and h["tcov"] >= args.min_tcov)
                if keep:
                    keep_ids.append(rid)
                w.writerow([rid, len(s), h["hmm"], f"{h['bitscore']:.2f}", f"{h['qcov']:.3f}", f"{h['tcov']:.3f}", keep,
                            meta_map.get(rid, {}).get("flipped","0"), meta_map.get(rid,{}).get("eos","")])

        # Write FASTA of keepers
        write_fasta(Path(args.out_fasta), id2seq, keep_ids)
        print(f"[ok] HMMER gate kept {len(keep_ids)} / {len(id2seq)} "
              f"(min_qcov={args.min_qcov}, min_tcov={args.min_tcov}, min_bitscore={args.min_bitscore})")
        print(f"[ok] Wrote: {args.out_fasta}")
        print(f"[ok] Report: {args.out_tsv}")

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
