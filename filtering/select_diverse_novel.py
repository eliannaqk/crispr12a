#!/usr/bin/env python3
"""
select_diverse_novel.py — Paper-consistent selection pipeline

Pipeline stages (drop-in for design flow):
  1) Enforce novelty (<80% ID to known) with coverage guard
  2) Compute segment-level diversity scores (e.g., 1–25/26–end and 1–514/515–end)
  3) Optionally gate novelty on the generated region based on seed type
  4) Self-cluster survivors at 90% identity (MMseqs easy-cluster)
  5) Rank within clusters and pick one representative using: LM perplexity (lower better),
     ESMFold mean pLDDT (higher better), optional DR-binding probability (higher better),
     and a mild novelty bonus.

Notes
 - MMseqs2 is invoked if available; otherwise we try apptainer/singularity container runners
   for the official soedinglab/mmseqs2 image.
 - For whole-protein novelty, we require both query and target coverage >= -c (cov-mode 0) so
   %ID reflects broad homology rather than a short local patch.
 - For segment (short) queries vs full-length known DB, we enforce coverage on the QUERY only
   (cov-mode 1) to avoid penalizing long target sequences.

Environment (per repo policy)
 - Prefer running in conda env: bio-utils (bioinformatics tools, MMseqs).
 - GPU not required. If you generate ESMFold pLDDT here, use oc-opencrispr-esm separately.

Example
  python filtering/select_diverse_novel.py \
    --generated generated.faa \
    --known known.faa \
    --metrics metrics.tsv \
    --outdir eval_outputs/selection_run \
    --seed-map seeds.tsv

Inputs
 - generated.faa: FASTA of candidates (headers used as IDs; up to first whitespace)
 - known.faa: FASTA of known CRISPR proteins (novelty guard DB)
 - metrics.tsv: TSV with columns: id, ppl, plddt, [dr_prob]
 - seeds.tsv (optional): TSV mapping id -> seed (1..4) to enable seed-aware gating.

Outputs (written under --outdir)
 - novelty/gen_vs_known.m8, novelty/keep_ids.txt, novelty/generated.novel80.faa
 - segments/<scheme>/segment_diversity.tsv (scheme '25' and '514' by default)
 - clusters/clusters_rep_seq.fasta, clusters/clusters_cluster.tsv
 - selection/cluster_winners.tsv (+ selection/cluster_winners.faa)
 - selection/selection_log.txt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import pandas as pd


# ----------------------------- Utility helpers ----------------------------- #


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def fasta_iter(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (id, seq) pairs from FASTA; id is header up to first whitespace."""
    hdr = None
    seq_chunks: List[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr.split()[0], "".join(seq_chunks)
                hdr = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if hdr is not None:
            yield hdr.split()[0], "".join(seq_chunks)


def write_fasta(records: Iterable[Tuple[str, str]], out_path: Path) -> None:
    with out_path.open("w") as out:
        for rid, seq in records:
            out.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i : i + 80] + "\n")


def try_find_mmseqs_runner() -> List[str]:
    """Return a command prefix to run mmseqs, preferring native, else apptainer/singularity container.

    Examples:
      ['mmseqs']
      ['apptainer', 'exec', 'docker://soedinglab/mmseqs2:latest', 'mmseqs']
    """
    if shutil.which("mmseqs"):
        return ["mmseqs"]
    for cntr in ("apptainer", "singularity"):
        if shutil.which(cntr):
            return [cntr, "exec", "docker://soedinglab/mmseqs2:latest", "mmseqs"]
    raise RuntimeError(
        "mmseqs2 not found and neither apptainer nor singularity is available."
    )


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    log(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def mmseqs_createdb(runner: List[str], fasta: Path, db: Path) -> None:
    run(runner + ["createdb", str(fasta), str(db)])


def mmseqs_search(
    runner: List[str],
    query_db: Path,
    target_db: Path,
    out_pref: Path,
    tmp_dir: Path,
    min_seq_id: float = 0.0,
    cov_mode: int = 0,
    cov: float = 0.8,
    threads: int = 8,
) -> None:
    ensure_dir(tmp_dir)
    cmd = (
        runner
        + [
            "search",
            str(query_db),
            str(target_db),
            str(out_pref),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
            "--cov-mode",
            str(cov_mode),
            "-c",
            str(cov),
            "--threads",
            str(threads),
        ]
    )
    run(cmd)


def mmseqs_convertalis(
    runner: List[str], query_db: Path, target_db: Path, alignment: Path, out_m8: Path
) -> None:
    fmt = "query,target,pident,qcov,tcov,alnlen,evalue,bits"
    cmd = runner + [
        "convertalis",
        str(query_db),
        str(target_db),
        str(alignment),
        str(out_m8),
        "--format-output",
        fmt,
    ]
    run(cmd)


def parse_m8_max_identity(m8_path: Path) -> Dict[str, float]:
    """Return max %identity per query id from an m8 file. If no file or empty, returns {}."""
    if not m8_path.exists() or m8_path.stat().st_size == 0:
        return {}
    df = pd.read_csv(
        m8_path,
        sep="\t",
        header=None,
        names=["q", "t", "pid", "qcov", "tcov", "alnlen", "evalue", "bits"],
    )
    g = df.groupby("q")["pid"].max()
    return g.to_dict()


# ----------------------------- Segment slicing ----------------------------- #


@dataclass(frozen=True)
class SegmentScheme:
    name: str
    n1: int
    n2: Optional[int]  # inclusive end for N segment; None -> empty
    c1: Optional[int]  # start for C segment; None -> empty

    def describe(self) -> str:
        nspan = f"{self.n1}-{self.n2}" if self.n2 else "(none)"
        cspan = f"{self.c1}-end" if self.c1 else "(none)"
        return f"N={nspan}, C={cspan}"


SCHEMES_DEFAULT = [
    SegmentScheme(name="25", n1=1, n2=25, c1=26),
    SegmentScheme(name="514", n1=1, n2=514, c1=515),
]


def slice_records(records: List[Tuple[str, str]], scheme: SegmentScheme) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    n_parts: List[Tuple[str, str]] = []
    c_parts: List[Tuple[str, str]] = []
    for rid, seq in records:
        s = seq
        if scheme.n2 and scheme.n1 <= len(s):
            nseq = s[max(0, scheme.n1 - 1) : min(len(s), scheme.n2)]
            if nseq:
                n_parts.append((f"{rid}|segN", nseq))
        if scheme.c1 and scheme.c1 <= len(s):
            cseq = s[scheme.c1 - 1 :]
            if cseq:
                c_parts.append((f"{rid}|segC", cseq))
    return n_parts, c_parts


# ------------------------- Seed-aware novelty gating ----------------------- #


# Seed windows for generated region (1-based, inclusive). Seeds 2/4 are length-dependent:
#  - Seed 1: seed=1..514 => generated=515..L
#  - Seed 3: seed=1..25  => generated=26..L
# For seeds 2 and 4 the generated N-terminus is variable-length:
#  - Seed 2: seed=714..L => generated=1..(L-713)
#  - Seed 4: seed=1203..L => generated=1..(L-1203)


def load_seed_map(path: Optional[Path], ids: Iterable[str]) -> Dict[str, int]:
    """Load mapping id -> seed (1..4). If not provided, try to parse from id (e.g., 'seed1')."""
    id_list = list(ids)
    mapping: Dict[str, int] = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            toks = re.split(r"[\t,]", line)
            if len(toks) < 2:
                continue
            rid, seed_str = toks[0].strip(), toks[1].strip()
            try:
                s = int(re.findall(r"\d+", seed_str)[0])
                mapping[rid] = s
            except Exception:
                continue
    # Heuristic: parse 'seedN' patterns in IDs
    for rid in id_list:
        if rid in mapping:
            continue
        m = re.search(r"seed[_-]?(\d)", rid, flags=re.IGNORECASE)
        if m:
            try:
                mapping[rid] = int(m.group(1))
            except Exception:
                pass
    return mapping


def slice_generated_region_per_seed(records: List[Tuple[str, str]], seed: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rid, seq in records:
        L = len(seq)
        if seed == 1:
            s1, e1 = 515, L
        elif seed == 3:
            s1, e1 = 26, L
        elif seed == 2:
            # Variable-length: generated 1..(L-713)
            s1, e1 = 1, max(0, L - 713)
        elif seed == 4:
            # Variable-length: generated 1..(L-1203)
            s1, e1 = 1, max(0, L - 1203)
        else:
            continue
        if e1 >= s1 and s1 >= 1:
            frag = seq[s1 - 1 : e1]
            if frag:
                out.append((rid, frag))
    return out


# ------------------------------ Main pipeline ------------------------------ #


def novelty_filter_whole(
    runner: List[str],
    gen_faa: Path,
    known_faa: Path,
    out_dir: Path,
    threads: int,
    cov: float,
    cutoff_id: float,
) -> Tuple[Path, Dict[str, float]]:
    """Whole-protein novelty: keep queries whose max %ID < cutoff_id (e.g., 80).

    Returns (survivor_fasta_path, whole_maxid_table).
    """
    nov_dir = out_dir / "novelty"
    ensure_dir(nov_dir)
    gen_db = nov_dir / "genDB"
    kn_db = nov_dir / "knDB"
    out_pref = nov_dir / "gen_vs_known"
    out_m8 = nov_dir / "gen_vs_known.m8"
    log("[Novelty] Building MMseqs DBs (generated, known)")
    mmseqs_createdb(runner, gen_faa, gen_db)
    mmseqs_createdb(runner, known_faa, kn_db)
    log(f"[Novelty] Search with cov-mode 0 (q& t) -c {cov}")
    mmseqs_search(runner, gen_db, kn_db, out_pref, nov_dir / "tmp", cov_mode=0, cov=cov, threads=threads)
    mmseqs_convertalis(runner, gen_db, kn_db, out_pref, out_m8)
    max_id = parse_m8_max_identity(out_m8)
    # Decide keepers
    keep_ids: List[str] = []
    all_ids = [rid for rid, _ in fasta_iter(gen_faa)]
    for rid in all_ids:
        pid = max_id.get(rid, -1.0)  # -1 means no hit -> keep
        if pid < cutoff_id:
            keep_ids.append(rid)
    (nov_dir / "keep_ids.txt").write_text("\n".join(keep_ids) + ("\n" if keep_ids else ""))
    # Write survivors
    survivors = [rec for rec in fasta_iter(gen_faa) if rec[0] in set(keep_ids)]
    out_faa = nov_dir / f"generated.novel{int(cutoff_id)}.faa"
    write_fasta(survivors, out_faa)
    log(f"[Novelty] Survivors: {len(survivors)} / {len(all_ids)} kept (<{cutoff_id}% ID)")
    return out_faa, max_id


def seed_aware_generated_region_gating(
    runner: List[str],
    survivor_faa: Path,
    known_faa: Path,
    out_dir: Path,
    seed_map_path: Optional[Path],
    threads: int,
    cutoff_id: float,
    cov_q_only: float,
) -> Path:
    """For each seed type, enforce novelty on the generated region only: max %ID (query coverage >= cov_q_only) < cutoff_id.

    Returns path to FASTA with seed-aware survivors (subset of survivor_faa). If no seeds detected, returns survivor_faa.
    """
    records = list(fasta_iter(survivor_faa))
    if not records:
        return survivor_faa
    seed_map = load_seed_map(seed_map_path, (rid for rid, _ in records))
    if not seed_map:
        log("[SeedGate] No seed map detected; skipping seed-aware gating.")
        return survivor_faa

    gate_dir = out_dir / "seed_gate"
    ensure_dir(gate_dir)

    # Build known DB once
    kn_db = gate_dir / "knDB"
    mmseqs_createdb(runner, known_faa, kn_db)

    keep: set[str] = set()
    all_ids = [rid for rid, _ in records]
    for seed in sorted(set(seed_map.values())):
        seed_ids = [rid for rid in all_ids if seed_map.get(rid) == seed]
        subset = [(rid, seq) for rid, seq in records if rid in seed_ids]
        if not subset:
            continue
        sliced = slice_generated_region_per_seed(subset, seed)
        if not sliced:
            log(f"[SeedGate] Seed {seed}: no valid generated fragments (length constraints); skipping.")
            continue
        q_faa = gate_dir / f"seed{seed}_generated_segments.faa"
        write_fasta(sliced, q_faa)
        q_db = gate_dir / f"seed{seed}_genDB"
        mmseqs_createdb(runner, q_faa, q_db)
        out_pref = gate_dir / f"seed{seed}_vs_known"
        out_m8 = gate_dir / f"seed{seed}_vs_known.m8"
        # For short query segments vs full-length known, enforce QUERY coverage only
        mmseqs_search(runner, q_db, kn_db, out_pref, gate_dir / f"tmp_seed{seed}", cov_mode=1, cov=cov_q_only, threads=threads)
        mmseqs_convertalis(runner, q_db, kn_db, out_pref, out_m8)
        max_id = parse_m8_max_identity(out_m8)
        for rid, _ in subset:
            pid = max_id.get(rid, -1.0)
            if pid < cutoff_id:
                keep.add(rid)
    survivors = [(rid, seq) for rid, seq in records if rid in keep]
    out_faa = gate_dir / f"generated.seed_gated_novel{int(cutoff_id)}.faa"
    write_fasta(survivors, out_faa)
    log(f"[SeedGate] Survivors after seed-aware gating: {len(survivors)} / {len(records)}")
    return out_faa


def compute_segment_diversity(
    runner: List[str],
    gen_faa: Path,
    known_faa: Path,
    out_dir: Path,
    schemes: List[SegmentScheme],
    threads: int,
) -> Dict[str, Path]:
    """For each scheme, compute per-sequence segN/segC maxID vs known DB and write TSV.

    We query short segments vs full-length known using cov-mode 1 (query-only coverage).
    Returns mapping scheme.name -> TSV path.
    """
    seg_root = out_dir / "segments"
    ensure_dir(seg_root)
    records = list(fasta_iter(gen_faa))
    rec_map = {rid: seq for rid, seq in records}
    tsvs: Dict[str, Path] = {}

    # Create known DB once
    kn_db = seg_root / "knDB"
    mmseqs_createdb(runner, known_faa, kn_db)

    for scheme in schemes:
        sdir = seg_root / scheme.name
        ensure_dir(sdir)
        n_parts, c_parts = slice_records(records, scheme)
        qn_faa = sdir / "gen_segN.faa"
        qc_faa = sdir / "gen_segC.faa"
        write_fasta(n_parts, qn_faa)
        write_fasta(c_parts, qc_faa)
        qn_db = sdir / "genNDB"
        qc_db = sdir / "genCDB"
        mmseqs_createdb(runner, qn_faa, qn_db)
        mmseqs_createdb(runner, qc_faa, qc_db)
        outN_pref = sdir / "segN_vs_known"
        outC_pref = sdir / "segC_vs_known"
        outN_m8 = sdir / "segN.m8"
        outC_m8 = sdir / "segC.m8"
        # Query-only coverage for short segments
        mmseqs_search(runner, qn_db, kn_db, outN_pref, sdir / "tmpN", cov_mode=1, cov=0.8, threads=threads)
        mmseqs_search(runner, qc_db, kn_db, outC_pref, sdir / "tmpC", cov_mode=1, cov=0.8, threads=threads)
        mmseqs_convertalis(runner, qn_db, kn_db, outN_pref, outN_m8)
        mmseqs_convertalis(runner, qc_db, kn_db, outC_pref, outC_m8)

        # Build TSV: id, segN_maxid, segC_maxid, segN_diversity, segC_diversity
        tN = parse_m8_max_identity(outN_m8)
        tC = parse_m8_max_identity(outC_m8)
        rows = []
        for rid, _ in records:
            nmax = tN.get(f"{rid}|segN", -1.0)
            cmax = tC.get(f"{rid}|segC", -1.0)
            # Missing -> -1 means no hit
            nmax = 0.0 if nmax < 0 else float(nmax)
            cmax = 0.0 if cmax < 0 else float(cmax)
            rows.append(
                {
                    "id": rid,
                    "segN_maxid": nmax,
                    "segC_maxid": cmax,
                    "segN_diversity": 1.0 - nmax / 100.0,
                    "segC_diversity": 1.0 - cmax / 100.0,
                }
            )
        df = pd.DataFrame(rows)
        tsv_path = sdir / "segment_diversity.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)
        tsvs[scheme.name] = tsv_path
        log(f"[Segments:{scheme.name}] Wrote {tsv_path} ({len(df)} rows)")
    return tsvs


def mmseqs_easy_cluster(
    runner: List[str], gen_faa: Path, out_dir: Path, min_seq_id: float, cov: float, threads: int
) -> Tuple[Path, Path]:
    clus_dir = out_dir / "clusters"
    ensure_dir(clus_dir)
    out_pref = clus_dir / "clusters"
    tmp_dir = clus_dir / "tmp"
    ensure_dir(tmp_dir)
    cmd = (
        runner
        + [
            "easy-cluster",
            str(gen_faa),
            str(out_pref),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
            "--cov-mode",
            "0",
            "-c",
            str(cov),
            "--cluster-mode",
            "0",
            "--threads",
            str(threads),
        ]
    )
    run(cmd)
    rep_faa = clus_dir / "clusters_rep_seq.fasta"
    cl_tsv = clus_dir / "clusters_cluster.tsv"
    log(f"[Cluster] reps: {rep_faa} map: {cl_tsv}")
    return rep_faa, cl_tsv


def zscore(s: pd.Series) -> pd.Series:
    mu = float(s.mean()) if len(s) else 0.0
    sd = float(s.std(ddof=0)) if len(s) else 1.0
    if sd == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def select_representatives(
    cl_tsv: Path,
    metrics_tsv: Path,
    segment_tsv: Path,
    whole_maxid: Dict[str, float],
    plddt_col: str,
    ppl_col: str,
    dr_col: Optional[str],
    plddt_min: float,
    ppl_max: Optional[float],
    w_ppl: float,
    w_dr: float,
    w_nov: float,
    nov_alpha: float,
    nov_beta: float,
    nov_gamma: float,
    out_dir: Path,
) -> Tuple[Path, Path]:
    sel_dir = out_dir / "selection"
    ensure_dir(sel_dir)
    cl = pd.read_csv(cl_tsv, sep="\t", header=None, names=["rep", "id"])
    m = pd.read_csv(metrics_tsv, sep="\t|,", engine="python")
    if "id" not in m.columns:
        raise ValueError("metrics file must have an 'id' column")
    if plddt_col not in m.columns:
        raise ValueError(f"metrics file missing plddt column '{plddt_col}'")
    if ppl_col not in m.columns:
        raise ValueError(f"metrics file missing ppl column '{ppl_col}'")
    sd = pd.read_csv(segment_tsv, sep="\t")
    df = cl.merge(m, on="id", how="left").merge(sd, on="id", how="left")
    # Fill missing metrics safely
    if dr_col and dr_col not in df.columns:
        log(f"[Select] DR column '{dr_col}' not in metrics; treating as 0.")
        df[dr_col] = 0.0
    for col in [dr_col, "segN_diversity", "segC_diversity"]:
        if col and col not in df.columns:
            df[col] = 0.0
    # Whole-protein maxid (%). Missing -> 0.
    df["whole_maxid"] = [whole_maxid.get(i, 0.0) for i in df["id"]]

    # Gates
    df = df[df[plddt_col] >= plddt_min]
    if ppl_max is not None:
        df = df[df[ppl_col] <= ppl_max]

    # Scores
    df["z_plddt"] = zscore(df[plddt_col])
    df["z_negppl"] = zscore(-df[ppl_col])
    nov_bonus = (
        nov_alpha * (1.0 - df["whole_maxid"] / 100.0)
        + nov_beta * df["segN_diversity"].fillna(0.0)
        + nov_gamma * df["segC_diversity"].fillna(0.0)
    )
    df["score"] = df["z_plddt"] + w_ppl * df["z_negppl"] + (w_dr * df.get(dr_col, pd.Series(0.0, index=df.index))) + w_nov * nov_bonus

    winners = (
        df.sort_values(["rep", "score"], ascending=[True, False])
        .groupby("rep", as_index=False)
        .first()
    )

    out_tsv = sel_dir / "cluster_winners.tsv"
    keep_cols = [
        "id",
        "rep",
        "score",
        plddt_col,
        ppl_col,
        dr_col if dr_col else None,
        "segN_diversity",
        "segC_diversity",
        "whole_maxid",
    ]
    keep_cols = [c for c in keep_cols if c]
    winners[keep_cols].to_csv(out_tsv, sep="\t", index=False)
    log(f"[Select] Wrote winners TSV: {out_tsv} (n={len(winners)})")

    # Optionally write winners FASTA by extracting from the clustered representative file map
    return out_tsv, sel_dir / "cluster_winners.faa"


def fasta_subseq_by_ids(src_faa: Path, ids: Iterable[str], out_faa: Path) -> None:
    ids_set = set(ids)
    write_fasta(((rid, seq) for rid, seq in fasta_iter(src_faa) if rid in ids_set), out_faa)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Novelty+Diversity+Clustering+Selection pipeline")
    ap.add_argument("--generated", required=True, help="FASTA of generated sequences")
    ap.add_argument("--known", required=True, help="FASTA of known sequences (novelty DB)")
    ap.add_argument("--metrics", required=True, help="TSV/CSV with columns: id, ppl, plddt, [dr_prob]")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--seed-map", default=None, help="Optional TSV: id\tseed(1..4) for seed-aware gating")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cov", type=float, default=0.8, help="Coverage threshold for whole-protein novelty (q & t)")
    ap.add_argument("--seg-cov", type=float, default=0.8, help="Query coverage for segment searches (cov-mode 1)")
    ap.add_argument("--novelty-cutoff", type=float, default=80.0, help="Max %%ID to known allowed (< cutoff)")
    ap.add_argument("--cluster-id", type=float, default=0.90, help="Clustering identity threshold (e.g., 0.90)")
    ap.add_argument("--plddt-col", default="plddt")
    ap.add_argument("--ppl-col", default="ppl")
    ap.add_argument("--dr-col", default="dr_prob")
    ap.add_argument("--plddt-min", type=float, default=80.0)
    ap.add_argument("--ppl-max", type=float, default=None)
    ap.add_argument("--w-ppl", type=float, default=0.7)
    ap.add_argument("--w-dr", type=float, default=0.3)
    ap.add_argument("--w-nov", type=float, default=0.2)
    ap.add_argument("--nov-alpha", type=float, default=0.5)
    ap.add_argument("--nov-beta", type=float, default=0.25)
    ap.add_argument("--nov-gamma", type=float, default=0.25)
    args = ap.parse_args(argv)

    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    # Resolve env runner for mmseqs2
    try:
        runner = try_find_mmseqs_runner()
    except RuntimeError as e:
        log(f"[Error] {e}")
        return 2
    log(f"[Env] Using MMseqs runner: {' '.join(runner)}")

    gen_faa = Path(args.generated)
    known_faa = Path(args.known)
    metrics_tsv = Path(args.metrics)
    seed_map = Path(args.seed_map) if args.seed_map else None

    # 1) Whole-protein novelty (<80% ID to known)
    survivor_faa, whole_maxid = novelty_filter_whole(
        runner, gen_faa, known_faa, out_dir, threads=args.threads, cov=args.cov, cutoff_id=args.novelty_cutoff
    )

    # 2) Seed-aware generated-region novelty gating (optional)
    survivor_seed_faa = seed_aware_generated_region_gating(
        runner,
        survivor_faa,
        known_faa,
        out_dir,
        seed_map,
        threads=args.threads,
        cutoff_id=args.novelty_cutoff,
        cov_q_only=args.seg_cov,
    )

    # 3) Segment-level diversity (two schemes by default)
    seg_tsvs = compute_segment_diversity(
        runner,
        survivor_seed_faa,
        known_faa,
        out_dir,
        schemes=SCHEMES_DEFAULT,
        threads=args.threads,
    )
    # Prefer the 514 scheme for Cas12a seeds as default in scoring
    seg_tsv = seg_tsvs.get("514") or next(iter(seg_tsvs.values()))

    # 4) Cluster survivors (self-cluster at 90% ID)
    rep_faa, cl_tsv = mmseqs_easy_cluster(
        runner, survivor_seed_faa, out_dir, min_seq_id=args.cluster_id, cov=args.cov, threads=args.threads
    )

    # 5) Rank within clusters and pick winners
    winners_tsv, winners_faa = select_representatives(
        cl_tsv,
        metrics_tsv,
        seg_tsv,
        whole_maxid,
        plddt_col=args.plddt_col,
        ppl_col=args.ppl_col,
        dr_col=args.dr_col,
        plddt_min=args.plddt_min,
        ppl_max=args.ppl_max,
        w_ppl=args.w_ppl,
        w_dr=args.w_dr,
        w_nov=args.w_nov,
        nov_alpha=args.nov_alpha,
        nov_beta=args.nov_beta,
        nov_gamma=args.nov_gamma,
        out_dir=out_dir,
    )
    # Extract FASTA for winners from survivor set
    # Read chosen IDs
    wdf = pd.read_csv(winners_tsv, sep="\t")
    fasta_subseq_by_ids(survivor_seed_faa, wdf["id"].tolist(), winners_faa)
    log(f"[Done] Winners FASTA: {winners_faa}")

    # Minimal log summary
    sel_log = out_dir / "selection" / "selection_log.txt"
    with sel_log.open("w") as f:
        f.write("Selection pipeline completed.\n")
        f.write(f"Whole-protein novelty cutoff: <{args.novelty_cutoff}% ID (cov-mode 0, -c {args.cov})\n")
        f.write(f"Segment query coverage (-c) for short segments: {args.seg_cov} (cov-mode 1)\n")
        f.write(f"Cluster ID threshold: {args.cluster_id}\n")
        f.write(f"Scoring weights: w_ppl={args.w_ppl}, w_dr={args.w_dr}, w_nov={args.w_nov}; nov(alpha,beta,gamma)={args.nov_alpha},{args.nov_beta},{args.nov_gamma}\n")
        f.write(f"Gates: plddt_min={args.plddt_min}; ppl_max={args.ppl_max}\n")
        f.write(f"Winners TSV: {winners_tsv}\n")
        f.write(f"Winners FASTA: {winners_faa}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
