"""Helpers to assemble per-seed metrics for composite CRISPR scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from Bio import SeqIO

TSV_SEP = "\t"


@dataclass
class SeedFasta:
    seed: str
    ids: List[str]


def _read_tsv(path: Path, *, dtype: Optional[dict[str, str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=TSV_SEP, dtype=dtype, low_memory=False)


def load_ppl(by_seed_dir: Path) -> pd.DataFrame:
    """Load LM perplexity metrics aggregated across seeds."""
    df = _read_tsv(by_seed_dir / "ppl_all.tsv")
    if df.empty:
        return df
    df = df.rename(columns={"id": "sequence_id", "seq_id": "sequence_id"})
    df["sequence_id"] = df["sequence_id"].astype(str)
    df["seed"] = df["seed"].astype(str)
    desired = ["seed", "sequence_id", "ppl", "nll", "n_tokens", "keep"]
    available = [c for c in desired if c in df.columns]
    df = df[available]
    return df.drop_duplicates(subset=["seed", "sequence_id"], keep="first")


def load_dr(by_seed_dir: Path) -> pd.DataFrame:
    """Load DR compatibility probabilities."""
    df = _read_tsv(by_seed_dir / "dr_all.tsv")
    if df.empty:
        return df
    df = df.rename(columns={"id": "sequence_id", "prob": "dr_prob", "logit": "dr_logit"})
    df["sequence_id"] = df["sequence_id"].astype(str)
    df["seed"] = df["seed"].astype(str)
    desired = ["seed", "sequence_id", "dr_prob", "dr_logit", "keep", "n_aa"]
    available = [c for c in desired if c in df.columns]
    df = df[available]
    return df.drop_duplicates(subset=["seed", "sequence_id"], keep="first")


def load_motif(by_seed_dir: Path) -> pd.DataFrame:
    """Load motif filter outcomes (optional)."""
    df = _read_tsv(by_seed_dir / "motif_all.tsv")
    if df.empty:
        return df
    df = df.rename(columns={"seq_id": "sequence_id"})
    df["sequence_id"] = df["sequence_id"].astype(str)
    df["seed"] = df["seed"].astype(str)
    return df.drop_duplicates(subset=["seed", "sequence_id"], keep="first")


def list_seed_dirs(by_seed_dir: Path) -> List[Path]:
    return sorted(p for p in by_seed_dir.iterdir() if p.is_dir() and p.name.startswith("seed"))


def load_pam_reports(by_seed_dir: Path) -> pd.DataFrame:
    """Collect PAM classifier probabilities per seed."""
    rows = []
    for seed_dir in list_seed_dirs(by_seed_dir):
        report_path = seed_dir / "step5_pam_filter" / "report.tsv"
        if not report_path.exists():
            continue
        rep = pd.read_csv(report_path, sep=TSV_SEP)
        rep = rep.rename(columns={"design_id": "sequence_id", "probability": "pam_prob", "logit": "pam_logit"})
        rep["seed"] = seed_dir.name
        rep["sequence_id"] = rep["sequence_id"].astype(str)
        rows.append(rep[["seed", "sequence_id", "pam_prob", "pam_logit", "predicted"]])
    if not rows:
        return pd.DataFrame(columns=["seed", "sequence_id", "pam_prob", "pam_logit", "predicted"])
    df = pd.concat(rows, ignore_index=True)
    return df.drop_duplicates(subset=["seed", "sequence_id"], keep="first")


def load_novelty_fastas(novelty_root: Path) -> List[SeedFasta]:
    """Load the set of sequence IDs that survive the 80% novelty filter for each seed."""
    seed_fastas: List[SeedFasta] = []
    for seed_dir in sorted(p for p in novelty_root.iterdir() if p.is_dir() and p.name.startswith("seed")):
        fasta_path = seed_dir / f"{seed_dir.name}_novel80_full.faa"
        if not fasta_path.exists():
            continue
        ids = [record.id for record in SeqIO.parse(fasta_path, "fasta")]
        seed_fastas.append(SeedFasta(seed=seed_dir.name, ids=ids))
    return seed_fastas


def load_esmfold_scores(roots: Iterable[Path]) -> pd.DataFrame:
    """Load ESMFold pLDDT summaries from one or more roots."""
    frames = []
    for root in roots:
        if not root.exists():
            continue
        for csv_path in root.rglob("esmfold_scores_*_full.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue
            cols = {c.lower(): c for c in df.columns}
            id_col = cols.get("id", df.columns[0])
            df = df.rename(columns={id_col: "sequence_id"})
            df["sequence_id"] = df["sequence_id"].astype(str)
            df["seed"] = df["sequence_id"].str.extract(r"\b(seed\d+)\b", expand=False)
            if df["seed"].isna().all():
                stem = csv_path.stem
                if "seed" in stem:
                    candidate = stem[stem.index("seed") :]
                    candidate = candidate.split("_")[0]
                    df["seed"] = candidate
            plddt_col = None
            for key in ["total_mean_plddt", "mean_plddt", "plddt"]:
                if key in cols:
                    plddt_col = cols[key]
                    break
            if plddt_col is None:
                continue
            out = df[["seed", "sequence_id", plddt_col]].copy()
            out = out.rename(columns={plddt_col: "plddt"})
            frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["seed", "sequence_id", "plddt"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["seed"])
    merged["seed"] = merged["seed"].astype(str)
    merged["sequence_id"] = merged["sequence_id"].astype(str)
    merged = merged.sort_values("plddt", ascending=False)
    return merged.drop_duplicates(subset=["seed", "sequence_id"], keep="first")
