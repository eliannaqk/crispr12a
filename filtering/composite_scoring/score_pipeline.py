#!/usr/bin/env python3
"""Compose composite scores for CRISPR candidates across seeds.

This script uses the novelty-filtered FASTA files as the candidate universe and merges
per-seed LM perplexity, ESMFold pLDDT, DR compatibility, and PAM compatibility metrics.
It outputs a ranked CSV ready for downstream clustering or manual review.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from . import data_loaders

_LOGGER = logging.getLogger(__name__)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((values - mean) / std).fillna(0.0)


def _prepare_candidate_frame(novel_fastas: Iterable[data_loaders.SeedFasta]) -> pd.DataFrame:
    records = []
    for entry in novel_fastas:
        for seq_id in entry.ids:
            records.append({"seed": entry.seed, "sequence_id": seq_id})
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No candidates found in the provided novelty-filtered FASTA directories.")
    return df


def _merge_metrics(base: pd.DataFrame, by_seed_dir: Path, esm_roots: list[Path], seed_filter: Optional[Set[str]] = None) -> pd.DataFrame:
    df = base.copy()

    ppl = data_loaders.load_ppl(by_seed_dir)
    if seed_filter and not ppl.empty:
        ppl = ppl[ppl["seed"].isin(seed_filter)]

    if not ppl.empty:
        _LOGGER.info("Merging PPL metrics (%d rows)", len(ppl))
        df = df.merge(ppl, on=["seed", "sequence_id"], how="left")
    else:
        _LOGGER.warning("ppl_all.tsv not found or empty; PPL scores will be missing.")

    dr = data_loaders.load_dr(by_seed_dir)
    if seed_filter and not dr.empty:
        dr = dr[dr["seed"].isin(seed_filter)]
    if not dr.empty:
        _LOGGER.info("Merging DR metrics (%d rows)", len(dr))
        df = df.merge(dr, on=["seed", "sequence_id"], how="left")
    else:
        _LOGGER.warning("dr_all.tsv not found or empty; DR probabilities will be missing.")

    motif = data_loaders.load_motif(by_seed_dir)
    if seed_filter and not motif.empty:
        motif = motif[motif["seed"].isin(seed_filter)]
    if not motif.empty:
        motif_cols = [c for c in motif.columns if c not in {"seed", "sequence_id"}]
        _LOGGER.info("Merging motif stats (%d rows)", len(motif))
        df = df.merge(motif[["seed", "sequence_id", *motif_cols]], on=["seed", "sequence_id"], how="left")

    pam = data_loaders.load_pam_reports(by_seed_dir)
    if seed_filter and not pam.empty:
        pam = pam[pam["seed"].isin(seed_filter)]
    if not pam.empty:
        _LOGGER.info("Merging PAM probabilities (%d rows)", len(pam))
        df = df.merge(pam, on=["seed", "sequence_id"], how="left", suffixes=("", "_pam"))
    else:
        _LOGGER.warning("No PAM reports found; pam_prob will default to 0.")

    esm = data_loaders.load_esmfold_scores(esm_roots)
    if seed_filter and not esm.empty:
        esm = esm[esm["seed"].isin(seed_filter)]
    if not esm.empty:
        _LOGGER.info("Merging ESMFold scores (%d rows)", len(esm))
        df = df.merge(esm, on=["seed", "sequence_id"], how="left")
    else:
        _LOGGER.warning("No ESMFold score CSVs detected; pLDDT will be missing.")

    return df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute composite CRISPR candidate scores.")
    parser.add_argument("--by-seed-dir", required=True, help="Directory containing per-seed metrics (ppl_all.tsv, dr_all.tsv, motif_all.tsv, seedN subdirs).")
    parser.add_argument("--novelty-root", required=True, help="Directory holding novelty-filtered FASTAs (seedN/seedN_novel80_full.faa).")
    parser.add_argument("--esmfold-root", default="esmfold_seed_outputs", help="Root directory for ESMFold seed outputs.")
    parser.add_argument("--esmfold-baseline-root", default="esmfold_baseline/full_model_single", help="Optional baseline ESMFold scores to merge (if IDs overlap).")
    parser.add_argument("--output-dir", default="metrics", help="Directory to place the output CSV(s).")
    parser.add_argument("--output-prefix", default="composite_scores", help="Prefix for generated CSV files.")
    parser.add_argument("--seeds", nargs="+", help="Subset of seeds to score (e.g., seed2 seed3).")
    parser.add_argument("--weight-lm", type=float, default=0.7, help="Weight for LM (−PPL) z-score.")
    parser.add_argument("--weight-dr", type=float, default=0.3, help="Weight for DR probability (0–1).")
    parser.add_argument("--weight-pam", type=float, default=0.2, help="Weight for PAM probability (0–1).")
    parser.add_argument("--weight-plddt", type=float, default=1.0, help="Multiplier on the pLDDT z-score (default 1.0).")
    parser.add_argument("--include-novelty", action="store_true", help="Include novelty bonus columns if they were merged beforehand.")
    parser.add_argument("--weight-novelty", type=float, default=0.2, help="Weight for novelty bonus if included.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    by_seed_dir = Path(args.by_seed_dir).expanduser().resolve()
    novelty_root = Path(args.novelty_root).expanduser().resolve()
    esm_root = Path(args.esmfold_root).expanduser().resolve()
    esm_baseline_root = Path(args.esmfold_baseline_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.info("Collecting novelty-filtered FASTAs from %s", novelty_root)
    novel_fastas = data_loaders.load_novelty_fastas(novelty_root)
    seed_filter = set(args.seeds) if args.seeds else None
    if seed_filter:
        novel_fastas = [f for f in novel_fastas if f.seed in seed_filter]
        if not novel_fastas:
            raise RuntimeError(f'No novelty FASTAs found for seeds: {sorted(seed_filter)}')
    base_df = _prepare_candidate_frame(novel_fastas)

    _LOGGER.info("Merging metrics for %d candidates", len(base_df))
    merged = _merge_metrics(base_df, by_seed_dir=by_seed_dir, esm_roots=[esm_root, esm_baseline_root], seed_filter=seed_filter)

    # Fill probability defaults where missing
    merged["dr_prob"] = pd.to_numeric(merged.get("dr_prob"), errors="coerce").fillna(0.0)
    merged["pam_prob"] = pd.to_numeric(merged.get("pam_prob"), errors="coerce").fillna(0.0)

    merged["ppl"] = pd.to_numeric(merged.get("ppl"), errors="coerce")
    merged["plddt"] = pd.to_numeric(merged.get("plddt"), errors="coerce")

    raw_cols = [c for c in ["seed", "sequence_id", "plddt", "ppl", "dr_prob", "pam_prob", "dr_logit", "pam_logit", "predicted"] if c in merged.columns]
    raw_cols += [c for c in ["overall_pass", "WED_pass", "PI_pass", "NUC_pass", "BH_pass", "LID_pass"] if c in merged.columns and c not in raw_cols]
    raw_df = merged[raw_cols].copy()

    merged["z_plddt"] = _zscore(merged["plddt"])
    merged["z_negppl"] = _zscore(-merged["ppl"])

    merged["novelty_bonus"] = 0.0
    if args.include_novelty:
        nov_cols = []
        for col in ["novelty_bonus", "segN_diversity", "segC_diversity", "whole_maxid"]:
            if col in merged.columns:
                nov_cols.append(col)
        if "novelty_bonus" not in nov_cols:
            whole = 1.0 - pd.to_numeric(merged.get("whole_maxid"), errors="coerce").fillna(100.0) / 100.0
            segn = pd.to_numeric(merged.get("segN_diversity"), errors="coerce").fillna(0.0)
            segc = pd.to_numeric(merged.get("segC_diversity"), errors="coerce").fillna(0.0)
            merged["novelty_bonus"] = 0.5 * whole + 0.25 * segn + 0.25 * segc
        merged["novelty_bonus"] = pd.to_numeric(merged.get("novelty_bonus"), errors="coerce").fillna(0.0)
    else:
        merged["novelty_bonus"] = 0.0

    merged["score"] = (
        args.weight_plddt * merged["z_plddt"]
        + args.weight_lm * merged["z_negppl"]
        + args.weight_dr * merged["dr_prob"]
        + args.weight_pam * merged["pam_prob"]
        + (args.weight_novelty * merged["novelty_bonus"] if args.include_novelty else 0.0)
    )

    merged["rank_within_seed"] = merged.groupby("seed")["score"].rank(method="first", ascending=False)

    out_cols = [
        "seed",
        "sequence_id",
        "score",
        "rank_within_seed",
        "plddt",
        "ppl",
        "dr_prob",
        "pam_prob",
        "z_plddt",
        "z_negppl",
        "novelty_bonus",
    ]
    for candidate in ["dr_logit", "pam_logit", "predicted", "overall_pass", "WED_pass", "PI_pass", "NUC_pass", "BH_pass", "LID_pass"]:
        if candidate in merged.columns and candidate not in out_cols:
            out_cols.append(candidate)

    out_cols = [c for c in out_cols if c in merged.columns]
    final_df = merged[out_cols].sort_values(["seed", "score"], ascending=[True, False])
    if seed_filter:
        final_df = final_df[final_df["seed"].isin(seed_filter)]

    if seed_filter:
        raw_df = raw_df[raw_df["seed"].isin(seed_filter)]
    raw_df = raw_df.sort_values(["seed", "sequence_id"]).reset_index(drop=True)
    raw_path = output_dir / f"{args.output_prefix}_raw_metrics.csv"
    raw_df.to_csv(raw_path, index=False)
    _LOGGER.info("Wrote raw metrics to %s", raw_path)

    all_path = output_dir / f"{args.output_prefix}_all_seeds.csv"
    final_df.to_csv(all_path, index=False)
    _LOGGER.info("Wrote composite rankings to %s", all_path)

    for seed, seed_df in final_df.groupby("seed"):
        seed_path = output_dir / f"{args.output_prefix}_{seed}.csv"
        seed_df.to_csv(seed_path, index=False)
        _LOGGER.info("Seed %s: %d sequences → %s", seed, len(seed_df), seed_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
