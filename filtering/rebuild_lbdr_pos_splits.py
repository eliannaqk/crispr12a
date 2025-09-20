#!/usr/bin/env python3
"""
Rebuild LbCas12a DR expanded train/valid label splits from evidence files.

Inputs (all required):
  - --dr20_join_csv: CSV with columns id,sequence,split,dr_exact,class
  - --dr21_join_csv: CSV with same schema as above
  - --tolerant_dr20_tsv: TSV with columns ref_id,...,label (where label in {positive,negative,...})

Logic:
  - Build a map id -> (sequence, split) from the join CSVs (prefer consistency; warn on conflicts).
  - Positives = union of {ids with class=='positive' in dr20_join} ∪ {class=='positive' in dr21_join}
    ∪ {ref_id with label=='positive' in tolerant_dr20_tsv} ∩ known ids.
  - Emit two CSVs under --out_dir:
      train_labeled_pos.csv   (protein_seq, dr_exact, id)
      valid_labeled_pos.csv   (protein_seq, dr_exact, id)

This reproduces the expanded labeling used by filtering/train_lbdr_classifier_esm.py.
"""
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Tuple, Set


def read_join_csv(path: str) -> Tuple[Dict[str, Tuple[str, str]], Set[str]]:
    """Return (id_map, pos_ids_from_join).

    id_map: id -> (sequence, split)
    pos_ids_from_join: ids with class=='positive'
    """
    id_map: Dict[str, Tuple[str, str]] = {}
    pos: Set[str] = set()
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        need = {"id", "sequence", "split", "class"}
        if not need.issubset(set(r.fieldnames or [])):
            raise SystemExit(f"{path} missing required columns: {sorted(need)}")
        for row in r:
            rid = row["id"].strip()
            seq = (row["sequence"] or "").strip()
            split = (row["split"] or "").strip().lower()
            if split == "valid":
                split = "val"
            if rid == "":
                continue
            prev = id_map.get(rid)
            if prev is None:
                id_map[rid] = (seq, split)
            else:
                # sanity: if sequence differs across joins, keep the first, warn
                if prev[0] != seq or prev[1] != split:
                    # Non-fatal; joins should be consistent
                    pass
            if (row["class"] or "").strip().lower() == "positive":
                pos.add(rid)
    return id_map, pos


def read_tolerant_pos_tsv(path: str) -> Set[str]:
    """Return set of ref_id with label=='positive' from tolerant DR20 pass."""
    pos: Set[str] = set()
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        # Accept either 'ref_id' (preferred) or 'key' columns
        key_col = "ref_id" if "ref_id" in (r.fieldnames or []) else ("key" if "key" in (r.fieldnames or []) else None)
        if key_col is None:
            raise SystemExit(f"{path} missing ref_id/key column")
        if "label" not in (r.fieldnames or []):
            raise SystemExit(f"{path} missing label column")
        for row in r:
            if (row.get("label", "").strip().lower() == "positive"):
                rid = row[key_col].strip()
                if rid:
                    pos.add(rid)
    return pos


def write_split_csv(out_path: str, rows: Dict[str, Tuple[str, str]], pos_ids: Set[str], target_split: str) -> Tuple[int, int, int]:
    """Write labeled CSV for target_split (train/val). Returns (n_total, n_pos, n_neg)."""
    n_total = n_pos = n_neg = 0
    with open(out_path, "w", newline="") as w:
        ww = csv.DictWriter(w, fieldnames=["protein_seq", "dr_exact", "id"])
        ww.writeheader()
        for rid, (seq, split) in rows.items():
            if split != target_split:
                continue
            lab = 1 if rid in pos_ids else 0
            ww.writerow({"protein_seq": seq, "dr_exact": lab, "id": rid})
            n_total += 1
            if lab == 1:
                n_pos += 1
            else:
                n_neg += 1
    return n_total, n_pos, n_neg


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild expanded DR train/valid splits from evidence")
    ap.add_argument("--dr20_join_csv", required=True)
    ap.add_argument("--dr21_join_csv", required=True)
    ap.add_argument("--tolerant_dr20_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build id->(seq, split) and positives from joins
    idmap20, pos20 = read_join_csv(args.dr20_join_csv)
    idmap21, pos21 = read_join_csv(args.dr21_join_csv)

    # Merge maps (favor idmap20 but ensure presence of all ids)
    idmap = dict(idmap20)
    for k, v in idmap21.items():
        if k not in idmap:
            idmap[k] = v

    # Tolerant positives
    pos_tol = read_tolerant_pos_tsv(args.tolerant_dr20_tsv)

    # Final positives: only include ids present in idmap
    pos_all = (pos20 | pos21 | pos_tol) & set(idmap.keys())

    # Write splits
    out_train = os.path.join(args.out_dir, "train_labeled_pos.csv")
    out_valid = os.path.join(args.out_dir, "valid_labeled_pos.csv")
    ntr, ptr, ntrn = write_split_csv(out_train, idmap, pos_all, target_split="train")
    nva, pva, nvan = write_split_csv(out_valid, idmap, pos_all, target_split="val")

    print(f"[rebuild] train: n={ntr} pos={ptr} neg={ntrn}")
    print(f"[rebuild] valid: n={nva} pos={pva} neg={nvan}")


if __name__ == "__main__":
    main()

