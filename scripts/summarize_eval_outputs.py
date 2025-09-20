#!/usr/bin/env python3
"""
Summarize evaluation outputs under eval_outputs/ into a CSV.

For each run (container directory), pick the best checkpoint by lowest
val mean loss (fallback to test mean loss if val is missing), and extract:
 - run_id (container path)
 - best_checkpoint (dirname, e.g., ba6000)
 - best_checkpoint_path (HF path if present in logs is not available; use run container + ckpt)
 - data_val_path, data_test_path (from run logs)
 - random_reverse (if detectable in logs; else blank)
 - bucketed_training (if detectable; else blank)
 - token_bucketed_batches (if detectable; else blank)
 - learning_rate (unknown from eval logs; blank)
 - val_mean_loss (model under test)
 - baseline_val_mean_loss (computed from Baseline PPL if present)
 - test_mean_loss (model under test)
 - baseline_test_mean_loss (computed from Baseline PPL if present)

Outputs scripts/eval_outputs_summary.csv relative to repo root by default
unless an output path is supplied as argv[1].
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
# Default eval roots to scan; can be overridden by argv paths
DEFAULT_EVAL_ROOTS = [
    ROOT / "eval_outputs",
    ROOT / "eval_outputs_gpu_aug21",
    ROOT / "eval_outputs_cpu_aug19",
    ROOT / "eval_outputs_cpu_aug21",
    ROOT / "eval_outputs_cpu_aug18",
    ROOT / "eval_outputs_cpu",
]


VAL_BLOCK_START = re.compile(r"^\s*Split\s*:\s*val\s*$")
TEST_BLOCK_START = re.compile(r"^\s*Split\s*:\s*test\s*$")
FILE_LINE = re.compile(r"^\s*File\s*:\s*(?P<path>.+)$")
MEAN_LOSS_PPL = re.compile(r"^\s*Mean loss\s*:\s*(?P<loss>[-+0-9.eE]+)\s*\|\s*PPL:\s*(?P<ppl>[-+0-9.eE]+)")
BASELINE_SPLIT = re.compile(r"^\s*Split\s*:\s*(?P<split>val|test)\s*$")
BASELINE_PPL = re.compile(r"^\s*Baseline PPL\s*:\s*(?P<ppl>[-+0-9.eE]+)")


def parse_runlog(runlog: Path) -> Dict[str, object]:
    """Parse a run.log and return metrics per split and config hints.

    Returns a dict with keys:
      val_loss, val_ppl, val_file
      test_loss, test_ppl, test_file
      baseline_val_ppl, baseline_test_ppl
      random_reverse, bucketed_training, token_bucketed_batches
    """
    res: Dict[str, object] = {
        "val_loss": None,
        "val_ppl": None,
        "val_file": None,
        "test_loss": None,
        "test_ppl": None,
        "test_file": None,
        "baseline_val_ppl": None,
        "baseline_test_ppl": None,
        "random_reverse": None,
        "bucketed_training": None,
        "token_bucketed_batches": None,
    }

    try:
        txt = runlog.read_text(errors="ignore").splitlines()
    except Exception:
        return res

    # Extract main blocks for model-under-test
    def parse_block(start_re: re.Pattern) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        file_path = None
        loss = None
        ppl = None
        in_block = False
        for line in txt:
            if not in_block and start_re.match(line):
                in_block = True
                continue
            if in_block:
                if line.strip().startswith("----"):
                    break
                m1 = FILE_LINE.match(line)
                if m1:
                    file_path = m1.group("path").strip()
                m2 = MEAN_LOSS_PPL.match(line)
                if m2:
                    try:
                        loss = float(m2.group("loss"))
                    except Exception:
                        loss = None
                    try:
                        ppl = float(m2.group("ppl"))
                    except Exception:
                        ppl = None
        return file_path, loss, ppl

    val_file, val_loss, val_ppl = parse_block(VAL_BLOCK_START)
    test_file, test_loss, test_ppl = parse_block(TEST_BLOCK_START)
    res.update({
        "val_file": val_file,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "test_file": test_file,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
    })

    # Extract baseline PPL from "Baseline vs Fine-tuned" section
    # We scan for Split lines and then Baseline PPL lines that follow shortly.
    current_split: Optional[str] = None
    for i, line in enumerate(txt):
        m = BASELINE_SPLIT.match(line)
        if m:
            current_split = m.group("split").strip()
            continue
        if current_split in ("val", "test"):
            m2 = BASELINE_PPL.match(line)
            if m2:
                try:
                    ppl_v = float(m2.group("ppl"))
                except Exception:
                    ppl_v = None
                if current_split == "val":
                    res["baseline_val_ppl"] = ppl_v
                elif current_split == "test":
                    res["baseline_test_ppl"] = ppl_v
                current_split = None

    # Try to detect flags if present in logs
    log_lower = "\n".join(txt).lower()
    # reverse flags
    if "reverse_on_splits" in log_lower:
        # naive extraction
        for line in txt:
            if "reverse_on_splits" in line:
                res["random_reverse"] = line.split(":", 1)[-1].strip()
                break
    elif "random reverse" in log_lower or "random_reverse" in log_lower:
        res["random_reverse"] = "true"

    # bucket flags
    if "token_bucketed_batches" in log_lower:
        # capture boolean-like token
        for line in txt:
            if "token_bucketed_batches" in line:
                res["token_bucketed_batches"] = line.split(":", 1)[-1].strip()
                break
    if "bucket" in log_lower:
        res["bucketed_training"] = "true"

    return res


def find_run_containers(eval_root: Path) -> Dict[Path, List[Path]]:
    """Return mapping of run container -> list of checkpoint dirs.

    A checkpoint dir is either a directory containing run.log, or one that
    contains a model_under_test/*.csv file (val/test preds).
    """
    containers: Dict[Path, List[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(eval_root):
        dpath = Path(dirpath)
        # A checkpoint dir contains run.log, or model_under_test CSVs
        is_ckpt = False
        if "run.log" in filenames:
            is_ckpt = True
        else:
            mut = dpath / "model_under_test"
            if mut.is_dir():
                if (mut / "val_preds.csv").exists() or (mut / "test_preds.csv").exists():
                    is_ckpt = True
        if is_ckpt:
            ckpt_dir = dpath
            container = ckpt_dir.parent
            # Skip the eval_outputs root being treated as a checkpoint
            if ckpt_dir == eval_root:
                continue
            # Require container to be within eval_root
            try:
                container.relative_to(eval_root)
            except Exception:
                continue
            containers.setdefault(container, []).append(ckpt_dir)
    return containers


def weighted_mean_from_csv(csv_path: Path) -> Optional[float]:
    try:
        import csv as _csv
        tot_loss = 0.0
        tot_tokens = 0.0
        with open(csv_path, newline="") as f:
            r = _csv.DictReader(f)
            for row in r:
                try:
                    loss = float(row.get("mean_loss", ""))
                    tokens = float(row.get("valid_tokens", ""))
                except Exception:
                    continue
                tot_loss += loss * tokens
                tot_tokens += tokens
        if tot_tokens > 0:
            return tot_loss / tot_tokens
    except Exception:
        return None
    return None


def index_wandb_configs(wandb_root: Path) -> Dict[str, Dict[str, object]]:
    """Scan wandb run config.yaml files and index training/eval flags by train run_id.

    Returns mapping: run_id (e.g., run-20250822-202947) -> {token_bucketed_batches, train_data_path, learning_rate, random_reverse}
    The run_id is extracted from evaluation.model_path or evaluation.ckpt path if present.
    """
    idx: Dict[str, Dict[str, object]] = {}
    if not wandb_root.exists():
        return idx
    for cfg in wandb_root.glob("run-*/files/config.yaml"):
        try:
            lines = cfg.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        current_section = []  # e.g., ['spec','data']
        info: Dict[str, object] = {}
        eval_paths: List[str] = []
        for raw in lines:
            line = raw.rstrip()
            # Track simple section stack by indentation and key ending with ':'
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            # Normalize key: value
            if line.strip().endswith(":"):
                key = line.strip().rstrip(":")
                # crude depth by indent // 2
                depth = indent // 2
                # adjust stack
                current_section = current_section[:depth]
                current_section.append(key)
                continue
            # Parse key: value pairs
            if ":" in line:
                k, v = line.strip().split(":", 1)
                v = v.strip()
                path_parts = current_section + [k]
                path = "/".join(path_parts)
                # Also consider a normalized path with '/value' stripped if present
                if len(path_parts) >= 2 and path_parts[1] == "value":
                    norm_parts = [path_parts[0]] + path_parts[2:]
                    norm_path = "/".join(norm_parts)
                else:
                    norm_path = path
                # capture values we care about
                if norm_path in (
                    "evaluation/model_path",
                    "evaluation/ckpt",
                ):
                    eval_paths.append(v)
                elif norm_path == "evaluation/random_reverse":
                    info.setdefault("eval_random_reverse", v.lower() in ("true", "1", "yes"))
                elif norm_path == "spec/data/token_bucketed_batches":
                    info["token_bucketed_batches"] = v.lower() in ("true", "1", "yes")
                elif norm_path == "spec/data/train_data_path":
                    info["train_data_path"] = v
                elif norm_path == "spec/data/random_reverse":
                    info["random_reverse"] = v.lower() in ("true", "1", "yes")
                elif norm_path == "spec/training/learning_rate":
                    try:
                        info["learning_rate"] = float(v)
                    except Exception:
                        info["learning_rate"] = v
        # Also parse companion debug.log for more signals
        dbg = cfg.parent.parent / "logs" / "debug.log"
        if dbg.exists():
            try:
                txt = dbg.read_text(errors="ignore")
            except Exception:
                txt = ""
            # evaluation model paths (collect any run-... matches)
            for m in re.finditer(r"(/run-\d{8}-\d{6}/huggingface/[^'\"\s]+)", txt):
                eval_paths.append(m.group(1))
            # training data path
            m = re.search(r"train_data_path[:'\"]?\s*[:=]\s*['\"]([^'\"\n]+)['\"]", txt)
            if m and "train_data_path" not in info:
                info["train_data_path"] = m.group(1)
            # token_bucketed_batches
            m = re.search(r"token_bucketed_batches[:'\"]?\s*[:=]\s*(True|False|true|false)", txt)
            if m and "token_bucketed_batches" not in info:
                info["token_bucketed_batches"] = m.group(1).lower() == "true"
            # learning rate
            m = re.search(r"learning_rate[:'\"]?\s*[:=]\s*([0-9.eE+-]+)", txt)
            if m and "learning_rate" not in info:
                try:
                    info["learning_rate"] = float(m.group(1))
                except Exception:
                    info["learning_rate"] = m.group(1)
            # random_reverse (prefer spec over evaluation)
            m = re.search(r"spec.*random_reverse[:'\"]?\s*[:=]\s*(True|False|true|false)", txt)
            if m and "random_reverse" not in info:
                info["random_reverse"] = m.group(1).lower() == "true"
            if "random_reverse" not in info:
                m = re.search(r"evaluation.*random_reverse[:'\"]?\s*[:=]\s*(True|False|true|false)", txt)
                if m:
                    info["eval_random_reverse"] = m.group(1).lower() == "true"

        # Extract run_id from any eval path
        for p in eval_paths:
            m = re.search(r"(run-\d{8}-\d{6})", p)
            if m:
                run_id = m.group(1)
                existing = idx.get(run_id, {})
                # Prefer training random_reverse; fallback to eval
                if "random_reverse" not in existing:
                    if "random_reverse" in info:
                        existing["random_reverse"] = info["random_reverse"]
                    elif "eval_random_reverse" in info:
                        existing["random_reverse"] = info["eval_random_reverse"]
                for key in ("token_bucketed_batches", "train_data_path", "learning_rate"):
                    if key in info and key not in existing:
                        existing[key] = info[key]
                idx[run_id] = existing
                break
    return idx


def parse_from_csv_fallback(ckpt_dir: Path, res: Dict[str, object]) -> Dict[str, object]:
    """Fill in missing losses from model_under_test and baseline_model CSVs if present."""
    # Model under test
    mut = ckpt_dir / "model_under_test"
    if mut.is_dir():
        val_csv = mut / "val_preds.csv"
        test_csv = mut / "test_preds.csv"
        if res.get("val_loss") is None and val_csv.exists():
            res["val_loss"] = weighted_mean_from_csv(val_csv)
        if res.get("test_loss") is None and test_csv.exists():
            res["test_loss"] = weighted_mean_from_csv(test_csv)

    # Baseline model
    base = ckpt_dir / "baseline_model"
    if base.is_dir():
        val_csv_b = base / "val_preds.csv"
        test_csv_b = base / "test_preds.csv"
        if res.get("baseline_val_ppl") is None and val_csv_b.exists():
            # we don't have baseline PPL, but we can compute baseline loss directly
            res["baseline_val_loss_from_csv"] = weighted_mean_from_csv(val_csv_b)
        if res.get("baseline_test_ppl") is None and test_csv_b.exists():
            res["baseline_test_loss_from_csv"] = weighted_mean_from_csv(test_csv_b)
    return res

def choose_best_ckpt(ckpt_dirs: List[Path]) -> Tuple[Optional[Path], Dict[str, object]]:
    """Pick best checkpoint by lowest val mean loss (fallback to test loss)."""
    best_dir = None
    best_metrics: Dict[str, object] = {}
    best_val = math.inf
    best_test = math.inf
    for ck in ckpt_dirs:
        m = parse_runlog(ck / "run.log")
        m = parse_from_csv_fallback(ck, m)
        val_loss = m.get("val_loss")
        test_loss = m.get("test_loss")
        if isinstance(val_loss, float):
            score = val_loss
            if score < best_val:
                best_val = score
                best_dir = ck
                best_metrics = m
        elif isinstance(test_loss, float):
            # only consider if no val available yet or better test
            score = test_loss
            if best_dir is None and score < best_test:
                best_test = score
                best_dir = ck
                best_metrics = m
    return best_dir, best_metrics


def main(out_path: Optional[Path] = None, roots: Optional[List[Path]] = None) -> int:
    roots = roots or [p for p in DEFAULT_EVAL_ROOTS if p.exists()]
    if not roots:
        print("No evaluation folders found.")
        return 1
    # Gather all containers from all roots
    containers: Dict[Path, List[Path]] = {}
    for root in roots:
        sub = find_run_containers(root)
        containers.update(sub)
    # Build W&B config index for enrichment if available
    wandb_idx = index_wandb_configs(ROOT / "wandb")
    rows: List[Dict[str, object]] = []
    for container, ckpts in sorted(containers.items()):
        best_dir, m = choose_best_ckpt(ckpts)
        if best_dir is None:
            continue
        # Run identifier: if container is eval root, use ckpt dir name; else relative container path
        # Include the root folder name as prefix for clarity
        run_id = best_dir.parent.name + "/" + best_dir.name
        best_ckpt = best_dir.name
        # Baseline losses from baseline PPL if present
        bv_ppl = m.get("baseline_val_ppl")
        bt_ppl = m.get("baseline_test_ppl")
        # Prefer direct baseline mean loss from CSVs if available
        bv_loss_csv = m.get("baseline_val_loss_from_csv")
        bt_loss_csv = m.get("baseline_test_loss_from_csv")
        def ln(x: Optional[float]) -> Optional[float]:
            try:
                return float(math.log(float(x))) if x is not None else None
            except Exception:
                return None
        # Build row and include derived improvement columns if baseline available
        row = {
            "run_id": run_id,
            "best_checkpoint": best_ckpt,
            "best_checkpoint_path": str(best_dir),
            "data_val_path": m.get("val_file"),
            "data_test_path": m.get("test_file"),
            # placeholders; may be enriched from W&B configs
            "random_reverse": m.get("random_reverse"),
            "bucketed_training": m.get("bucketed_training"),
            "token_bucketed_batches": m.get("token_bucketed_batches"),
            "learning_rate": None,
            "val_mean_loss": m.get("val_loss"),
            "baseline_val_mean_loss": (bv_loss_csv if isinstance(bv_loss_csv, float) else ln(bv_ppl)),
            "test_mean_loss": m.get("test_loss"),
            "baseline_test_mean_loss": (bt_loss_csv if isinstance(bt_loss_csv, float) else ln(bt_ppl)),
        }
        # Enrich from W&B configs using train run_id inferred from path
        try:
            m_run = re.search(r"(run-\d{8}-\d{6})", str(best_dir))
            if m_run:
                rid = m_run.group(1)
                wb = wandb_idx.get(rid, {})
                if "random_reverse" in wb and row.get("random_reverse") in (None, "", False):
                    row["random_reverse"] = wb["random_reverse"]
                if "token_bucketed_batches" in wb and not row.get("token_bucketed_batches"):
                    row["token_bucketed_batches"] = wb["token_bucketed_batches"]
                if "learning_rate" in wb and row.get("learning_rate") in (None, ""):
                    row["learning_rate"] = wb["learning_rate"]
                if "train_data_path" in wb and not row.get("train_data_path"):
                    # capture training dataset path separately (always safe to add)
                    row["train_data_path"] = wb["train_data_path"]
        except Exception:
            pass
        # Derived improvements
        try:
            v = row["val_mean_loss"]
            b = row["baseline_val_mean_loss"]
            row["improvement_delta_val"] = (b - v) if (isinstance(v, float) and isinstance(b, float)) else None
            row["improvement_ratio_val"] = (v / b) if (isinstance(v, float) and isinstance(b, float) and b != 0) else None
        except Exception:
            row["improvement_delta_val"] = None
            row["improvement_ratio_val"] = None
        rows.append(row)

    if not rows:
        print("No evaluation logs found under eval_outputs/ with parseable metrics.")
        return 2

    out = out_path or (ROOT / "eval_outputs_summary.csv")
    fieldnames = [
        "run_id",
        "best_checkpoint",
        "best_checkpoint_path",
        "data_val_path",
        "data_test_path",
        "train_data_path",
        "random_reverse",
        "bucketed_training",
        "token_bucketed_batches",
        "learning_rate",
        "val_mean_loss",
        "baseline_val_mean_loss",
        "test_mean_loss",
        "baseline_test_mean_loss",
        "improvement_delta_val",
        "improvement_ratio_val",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Brief best-run summary to stdout
    # Best by absolute val loss
    rows_with_val = [r for r in rows if isinstance(r.get("val_mean_loss"), float)]
    best_by_val = min(rows_with_val, key=lambda r: r["val_mean_loss"]) if rows_with_val else None
    # Best by improvement vs baseline (delta)
    rows_with_imp = [r for r in rows if isinstance(r.get("improvement_delta_val"), float)]
    best_by_delta = max(rows_with_imp, key=lambda r: r["improvement_delta_val"]) if rows_with_imp else None
    # Best by improvement ratio (lower is better)
    rows_with_ratio = [r for r in rows if isinstance(r.get("improvement_ratio_val"), float)]
    best_by_ratio = min(rows_with_ratio, key=lambda r: r["improvement_ratio_val"]) if rows_with_ratio else None

    print(f"Wrote summary: {out} ({len(rows)} rows)")
    if best_by_val:
        print(f"Best val loss: {best_by_val['val_mean_loss']:.5f} @ {best_by_val['run_id']} -> {best_by_val['best_checkpoint']}")
    if best_by_delta:
        print(f"Best improvement (delta): {best_by_delta['improvement_delta_val']:.5f} @ {best_by_delta['run_id']} -> {best_by_delta['best_checkpoint']} (baseline {best_by_delta['baseline_val_mean_loss']}, val {best_by_delta['val_mean_loss']})")
    if best_by_ratio:
        print(f"Best improvement (ratio): {best_by_ratio['improvement_ratio_val']:.5f} @ {best_by_ratio['run_id']} -> {best_by_ratio['best_checkpoint']} (baseline {best_by_ratio['baseline_val_mean_loss']}, val {best_by_ratio['val_mean_loss']})")
    return 0


if __name__ == "__main__":
    # Usage: python scripts/summarize_eval_outputs.py [out.csv] [root1 root2 ...]
    out: Optional[Path] = None
    roots: List[Path] = []
    args = sys.argv[1:]
    if args:
        # First argument that endswith .csv is out path
        if args[0].lower().endswith('.csv'):
            out = Path(args[0]).resolve()
            roots = [Path(p).resolve() for p in args[1:]] if len(args) > 1 else []
        else:
            roots = [Path(p).resolve() for p in args]
    raise SystemExit(main(out, roots or None))
