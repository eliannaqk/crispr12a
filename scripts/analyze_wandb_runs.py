#!/usr/bin/env python3
"""
Aggregate W&B metrics for TTTV core-aware classifier runs.

Pulls runs by name substrings and reports best AUROC and last-epoch metrics,
including SNR regression MSE/correlation and logit/prob correlation with SNR.

Usage:
  python scripts/analyze_wandb_runs.py \
    --entity eqk3 --project crispr_filtering \
    --names coreaware_strict2strict_bcebal_snr_,coreaware_strict2strict_nosnr_bcebal_,coreaware_perm2perm_bcebal_snr_,coreaware_perm2perm_nosnr_bcebal_

Requires W&B login (wandb login) or WANDB_API_KEY in env.
"""
from __future__ import annotations
import argparse, sys, math, re
from typing import Dict, Any, List

try:
    import wandb
except Exception as e:
    print("[error] wandb not available. pip install wandb and login.", file=sys.stderr)
    raise


def best_of_history(run, key: str):
    """Return (best_val, best_step) for a metric key using run.history()."""
    try:
        hist = run.history(keys=[key, 'epoch'], pandas=False)
        best_val = None
        best_step = None
        for row in hist:
            val = row.get(key, None)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                continue
            if best_val is None:
                best_val, best_step = val, row.get('epoch')
            else:
                # maximize AUROC, minimize losses
                if key.lower().endswith('auroc'):
                    if val > best_val: best_val, best_step = val, row.get('epoch')
                else:
                    if val < best_val: best_val, best_step = val, row.get('epoch')
        return best_val, best_step
    except Exception:
        return None, None


def last_of_history(run, keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        hist = list(run.history(keys=keys + ['epoch'], pandas=False))
        if not hist:
            return out
        last = hist[-1]
        for k in keys:
            out[k] = last.get(k, None)
        out['epoch'] = last.get('epoch', None)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entity', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--names', required=True, help='Comma-separated substrings to match run.name')
    ap.add_argument('--max', type=int, default=400, help='Max runs to scan')
    args = ap.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", per_page=min(200, args.max))
    patterns = [s.strip() for s in args.names.split(',') if s.strip()]

    selected = []
    for r in runs:
        nm = r.name or ''
        if any(p in nm for p in patterns):
            selected.append(r)

    if not selected:
        print("[info] No runs matched the provided names yet.")
        return

    # Sort by created time ascending
    selected.sort(key=lambda r: r.created_at)

    print("run_name,finished,state,epoch,last_valid_auroc,best_valid_auroc,best_auroc_epoch,last_valid_loss,last_valid_snr_mse,last_valid_snr_r,last_corr_prob_y_snr,valid_tp,last_valid_tn,last_valid_fp,last_valid_fn")
    for r in selected:
        best_auc, best_ep = best_of_history(r, 'valid_auroc')
        last = last_of_history(r, [
            'valid_auroc','valid_loss','valid_snr_mse','valid_snr_r','valid_corr_prob_y_snr',
            'valid_tp@0.5','valid_tn@0.5','valid_fp@0.5','valid_fn@0.5'
        ])
        is_finished = getattr(r, 'state', None) in ('finished', 'crashed', 'failed')
        row = [
            r.name,
            'yes' if is_finished else 'no',
            getattr(r, 'state', None),
            last.get('epoch'),
            last.get('valid_auroc'),
            best_auc,
            best_ep,
            last.get('valid_loss'),
            last.get('valid_snr_mse'),
            last.get('valid_snr_r'),
            last.get('valid_corr_prob_y_snr'),
            last.get('valid_tp@0.5'),
            last.get('valid_tn@0.5'),
            last.get('valid_fp@0.5'),
            last.get('valid_fn@0.5'),
        ]
        print(','.join('' if v is None else str(v) for v in row))


if __name__ == '__main__':
    main()

