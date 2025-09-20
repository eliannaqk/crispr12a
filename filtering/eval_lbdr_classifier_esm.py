#!/usr/bin/env python3
"""
Evaluate the LbCas12a DR classifier on train/valid CSVs, compute confusion
matrix metrics (accuracy, PPV, NPV, sensitivity, specificity, F1), and
optionally log them to the original W&B run.

Usage example:
  python filtering/eval_lbdr_classifier_esm.py \
    --train_csv /path/to/train_labeled_pos.csv \
    --valid_csv /path/to/valid_labeled_pos.csv \
    --model_path /path/to/lbdr_cls.pt \
    --threshold 0.5 \
    --wandb-project crispr_filtering \
    --wandb-entity eqk3 \
    --wandb-run-id 9e73j1ij
"""
import argparse
import os
from typing import Dict, Tuple, Optional

import torch
from torch import nn

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

# Reuse training utilities
from .train_lbdr_classifier_esm import (
    load_rows,
    embed_with_cls_concat_two,
    MLP,
)
import esm  # fair-esm


def safe_div(n: float, d: float) -> float:
    return float('nan') if d == 0 else (n / d)


@torch.no_grad()
def logits_for_rows(rows, backbone, alphabet, clf: nn.Module, device: str, batch_embed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed rows (optionally batched by count of sequences; 0 = one-by-one),
    run classifier, and return (logits, labels) on CPU tensors.
    """
    feats = []
    labels = []
    # One-by-one embedding to reuse embed_with_cls_concat_two logic
    for r in rows:
        z = embed_with_cls_concat_two(r['protein_seq'], backbone, alphabet, device)
        if z is None:
            continue
        feats.append(z)
        labels.append(int(r['dr_exact']))
    if not feats:
        return torch.empty(0), torch.empty(0)
    X = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    logits = clf(X).detach().cpu()
    return logits, y.detach().cpu()


def confusion_metrics(y_true: torch.Tensor, logits: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute confusion matrix and derived metrics.
    y_true/logits are CPU tensors; logits are raw scores before sigmoid.
    """
    if y_true.numel() == 0:
        return {
            'n_total': 0,
            'n_pos': 0,
            'n_neg': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'accuracy': float('nan'),
            'ppv': float('nan'),
            'npv': float('nan'),
            'sensitivity': float('nan'),
            'specificity': float('nan'),
            'f1': float('nan'),
        }
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int32)
    y = y_true.to(torch.int32)

    tp = int(((preds == 1) & (y == 1)).sum().item())
    tn = int(((preds == 0) & (y == 0)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())
    n_total = int(y.numel())
    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())

    accuracy = safe_div(tp + tn, n_total)
    ppv = safe_div(tp, tp + fp)  # precision
    npv = safe_div(tn, tn + fn)
    sensitivity = safe_div(tp, tp + fn)  # recall/TPR
    specificity = safe_div(tn, tn + fp)  # TNR
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        'n_total': n_total,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'n_correct': tp + tn,
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate lbdr classifier on train/valid and log metrics.")
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--valid_csv', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--wandb-project', default=None)
    ap.add_argument('--wandb-entity', default=None)
    ap.add_argument('--wandb-run-id', default=None, help='If set, log to this existing run id')
    args = ap.parse_args()

    # Load data
    train_rows = load_rows(args.train_csv)
    valid_rows = load_rows(args.valid_csv)

    # Load ESM backbone
    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    backbone = backbone.to(args.device).eval()

    # Load classifier
    ckpt = torch.load(args.model_path, map_location='cpu')
    clf = MLP(in_dim=int(ckpt.get('in_dim', 2560)), hidden=512, p_drop=0.0)
    clf.load_state_dict(ckpt['state_dict'])
    clf = clf.to(args.device).eval()

    # Compute logits
    tr_logits, tr_y = logits_for_rows(train_rows, backbone, alphabet, clf, args.device)
    va_logits, va_y = logits_for_rows(valid_rows, backbone, alphabet, clf, args.device)

    # Metrics
    tr = confusion_metrics(tr_y, tr_logits, threshold=args.threshold)
    va = confusion_metrics(va_y, va_logits, threshold=args.threshold)

    # Print summary
    def fmt(d: Dict[str, float]) -> str:
        return (
            f"n={d['n_total']} (pos={d['n_pos']}, neg={d['n_neg']}), "
            f"correct={d['n_correct']}, acc={d['accuracy']:.4f}, "
            f"ppv={d['ppv']:.4f}, npv={d['npv']:.4f}, sens={d['sensitivity']:.4f}, "
            f"spec={d['specificity']:.4f}, f1={d['f1']:.4f}"
        )

    print("[Eval] Train:", fmt(tr))
    print("[Eval] Valid:", fmt(va))

    # Log to W&B if configured
    if args.wandb_project and _WANDB_AVAILABLE:
        init_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity:
            init_kwargs['entity'] = args.wandb_entity
        if args.wandb_run_id:
            init_kwargs['id'] = args.wandb_run_id
            init_kwargs['resume'] = 'allow'
        run = wandb.init(**init_kwargs)
        wandb.log({
            'eval/threshold': args.threshold,
            # Train metrics
            'train/n_total': tr['n_total'],
            'train/n_pos': tr['n_pos'],
            'train/n_neg': tr['n_neg'],
            'train/n_correct': tr['n_correct'],
            'train/accuracy': tr['accuracy'],
            'train/ppv': tr['ppv'],
            'train/npv': tr['npv'],
            'train/sensitivity': tr['sensitivity'],
            'train/specificity': tr['specificity'],
            'train/f1': tr['f1'],
            # Valid metrics
            'valid/n_total': va['n_total'],
            'valid/n_pos': va['n_pos'],
            'valid/n_neg': va['n_neg'],
            'valid/n_correct': va['n_correct'],
            'valid/accuracy': va['accuracy'],
            'valid/ppv': va['ppv'],
            'valid/npv': va['npv'],
            'valid/sensitivity': va['sensitivity'],
            'valid/specificity': va['specificity'],
            'valid/f1': va['f1'],
        })
        run.finish()


if __name__ == '__main__':
    main()
