#!/usr/bin/env python3
import argparse
import torch
from torch import nn
from typing import Dict

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

from .train_lbdr_classifier_esm import (
    load_rows,
    embed_with_cls_concat_two,
    MLP,
)
import esm

def safe_div(n: float, d: float) -> float:
    return float('nan') if d == 0 else (n / d)

@torch.no_grad()
def confusion_metrics(y_true: torch.Tensor, logits: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.int32)
    y = y_true.to(torch.int32)
    tp = int(((preds == 1) & (y == 1)).sum().item())
    tn = int(((preds == 0) & (y == 0)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())
    n_total = int(y.numel())
    accuracy = safe_div(tp + tn, n_total)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        'n_total': n_total,
        'n_correct': tp + tn,
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
    }

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Quick valid-only eval and W&B log.")
    ap.add_argument('--valid_csv', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--wandb-project', default=None)
    ap.add_argument('--wandb-entity', default=None)
    ap.add_argument('--wandb-run-id', default=None)
    args = ap.parse_args()

    valid_rows = load_rows(args.valid_csv)
    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    backbone = backbone.to(args.device).eval()
    ckpt = torch.load(args.model_path, map_location='cpu')
    clf = MLP(in_dim=int(ckpt.get('in_dim', 2560)), hidden=512, p_drop=0.0)
    clf.load_state_dict(ckpt['state_dict'])
    clf = clf.to(args.device).eval()

    feats, labels = [], []
    for r in valid_rows:
        z = embed_with_cls_concat_two(r['protein_seq'], backbone, alphabet, args.device)
        if z is None:
            continue
        feats.append(z)
        labels.append(int(r['dr_exact']))
    X = torch.stack(feats).to(args.device)
    y = torch.tensor(labels, dtype=torch.float32, device=args.device)
    logits = clf(X).detach().cpu()
    ycpu = y.detach().cpu()

    m = confusion_metrics(ycpu, logits, args.threshold)
    print(f"[Valid-only] n={m['n_total']}, correct={m['n_correct']}, acc={m['accuracy']:.4f}, ppv={m['ppv']:.4f}, npv={m['npv']:.4f}, sens={m['sensitivity']:.4f}, spec={m['specificity']:.4f}, f1={m['f1']:.4f}")

    if args.wandb_project and _WANDB_AVAILABLE:
        init_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity: init_kwargs['entity'] = args.wandb_entity
        if args.wandb_run_id:
            init_kwargs['id'] = args.wandb_run_id
            init_kwargs['resume'] = 'allow'
        run = wandb.init(**init_kwargs)
        wandb.log({
            'valid_quick/n_total': m['n_total'],
            'valid_quick/n_correct': m['n_correct'],
            'valid_quick/accuracy': m['accuracy'],
            'valid_quick/ppv': m['ppv'],
            'valid_quick/npv': m['npv'],
            'valid_quick/sensitivity': m['sensitivity'],
            'valid_quick/specificity': m['specificity'],
            'valid_quick/f1': m['f1'],
        })
        run.finish()

if __name__ == '__main__':
    main()

