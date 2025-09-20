#!/usr/bin/env python3
import argparse, os, sys, csv, random, numpy as np, torch
from typing import Optional, Tuple
from torch import nn
import torch.nn.functional as F

try:
    import esm  # pip install fair-esm
except ImportError:
    print("Please: pip install fair-esm", file=sys.stderr); raise

# Optional Weights & Biases
_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"protein_seq","pam_tttv"}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing required columns: {need - set(reader.fieldnames or [])}")
        have_snr = 'snr' in (reader.fieldnames or [])
        for r in reader:
            try:
                lbl = int(r["pam_tttv"])
            except Exception:
                continue
            item = {"protein_seq": r["protein_seq"], "pam_tttv": lbl}
            if have_snr:
                try:
                    snr_val = float(r.get("snr", ""))
                except Exception:
                    snr_val = None
                item["snr"] = snr_val
            rows.append(item)
    return rows

def embed_with_cls_concat_two(seq: str, model, alphabet, device,
                              last_layer: int = 33, max_tokens_per_chunk: int = 1022,
                              drop_len_threshold: int = 20000):
    if len(seq) > drop_len_threshold:
        return None
    model.eval()
    toks = alphabet.get_batch_converter()
    def _cls_for(subseq: str):
        batch = [("protein", subseq)]
        _, _, tokens = toks(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[last_layer], return_contacts=False)
            rep = out["representations"][last_layer][0]
        return rep[0].cpu()
    if len(seq) <= max_tokens_per_chunk:
        cls1 = _cls_for(seq)
        cls2 = torch.zeros_like(cls1)
        return torch.cat([cls1, cls2], dim=-1)
    head = seq[:max_tokens_per_chunk]
    tail = seq[-max_tokens_per_chunk:]
    return torch.cat([_cls_for(head), _cls_for(tail)], dim=-1)

def embed_cls_tail_only(seq: str, model, alphabet, device,
                        last_layer: int = 33, max_tokens_per_chunk: int = 1022,
                        drop_len_threshold: int = 200000):
    """
    CLS embedding from the last <=1022 AA window (tail-only).
    """
    # Keep a high threshold; we still embed only the tail chunk.
    model.eval()
    toks = alphabet.get_batch_converter()
    subseq = seq[-max_tokens_per_chunk:]
    batch = [("protein", subseq)]
    _, _, tokens = toks(batch)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[last_layer], return_contacts=False)
        rep = out["representations"][last_layer][0]
    return rep[0].cpu()

class MLPHeads(nn.Module):
    def __init__(self, in_dim=1280, hidden=512, p_drop=0.2, with_snr: bool = False):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )
        self.clf_head = nn.Linear(hidden, 1)
        self.with_snr = with_snr
        if with_snr:
            self.snr_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.feat(x)
        logits = self.clf_head(h).squeeze(-1)
        if self.with_snr:
            snr_pred = self.snr_head(h).squeeze(-1)
            return logits, snr_pred
        return logits, None

def preembed(rows, model, alphabet, device, embedder=embed_with_cls_concat_two) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
    feats, labels, snr_vals, snr_mask, dropped = [], [], [], [], 0
    for r in rows:
        z = embedder(r["protein_seq"], model, alphabet, device)
        if z is None:
            dropped += 1; continue
        feats.append(z); labels.append(int(r["pam_tttv"]))
        snr = r.get("snr", None)
        if isinstance(snr, (int, float)):
            snr_vals.append(float(snr))
            snr_mask.append(1.0)
        else:
            snr_vals.append(0.0)
            snr_mask.append(0.0)
    X = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    y_snr = torch.tensor(snr_vals, dtype=torch.float32, device=device) if any(m>0 for m in snr_mask) else None
    m_snr = torch.tensor(snr_mask, dtype=torch.float32, device=device)
    return X, y, y_snr, m_snr, dropped

@torch.no_grad()
def evaluate(model, X, y, batch_size=256) -> Tuple[float, float]:
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    all_logits = []; total = 0.0
    for i in range(0, len(y), batch_size):
        logits, _ = model(X[i:i+batch_size])
        loss = crit(logits, y[i:i+batch_size])
        total += loss.item() * len(logits)
        all_logits.append(logits.detach().cpu())
    loss_mean = total / len(y)
    scores = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = y.detach().cpu().numpy()
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, scores)
    except Exception:
        import numpy as _np
        pos = scores[labels == 1]; neg = scores[labels == 0]
        auroc = (pos.reshape(-1,1) > neg.reshape(1,-1)).mean() if len(pos)*len(neg) else float('nan')
    return loss_mean, float(auroc)

@torch.no_grad()
def eval_snr_metrics(model, X, y_snr, m_snr, batch_size=256, log1p: bool = False):
    if y_snr is None or m_snr is None:
        return None
    if float(m_snr.sum().item()) <= 0:
        return None
    preds = []
    tgts = []
    for i in range(0, len(m_snr), batch_size):
        logits, snr_pred = model(X[i:i+batch_size])
        if snr_pred is None:
            return None
        mask = (m_snr[i:i+batch_size] > 0.5)
        if mask.any():
            p = snr_pred[mask]
            t = y_snr[i:i+batch_size][mask]
            if log1p:
                t = torch.log1p(torch.clamp(t, min=0.0))
            preds.append(p.detach().cpu())
            tgts.append(t.detach().cpu())
    if not preds:
        return None
    p = torch.cat(preds).numpy()
    t = torch.cat(tgts).numpy()
    mse = float(((p - t) ** 2).mean())
    try:
        r = float(np.corrcoef(p, t)[0, 1]) if p.size > 1 else float('nan')
    except Exception:
        r = float('nan')
    return {'valid_snr_mse': mse, 'valid_snr_r': r}

@torch.no_grad()
def eval_logit_snr_corr(model, X, y_snr, m_snr, batch_size=256, log1p: bool = False):
    if y_snr is None or m_snr is None:
        return None
    if float(m_snr.sum().item()) <= 0:
        return None
    logits_cat = []
    snr_cat = []
    for i in range(0, len(m_snr), batch_size):
        logits, _ = model(X[i:i+batch_size])
        mask = (m_snr[i:i+batch_size] > 0.5)
        if mask.any():
            l = logits[mask]
            t = y_snr[i:i+batch_size][mask]
            if log1p:
                t = torch.log1p(torch.clamp(t, min=0.0))
            logits_cat.append(l.detach().cpu())
            snr_cat.append(t.detach().cpu())
    if not logits_cat:
        return None
    L = torch.cat(logits_cat).numpy()
    T = torch.cat(snr_cat).numpy()
    try:
        r_logit = float(np.corrcoef(L, T)[0, 1]) if L.size > 1 else float('nan')
        r_prob = float(np.corrcoef(1/(1+np.exp(-L)), T)[0, 1]) if L.size > 1 else float('nan')
    except Exception:
        r_logit = float('nan'); r_prob = float('nan')
    return {'valid_corr_logit_y_snr': r_logit, 'valid_corr_prob_y_snr': r_prob}

@torch.no_grad()
def confusion_counts(model, X, y, batch_size=256, threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """Return (tp, tn, fp, fn) at the given threshold."""
    model.eval()
    tp = tn = fp = fn = 0
    for i in range(0, len(y), batch_size):
        logits, _ = model(X[i:i+batch_size])
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.float32)
        tgt = y[i:i+batch_size]
        tp += int(((preds == 1) & (tgt == 1)).sum().item())
        tn += int(((preds == 0) & (tgt == 0)).sum().item())
        fp += int(((preds == 1) & (tgt == 0)).sum().item())
        fn += int(((preds == 0) & (tgt == 1)).sum().item())
    return tp, tn, fp, fn

def main():
    ap = argparse.ArgumentParser(description="ESM2 fine-tune for TTTV recognition (Cas12a).")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--valid_csv", default=None)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument('--wandb-project', default=None, help="W&B project (e.g., crispr_filtering)")
    ap.add_argument('--wandb-entity', default=None, help="W&B entity/org (optional)")
    ap.add_argument('--wandb-run-name', default=None, help="W&B run name (optional)")
    ap.add_argument('--early-stop-patience', type=int, default=0, help="Stop if val loss doesn't improve for N epochs (requires valid set)")
    ap.add_argument('--pos-weight-scale', type=float, default=1.0, help="Multiply computed pos_weight by this factor")
    ap.add_argument('--pos-weight-min', type=float, default=0.0, help="Floor for pos_weight after scaling (0 to disable)")
    ap.add_argument('--pos-weight-fixed', type=float, default=None, help="If set, use this fixed pos_weight and ignore computed one")
    ap.add_argument('--embed-policy', default='concat_head_tail1022', choices=['concat_head_tail1022','cls_tail_1022'], help="How to embed sequence")
    ap.add_argument('--balance-oversample', action='store_true', help='Oversample minority class each epoch to balance batches (ignored if --bce-weighting balanced)')
    ap.add_argument('--aux-snr-weight', type=float, default=0.0, help='Weight for auxiliary SNR regression (0 disables)')
    ap.add_argument('--aux-snr-log1p', action='store_true', help='Regress log1p(SNR) instead of raw SNR')
    # SNR-driven sample weighting for classifier loss (uses CSV column `snr` if present)
    ap.add_argument('--snr-sample-weight-mode', default='none', choices=['none','linear','log1p'],
                    help="Per-sample weight factor from SNR: 'linear' -> snr, 'log1p' -> log1p(max(snr,0)).")
    ap.add_argument('--snr-weight-min', type=float, default=0.5, help='Clamp minimum SNR-derived weight (after transform)')
    ap.add_argument('--snr-weight-max', type=float, default=3.0, help='Clamp maximum SNR-derived weight (after transform)')
    ap.add_argument('--snr-weight-norm', default='mean', choices=['none','mean'],
                    help="Normalize SNR weight vector (e.g., divide by mean so avg≈1). Applies before clamping.")
    ap.add_argument('--bce-weighting', default='none', choices=['none','balanced'], help='Use per-class BCE weights instead of oversampling')
    ap.add_argument('--bce-neg-mult', type=float, default=1.0, help='Multiply negative-class weight for BCE (use >1 when negatives are fewer)')
    ap.add_argument('--bce-pos-mult', type=float, default=1.0, help='Multiply positive-class weight for BCE (use >1 when positives are fewer)')
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows = load_rows(args.train_csv)
    valid_rows = load_rows(args.valid_csv) if args.valid_csv else None

    # Pre-embedding class counts (visibility + early W&B logging)
    def _count_pos_neg(rows):
        if rows is None:
            return (0, 0)
        p = sum(1 for r in rows if int(r["pam_tttv"]) == 1)
        n = sum(1 for r in rows if int(r["pam_tttv"]) == 0)
        return (p, n)
    tr_pos, tr_neg = _count_pos_neg(train_rows)
    va_pos, va_neg = _count_pos_neg(valid_rows)
    print(f"[dataset] train: pos={tr_pos} neg={tr_neg} total={tr_pos+tr_neg}")
    if valid_rows is not None:
        print(f"[dataset] valid: pos={va_pos} neg={va_neg} total={va_pos+va_neg}")

    # Early W&B init (so run appears before long embedding step)
    use_wandb = bool(args.wandb_project) and _WANDB_AVAILABLE
    if use_wandb:
        init_kwargs = dict(project=args.wandb_project)
        if args.wandb_entity:
            init_kwargs['entity'] = args.wandb_entity
        if args.wandb_run_name:
            init_kwargs['name'] = args.wandb_run_name
        run = wandb.init(**init_kwargs)
        wandb.config.update({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'esm_model': 'esm2_t33_650M_UR50D',
            'task': 'pam_tttv',
            'train_size': len(train_rows),
            'valid_size': 0 if valid_rows is None else len(valid_rows),
            'train_pos': tr_pos,
            'train_neg': tr_neg,
            'valid_pos': va_pos,
            'valid_neg': va_neg,
        }, allow_val_change=True)

    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    embed_dim = backbone.embed_dim
    backbone = backbone.to(device).eval()

    # Select embedding policy
    if args.embed_policy == 'concat_head_tail1022':
        embed_fn = embed_with_cls_concat_two
        in_dim = embed_dim * 2
        embed_policy_name = 'concat_head_tail1022'
    else:
        embed_fn = embed_cls_tail_only
        in_dim = embed_dim
        embed_policy_name = 'cls_tail_1022'

    Xtr, ytr, ytr_snr, mtr_snr, drop_tr = preembed(train_rows, backbone, alphabet, device, embedder=embed_fn)
    if drop_tr: print(f"[warn] dropped {drop_tr} long sequences in train.", file=sys.stderr)
    print(f"[stats] train_embedded: n={len(ytr)} pos={(ytr==1).sum().item()} neg={(ytr==0).sum().item()}")
    if valid_rows:
        Xva, yva, yva_snr, mva_snr, drop_va = preembed(valid_rows, backbone, alphabet, device, embedder=embed_fn)
        if drop_va: print(f"[warn] dropped {drop_va} long sequences in valid.", file=sys.stderr)
        print(f"[stats] valid_embedded: n={len(yva)} pos={(yva==1).sum().item()} neg={(yva==0).sum().item()}")
    else:
        Xva = yva = None
        yva_snr = mva_snr = None

    # class imbalance / positive class weight
    pos = float((ytr == 1).sum().item()); neg = float((ytr == 0).sum().item())
    if args.pos_weight_fixed is not None:
        pos_w = float(args.pos_weight_fixed)
        pw_note = 'fixed'
    else:
        base = (neg / (pos + 1e-6)) if pos > 0 else 1.0
        pos_w = base * float(args.pos_weight_scale)
        if args.pos_weight_min and pos_w < float(args.pos_weight_min):
            pos_w = float(args.pos_weight_min)
        pw_note = 'scaled'
    print(f"[stats] train: n={len(ytr)}  pos={int(pos)} neg={int(neg)}  pos_weight={pos_w:.3f} ({pw_note})")

    clf = MLPHeads(in_dim=in_dim, hidden=512, p_drop=0.2, with_snr=bool(args.aux_snr_weight>0)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr)
    use_balanced = (args.bce_weighting == 'balanced')
    if not use_balanced:
        # We'll compute BCE per-sample to allow optional SNR-derived weights; keep class pos_weight.
        criterion_cls = None
        sample_weights = torch.ones_like(ytr, dtype=torch.float32, device=device)
        w_pos = w_neg = None  # not used in this branch
    else:
        criterion_cls = None
        # Build class-balanced sample weights
        n_pos = int((ytr == 1).sum().item())
        n_neg = int((ytr == 0).sum().item())
        n_tot = max(1, n_pos + n_neg)
        w_pos = (n_tot / (2.0 * max(1, n_pos))) * float(args.bce_pos_mult)
        w_neg = (n_tot / (2.0 * max(1, n_neg))) * float(args.bce_neg_mult)
        sample_weights = torch.where(ytr > 0.5, torch.tensor(w_pos, device=device), torch.tensor(w_neg, device=device))
        print(f"[bce] weighting=balanced  n_pos={n_pos} n_neg={n_neg}  w_pos={w_pos:.4f} w_neg={w_neg:.4f}", file=sys.stderr)
    # Optional SNR-based sample weighting (train set only)
    snr_weight_vec = None
    if args.snr_sample_weight_mode != 'none' and ytr_snr is not None and float(mtr_snr.sum().item()) > 0:
        snr_vals = torch.clamp(ytr_snr, min=0.0)
        if args.snr_sample_weight_mode == 'log1p':
            w_snr = torch.log1p(snr_vals)
        elif args.snr_sample_weight_mode == 'linear':
            w_snr = snr_vals.clone()
        else:
            w_snr = torch.ones_like(snr_vals)
        # Normalize by mean (over available SNRs) to keep average ≈ 1
        if args.snr_weight_norm == 'mean':
            denom = w_snr[mtr_snr > 0.5].mean() if (mtr_snr > 0.5).any() else None
            if denom is not None and torch.isfinite(denom):
                w_snr = w_snr / (denom + 1e-8)
        # Clamp and fill missing with 1.0
        w_snr = torch.clamp(w_snr, min=float(args.snr_weight_min), max=float(args.snr_weight_max))
        snr_weight_vec = torch.where(mtr_snr > 0.5, w_snr, torch.tensor(1.0, device=device))
    criterion_snr = nn.MSELoss(reduction='none') if args.aux_snr_weight>0 else None

    # W&B config updates after we know embedding policy and pos_weight
    if use_wandb:
        wandb.config.update({
            'embed_policy': embed_policy_name,
            'pos_weight': pos_w,
            'pos_weight_scale': args.pos_weight_scale,
            'pos_weight_min': args.pos_weight_min,
            'pos_weight_fixed': args.pos_weight_fixed,
            'bce_weighting': args.bce_weighting,
            'bce_neg_mult': args.bce_neg_mult,
            'bce_pos_mult': args.bce_pos_mult,
            'w_pos': None if not use_balanced else float(w_pos),
            'w_neg': None if not use_balanced else float(w_neg),
            'snr_sample_weight_mode': args.snr_sample_weight_mode,
            'snr_weight_min': args.snr_weight_min,
            'snr_weight_max': args.snr_weight_max,
            'snr_weight_norm': args.snr_weight_norm,
        }, allow_val_change=True)
        if 'snr_weight_vec' in locals() and snr_weight_vec is not None:
            try:
                wandb.config.update({
                    'snr_weight_mean': float(torch.mean(snr_weight_vec).item()),
                    'snr_weight_min_actual': float(torch.min(snr_weight_vec).item()),
                    'snr_weight_max_actual': float(torch.max(snr_weight_vec).item()),
                }, allow_val_change=True)
            except Exception:
                pass

    # train
    n = len(ytr)
    best_val = float('inf')
    best_state = None
    patience_left = args.early_stop_patience if (args.early_stop_patience and Xva is not None) else 0
    for ep in range(1, args.epochs+1):
        clf.train()
        if args.balance_oversample and not use_balanced:
            pos_idx = torch.nonzero(ytr == 1, as_tuple=False).flatten()
            neg_idx = torch.nonzero(ytr == 0, as_tuple=False).flatten()
            if len(pos_idx) and len(neg_idx):
                # Oversample minority to match majority
                if len(pos_idx) < len(neg_idx):
                    extra = pos_idx[torch.randint(0, len(pos_idx), (len(neg_idx)-len(pos_idx),), device=device)]
                    perm = torch.cat([neg_idx, pos_idx, extra])
                else:
                    extra = neg_idx[torch.randint(0, len(neg_idx), (len(pos_idx)-len(neg_idx),), device=device)]
                    perm = torch.cat([pos_idx, neg_idx, extra])
                perm = perm[torch.randperm(len(perm), device=device)]
            else:
                perm = torch.randperm(n, device=device)
        else:
            perm = torch.randperm(n, device=device)
        total = 0.0
        total_snr = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i+args.batch_size]
            logits, snr_pred = clf(Xtr[idx])
            # Build per-sample weights for classifier loss
            if use_balanced:
                w = sample_weights[idx]
            else:
                w = torch.ones_like(ytr[idx], dtype=torch.float32, device=device)
            if snr_weight_vec is not None:
                w = w * snr_weight_vec[idx]
            # Compute BCE with optional class pos_weight (unbalanced branch) and per-sample weights
            if not use_balanced:
                lvec = F.binary_cross_entropy_with_logits(
                    logits, ytr[idx], reduction='none',
                    pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device)
                )
            else:
                lvec = F.binary_cross_entropy_with_logits(logits, ytr[idx], reduction='none')
            loss_cls = (lvec * w).sum() / (w.sum() + 1e-8)
            loss = loss_cls
            if criterion_snr is not None and ytr_snr is not None:
                tgt = ytr_snr[idx]
                if args.aux_snr_log1p:
                    tgt = torch.log1p(torch.clamp(tgt, min=0.0))
                mask = mtr_snr[idx]
                if mask.sum() > 0:
                    loss_snr_vec = criterion_snr(snr_pred, tgt)
                    loss_snr = (loss_snr_vec * mask).sum() / mask.sum()
                    loss = loss + float(args.aux_snr_weight) * loss_snr
                    total_snr += loss_snr.item() * len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss_cls.item() * len(idx)
        tr_loss = total / n
        if Xva is not None:
            va_loss, va_auc = evaluate(clf, Xva, yva)
            # confusion metrics at threshold 0.5 (log to stdout and W&B)
            tp, tn, fp, fn = confusion_counts(clf, Xva, yva, batch_size=args.batch_size, threshold=0.5)
            print(f"epoch {ep}: train_loss={tr_loss:.4f}  valid_loss={va_loss:.4f}  valid_auroc={va_auc:.3f}  TP={tp} TN={tn} FP={fp} FN={fn}")
            # Optional validation SNR metrics
            snr_eval = eval_snr_metrics(clf, Xva, yva_snr, mva_snr, batch_size=args.batch_size, log1p=args.aux_snr_log1p) if args.aux_snr_weight>0 else None
            corr_eval = eval_logit_snr_corr(clf, Xva, yva_snr, mva_snr, batch_size=args.batch_size, log1p=args.aux_snr_log1p)
            if snr_eval is not None:
                print(f"[snr] valid_mse={snr_eval['valid_snr_mse']:.4f}  valid_r={snr_eval['valid_snr_r']:.3f}")
            if corr_eval is not None:
                print(f"[snr] corr(logit,y_snr)={corr_eval['valid_corr_logit_y_snr']:.3f}  corr(prob,y_snr)={corr_eval['valid_corr_prob_y_snr']:.3f}")
            if use_wandb:
                try:
                    lr_cur = opt.param_groups[0].get('lr', args.lr)
                except Exception:
                    lr_cur = args.lr
                wandb.log({
                    'epoch': ep,
                    'train_loss': tr_loss,
                    'valid_loss': va_loss,
                    'valid_auroc': float(va_auc),
                    'lr': lr_cur,
                    'valid_tp@0.5': tp,
                    'valid_tn@0.5': tn,
                    'valid_fp@0.5': fp,
                    'valid_fn@0.5': fn,
                    'train_loss_snr': (total_snr / n) if (criterion_snr is not None and n>0) else None,
                    **(snr_eval or {}),
                    **(corr_eval or {}),
                }, step=ep)
            # Early stopping on validation loss
            if args.early_stop_patience:
                if va_loss < best_val - 1e-6:
                    best_val = va_loss
                    best_state = {k: v.detach().cpu() for k, v in clf.state_dict().items()}
                    patience_left = args.early_stop_patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"[early-stop] stop at epoch {ep} (no val improvement for {args.early_stop_patience} epochs)")
                        break
        else:
            print(f"epoch {ep}: train_loss={tr_loss:.4f}")
            if use_wandb:
                try:
                    lr_cur = opt.param_groups[0].get('lr', args.lr)
                except Exception:
                    lr_cur = args.lr
                wandb.log({'epoch': ep, 'train_loss': tr_loss, 'lr': lr_cur}, step=ep)

    # Restore best state if early stopped
    if best_state is not None:
        clf.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save({
        "state_dict": clf.state_dict(),
        "in_dim": in_dim,
        "hidden": 512,
        "task": "pam_tttv",
        "esm": "esm2_t33_650M_UR50D",
        "cls_policy": embed_policy_name,
    }, args.out_model)
    print(f"[saved] {args.out_model}")
    if use_wandb:
        try:
            art = wandb.Artifact('pam_tttv_classifier', type='model')
            art.add_file(args.out_model)
            wandb.log_artifact(art)
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
