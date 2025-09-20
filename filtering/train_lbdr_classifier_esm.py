#!/usr/bin/env python3
import argparse, os, sys, csv, math, torch
import random
import numpy as np
from typing import List, Tuple, Optional
from torch import nn
from torch.utils.data import Dataset, DataLoader

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

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -----------------------
# Data
# -----------------------
class SeqClsDataset(Dataset):
    """
    Expects CSV with at least:
      - 'protein_seq'
      - 'dr_exact' (binary label: 0/1)  -> LbCas12a DR-exact target
    """
    def __init__(self, rows):
        self.rows = [r for r in rows if r.get('dr_exact') is not None]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return r['protein_seq'], int(r['dr_exact'])

def load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        need = {'protein_seq', 'dr_exact'}
        missing = [c for c in need if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")
        for r in reader:
            dr = r['dr_exact']
            rows.append({
                'protein_seq': r['protein_seq'],
                'dr_exact': int(dr) if dr not in ('', 'None') else None
            })
    return rows

# -----------------------
# ESM-2 embeddings (CLS)
# -----------------------
def embed_with_cls_concat_two(
    seq: str,
    model,
    alphabet,
    device,
    last_layer: int = 33,
    max_tokens_per_chunk: int = 1022,
    drop_len_threshold: int = 20000,
):
    """
    Return 2x-CLS vector: [CLS(head <=1022 aa), CLS(tail <=1022 aa)].
    If seq fits in one window, pad the second CLS with zeros.
    """
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
            rep = out["representations"][last_layer][0]  # (L+1, D)
        return rep[0].cpu()  # CLS at index 0

    if len(seq) <= max_tokens_per_chunk:
        cls1 = _cls_for(seq)
        cls2 = torch.zeros_like(cls1)
        return torch.cat([cls1, cls2], dim=-1)

    head = seq[:max_tokens_per_chunk]
    tail = seq[-max_tokens_per_chunk:]
    cls_head = _cls_for(head)
    cls_tail = _cls_for(tail)
    return torch.cat([cls_head, cls_tail], dim=-1)

# -----------------------
# Classifier
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1280, hidden=512, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # -> (B,)
        return self.net(x).squeeze(-1)

# -----------------------
# Train/eval loops
# -----------------------
def preembed(rows, model, alphabet, device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    feats, labels, dropped = [], [], 0
    for seq, lbl in [(r['protein_seq'], int(r['dr_exact'])) for r in rows]:
        z = embed_with_cls_concat_two(seq, model, alphabet, device)
        if z is None:
            dropped += 1; continue
        feats.append(z); labels.append(lbl)
    X = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return X, y, dropped

@torch.no_grad()
def evaluate(clf, X, y, batch_size=256) -> Tuple[float, float]:
    """
    Returns (loss_bce, auroc_approx). AUROC is computed by a fast rank-based approximation
    if scikit-learn is not installed.
    """
    clf.eval()
    crit = nn.BCEWithLogitsLoss()
    all_logits = []
    total = 0.0
    for i in range(0, len(y), batch_size):
        logits = clf(X[i:i+batch_size])
        loss = crit(logits, y[i:i+batch_size])
        total += loss.item() * len(logits)
        all_logits.append(logits.detach().cpu())
    mean_loss = total / len(y)
    scores = torch.sigmoid(torch.cat(all_logits)).numpy()
    labels = y.detach().cpu().numpy()

    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, scores)
    except Exception:
        # Fallback: Wilcoxon/Mann–Whitney U approximation to AUROC
        pos = scores[labels == 1]; neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            auroc = float('nan')
        else:
            # AUROC = P(score_pos > score_neg)
            import numpy as _np
            auroc = (pos.reshape(-1,1) > neg.reshape(1,-1)).mean()
    return mean_loss, auroc

def train_lb_dr(
    train_rows,
    valid_rows: Optional[list],
    outpath: str,
    epochs=6,
    batch_size=32,
    lr=1e-3,
    device='cuda',
    esm_layer=33,
    early_stop_patience: int = 0,
    pos_weight_scale: float = 1.0,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    print(f"[LbCas12a DR] train_n={len(train_rows)} valid_n={0 if valid_rows is None else len(valid_rows)}")

    # Load ESM-2 650M (t33)
    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    embed_dim = backbone.embed_dim
    backbone = backbone.to(device).eval()

    # Pre-embed
    Xtr, ytr, drop_tr = preembed(train_rows, backbone, alphabet, device)
    if drop_tr:
        print(f"Dropped {drop_tr} train sequences longer than 20000 aa.")
    Xva = yva = None
    if valid_rows:
        Xva, yva, drop_va = preembed(valid_rows, backbone, alphabet, device)
        if drop_va:
            print(f"Dropped {drop_va} valid sequences longer than 20000 aa.")

    # Class imbalance
    pos = float((ytr == 1).sum().item())
    neg = float((ytr == 0).sum().item())
    pos_w = (neg / (pos + 1e-6)) if pos > 0 else 1.0
    if pos_weight_scale and pos_weight_scale > 0:
        pos_w *= float(pos_weight_scale)
    print(f"Class balance (train): pos={int(pos)} neg={int(neg)} pos_weight={pos_w:.2f} (scale={pos_weight_scale:.2f})")

    # Classifier
    clf = MLP(in_dim=embed_dim * 2, hidden=512, p_drop=0.2).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    # W&B init (optional)
    use_wandb = bool(wandb_project) and _WANDB_AVAILABLE
    if use_wandb:
        wandb_kwargs = dict(project=wandb_project)
        if wandb_entity:
            wandb_kwargs.update(entity=wandb_entity)
        if wandb_run_name:
            wandb_kwargs.update(name=wandb_run_name)
        wandb.init(**wandb_kwargs)
        wandb.config.update({
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'esm_model': 'esm2_t33_650M_UR50D',
            'embed_policy': 'concat_head_tail1022',
            'task': 'lb_dr_exact',
            'train_size': len(train_rows),
            'valid_size': 0 if valid_rows is None else len(valid_rows),
            'pos_weight': pos_w,
        }, allow_val_change=True)

    # Training (features are precomputed; simple in‑memory loop)
    n = len(ytr)
    best_val = float('inf')
    best_state = None
    patience_left = early_stop_patience if (early_stop_patience and valid_rows) else 0
    for ep in range(1, epochs + 1):
        clf.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = clf(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)

        tr_loss = total / n
        # Quick validation if provided
        log_dict = {'epoch': ep, 'train_loss': tr_loss}
        if Xva is not None:
            va_loss, va_auc = evaluate(clf, Xva, yva)
            log_dict.update({'valid_loss': va_loss, 'valid_auroc': float(va_auc) if va_auc==va_auc else None})
            print(f"epoch {ep}: train_loss={tr_loss:.4f}  valid_loss={va_loss:.4f}  valid_auroc={va_auc:.3f}")
            # Early stopping on validation loss
            if va_loss < best_val - 1e-6:
                best_val = va_loss
                best_state = {k: v.detach().cpu() for k, v in clf.state_dict().items()}
                if early_stop_patience:
                    patience_left = early_stop_patience
            else:
                if early_stop_patience:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"Early stopping at epoch {ep} (no val improvement for {early_stop_patience} epochs)")
                        break
        else:
            print(f"epoch {ep}: train_loss={tr_loss:.4f}")
        if use_wandb:
            # also record learning rate from optimizer
            try:
                lr_cur = opt.param_groups[0].get('lr', lr)
            except Exception:
                lr_cur = lr
            log_dict['lr'] = lr_cur
            wandb.log(log_dict, step=ep)
        # End epoch
    # Restore best state if early stopped
    if best_state is not None:
        clf.load_state_dict(best_state)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    torch.save({
        'state_dict': clf.state_dict(),
        'in_dim': embed_dim * 2,
        'hidden': 512,
        'task': 'lb_dr_exact',
        'esm': 'esm2_t33_650M_UR50D',
        'cls_policy': 'concat_head_tail1022',
    }, outpath)
    print(f"Saved LbCas12a DR classifier to {outpath}")
    if use_wandb:
        # Log artifact path
        try:
            art = wandb.Artifact('lbdr_classifier', type='model')
            art.add_file(outpath)
            wandb.log_artifact(art)
        except Exception:
            pass
        wandb.finish()

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Train ESM‑2([CLS])→MLP classifier for LbCas12a DR‑exact compatibility.")
    ap.add_argument('--train_csv', required=True, help="CSV with columns: protein_seq, dr_exact")
    ap.add_argument('--valid_csv', default=None, help="Optional CSV for validation (same schema)")
    ap.add_argument('--out_model', required=True, help="Path to write classifier .pt")
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--early-stop-patience', type=int, default=0, help="Stop if val loss doesn't improve for N epochs")
    ap.add_argument('--pos-weight-scale', type=float, default=1.0, help="Multiply computed pos_weight by this factor")
    ap.add_argument('--wandb-project', default=None, help="W&B project (e.g., crispr_filtering)")
    ap.add_argument('--wandb-entity', default=None, help="W&B entity/org (optional)")
    ap.add_argument('--wandb-run-name', default=None, help="W&B run name (optional)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_rows = load_rows(args.train_csv)
    valid_rows = load_rows(args.valid_csv) if args.valid_csv else None

    train_lb_dr(
        train_rows=train_rows,
        valid_rows=valid_rows,
        outpath=args.out_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        early_stop_patience=args.early_stop_patience,
        pos_weight_scale=args.pos_weight_scale,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

if __name__ == '__main__':
    main()
