# train_esm2_cas12a_classifiers.py
#!/usr/bin/env python3
import argparse, os, sys, csv, math, torch
import random
import numpy as np
from typing import List, Tuple
from torch import nn
from torch.utils.data import Dataset, DataLoader
try:
    import esm  # pip install fair-esm
except ImportError:
    print("Please: pip install fair-esm", file=sys.stderr); raise

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class SeqClsDataset(Dataset):
    def __init__(self, rows, task_key):
        # rows: list of dicts with 'protein_seq', 'pam_tttv', 'dr_exact'
        self.rows = [r for r in rows if r.get(task_key) is not None]
        self.task_key = task_key

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]['protein_seq'], int(self.rows[i][self.task_key])

def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'protein_seq': r['protein_seq'],
                'pam_tttv': int(r['pam_tttv']) if r['pam_tttv'] != '' else None,
                'dr_exact': int(r['dr_exact']) if r['dr_exact'] not in ('', 'None') else None
            })
    return rows

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
    Build a fixed 2x-CLS embedding for a protein sequence.

    - If len(seq) > drop_len_threshold (default 20000 aa): return None (caller should drop example).
    - If len(seq) <= max_tokens_per_chunk (1022): take CLS on full sequence as first vector, and
      pad the second vector with zeros. Return concat [CLS_first, zeros].
    - Else (longer): take at most TWO chunks — the first `max_tokens_per_chunk` residues and the
      last `max_tokens_per_chunk` residues — run each separately and concatenate their CLS vectors
      [CLS_head, CLS_tail]. This ensures at most two embeddings regardless of length.
    """
    if len(seq) > drop_len_threshold:
        return None
    model.eval()
    toks = alphabet.get_batch_converter()

    def _cls_for(subseq: str):
        batch = [("protein", subseq)]
        _, _, tokens = toks(batch)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[last_layer], return_contacts=False)
        rep = out["representations"][last_layer][0]
        return rep[0].cpu()

    with torch.no_grad():
        if len(seq) <= max_tokens_per_chunk:
            cls1 = _cls_for(seq)
            cls2 = torch.zeros_like(cls1)
            return torch.cat([cls1, cls2], dim=-1)
        # Two-chunk policy: N-terminus and C-terminus windows
        head = seq[:max_tokens_per_chunk]
        tail = seq[-max_tokens_per_chunk:]
        cls_head = _cls_for(head)
        cls_tail = _cls_for(tail)
        return torch.cat([cls_head, cls_tail], dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim=1280, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_one(rows, task_key, outpath, epochs=6, batch_size=32, lr=1e-3, device='cuda'):
    print(f"Training task={task_key}, n={len(rows)}")
    ds = SeqClsDataset(rows, task_key)
    # class weights (handle imbalance)
    y = np.array([lbl for _, lbl in [ds[i] for i in range(len(ds))]])
    pos_w = (len(y)-y.sum()) / (y.sum() + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    # load ESM-2 650M
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device); model.eval()
    clf = MLP(in_dim=model.embed_dim * 2, hidden=512).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)

    # pre-embed all sequences to speed training
    feats, labels = [], []
    dropped = 0
    for i in range(len(ds)):
        seq, lbl = ds[i]
        z = embed_with_cls_concat_two(seq, model, alphabet, device)
        if z is None:
            dropped += 1
            continue
        feats.append(z)
        labels.append(lbl)
    if dropped:
        print(f"Dropped {dropped} sequences longer than 20000 aa for task={task_key}.")
    X = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    for ep in range(1, epochs+1):
        clf.train()
        perm = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch_size):
            idx = perm[i:i+batch_size]
            logits = clf(X[idx])
            loss = criterion(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {ep}: loss={total/len(y):.4f}")

    torch.save({'state_dict': clf.state_dict(),
                'in_dim': model.embed_dim * 2,
                'hidden': 512,
                'task': task_key}, outpath)
    print(f"Saved {task_key} classifier to {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.train_csv)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_one(rows, 'pam_tttv', os.path.join(args.out_dir, 'cls_pam_tttv.pt'),
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    train_one(rows, 'dr_exact', os.path.join(args.out_dir, 'cls_dr_exact.pt'),
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

if __name__ == '__main__':
    main()
