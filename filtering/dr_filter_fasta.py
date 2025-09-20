#!/usr/bin/env python3
"""
Step 4: LbCas12a DR classifier filtering over FASTA.

Reads an input FASTA (output of Step 3: PPL filter), runs an ESM‑2(CLS)->MLP
classifier to predict DR binding, and writes:
  - out_fasta: sequences with prob >= threshold
  - out_tsv: per-sequence report with id, n_aa, logit, prob, keep

Example:
  python filtering/dr_filter_fasta.py \
    --classifier_pt /path/to/lbdr_cls.pt \
    --in_fasta seed1.step3_ppl_pass.fasta \
    --out_fasta seed1.step4_dr_pass.fasta \
    --out_tsv seed1.step4_dr_report.tsv \
    --threshold 0.5 --device cuda --batch-size 16
"""
import argparse
import csv
import os
from typing import Iterator, Tuple, List, Optional

import torch
from torch import nn

# Support both "python -m filtering.dr_filter_fasta" (package context)
# and "python filtering/dr_filter_fasta.py" (script context).
try:
    from .train_lbdr_classifier_esm import (
        embed_with_cls_concat_two,
        MLP,
    )
except Exception:  # fallback when executed as a script without package context
    from filtering.train_lbdr_classifier_esm import (
        embed_with_cls_concat_two,
        MLP,
    )
import esm  # fair-esm


def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    rid = None
    buf = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if rid is not None:
                    yield rid, ''.join(buf)
                rid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if rid is not None:
            yield rid, ''.join(buf)


def _batched(it: Iterator[Tuple[str, str]], batch_size: int):
    buf: List[Tuple[str, str]] = []
    for x in it:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


@torch.no_grad()
def embed_batch_cls_concat_two(
    seqs: List[str],
    model,
    alphabet,
    device: str,
    last_layer: int = 33,
    max_tokens_per_chunk: int = 1022,
    drop_len_threshold: int = 20000,
) -> List[Optional[torch.Tensor]]:
    """Return list of 2x-CLS vectors (CPU tensors) aligned with input seqs.
    Returns None for any sequence over drop_len_threshold.
    """
    model.eval()
    toks = alphabet.get_batch_converter()
    n = len(seqs)
    out: List[Optional[torch.Tensor]] = [None] * n

    # Indices by category
    fit_idx = [i for i, s in enumerate(seqs) if len(s) <= max_tokens_per_chunk]
    long_idx = [i for i, s in enumerate(seqs) if max_tokens_per_chunk < len(s) <= drop_len_threshold]

    # Fit sequences: one pass, pad second CLS with zeros
    if fit_idx:
        batch = [(f"prot_{i}", seqs[i]) for i in fit_idx]
        _, _, tokens = toks(batch)
        tokens = tokens.to(device)
        rep = model(tokens, repr_layers=[last_layer], return_contacts=False)["representations"][last_layer]
        for j, i in enumerate(fit_idx):
            cls = rep[j, 0].detach().cpu()
            out[i] = torch.cat([cls, torch.zeros_like(cls)], dim=-1)

    # Long sequences: two passes (heads, tails)
    if long_idx:
        heads = [(f"h_{i}", seqs[i][:max_tokens_per_chunk]) for i in long_idx]
        tails = [(f"t_{i}", seqs[i][-max_tokens_per_chunk:]) for i in long_idx]
        # Heads
        _, _, t_h = toks(heads)
        rep_h = model(t_h.to(device), repr_layers=[last_layer], return_contacts=False)["representations"][last_layer]
        # Tails
        _, _, t_t = toks(tails)
        rep_t = model(t_t.to(device), repr_layers=[last_layer], return_contacts=False)["representations"][last_layer]
        for j, i in enumerate(long_idx):
            h = rep_h[j, 0].detach().cpu()
            t = rep_t[j, 0].detach().cpu()
            out[i] = torch.cat([h, t], dim=-1)

    return out


@torch.no_grad()
def classify_fasta(
    in_fasta: str,
    backbone,
    alphabet,
    clf: nn.Module,
    device: str,
    threshold: float,
    batch_size: int = 16,
):
    """Yield per-seq rows: (id, n_aa, logit, prob, keep, seq) using batched ESM/MLP.
    Batch size controls how many sequences are embedded per forward. Adjust to GPU memory.
    """
    for batch in _batched(iter_fasta(in_fasta), batch_size):
        ids = [sid for sid, _ in batch]
        seqs = [seq.strip().upper() for _, seq in batch]
        n_aa = [len(s) for s in seqs]
        Z = embed_batch_cls_concat_two(seqs, backbone, alphabet, device)
        # Compute logits in a single classifier pass for valid embeddings
        valid_idx = [i for i, z in enumerate(Z) if z is not None]
        logits_cpu: List[float] = [float('nan')] * len(batch)
        probs_cpu: List[float] = [float('nan')] * len(batch)
        if valid_idx:
            X = torch.stack([Z[i] for i in valid_idx]).to(device)
            logits = clf(X).detach().cpu()
            probs = torch.sigmoid(logits)
            for k, i in enumerate(valid_idx):
                logits_cpu[i] = float(logits[k].item())
                probs_cpu[i] = float(probs[k].item())
        # Yield
        for i in range(len(batch)):
            logit = logits_cpu[i]
            prob = probs_cpu[i]
            keep = 1 if (prob == prob and prob >= threshold) else 0
            yield (ids[i], n_aa[i], logit, prob, keep, seqs[i])


def main():
    ap = argparse.ArgumentParser(description="LbDR classifier filter over FASTA")
    ap.add_argument('--classifier_pt', required=True, help='Path to lbdr_cls.pt')
    ap.add_argument('--in_fasta', required=True)
    ap.add_argument('--out_fasta', required=True)
    ap.add_argument('--out_tsv', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=16, help='Number of sequences per ESM/MLP forward pass (memory dependent)')
    args = ap.parse_args()

    # Load backbone and classifier
    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    backbone = backbone.to(args.device).eval()
    ckpt = torch.load(args.classifier_pt, map_location='cpu')
    in_dim = int(ckpt.get('in_dim', 2560))
    clf = MLP(in_dim=in_dim, hidden=512, p_drop=0.0)
    clf.load_state_dict(ckpt['state_dict'])
    clf = clf.to(args.device).eval()

    os.makedirs(os.path.dirname(args.out_fasta) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)

    n_total = 0
    n_kept = 0
    with open(args.out_fasta, 'w') as fout, open(args.out_tsv, 'w', newline='') as frpt:
        w = csv.writer(frpt, delimiter='\t')
        w.writerow(['id', 'n_aa', 'logit', 'prob', 'keep'])
        for sid, n_aa, logit, prob, keep, seq in classify_fasta(
            args.in_fasta, backbone, alphabet, clf, args.device, args.threshold, args.batch_size
        ):
            n_total += 1
            n_kept += int(keep)
            w.writerow([sid, n_aa, f"{logit:.6f}", f"{prob:.6f}", keep])
            if keep == 1:
                fout.write(f">{sid}\n{seq}\n")

    rate = (n_kept / n_total) if n_total > 0 else 0.0
    print(f"[dr] Kept {n_kept}/{n_total} sequences (threshold={args.threshold}, pass_rate={rate:.4f})")
    print(f"[ok] Wrote: {args.out_fasta}")
    print(f"[ok] Report: {args.out_tsv}")


if __name__ == '__main__':
    main()
