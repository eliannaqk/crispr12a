#!/usr/bin/env python3
"""
ESM-based PAM filter (core-aware strict→strict classifier) for full-length designs.

Inputs:
  - --designs: FASTA of protein designs (e.g., seedN.step4_motif_pass.fasta)
  - --model: Path to .pt checkpoint saved by filtering/train_pam_tttv_classifier_esm.py

Outputs in --outdir:
  - pass.faa: FASTA of sequences predicted to bind PAM (prob >= --threshold)
  - fail.faa: FASTA of sequences predicted negative
  - report.tsv: per-sequence probabilities and decisions

The checkpoint carries the embedding policy (head+tail concat or tail-only) and
input dimension so we can reproduce feature extraction.
"""

import argparse, os, sys, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

try:
    import esm  # fair-esm
except Exception as e:
    print("[error] fair-esm is required (pip install fair-esm)", file=sys.stderr)
    raise


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    seqs: List[Tuple[str, str]] = []
    cur_id: str = ''
    cur_seq: List[str] = []
    with open(path) as f:
        for line in f:
            if not line:
                continue
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if cur_id:
                    seqs.append((cur_id, ''.join(cur_seq)))
                cur_id = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line.strip())
    if cur_id:
        seqs.append((cur_id, ''.join(cur_seq)))
    return seqs


def write_fasta(path: Path, items: List[Tuple[str, str]]) -> None:
    with open(path, 'w') as out:
        for sid, seq in items:
            out.write(f'>{sid}\n')
            # wrap at 80 cols for readability
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")


class MLPHeads(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, p_drop: float = 0.0, with_snr: bool = False):
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
        snr_pred = None
        if self.with_snr:
            snr_pred = self.snr_head(h).squeeze(-1)
        return logits, snr_pred


@dataclass
class EmbedConfig:
    policy: str  # 'concat_head_tail1022' | 'cls_tail_1022'
    last_layer: int = 33
    max_chunk: int = 1022


@torch.no_grad()
def embed_batch_tail_only(records: List[Tuple[str, str]], model, alphabet, device: str, cfg: EmbedConfig) -> torch.Tensor:
    """Return [N, D] CLS embeddings from the last <=1022 AA window (tail-only)."""
    batch_converter = alphabet.get_batch_converter()
    embs: List[torch.Tensor] = []
    B = 32
    for i in range(0, len(records), B):
        chunk = records[i:i+B]
        data = [(sid, seq[-cfg.max_chunk:]) for sid, seq in chunk]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[cfg.last_layer], return_contacts=False)
        reps = out["representations"][cfg.last_layer]  # [B, L, D]
        cls = reps[:, 0, :].cpu()  # [B, D]
        embs.append(cls)
    return torch.cat(embs, dim=0)


@torch.no_grad()
def embed_batch_concat_head_tail(records: List[Tuple[str, str]], model, alphabet, device: str, cfg: EmbedConfig) -> torch.Tensor:
    """Return [N, 2D] CLS embeddings concatenating first and last <=1022 AA windows."""
    batch_converter = alphabet.get_batch_converter()
    B = 32
    heads: List[torch.Tensor] = []
    tails: List[torch.Tensor] = []
    # Heads
    for i in range(0, len(records), B):
        chunk = records[i:i+B]
        data = [(sid, seq[:cfg.max_chunk]) for sid, seq in chunk]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[cfg.last_layer], return_contacts=False)
        reps = out["representations"][cfg.last_layer]
        cls = reps[:, 0, :].cpu()
        heads.append(cls)
    # Tails
    for i in range(0, len(records), B):
        chunk = records[i:i+B]
        data = [(sid, seq[-cfg.max_chunk:]) for sid, seq in chunk]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[cfg.last_layer], return_contacts=False)
        reps = out["representations"][cfg.last_layer]
        cls = reps[:, 0, :].cpu()
        tails.append(cls)
    H = torch.cat(heads, dim=0)
    T = torch.cat(tails, dim=0)
    return torch.cat([H, T], dim=-1)


def load_checkpoint(model_path: Path) -> Dict:
    ckpt = torch.load(model_path, map_location='cpu')
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise ValueError(f"Invalid checkpoint: {model_path}")
    return ckpt


def main():
    ap = argparse.ArgumentParser(description="Filter proteins by PAM-binding probability using an ESM2 classifier.")
    ap.add_argument('--designs', required=True, help='Input FASTA of full-length protein designs')
    ap.add_argument('--outdir', required=True, help='Output directory for pass/fail/report')
    ap.add_argument('--model', required=True, help='Path to .pt checkpoint saved by train_pam_tttv_classifier_esm.py')
    ap.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for PASS (>=)')
    ap.add_argument('--device', default=None, help='cuda|cpu (auto if not set)')
    ap.add_argument('--batch', type=int, default=32, help='Batch size for embedding inference')
    args = ap.parse_args()

    in_fa = Path(args.designs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)

    # Read designs
    records = read_fasta(in_fa)
    if not records:
        print(f"[warn] No sequences in {in_fa}")
        # Emit empty outputs for consistency
        write_fasta(outdir / 'pass.faa', [])
        write_fasta(outdir / 'fail.faa', [])
        with open(outdir / 'report.tsv', 'w') as w:
            w.write('design_id\tprobability\tpredicted\tlogit\n')
        return

    # Load backbone + classifier
    device = (args.device or ('cuda' if torch.cuda.is_available() else 'cpu')).lower()
    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    backbone = backbone.to(device).eval()

    ckpt = load_checkpoint(model_path)
    in_dim = int(ckpt.get('in_dim', backbone.embed_dim))
    hidden = int(ckpt.get('hidden', 512))
    cls_policy = str(ckpt.get('cls_policy', 'concat_head_tail1022'))
    sd = ckpt['state_dict']
    has_snr = any(k.startswith('snr_head.') for k in sd.keys())
    clf = MLPHeads(in_dim=in_dim, hidden=hidden, p_drop=0.0, with_snr=has_snr)
    clf.load_state_dict(sd)
    clf = clf.to(device).eval()

    cfg = EmbedConfig(policy=cls_policy)

    # Embed
    if cfg.policy == 'cls_tail_1022':
        X = embed_batch_tail_only(records, backbone, alphabet, device, cfg)
    else:
        X = embed_batch_concat_head_tail(records, backbone, alphabet, device, cfg)
    X = X.to(device)

    # Classify
    with torch.no_grad():
        logits, _ = clf(X)
        probs = torch.sigmoid(logits)
    probs_cpu = probs.detach().cpu().tolist()
    logits_cpu = logits.detach().cpu().tolist()

    # Partition pass/fail
    pass_items: List[Tuple[str, str]] = []
    fail_items: List[Tuple[str, str]] = []
    for (sid, seq), p, z in zip(records, probs_cpu, logits_cpu):
        if p >= args.threshold:
            pass_items.append((sid, seq))
        else:
            fail_items.append((sid, seq))

    write_fasta(outdir / 'pass.faa', pass_items)
    write_fasta(outdir / 'fail.faa', fail_items)
    with open(outdir / 'report.tsv', 'w') as w:
        w.write('design_id\tprobability\tpredicted\tlogit\n')
        for (sid, _), p, z in zip(records, probs_cpu, logits_cpu):
            lab = 'PASS' if p >= args.threshold else 'FAIL'
            w.write(f"{sid}\t{p:.6f}\t{lab}\t{z:.6f}\n")

    print(f"[pam] in={len(records)} pass={len(pass_items)} fail={len(fail_items)} thr={args.threshold}")
    print(f"[pam] wrote: {outdir/'pass.faa'}  {outdir/'fail.faa'}  {outdir/'report.tsv'}")


if __name__ == '__main__':
    main()

