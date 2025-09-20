#!/usr/bin/env python3
"""
Score generated protein sequences with the LbDR classifier.

Input CSV format (from generate.py):
  columns: context_name, context, sequence
  - `sequence` may end with EOS tokens ('1' or '2') and often repeats the context
    as prefix; we normalize by removing any non-amino-acid characters (retain
    ACDEFGHIKLMNPQRSTVWY) and uppercasing.

Outputs a CSV with columns:
  context_name,context,sequence,protein_seq,logit,prob,label

Example:
  python -m filtering.score_generated_with_lbdr \
    --generated-dir /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000 \
    --classifier-pt /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/newCas12a/cas12a_dr_datasets/lbdr_cls.pt \
    --out-csv /home/eqk3/project_pi_mg269/eqk3/crisprData/atlas/generated_sequences/run-20250828-193254-ba4000-sweep/generations/ba4000/lbdr_scores_filtered.csv
"""
import argparse
import csv
import glob
import os
import re
from typing import Iterable, Tuple

import torch
from torch import nn

from .train_lbdr_classifier_esm import (
    embed_with_cls_concat_two,
    MLP,
)
import esm


AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]", re.IGNORECASE)


def normalize_protein(seq: str) -> str:
    """Uppercase and remove any non-20AA characters (digits like '1','2')."""
    return AA_RE.sub("", seq).upper()


def iter_generated_csvs(generated_dir: str, csv_glob: str) -> Iterable[str]:
    pattern = os.path.join(generated_dir, csv_glob)
    files = sorted(glob.glob(pattern))
    for p in files:
        yield p


def read_generated_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Expect keys: context_name, context, sequence
            ctxn = r.get('context_name', '')
            ctx = r.get('context', '')
            seq = r.get('sequence', '')
            yield (ctxn, ctx, seq)


@torch.no_grad()
def score_sequences(rows: Iterable[Tuple[str, str, str]], backbone, alphabet, clf: nn.Module, device: str, batch_write: int = 512):
    """Yield scored rows. Each input row is (context_name, context, sequence)."""
    for ctxn, ctx, seq in rows:
        prot = normalize_protein(seq)
        if not prot:
            yield (ctxn, ctx, seq, prot, float('nan'), float('nan'), -1)
            continue
        z = embed_with_cls_concat_two(prot, backbone, alphabet, device)
        if z is None:
            yield (ctxn, ctx, seq, prot, float('nan'), float('nan'), -1)
            continue
        z = z.to(device)
        logit = clf(z.unsqueeze(0)).squeeze(0).item()
        prob = float(torch.sigmoid(torch.tensor(logit)))
        label = 1 if prob >= 0.5 else 0
        yield (ctxn, ctx, seq, prot, logit, prob, label)


def main():
    ap = argparse.ArgumentParser(description="Score generated sequences with LbDR classifier")
    ap.add_argument('--generated-dir', required=True, help='Directory with generated CSVs (contains *_filtered.csv from generate.py)')
    ap.add_argument('--csv-glob', default='*_filtered.csv', help='Glob under generated-dir to select CSVs')
    ap.add_argument('--classifier-pt', required=True, help='Path to lbdr_cls.pt')
    ap.add_argument('--out-csv', required=True, help='Path to write combined scored CSV')
    ap.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for pass (label=1) summary')
    ap.add_argument('--summary-csv', default=None, help='Optional path to write per-file summary CSV')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    # Load ESM backbone and classifier
    backbone, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    backbone = backbone.to(args.device).eval()
    ckpt = torch.load(args.classifier_pt, map_location='cpu')
    in_dim = int(ckpt.get('in_dim', 2560))
    clf = MLP(in_dim=in_dim, hidden=512, p_drop=0.0)
    clf.load_state_dict(ckpt['state_dict'])
    clf = clf.to(args.device).eval()

    # Prepare output
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_cols = ['context_name', 'context', 'sequence', 'protein_seq', 'logit', 'prob', 'label']
    summaries = []  # per-file summaries
    total_all = dict(n_total=0, n_scored=0, n_passed=0)

    with open(args.out_csv, 'w', newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(out_cols)
        # Iterate files
        for csv_in in iter_generated_csvs(args.generated_dir, args.csv_glob):
            n_total = 0
            n_scored = 0
            n_passed = 0
            # Stream rows
            for row in read_generated_rows(csv_in):
                n_total += 1
                wrote_any = False
                for scored in score_sequences([row], backbone, alphabet, clf, args.device):
                    writer.writerow(scored)
                    wrote_any = True
                    # scored tuple: (ctxn, ctx, seq, prot, logit, prob, label)
                    prob = scored[5]
                    label = scored[6]
                    # label == -1 marks dropped/invalid
                    if label != -1:
                        n_scored += 1
                        if prob >= args.threshold:
                            n_passed += 1
                if not wrote_any:
                    # Should not happen, but keep accounting consistent
                    pass
            pass_rate = (n_passed / n_scored) if n_scored > 0 else float('nan')
            summaries.append((csv_in, n_total, n_scored, n_passed, pass_rate))
            total_all['n_total'] += n_total
            total_all['n_scored'] += n_scored
            total_all['n_passed'] += n_passed

    # Print summary to stdout
    print(f"Wrote scores to {args.out_csv}")
    print("Summary (per file):")
    for path, n_total, n_scored, n_passed, pass_rate in summaries:
        print(f" - {os.path.basename(path)}: total={n_total} scored={n_scored} passed={n_passed} pass_rate={pass_rate:.4f}")
    if total_all['n_scored'] > 0:
        overall_rate = total_all['n_passed'] / total_all['n_scored']
    else:
        overall_rate = float('nan')
    print(f"Overall: total={total_all['n_total']} scored={total_all['n_scored']} passed={total_all['n_passed']} pass_rate={overall_rate:.4f} (threshold={args.threshold})")

    # Optional summary CSV
    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        with open(args.summary_csv, 'w', newline='') as sf:
            sw = csv.writer(sf)
            sw.writerow(['file', 'n_total', 'n_scored', 'n_passed', 'pass_rate', 'threshold'])
            for path, n_total, n_scored, n_passed, pass_rate in summaries:
                sw.writerow([path, n_total, n_scored, n_passed, f"{pass_rate:.6f}", args.threshold])
            sw.writerow(['__OVERALL__', total_all['n_total'], total_all['n_scored'], total_all['n_passed'], f"{overall_rate:.6f}", args.threshold])
        print(f"Wrote summary to {args.summary_csv}")


if __name__ == '__main__':
    main()
