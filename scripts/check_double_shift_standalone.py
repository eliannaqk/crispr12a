import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer
from opencrispr_repro.data import SeqDataset, progen2_collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF folder (e.g., .../huggingface/baXXXX) or pretrained size")
    ap.add_argument("--data", required=True, help="Path to val.csv/tsv")
    ap.add_argument("--sequence-col", default="sequence")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample", type=int, default=64, help="Optional sample size")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(ModelSchema(name="progen2", path=args.model_path)).to(device).eval()
    tokenizer = get_tokenizer(ModelSchema(name="progen2", path=args.model_path))
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0

    ds = SeqDataset(csv_fname=args.data, sequence_col=args.sequence_col, label_col=None)
    if args.sample and args.sample < len(ds):
        idx = np.random.RandomState(0).permutation(len(ds))[: args.sample]
        ds = Subset(ds, idx.tolist())

    def collate(items):
        return progen2_collate_fn(items, tokenizer=tokenizer, random_reverse=False)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    n1 = n2 = 0
    s1 = s2 = 0.0
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)  # no labels
            logits = out.logits
            V = logits.size(-1)

            # shift 1
            lg1 = logits[..., :-1, :].contiguous()
            lb1 = labels[..., 1:].contiguous()
            ce1 = F.cross_entropy(lg1.view(-1, V), lb1.view(-1), ignore_index=pad_id, reduction="sum")
            t1 = int((lb1 != pad_id).sum().item())
            s1 += float(ce1.item())
            n1 += t1

            # shift 2
            if logits.size(1) >= 3:
                lg2 = logits[..., :-2, :].contiguous()
                lb2 = labels[..., 2:].contiguous()
                ce2 = F.cross_entropy(lg2.view(-1, V), lb2.view(-1), ignore_index=pad_id, reduction="sum")
                t2 = int((lb2 != pad_id).sum().item())
                s2 += float(ce2.item())
                n2 += t2

    mean1 = s1 / max(1, n1)
    mean2 = s2 / max(1, n2) if n2 > 0 else float("nan")
    print("pad_id:", pad_id)
    print("token-weighted CE (1-shift):", mean1)
    print("token-weighted CE (2-shift):", mean2)
    print("tokens (1-shift):", n1, "tokens (2-shift):", n2)
    print("If your training-time 'loss/eval/total' ≈ 2-shift, it indicates double shifting in the wrapper.")


if __name__ == "__main__":
    main()

