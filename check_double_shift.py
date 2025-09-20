#!/usr/bin/env python
import os
import json
import math
import argparse
from typing import Optional

import torch
import torch.nn.functional as F
import pandas as pd

from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer


def load_sequences(csv_dir: str, sequence_col: str, max_rows: int) -> list[str]:
    # Prefer val/valid.csv else train.csv
    for fname in ["valid.csv", "val.csv", "validation.csv", "dev.csv", "train.csv"]:
        p = os.path.join(csv_dir, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            seqs = df[sequence_col].astype(str).tolist()
            return seqs[:max_rows]
    raise FileNotFoundError(f"No CSV found under {csv_dir} (expected valid/val/dev/train)")


def wrap_and_tokenize(seqs: list[str], tokenizer, pad_right: bool = True):
    # Mirror training/eval: add boundary tokens "1" and "2"
    wrapped = ["1" + s + "2" for s in seqs]
    try:
        if pad_right:
            tokenizer.padding_side = "right"
    except Exception:
        pass
    batch = tokenizer(wrapped, return_tensors="pt", padding=True)
    # teacher forcing: labels = input_ids
    batch["labels"] = batch["input_ids"].clone()
    return batch


@torch.inference_mode()
def compute_losses(model, tokenizer, batch: dict, pad_id: int):
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in batch.items()}
    out = model(**inputs, return_dict=True)
    logits = out.logits.float()
    labels = inputs["labels"]

    # Manual single-shift (match model): logits[..., :-1] vs labels[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    V = shift_logits.size(-1)
    loss_manual_single = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=pad_id,
        reduction="mean",
    )

    # Emulate pre-shifted labels (wrapper shift) before passing to a model that also shifts internally
    # Create labels pre-shifted left by 1 (drop first token, pad at end with pad_id)
    pre = torch.empty_like(labels)
    pre[..., :-1] = labels[..., 1:]
    pre[..., -1] = pad_id
    out_pre = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), labels=pre, return_dict=True)
    logits_pre = out_pre.logits.float()
    # Manual double-shift relative to original (equivalent to model shifting again)
    # Compare logits[..., :-2] vs labels[..., 2:]
    if logits_pre.size(1) >= 3 and labels.size(1) >= 3:
        ds_logits = logits_pre[..., :-2, :].contiguous()
        ds_labels = labels[..., 2:].contiguous()
        loss_manual_double = F.cross_entropy(
            ds_logits.view(-1, V),
            ds_labels.view(-1),
            ignore_index=pad_id,
            reduction="mean",
        )
        loss_model_with_preshift = out_pre.loss
    else:
        loss_manual_double = torch.tensor(float("nan"))
        loss_model_with_preshift = torch.tensor(float("nan"))

    # Token counts for context
    am = inputs.get("attention_mask")
    if am is not None:
        valid_tokens_single = int((shift_labels != pad_id).sum().item())
        valid_tokens_double = int((labels[..., 2:] != pad_id).sum().item()) if labels.size(1) >= 3 else 0
    else:
        valid_tokens_single = shift_labels.numel()
        valid_tokens_double = labels[..., 2:].numel() if labels.size(1) >= 3 else 0

    return {
        "loss_model_single": float(out.loss.item()) if out.loss is not None else float("nan"),
        "loss_manual_single": float(loss_manual_single.item()),
        "loss_model_with_preshift": float(loss_model_with_preshift.item()) if torch.is_tensor(loss_model_with_preshift) else float("nan"),
        "loss_manual_double": float(loss_manual_double.item()) if torch.is_tensor(loss_manual_double) else float("nan"),
        "valid_tokens_single": valid_tokens_single,
        "valid_tokens_double": valid_tokens_double,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF folder to load model from (e.g., .../huggingface/baXXXX)")
    ap.add_argument("--data-dir", required=True, help="Directory with train/val CSVs")
    ap.add_argument("--sequence-col", default="sequence")
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--output", default="double_shift_check.json")
    args = ap.parse_args()

    print("[env] device: cpu (forcing CPU for this check)")
    model = get_model(ModelSchema(name="progen2", path=args.model_path))
    model.eval()
    tokenizer = get_tokenizer(ModelSchema(name="progen2", path=args.model_path))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    seqs = load_sequences(args.data_dir, args.sequence_col, args.max_rows)
    batch = wrap_and_tokenize(seqs, tokenizer, pad_right=True)

    # Forward passes and manual checks
    metrics = compute_losses(model, tokenizer, batch, pad_id=pad_id)

    # Simple decision heuristic
    decision = {
        "single_manual_matches_model": abs(metrics["loss_model_single"] - metrics["loss_manual_single"]) < 1e-6,
        "double_looks_like_preshift": (
            (not math.isnan(metrics["loss_model_with_preshift"]))
            and abs(metrics["loss_model_with_preshift"] - metrics.get("loss_manual_double", float("nan"))) < 1e-6
        ),
        "double_shift_lower_than_single": (
            not math.isnan(metrics.get("loss_manual_double", float("nan")))
            and metrics.get("loss_manual_double", float("inf")) < metrics["loss_manual_single"]
        ),
    }

    summary = {
        "model_path": args.model_path,
        "pad_id": pad_id,
        "num_sequences": len(seqs),
        **metrics,
        **{f"decision/{k}": v for k, v in decision.items()},
    }

    print("[summary]")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()

