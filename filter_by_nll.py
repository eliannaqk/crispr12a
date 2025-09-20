import os
import json
from typing import Any, Optional

import click
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer


def _read_config_file(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        if config_path.endswith((".yml", ".yaml")):
            try:
                from ruamel.yaml import YAML
                yaml = YAML(typ="safe")
                return yaml.load(f)
            except Exception:
                f.seek(0)
                return json.load(f)
        else:
            return json.load(f)


def _collate_eval(seqs: list[str], tokenizer):
    batch = tokenizer(seqs, return_tensors="pt", padding=True)
    return {
        "input_ids": batch["input_ids"],
        "labels": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }


def _compute_mean_nll(model, tokenizer, sequences: list[str], batch_size: int, device: str) -> list[float]:
    vals: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            batch = _collate_eval(chunk, tokenizer)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, return_dict=True)

            logits = out.logits  # [B, T, V]
            labels = batch["labels"]  # [B, T]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            per_token_loss = F.cross_entropy(
                shift_logits.reshape(-1, vocab_size),
                shift_labels.reshape(-1),
                ignore_index=0,
                reduction="none",
            ).view(shift_labels.size())  # [B, T-1]
            valid_mask = (shift_labels != 0).to(per_token_loss.dtype)
            valid_counts = torch.clamp(valid_mask.sum(dim=1), min=1)
            per_seq_mean_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_counts
            vals.extend([float(x.item()) for x in per_seq_mean_loss])
    return vals


@click.command()
@click.option("--input-csv", required=True, help="CSV with sequences to score (expects a 'sequence' column by default)")
@click.option("--sequence-col", default="sequence", show_default=True, help="Column name containing sequences")
@click.option("--output-csv", required=True, help="Where to write the filtered CSV")
@click.option("--top-k", type=int, default=None, help="Keep only the best K sequences (lowest mean NLL)")
@click.option("--batch-size", type=int, default=8, show_default=True, help="Batch size for scoring")
@click.option("--model-path", required=False, help="Base HF model path (if not inferrable from config)")
@click.option("--ckpt", required=False, help="Optional fine-tuned HF folder or checkpoint to load")
@click.option("--config", "config_path", required=False, help="Optional training config to resolve model.path")
def main(input_csv: str, sequence_col: str, output_csv: str, top_k: Optional[int], batch_size: int,
         model_path: Optional[str], ckpt: Optional[str], config_path: Optional[str]):
    # Resolve model path
    base_model_path: Optional[str] = model_path
    if base_model_path is None and config_path is not None:
        try:
            cfg = _read_config_file(config_path)
            if isinstance(cfg, dict):
                base_model_path = cfg.get("model", {}).get("path")  # type: ignore
        except Exception:
            base_model_path = None
    if base_model_path is None:
        raise click.UsageError("--model-path not provided and could not resolve from --config.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer(ModelSchema(name="progen2"))
    model = get_model(ModelSchema(name="progen2", path=(ckpt or base_model_path)))
    model.to(device)
    model.eval()

    df = pd.read_csv(input_csv)
    if sequence_col not in df.columns:
        raise click.UsageError(f"Column '{sequence_col}' not found in {input_csv}")
    sequences = df[sequence_col].astype(str).tolist()

    scores = _compute_mean_nll(model, tokenizer, sequences, batch_size, device)
    df_out = df.copy()
    df_out["mean_nll"] = scores
    df_out["avg_logprob"] = -df_out["mean_nll"]
    df_out.sort_values(by=["mean_nll", "avg_logprob"], ascending=[True, False], inplace=True)
    if top_k is not None:
        df_out = df_out.head(top_k).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote filtered sequences to: {output_csv}")


if __name__ == "__main__":
    main()

