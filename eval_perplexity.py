import os
import re
import math
import json
from typing import Any, Tuple, Dict, Optional

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer
from opencrispr_repro.data import SeqDataset, progen2_collate_fn  # keep import for parity, not used directly


# ----------------------------
# Utilities
# ----------------------------

def _find_latest_hf_dir(save_folder: str) -> Optional[str]:
    """Find latest HuggingFace export directory under save_folder/huggingface/baXXXX."""
    hf_root = os.path.join(save_folder, "huggingface")
    if not os.path.isdir(hf_root):
        return None
    step_dirs = [d for d in os.listdir(hf_root) if d.startswith("ba") and d[2:].isdigit()]
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda s: int(s[2:]), reverse=True)
    return os.path.join(hf_root, step_dirs[0])


def progen2_collate_eval(items, tokenizer, random_reverse: bool, pad_id: int):
    """Eval collate mirroring training with explicit pad masking.

    - Wrap with '1' + seq + '2'
    - Optional reversal
    - Right padding
    - labels = input_ids but pads and first token are set to -100 (HF ignore index)
    """
    wrapped = ["1" + it["sequence"] + "2" for it in items]
    if random_reverse:
        # 50/50 reversal like training's progen2_collate_fn(random_reverse=True)
        seqs = [w if np.random.random() < 0.5 else w[::-1] for w in wrapped]
    else:
        seqs = wrapped

    # Ensure right padding already set on tokenizer before DataLoader (outside)
    batch = tokenizer(seqs, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = input_ids.clone()
    labels = labels.masked_fill(attn == 0, -100)
    if labels.size(1) > 0:
        labels[:, 0] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
    }


def hf_like_teacher_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Match model-internal CE: fp32, token-level mean, ignore_index for pads."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def _load_model_from_ckpt(base_model_path: str, ckpt_path: Optional[str]) -> Tuple[Any, str]:
    """
    Build ProGen2 model and optionally load weights from a checkpoint path.

    Returns:
        (model, resolved_path)  # resolved_path = folder or file from which weights were actually loaded
    """
    model = get_model(ModelSchema(name="progen2", path=base_model_path))
    resolved = base_model_path
    if not ckpt_path:
        return model, resolved

    ckpt_path = os.path.realpath(ckpt_path)

    # Folder: Composer saves or HF export
    if os.path.isdir(ckpt_path):
        # Handle Composer autoresume semantics
        latest_path = os.path.join(ckpt_path, "latest")
        resolved_latest: Optional[str] = None
        if os.path.exists(latest_path):
            if os.path.islink(latest_path):
                try:
                    resolved_latest = os.readlink(latest_path)
                    if not os.path.isabs(resolved_latest):
                        resolved_latest = os.path.join(ckpt_path, resolved_latest)
                except Exception:
                    resolved_latest = None
            else:
                try:
                    with open(latest_path, "r") as f:
                        p = f.read().strip()
                        if p:
                            resolved_latest = p if os.path.isabs(p) else os.path.join(ckpt_path, p)
                except Exception:
                    resolved_latest = None

        if resolved_latest is not None:
            # If points to a Composer step dir with distcp shards, prefer HF sibling export
            if os.path.isdir(resolved_latest) and os.path.exists(os.path.join(resolved_latest, "__0_0.distcp")):
                ba_name = os.path.basename(resolved_latest)
                hf_sibling = os.path.join(os.path.dirname(resolved_latest), "huggingface", ba_name)
                if os.path.isdir(hf_sibling):
                    return get_model(ModelSchema(name="progen2", path=hf_sibling)), hf_sibling
            # If points directly to a file, try to load it
            if os.path.isfile(resolved_latest):
                try:
                    ckpt = torch.load(resolved_latest, map_location="cpu")
                    state = ckpt.get("state", {}).get("model", ckpt) if isinstance(ckpt, dict) else ckpt
                    if isinstance(state, dict) and "model" in state and not any(k.startswith("model.") for k in state.keys()):
                        state = state["model"]
                    model.load_state_dict(state, strict=False)
                    return model, resolved_latest
                except Exception:
                    pass

        # Composer save folder: pick best rank0 .pt
        try:
            pt_files = [f for f in os.listdir(ckpt_path) if f.endswith(".pt")]
            best_step, best_fname = -1, None
            for fname in pt_files:
                m = re.match(r"ba(\d+)-rank(\d+)\.pt$", fname)
                if m and m.group(2) == "0":
                    step = int(m.group(1))
                    if step > best_step:
                        best_step, best_fname = step, fname
            if best_fname is None and pt_files:
                rank0_files = [f for f in pt_files if "rank0" in f]
                best_fname = rank0_files[0] if rank0_files else pt_files[0]
            if best_fname is not None:
                ckpt_file = os.path.join(ckpt_path, best_fname)
                ckpt = torch.load(ckpt_file, map_location="cpu")
                if isinstance(ckpt, dict) and "state" in ckpt and isinstance(ckpt["state"], dict) and "model" in ckpt["state"]:
                    state = ckpt["state"]["model"]
                elif isinstance(ckpt, dict) and "model" in ckpt:
                    state = ckpt["model"]
                else:
                    state = ckpt
                model.load_state_dict(state, strict=False)
                return model, ckpt_file
        except Exception:
            pass

        # Try latest HF export under save_folder/huggingface/baXXXX
        hf_dir = _find_latest_hf_dir(ckpt_path)
        if hf_dir is not None:
            return get_model(ModelSchema(name="progen2", path=hf_dir)), hf_dir

        # If this is a 'baXXXX' dir with distcp shards, prefer sibling HF export
        base = os.path.basename(ckpt_path)
        if base.startswith("ba") and base[2:].isdigit() and os.path.exists(os.path.join(ckpt_path, "__0_0.distcp")):
            hf_sibling = os.path.join(os.path.dirname(ckpt_path), "huggingface", base)
            if os.path.isdir(hf_sibling):
                return get_model(ModelSchema(name="progen2", path=hf_sibling)), hf_sibling

        # Prefer safetensors if present
        st_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
        if st_files:
            try:
                from safetensors.torch import load_file  # type: ignore
                state = load_file(os.path.join(ckpt_path, st_files[0]))
                model.load_state_dict(state, strict=True)
                return model, os.path.join(ckpt_path, st_files[0])
            except Exception:
                pass

        # Fallback to PyTorch bin shards
        bin_files = sorted(
            f for f in os.listdir(ckpt_path) if f.startswith("pytorch_model") and f.endswith(".bin")
        )
        if bin_files:
            state_full: Dict[str, Any] = {}
            for bf in bin_files:
                shard = torch.load(os.path.join(ckpt_path, bf), map_location="cpu")
                if isinstance(shard, dict):
                    state_full.update(shard)
            model.load_state_dict(state_full, strict=True)
            return model, ckpt_path

    # Single safetensors
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore
        state = load_file(ckpt_path)
        model.load_state_dict(state, strict=True)
        return model, ckpt_path

    # Generic torch checkpoint file
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state" in ckpt and isinstance(ckpt["state"], dict) and "model" in ckpt["state"]:
            state = ckpt["state"]["model"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        model.load_state_dict(state, strict=True)
        return model, ckpt_path

    raise FileNotFoundError(f"Unrecognized checkpoint path or missing HF export next to distcp: {ckpt_path}")


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


def _auto_find_splits(data_dir: str) -> Tuple[Optional[str], Optional[str], str]:
    """Try to find train/val/test files by common names."""
    candidates: Dict[str, list[str]] = {
        "train": ["train.csv", "train.tsv"],
        "val":   ["val.csv","val.tsv","valid.csv","valid.tsv","validation.csv","validation.tsv","dev.csv","dev.tsv"],
        "test":  ["test.csv","test.tsv","eval.csv","eval.tsv","evaluation.csv","evaluation.tsv"],
    }
    found: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
    files = set(os.listdir(data_dir))
    for split, names in candidates.items():
        for n in names:
            if n in files:
                found[split] = os.path.join(data_dir, n)
                break
    if found["test"] is None:
        raise FileNotFoundError(
            f"Could not find a test file in {data_dir}. Expected one of: {', '.join(candidates['test'])}"
        )
    return found["train"], found["val"], found["test"]


def _extract_predicted_sequence(pred_tokens: list[str]) -> str:
    """Extract sequence between the first '1' and next '2' token; fallback to joined tokens."""
    s = "".join(pred_tokens)
    if "1" in s and "2" in s:
        i = s.find("1"); j = s.find("2", i + 1)
        if i != -1 and j != -1 and j > i:
            return s[i + 1 : j]
    return s.strip()


# ----------------------------
# Core evaluation
# ----------------------------

def _evaluate_dataset(model,
                     tokenizer,
                     csv_path: str,
                     sequence_col: str,
                     batch_size: int,
                     device: str,
                     split_name: str,
                     output_dir: str,
                     pad_id: int,
                     random_reverse: bool,
                     debug: bool = False,
                     sample_size: int | None = None,
                     seed: int = 42) -> dict:
    # Build dataset (optionally sample)
    base_ds = SeqDataset(csv_fname=csv_path, sequence_col=sequence_col, label_col=None)
    if sample_size is not None:
        num = len(base_ds)
        k = min(sample_size, num)
        if k < num:
            gen = torch.Generator(); gen.manual_seed(seed)
            idx = torch.randperm(num, generator=gen)[:k].tolist()
        else:
            idx = list(range(num))
        ds = Subset(base_ds, idx)
    else:
        ds = base_ds

    # Ensure right padding BEFORE DataLoader is built
    try: tokenizer.padding_side = "right"
    except Exception: pass

    def _collate_with_seq(items):
        batch = progen2_collate_eval(items, tokenizer=tokenizer, random_reverse=random_reverse, pad_id=pad_id)
        batch["sequences"] = [it["sequence"] for it in items]
        return batch

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=_collate_with_seq)

    per_seq_records: list[dict[str, Any]] = []
    total_valid_tokens = 0
    token_loss_sum_teacher = 0.0    # token-weighted mean (teacher CE)
    token_loss_sum_manual = 0.0     # token-weighted mean (manual CE)
    model_token_loss_sum = 0.0      # token-weighted mean (model-reported loss)
    batch_means: list[float] = []
    model_batch_means: list[float] = []
    seq_means: list[float] = []
    total_batches = 0

    did_first_batch_diag = False

    with torch.inference_mode():
        for batch in dl:
            sequences = batch.pop("sequences")
            batch = {k: v.to(device) for k, v in batch.items()}
            # Do not pass labels into the model because this ProGen2 head uses
            # ignore_index=0 internally, while our collate masks labels with -100.
            # We compute the teacher-forced CE manually below with ignore_index=-100.
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_dict=True)

            # One-time diagnostics
            if debug and not did_first_batch_diag:
                try:
                    print("pad/bos/eos:", pad_id,
                          getattr(tokenizer, "bos_token_id", None),
                          getattr(tokenizer, "eos_token_id", None))
                    uniq = torch.unique(batch["labels"]).detach().cpu()
                    num_ignored = int((batch["labels"] == -100).sum().item())
                    total_labels = int(batch["labels"].numel())
                    print("labels unique (showing up to 10):", uniq[:10].tolist(), "...",
                          "ignored(-100)=", num_ignored, "/", total_labels)
                except Exception:
                    pass
                try:
                    loss_model = float(out.loss.item()) if getattr(out, "loss", None) is not None else float("nan")
                    shift_logits_d = out.logits[..., :-1, :].contiguous().float()
                    shift_labels_d = batch["labels"][..., 1:].contiguous()
                    V = shift_logits_d.size(-1)
                    manual = F.cross_entropy(
                        shift_logits_d.view(-1, V),
                        shift_labels_d.view(-1),
                        ignore_index=-100,
                        reduction="mean",
                    ).item()
                    print("model loss vs manual:", loss_model, manual)
                except Exception:
                    pass
                did_first_batch_diag = True

            # Per-sequence losses & PPL
            logits = out.logits           # [B, T, V]
            labels = batch["labels"]      # [B, T]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            V = shift_logits.size(-1)

            per_token_loss = F.cross_entropy(
                shift_logits.reshape(-1, V),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.size())    # [B, T-1]

            valid_mask = (shift_labels != -100).to(per_token_loss.dtype)  # [B, T-1]
            valid_counts = valid_mask.sum(dim=1)                             # [B]
            valid_counts_safe = torch.clamp(valid_counts, min=1)
            per_seq_mean_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_counts_safe  # [B]
            per_seq_ppl = torch.exp(per_seq_mean_loss)

            # Aggregations
            ce_mean = hf_like_teacher_loss(logits, labels, ignore_index=-100)  # teacher CE
            batch_valid_tokens = int(valid_mask.sum().item())

            token_loss_sum_teacher += float(ce_mean.item() * batch_valid_tokens)
            token_loss_sum_manual  += float((per_token_loss * valid_mask).sum().item())
            if getattr(out, "loss", None) is not None:
                ce_model = float(out.loss.item())
                model_token_loss_sum += ce_model * batch_valid_tokens
                model_batch_means.append(ce_model)

            total_valid_tokens += batch_valid_tokens
            batch_means.append(float(ce_mean.item()))
            seq_means.extend([float(x.item()) for x in per_seq_mean_loss])
            total_batches += 1

            # Greedy predictions under teacher forcing
            pred_ids = shift_logits.argmax(dim=-1)  # [B, T-1]
            labels_mask = (shift_labels != pad_id)
            for i in range(pred_ids.size(0)):
                ids_kept = pred_ids[i][labels_mask[i]].detach().cpu().tolist()
                try:
                    tokens = tokenizer.convert_ids_to_tokens(ids_kept, skip_special_tokens=False)
                except Exception:
                    tokens = []
                pred_seq = _extract_predicted_sequence(tokens) if tokens else ""
                per_seq_records.append({
                    "sequence": sequences[i],
                    "predicted": pred_seq,
                    "mean_loss": float(per_seq_mean_loss[i].item()),
                    "perplexity": float(per_seq_ppl[i].item()),
                    "valid_tokens": int(valid_counts[i].item()),
                })

    if total_valid_tokens == 0:
        raise RuntimeError(f"No valid tokens found in split '{split_name}' (file={csv_path}). Check sequence column and tokenizer.")

    # Final scalars
    mean_loss_teacher = token_loss_sum_teacher / total_valid_tokens
    mean_loss_manual  = token_loss_sum_manual  / total_valid_tokens
    model_loss_token_weighted = float(model_token_loss_sum / total_valid_tokens) if total_valid_tokens > 0 and (model_batch_means) else float('nan')
    mean_loss_batch_unweighted = float(np.mean(batch_means)) if batch_means else float('nan')
    model_loss_batch_unweighted = float(np.mean(model_batch_means)) if model_batch_means else float('nan')
    mean_loss_sequence_unweighted = float(np.mean(seq_means)) if seq_means else float('nan')
    ppl = math.exp(mean_loss_teacher)

    # Save per-seq predictions
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{split_name}_preds.csv")
    pd.DataFrame(per_seq_records).to_csv(out_csv, index=False)

    return {
        "split": split_name,
        "file": csv_path,
        "num_sequences": len(per_seq_records),
        "total_batches": total_batches,
        "total_valid_tokens": int(total_valid_tokens),
        "mean_loss": float(mean_loss_teacher),  # canonical
        "mean_loss_teacher": float(mean_loss_teacher),
        "mean_loss_manual": float(mean_loss_manual),
        "mean_loss_batch_unweighted": float(mean_loss_batch_unweighted),
        "mean_loss_sequence_unweighted": float(mean_loss_sequence_unweighted),
        "model_loss_token_weighted": float(model_loss_token_weighted),
        "model_loss_batch_unweighted": float(model_loss_batch_unweighted),
        "perplexity": float(ppl),
        "csv_path": out_csv,
        "df": pd.DataFrame(per_seq_records),  # for W&B table
    }


# ----------------------------
# CLI
# ----------------------------

@click.command()
@click.option("--ckpt", required=False, help="Path to checkpoint: Composer .pt, HF folder, LoRA adapter, or .safetensors")
@click.option("--model-path", required=False, help="Path to base ProGen2 HF model folder (used if needed to build the model)")
@click.option("--baseline-model-path", required=False, help="Optional baseline model path for side-by-side comparison (HF folder)")
@click.option("--data-dir", required=True, help="Directory containing train/val/test CSV/TSV files")
@click.option("--sequence-col", default="sequence", show_default=True, help="Column name for sequences in CSV")
@click.option("--batch-size", default=8, show_default=True, help="Evaluation batch size")
@click.option("--config", "config_path", required=False, help="Optional training YAML/JSON; used to resolve model path if not provided")
@click.option("--output-dir", default="eval_outputs", show_default=True, help="Where to save per-sequence predictions CSVs")
@click.option("--seed", default=42, show_default=True, help="Random seed for subsampling")
@click.option("--wandb-project", default="crispr12a", show_default=True, help="W&B project name")
@click.option("--wandb-entity", default="eqk3", show_default=True, help="W&B entity/user")
@click.option("--wandb-run-name", default=None, help="Optional W&B run name")
@click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
@click.option("--only-val-and-train-sample", is_flag=True, default=False,
              help="Evaluate only on valid.csv and a random subsample of train.csv sized to the validation set.")
@click.option("--random-reverse", is_flag=True, default=False,
              help="Apply 50/50 full-sequence reversal augmentation at eval time (to match training-time eval if it used random_reverse).")
@click.option("--reverse-on-splits", type=click.Choice(["none","train","all"]), default="none",
              help="Which splits apply random_reverse; default 'none'. If 'train', only the train_sample_* splits are reversed.")
@click.option("--debug", is_flag=True, default=False, help="Print one-time diagnostics and loss parity check.")
def main(ckpt: str | None, model_path: str | None, baseline_model_path: str | None, data_dir: str,
         sequence_col: str, batch_size: int, config_path: str | None,
         output_dir: str, seed: int, wandb_project: str, wandb_entity: str,
         wandb_run_name: str | None, no_wandb: bool,
         only_val_and_train_sample: bool, random_reverse: bool, reverse_on_splits: str, debug: bool):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Using device: {device}")

    # Resolve base model path if needed
    base_model_path: Optional[str] = model_path
    if base_model_path is None and config_path is not None:
        try:
            cfg = _read_config_file(config_path)
            if isinstance(cfg, dict):
                base_model_path = cfg.get("model", {}).get("path")  # type: ignore
        except Exception:
            base_model_path = None
    if base_model_path is None:
        raise click.UsageError("--model-path not provided and could not resolve from --config. Please supply --model-path.")

    # Load model under test
    model_fine, resolved_model_path = _load_model_from_ckpt(base_model_path, ckpt)
    # Tokenizer (repo tokenizer; pad_id is 0 in your codebase, but read it anyway)
    tok_path = resolved_model_path if resolved_model_path and os.path.isdir(resolved_model_path) else base_model_path
    tokenizer = get_tokenizer(ModelSchema(name="progen2", path=tok_path))
    try: tokenizer.padding_side = "right"
    except Exception: pass
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0

    # Sanity prints
    try: print("Tokenizer vocab size:", len(tokenizer))
    except Exception: pass
    print("pad/bos/eos ids:", getattr(tokenizer, "pad_token_id", None),
          getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "eos_token_id", None))
    print("padding_side:", getattr(tokenizer, "padding_side", None))

    model_fine = model_fine.to(device)
    model_fine.eval()

    # W&B
    wb = None
    if not no_wandb:
        try:
            import wandb
            wb = wandb
            run = wb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, job_type="eval")
            wb.config.update({
                "evaluation": {
                    "model_path": model_path,
                    "resolved_model_path": resolved_model_path,
                    "sequence_col": sequence_col,
                    "batch_size": batch_size,
                    "seed": seed,
                    "pad_id": pad_id,
                    "random_reverse": random_reverse,
                    "reverse_on_splits": reverse_on_splits,
                }
            }, allow_val_change=True)
            if ckpt is not None:
                wb.config.update({"evaluation": {"ckpt": ckpt}}, allow_val_change=True)
        except ImportError:
            print("wandb not installed; proceeding without W&B logging. Install with `pip install wandb`.\n")
            wb = None

    # Determine dataset files
    train_csv = val_csv = test_csv = None
    if only_val_and_train_sample:
        files = set(os.listdir(data_dir))
        for name in ["valid.csv","val.csv","validation.csv","dev.csv","valid.tsv","val.tsv","validation.tsv","dev.tsv"]:
            if name in files:
                val_csv = os.path.join(data_dir, name); break
        for name in ["train.csv","train.tsv"]:
            if name in files:
                train_csv = os.path.join(data_dir, name); break
        if val_csv is None:
            raise FileNotFoundError("--only-val-and-train-sample set but no validation file found.")
        if train_csv is None:
            raise FileNotFoundError("--only-val-and-train-sample set but no train.csv/.tsv found.")
    else:
        train_csv, val_csv, test_csv = _auto_find_splits(data_dir)

    # Output dirs
    out_fine = os.path.join(output_dir, "model_under_test")
    out_base = os.path.join(output_dir, "baseline_model") if baseline_model_path else None

    # Helper to decide reversal by split
    def _rev_for(split: str) -> bool:
        if not random_reverse:
            return False
        if reverse_on_splits == "all":
            return True
        if reverse_on_splits == "train":
            return split.startswith("train")
        return False  # "none"

    # Evaluate fine-tuned model
    results_fine = []
    res_fine_test = None
    res_fine_val = None
    if not only_val_and_train_sample:
        if test_csv is None:
            raise FileNotFoundError("No test file found. Provide test.csv or use --only-val-and-train-sample.")
        res_fine_test = _evaluate_dataset(model_fine, tokenizer, test_csv, sequence_col, batch_size,
                                          device, "test", out_fine, pad_id, _rev_for("test"), debug)
        results_fine.append(res_fine_test)
    if val_csv:
        res_fine_val = _evaluate_dataset(model_fine, tokenizer, val_csv, sequence_col, batch_size,
                                         device, "val", out_fine, pad_id, _rev_for("val"), debug)
        results_fine.append(res_fine_val)
    if train_csv:
        if res_fine_val is not None:
            results_fine.append(
                _evaluate_dataset(
                    model_fine,
                    tokenizer,
                    train_csv,
                    sequence_col,
                    batch_size,
                    device,
                    "train_sample_like_val",
                    out_fine,
                    pad_id,
                    _rev_for("train_sample_like_val"),
                    debug=debug,
                    sample_size=res_fine_val["num_sequences"],
                    seed=seed,
                )
            )
        if (not only_val_and_train_sample) and (res_fine_test is not None):
            results_fine.append(
                _evaluate_dataset(
                    model_fine,
                    tokenizer,
                    train_csv,
                    sequence_col,
                    batch_size,
                    device,
                    "train_sample_like_test",
                    out_fine,
                    pad_id,
                    _rev_for("train_sample_like_test"),
                    debug=debug,
                    sample_size=res_fine_test["num_sequences"],
                    seed=seed,
                )
            )

    # Optional: evaluate baseline model side-by-side
    results_base = []
    if baseline_model_path is not None:
        model_base = get_model(ModelSchema(name="progen2", path=baseline_model_path)).to(device)
        model_base.eval()
        tokenizer_base = get_tokenizer(ModelSchema(name="progen2", path=baseline_model_path))
        try: tokenizer_base.padding_side = "right"
        except Exception: pass
        pad_id_base = tokenizer_base.pad_token_id if getattr(tokenizer_base, "pad_token_id", None) is not None else 0
        if out_base is not None:
            os.makedirs(out_base, exist_ok=True)

        if not only_val_and_train_sample:
            res_base_test = _evaluate_dataset(model_base, tokenizer_base, test_csv, sequence_col, batch_size,
                                              device, "test", out_base, pad_id_base, _rev_for("test"), debug)
            results_base.append(res_base_test)
        if val_csv:
            res_base_val = _evaluate_dataset(model_base, tokenizer_base, val_csv, sequence_col, batch_size,
                                             device, "val", out_base, pad_id_base, _rev_for("val"), debug)
            results_base.append(res_base_val)
        if train_csv:
            if val_csv and res_fine_val is not None:
                results_base.append(
                    _evaluate_dataset(
                        model_base,
                        tokenizer_base,
                        train_csv,
                        sequence_col,
                        batch_size,
                        device,
                        "train_sample_like_val",
                        out_base,
                        pad_id_base,
                        _rev_for("train_sample_like_val"),
                        debug=debug,
                        sample_size=res_fine_val["num_sequences"],
                        seed=seed,
                    )
                )
            if not only_val_and_train_sample and res_fine_test is not None:
                results_base.append(
                    _evaluate_dataset(
                        model_base,
                        tokenizer_base,
                        train_csv,
                        sequence_col,
                        batch_size,
                        device,
                        "train_sample_like_test",
                        out_base,
                        pad_id_base,
                        _rev_for("train_sample_like_test"),
                        debug=debug,
                        sample_size=res_fine_test["num_sequences"],
                        seed=seed,
                    )
                )

    # Summaries
    print("=======================")
    print("Evaluation summary (model under test)")
    print("=======================")
    total_samples = 0
    for rf in results_fine:
        split = rf["split"]
        print(f"Split                 : {split}")
        print(f"File                  : {rf['file']}")
        print(f"Num sequences         : {rf['num_sequences']}")
        print(f"Batches (size={batch_size}): {rf['total_batches']}")
        print(f"Valid tokens counted  : {rf['total_valid_tokens']}")
        print(f"Mean loss (teacher)   : {rf['mean_loss']:.4f} | PPL: {rf['perplexity']:.2f}")
        print(f"Predictions CSV       : {rf['csv_path']}")
        print("-----------------------")
        total_samples += rf["num_sequences"]

    if results_base:
        print("=======================")
        print("Baseline vs Fine-tuned (per split)")
        print("=======================")
        base_by_split = {r["split"]: r for r in results_base}
        fine_by_split = {r["split"]: r for r in results_fine}
        for split in [s for s in ["test","val","train_sample_like_val","train_sample_like_test"] if s in fine_by_split]:
            rb = base_by_split.get(split); rf = fine_by_split.get(split)
            if rb and rf:
                delta = rb["perplexity"] - rf["perplexity"]
                pct = (delta / rb["perplexity"] * 100.0) if rb["perplexity"] != 0 else 0.0
                print(f"Split                 : {split}")
                print(f"Baseline PPL          : {rb['perplexity']:.2f}")
                print(f"Fine-tuned PPL        : {rf['perplexity']:.2f}")
                print(f"Delta (abs)           : {delta:.2f}  ({pct:+.2f}%)")
                print("-----------------------")

    # W&B logging
    if wb is not None and wb.run is not None:
        for rf in results_fine:
            split = rf["split"]
            try:
                table_fine = wb.Table(dataframe=rf["df"])  # type: ignore
            except Exception:
                table_fine = None
            wb.log({
                f"{split}/mean_loss": rf["mean_loss"],
                f"{split}/mean_loss_teacher": rf["mean_loss_teacher"],
                f"{split}/mean_loss_manual": rf["mean_loss_manual"],
                f"{split}/mean_loss_batch_unweighted": rf["mean_loss_batch_unweighted"],
                f"{split}/mean_loss_sequence_unweighted": rf["mean_loss_sequence_unweighted"],
                f"{split}/model_loss_token_weighted": rf["model_loss_token_weighted"],
                f"{split}/model_loss_batch_unweighted": rf["model_loss_batch_unweighted"],
                f"{split}/perplexity": rf["perplexity"],
                f"{split}/num_sequences": rf["num_sequences"],
                f"{split}/total_valid_tokens": rf["total_valid_tokens"],
                **({f"{split}/predictions": table_fine} if table_fine is not None else {}),
            })
        if results_base:
            for rb in results_base:
                split = rb["split"]
                try:
                    table_base = wb.Table(dataframe=rb["df"])  # type: ignore
                except Exception:
                    table_base = None
                wb.log({
                    f"baseline/{split}/mean_loss": rb["mean_loss"],
                    f"baseline/{split}/mean_loss_teacher": rb["mean_loss_teacher"],
                    f"baseline/{split}/mean_loss_manual": rb["mean_loss_manual"],
                    f"baseline/{split}/mean_loss_batch_unweighted": rb["mean_loss_batch_unweighted"],
                    f"baseline/{split}/mean_loss_sequence_unweighted": rb["mean_loss_sequence_unweighted"],
                    f"baseline/{split}/model_loss_token_weighted": rb["model_loss_token_weighted"],
                    f"baseline/{split}/model_loss_batch_unweighted": rb["model_loss_batch_unweighted"],
                    f"baseline/{split}/perplexity": rb["perplexity"],
                    f"baseline/{split}/num_sequences": rb["num_sequences"],
                    f"baseline/{split}/total_valid_tokens": rb["total_valid_tokens"],
                    **({f"baseline/{split}/predictions": table_base} if table_base is not None else {}),
                })
            # Deltas (fine - base) for shared splits
            base_by_split = {r["split"]: r for r in results_base}
            fine_by_split = {r["split"]: r for r in results_fine}
            for split in set(base_by_split).intersection(fine_by_split):
                rb = base_by_split[split]; rf = fine_by_split[split]
                wb.log({
                    f"delta/{split}/mean_loss_teacher": rf["mean_loss_teacher"] - rb["mean_loss_teacher"],
                    f"delta/{split}/mean_loss_batch_unweighted": rf["mean_loss_batch_unweighted"] - rb["mean_loss_batch_unweighted"],
                    f"delta/{split}/model_loss_token_weighted": rf.get("model_loss_token_weighted", float('nan')) - rb.get("model_loss_token_weighted", float('nan')),
                    f"delta/{split}/model_loss_batch_unweighted": rf.get("model_loss_batch_unweighted", float('nan')) - rb.get("model_loss_batch_unweighted", float('nan')),
                    f"delta/{split}/perplexity": rf["perplexity"] - rb["perplexity"],
                })

        wb.summary["evaluation/total_num_sequences"] = int(sum(r["num_sequences"] for r in results_fine))
        wb.summary["evaluation/num_splits"] = len(results_fine)


if __name__ == "__main__":
    main()
