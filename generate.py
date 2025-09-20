import json
import os
import re
import contextlib

import click
import pandas as pd
import torch
from ruamel.yaml import YAML
from tqdm import tqdm

from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer

ROOT_DIR = os.path.dirname(__file__)
MAX_LEN = 2000

def kmer_n_repeat(k: int, n: int):
    return "(.)" * k + "".join([f"\\{i+1}" for i in range(k)]) * (n-1)


def has_kmer_repeats(seq: str):
    k_to_n = [6, 4, 3, 3, 3, 3, 2]  # Thresholds apply to >99.5% of all naturals
    return any(re.search(kmer_n_repeat(k, n), seq) for k, n in enumerate(k_to_n, start=1))


def is_valid_seq(seq: str, eos: str):
    # Remove minimum length requirement; keep upper bound, k-mer repeat filter, and EOS check
    return (len(seq) <= MAX_LEN) and not has_kmer_repeats(seq) and seq.endswith(eos)


def read_config_file(config_path: str) -> dict:
    with open(config_path) as f:
        if config_path.endswith(".yml") or config_path.endswith(".yaml"):
            config = YAML(typ="safe").load(f)
        else:
            config = json.load(f)
    return config


@click.command()
@click.option("--model-path", "model_path", required=True, help="Path where model is stored (HF folder, e.g., huggingface/baXXXX)")
@click.option("--config", "config_path", required=True, help="JSON or YML w/ generation hyperparams")
@click.option("--save-folder", "save_folder", required=False, default=None, help="Optional folder to save generations (defaults to repo ./generations)")
@click.option("--job-idx", "job_idx", type=int, default=None, required=False, help="Job index (for parallel jobs)")
@click.option("--precision", type=click.Choice(["fp32", "fp16", "bf16"]), default="bf16", show_default=True, help="Computation precision for model + autocast (CUDA only)")
@click.option("--autocast/--no-autocast", default=True, show_default=True, help="Use torch.cuda.amp.autocast during generation when precision != fp32")
def main(model_path: str, config_path: str, save_folder: str | None, job_idx: int, precision: str, autocast: bool):
    with open(config_path) as f:
        config = YAML(typ="safe").load(f)
    tokenizer = get_tokenizer(ModelSchema(name="progen2"))

    # Files in which to save generations
    # If save_folder is provided, organize as: <save_folder>/generations/<model_tag>/<config_basename>_{raw,filtered}.csv
    # where model_tag uses the leaf folder name of model_path (e.g., ba8000).
    cfg_base = os.path.basename(config_path)
    cfg_stem = re.sub(r"\.(yml|yaml|json)$", "", cfg_base)
    if save_folder:
        base = os.path.basename(os.path.normpath(model_path))
        model_tag = base if base else "model"
        gen_dir = os.path.join(save_folder, "generations", model_tag)
    else:
        gen_dir = os.path.join(ROOT_DIR, "generations")
    base_file = os.path.join(gen_dir, cfg_stem)
    raw_file = base_file + "_raw.csv"
    filt_file = base_file + "_filtered.csv"
    if job_idx is not None:
        raw_file += f".{job_idx}"
        filt_file += f".{job_idx}"

    model = get_model(ModelSchema(name="progen2", path=model_path))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set model dtype/device; on CUDA allow fp16/bf16
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    target_dtype = dtype_map.get(precision, torch.float32)
    if device == "cuda" and target_dtype != torch.float32:
        try:
            model.to(device=device, dtype=target_dtype)
        except Exception:
            # Fallback to device-only move if dtype cast not supported
            model.to(device=device)
    else:
        model.to(device=device)

    # Prepare output files and headers
    os.makedirs(os.path.dirname(base_file), exist_ok=True)
    header = "context_name,context,sequence\n"
    if not os.path.exists(raw_file):
        with open(raw_file, "w") as f:
            f.write(header)
    if not os.path.exists(filt_file):
        with open(filt_file, "w") as f:
            f.write(header)

    def count_existing_for_context(path: str, ctx_value: str) -> int:
        if not os.path.exists(path):
            return 0
        n = 0
        with open(path) as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                parts = line.rstrip("\n").split(",", 2)
                if len(parts) >= 2 and parts[1] == ctx_value:
                    n += 1
        return n

    # Generate samples and append them to the file of saved generations
    if isinstance(config, dict):
        config = [config]
    for sub_config in config:
        n_raw, n_filt = 0, 0
        total_samples = int(sub_config.pop("num_samples", 10000))
        batch_n = int(sub_config.pop("batch_size", 10))
        ctx_dict = sub_config.pop("context")
        ctx_name = ctx_dict["name"]
        ctx = ctx_dict["seq"]
        # Determine orientation and EOS token more robustly
        if ctx.startswith("1"):
            eos = "2"
        elif ctx.startswith("2"):
            eos = "1"
        elif ctx.endswith("2"):
            ctx = ctx[::-1]
            eos = "1"
        else:
            eos = "2"
        eos_id = tokenizer.encode(eos)[0]

        # Resume support: stop when raw (pre-filter) samples for this context reach total_samples
        i = count_existing_for_context(raw_file, ctx)
        with tqdm(total=total_samples, desc="Generations (raw/filtered)") as pbar:
            pbar.update(i)
            device = next(model.parameters()).device
            while i < total_samples:
                this_n = min(batch_n, total_samples - i)
                batch_ctx = [ctx] * this_n
                inputs = tokenizer(batch_ctx, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(device)
                attn = inputs.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)
                use_amp = (device.type == "cuda") and autocast and (target_dtype != torch.float32)
                amp_ctx = torch.cuda.amp.autocast(dtype=target_dtype) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_length=MAX_LEN,
                        eos_token_id=eos_id,
                        do_sample=sub_config.get("do_sample", True),
                        temperature=sub_config.get("temperature", 1.0),
                        top_p=sub_config.get("top_p", 1.0),
                        top_k=sub_config.get("top_k", 0),
                        repetition_penalty=sub_config.get("repetition_penalty", 1.0),
                    )
                seqs = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
                # Truncate to remaining needed raw samples for this context
                seqs = seqs[: (total_samples - i)]
                # Write all raw sequences (before filtering)
                with open(raw_file, "a") as fraw:
                    for seq in seqs:
                        fraw.write(f"{ctx_name},{ctx},{seq}\n")
                # Apply filters and write passing sequences
                passing = [s for s in seqs if is_valid_seq(s, eos)]
                if passing:
                    with open(filt_file, "a") as ff:
                        for seq in passing:
                            ff.write(f"{ctx_name},{ctx},{seq}\n")
                # Update counters and progress (stop condition is raw count)
                n_raw += len(seqs)
                n_filt += len(passing)
                i += len(seqs)
                pbar.set_description(f"Generations (raw={n_raw} filtered={n_filt})")
                pbar.update(len(seqs))

if __name__ == "__main__":
    main()
