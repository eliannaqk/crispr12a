#!/usr/bin/env python
"""
debug_model.py – quick diagnostics for opencrispr_repro

Usage
-----
python debug_model.py \
    --config cas12a_ft_finetuneapi.yaml \
    --ckpt   /path/to/last_checkpoint (optional) \
    --n-head 4

Tips
----
* Run inside the same environment / GPU as training.
* Increase --n-head if you want more examples printed.
"""
from __future__ import annotations
import argparse, json, os, sys, textwrap, logging, torch, pandas as pd
from pathlib import Path
from ruamel.yaml import YAML
from tabulate import tabulate     # pip install tabulate (nice printing)
from torch.nn.functional import softmax
from safetensors.torch import load_file

# ---------- project imports ----------
# Everything below lives in the repo you already cloned
from opencrispr_repro.schema import FinetuneAPI            # :contentReference[oaicite:6]{index=6}
from opencrispr_repro.model  import get_model, get_tokenizer  # :contentReference[oaicite:7]{index=7}
from opencrispr_repro.data   import SeqDataset, progen2_collate_fn  # :contentReference[oaicite:8]{index=8}
# -------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
yaml_loader = YAML(typ="safe")
log = logging.getLogger("DEBUG_SCRIPT")

def read_cfg(p: str) -> dict:
    with open(p) as f:
        return yaml_loader.load(f) if p.endswith((".yml", ".yaml")) else json.load(f)

def pretty_header(msg: str):
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))

def print_token_mapping(tokenizer, encoded_batch, n=2):
    ids = encoded_batch["input_ids"][:n].tolist()
    for row in ids:
        toks = [tokenizer.decode([t]) for t in row]
        print(tabulate({"idx": range(len(row)), "id": row, "token": toks},
                       headers="keys", tablefmt="psql"))



# ---------- checkpoint loading helpers ----------

def _find_latest_hf_adapter_dir(root_dir: str) -> str | None:
    """Given a Composer save_folder or a huggingface subfolder, find latest baXXXX adapter dir."""
    import re
    # If given the huggingface subfolder, search within; else try to enter it
    hf_dir = root_dir
    if os.path.isdir(os.path.join(root_dir, "huggingface")):
        hf_dir = os.path.join(root_dir, "huggingface")
    if not os.path.isdir(hf_dir):
        return None
    try:
        step_dirs = [d for d in os.listdir(hf_dir) if re.match(r"ba\d+", d)]
        if not step_dirs:
            return None
        step_dirs.sort(key=lambda s: int(s[2:]), reverse=True)
        for d in step_dirs:
            cand = os.path.join(hf_dir, d)
            if os.path.exists(os.path.join(cand, "adapter_config.json")):
                return cand
    except Exception:
        return None
    return None


def load_weights(model, ckpt_path: str):
    """Load weights into model.

    Supports:
    - Hugging Face folder with full weights
    - Composer/GPU trainer checkpoints (.pt)
    - PEFT LoRA adapter folders (with adapter_config.json)
    - Single *.safetensors files (adapter_model.safetensors or full shards)

    Returns possibly-updated model instance (e.g., wrapped with PEFT).
    """
    ckpt_path = os.path.realpath(ckpt_path)

    def is_peft_adapter_dir(path: str) -> bool:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json"))

    def is_peft_adapter_file(path: str) -> bool:
        return path.endswith(".safetensors") and os.path.exists(
            os.path.join(os.path.dirname(path), "adapter_config.json")
        )

    # Handle PEFT LoRA adapter (folder or single safetensors alongside adapter_config.json)
    if is_peft_adapter_dir(ckpt_path) or is_peft_adapter_file(ckpt_path):
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "peft library required to load LoRA adapter checkpoints. Install with `pip install peft`."
            ) from e

        adapter_dir = ckpt_path if is_peft_adapter_dir(ckpt_path) else os.path.dirname(ckpt_path)

        # If the current model already has PEFT layers, load the adapter into it; otherwise wrap base model.
        if hasattr(model, "peft_config"):
            model.load_adapter(adapter_dir, adapter_name="default", is_trainable=False)
            model.set_adapter("default")
            log.info(f"Loaded LoRA adapter from {adapter_dir} into existing PEFT model (adapter='default')")
            return model
        else:
            peft_model = PeftModel.from_pretrained(model, adapter_dir)
            log.info(f"Wrapped base model with LoRA adapter from {adapter_dir} (adapter='default')")
            return peft_model

    # Composer save_folder with HuggingFace adapters underneath
    if os.path.isdir(ckpt_path):
        latest_adapter = _find_latest_hf_adapter_dir(ckpt_path)
        if latest_adapter is not None:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError("peft library required to load LoRA adapter checkpoints. Install with `pip install peft`.") from e
            if hasattr(model, "peft_config"):
                model.load_adapter(latest_adapter, adapter_name="default", is_trainable=False)
                model.set_adapter("default")
                log.info(f"Loaded latest HF LoRA adapter {latest_adapter} into existing PEFT model (adapter='default')")
                return model
            else:
                peft_model = PeftModel.from_pretrained(model, latest_adapter)
                log.info(f"Wrapped base model with latest HF LoRA adapter {latest_adapter} (adapter='default')")
                return peft_model

        # Hugging Face-style folder with full weights
        safetensor_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
        if safetensor_files:
            state = load_file(os.path.join(ckpt_path, safetensor_files[0]))
        else:
            bin_files = sorted(
                f for f in os.listdir(ckpt_path) if f.startswith("pytorch_model") and f.endswith(".bin")
            )
            if not bin_files:
                raise FileNotFoundError(
                    f"No weight files (.safetensors or pytorch_model*.bin) found in {ckpt_path}"
                )
            state = {}
            for bf in bin_files:
                shard = torch.load(os.path.join(ckpt_path, bf), map_location="cpu")
                state.update(shard)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(
            f"Loaded weights from {ckpt_path} (missing={len(missing)} unexpected={len(unexpected)})"
        )
        return model

    # Single safetensors (assume full state dict shard)
    if ckpt_path.endswith(".safetensors"):
        state = load_file(ckpt_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(f"Loaded safetensors {ckpt_path} (missing={len(missing)} unexpected={len(unexpected)})")
        return model

    # Composer or generic torch checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state" in ckpt and "model" in ckpt["state"]:
        state = ckpt["state"]["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info(f"Loaded Composer checkpoint {ckpt_path} (missing={len(missing)} unexpected={len(unexpected)})")
    return model


def print_lora_diagnostics(model):
    try:
        import peft  # noqa: F401
    except Exception:
        return
    print("\n===========================")
    print("LoRA diagnostics")
    print("===========================")
    using_peft = hasattr(model, "peft_config")
    print(f"PEFT attached         : {using_peft}")
    if not using_peft:
        return
    try:
        active = getattr(model, "active_adapter", None)
        peft_cfg = getattr(model, "peft_config", {})
        adapters = list(peft_cfg.keys()) if isinstance(peft_cfg, dict) else []
        print(f"Active adapter        : {active}")
        print(f"Available adapters    : {adapters}")
        # Print target_modules if present
        if active and isinstance(peft_cfg, dict) and active in peft_cfg:
            tm = getattr(peft_cfg[active], "target_modules", None)
            if tm is not None:
                print(f"Target modules        : {list(tm)}")
    except Exception:
        pass
    # Collect modules that have LoRA injected
    lora_module_names = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_module_names.append(name)
    print(f"Modules with LoRA     : {len(lora_module_names)}")
    # Show a few examples
    for name in sorted(lora_module_names)[:10]:
        print(f"  - {name}")
    # Count LoRA parameters
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            lora_params += p.numel()
    print(f"LoRA params           : {lora_params:,} / total {total_params:,}")


def main(cfg_path: str, ckpt_path: str | None, n_head: int, log_file: str | None):
    # ---------- config & objects ----------
    cfg  = FinetuneAPI(**read_cfg(cfg_path))    # same validation path the Trainer uses
    tok  = get_tokenizer(cfg.model)             # ProGen2 tokenizer
    base_model = get_model(cfg.model)

    # Apply LoRA if enabled in config
    if cfg.algorithms and cfg.algorithms.lora and cfg.algorithms.lora.enabled:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft library required for LoRA diagnostics. Install with `pip install peft`. ")
        lora_cfg = cfg.algorithms.lora
        peft_conf = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        base_model = get_peft_model(base_model, peft_conf)
    model = base_model.to(DEVICE).eval()
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        model = load_weights(model, ckpt_path)
        print_lora_diagnostics(model)

    # ---------- peek at valid.csv ----------
    pretty_header("Head of validation file")
    df_val = pd.read_csv(cfg.data.val_data_path)
    #print(df_val.head(n_head).to_markdown(index=False))

    # Print dimensions of first n_head examples
    print(f"\nDimensions of first {n_head} examples:")
    for i in range(n_head):
        seq = df_val.iloc[i]['sequence']
        print(f"Example {i+1}: length = {len(seq)}")
    #print(df_val.head(n_head).to_markdown(index=False))
    # Get dimensions of validation dataframe
    print(f"\nValidation CSV dimensions: {df_val.shape[0]} rows x {df_val.shape[1]} columns")
    
    # # Read and get dimensions of corresponding FASTA file
    fasta_path = cfg.data.val_data_path.replace('.csv', '.fasta')
    if os.path.exists(fasta_path):
        fasta_seqs = []
        with open(fasta_path) as f:
            for line in f:
                if not line.startswith('>'):  # Skip header lines
                    fasta_seqs.append(line.strip())
        print(f"FASTA file dimensions: {len(fasta_seqs)} sequences")
    else:
        print(f"No corresponding FASTA file found at: {fasta_path}")

    # ---------- encode & forward ----------
    pretty_header("Tokenizer encoding preview")
    val_ds = SeqDataset(cfg.data.val_data_path,
                        sequence_col=cfg.data.sequence_col,
                        label_col=None)
    batch_items = [val_ds[i] for i in range(n_head)]
    batch      = progen2_collate_fn(batch_items, tokenizer=tok)      # :contentReference[oaicite:9]{index=9}
    #print_token_mapping(tok, batch, n=n_head)

    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch,
                    output_hidden_states=True,
                    return_dict=True)
    print(f"\nLoss on this mini‑batch       : {out.loss.item():.4f}")
    print(f"Perplexity (exp(loss))        : {torch.exp(out.loss).item():.2f}")

    # If using PEFT, compare loss with adapters disabled to verify effect
    if hasattr(model, "disable_adapter"):
        try:
            with model.disable_adapter():
                out_base = model(**batch, output_hidden_states=False, return_dict=True)
            base_loss = out_base.loss.item()
            delta = out.loss.item() - base_loss
            print("\n------------------------------")
            print("LoRA effect check")
            print("------------------------------")
            print(f"Loss with adapters      : {out.loss.item():.4f}")
            print(f"Loss without adapters   : {base_loss:.4f}")
            print(f"Delta (with - without)  : {delta:+.4f}")
        except Exception:
            pass

    # ---------- layer‑stats ----------
    pretty_header("Hidden‑state layer statistics")
    h_stats = []
    for idx, h in enumerate(out.hidden_states):
        mu = h.mean().item(); sigma = h.std().item()
        h_stats.append((idx, mu, sigma))
        print(f"Layer {idx:>2}: mean={mu:+.4f}  std={sigma:.4f}")

    # Optional: track LR / loss over time from training logs -------------
    if log_file:
        pretty_header("Training‑loss trace")
        log_df = (pd.read_csv(log_file, sep="\t")      # Composer default TSV
                    .query("key == 'loss/train'")
                    .sort_values('batch'))
        print(log_df[['batch', 'value']].head(20).to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   default=None,
                        help="Folder containing HuggingFace files (optional)")
    parser.add_argument("--n-head", type=int, default=5,
                        help="How many validation rows to show/encode")
    parser.add_argument("--log-file", default=None,
                        help="Composer log TSV to inspect loss curve")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
    # Ensure our custom logger emits INFO-level messages
    logging.getLogger("DEBUG_SCRIPT").setLevel(logging.INFO)
    main(args.config, args.ckpt, args.n_head, args.log_file)
