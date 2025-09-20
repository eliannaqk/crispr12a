from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import click
import torch
import torch.nn.functional as F
from composer.core import Callback, Event, State
from composer.loggers import Logger
from ruamel.yaml import YAML

from opencrispr_repro.schema import FinetuneAPI


def _read_config_file(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        if config_path.endswith((".yml", ".yaml")):
            return YAML(typ="safe").load(f)
        return json.load(f)


class TrainStepwiseProbe(Callback):
    """Logs per-step Composer training loss vs HF forward loss vs manual CE.

    Hooks AFTER_LOSS each train batch; logs under debug/stepprobe/* and prints a short line.
    """

    def __init__(self, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self._hf_model = None

    def _get_hf_model(self, state: State):
        if self._hf_model is not None:
            return self._hf_model
        m = state.model
        for attr in ("model", "module", "_model", "_original_model"):
            if hasattr(m, attr):
                self._hf_model = getattr(m, attr)
                return self._hf_model
        self._hf_model = m
        return self._hf_model

    def run_event(self, event: Event, state: State, logger: Logger) -> None:  # type: ignore[override]
        if event != Event.AFTER_LOSS:
            return

        batch = state.batch
        if not isinstance(batch, dict) or ("input_ids" not in batch or "labels" not in batch):
            return

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]
        pad_id = self.pad_id

        hf = self._get_hf_model(state)
        hf.eval()

        with torch.no_grad():
            try:
                loss_train = float(state.loss) if state.loss is not None else float("nan")
            except Exception:
                loss_train = float("nan")

            try:
                out_hf = hf(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                hf_loss = float(out_hf.loss.detach().cpu().item()) if getattr(out_hf, "loss", None) is not None else float("nan")
            except Exception:
                hf_loss = float("nan")

            try:
                out_logits = hf(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = out_logits.logits
                V = logits.size(-1)
                lg1 = logits[..., :-1, :].contiguous()
                lb1 = labels[..., 1:].contiguous()
                ce1 = F.cross_entropy(lg1.view(-1, V), lb1.view(-1), ignore_index=pad_id, reduction="mean")
                ce1_val = float(ce1.item())
            except Exception:
                ce1_val = float("nan")

            # Diagnostics: effective tokens, pad ratio, and length stats
            try:
                if attention_mask is not None:
                    lengths = attention_mask.sum(dim=-1).to(torch.int64)
                    tokens_in_batch = int(lengths.sum().item())
                    pad_ratio = float(1.0 - attention_mask.float().mean().item())
                else:
                    # Fallback: infer from non-pad tokens in input_ids
                    nonpad = (input_ids != pad_id)
                    lengths = nonpad.sum(dim=-1).to(torch.int64)
                    tokens_in_batch = int(lengths.sum().item())
                    pad_ratio = float(1.0 - nonpad.float().mean().item())

                # Effective tokens contributing to shifted CE (labels are shifted right by 1)
                effective_tokens = int((lb1 != pad_id).sum().item())
                bsz = int(input_ids.size(0))
                len_min = int(lengths.min().item()) if lengths.numel() > 0 else 0
                len_max = int(lengths.max().item()) if lengths.numel() > 0 else 0
                len_mean = float(lengths.float().mean().item()) if lengths.numel() > 0 else 0.0
                # median via percentile on CPU to avoid GPU sync cost spikes on some backends
                len_med = int(torch.quantile(lengths.float().cpu(), q=0.5).item()) if lengths.numel() > 0 else 0
            except Exception:
                tokens_in_batch = 0
                effective_tokens = 0
                bsz = int(input_ids.size(0)) if hasattr(input_ids, 'size') else 0
                len_min = len_max = len_med = 0
                len_mean = 0.0
                pad_ratio = float('nan')

        logger.log_metrics({
            "debug/stepprobe/loss_train": loss_train,
            "debug/stepprobe/hf_loss": hf_loss,
            "debug/stepprobe/ce_shift1": ce1_val,
            "debug/stepprobe/tokens_effective": effective_tokens,
            "debug/stepprobe/tokens_in_batch": tokens_in_batch,
            "debug/stepprobe/pad_ratio": pad_ratio,
            "debug/stepprobe/len_min": len_min,
            "debug/stepprobe/len_p50": len_med,
            "debug/stepprobe/len_mean": len_mean,
            "debug/stepprobe/len_max": len_max,
            "debug/stepprobe/batch_size": bsz,
        })
        print(
            f"[StepProbe] step={state.timestamp.batch.value} "
            f"loss_train={loss_train:.5f} hf_loss={hf_loss:.5f} ce1={ce1_val:.5f} "
            f"eff_toks={effective_tokens} toks={tokens_in_batch} pad={pad_ratio:.3f} "
            f"len[min/50/mean/max]={len_min}/{len_med}/{len_mean:.1f}/{len_max} bsz={bsz}"
        )


@click.command()
@click.option("--config", "config_path", required=True, help="JSON or YML file based on schema.FinetuneAPI")
@click.option("--steps", default=10, show_default=True, help="Number of training batches to run")
@click.option("--preserve-config/--no-preserve-config", default=True, show_default=True,
              help="Preserve YAML trainer/data settings (FSDP, batch size, workers)")
@click.option("--hf-path", default=None, help="Optional HF checkpoint dir to load model weights from (overrides cfg.model.path)")
@click.option("--override-batch-size", type=int, default=None, help="Override cfg.data.batch_size for the probe")
@click.option("--override-microbatch-size", type=int, default=None,
              help="Override Composer device_train_microbatch_size to enable gradient accumulation")
@click.option("--enable-token-bucketing", is_flag=True, default=False,
              help="Enable token_bucketed_batches regardless of YAML")
@click.option("--target-tokens-per-microbatch", type=int, default=None,
              help="When token bucketing is enabled, approximate tokens per microbatch")
def main(config_path: str, steps: int, preserve_config: bool, hf_path: str | None,
         override_batch_size: int | None, override_microbatch_size: int | None,
         enable_token_bucketing: bool, target_tokens_per_microbatch: int | None):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    raw_cfg = _read_config_file(config_path)
    cfg = FinetuneAPI(**raw_cfg)
    # Optionally override model path to load a specific HF checkpoint
    if hf_path:
        try:
            cfg.model.path = hf_path  # type: ignore[attr-defined]
        except Exception:
            pass

    # Build a patched Trainer: disable FSDP for the probe and attach the stepwise probe
    import opencrispr_repro.trainer as trainer_mod
    try:
        from composer import Trainer as ComposerTrainer
    except Exception:
        from composer.trainer import Trainer as ComposerTrainer  # pragma: no cover

    class PatchedTrainer(ComposerTrainer):
        def __init__(self, *args, callbacks=None, **kwargs):  # type: ignore[no-untyped-def]
            if not preserve_config:
                # Optionally disable FSDP to reduce memory for probes
                kwargs.pop("parallelism_config", None)
            # Enable gradient accumulation by overriding device microbatch size if requested
            if override_microbatch_size is not None and override_microbatch_size > 0:
                kwargs["device_train_microbatch_size"] = int(override_microbatch_size)
            cbs = list(callbacks) if callbacks is not None else []
            cbs.append(TrainStepwiseProbe(pad_id=0))
            super().__init__(*args, callbacks=cbs, **kwargs)

    trainer_mod.Trainer = PatchedTrainer  # type: ignore[attr-defined]

    # Optionally tighten data settings to keep memory small or apply explicit overrides
    try:
        if not preserve_config:
            # Defaults for low-memory probing
            cfg.data.token_bucketed_batches = False  # type: ignore[attr-defined]
            cfg.data.batch_size = 1  # type: ignore[attr-defined]
            cfg.data.num_workers = 0  # type: ignore[attr-defined]
        # Explicit overrides
        if override_batch_size is not None and override_batch_size > 0:
            cfg.data.batch_size = int(override_batch_size)  # type: ignore[attr-defined]
        if enable_token_bucketing:
            cfg.data.token_bucketed_batches = True  # type: ignore[attr-defined]
        if target_tokens_per_microbatch is not None and target_tokens_per_microbatch > 0:
            cfg.data.target_tokens_per_microbatch = int(target_tokens_per_microbatch)  # type: ignore[attr-defined]
    except Exception:
        pass

    trainer = trainer_mod.get_trainer(cfg)
    trainer.fit(duration=f"{int(max(1, steps))}ba")


if __name__ == "__main__":
    main()
