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


class TrainLossProbe(Callback):
    """Capture Composer's training loss (state.loss) and compare to HF/wrapper forward losses on the same batch.

    Runs once on the first training batch after loss is computed.
    Logs to metrics with keys under `debug/trainprobe/*` and prints to stdout.
    """

    def __init__(self, pad_id: int = 0) -> None:
        super().__init__()
        self._done = False
        self.pad_id = int(pad_id)

    def _get_hf_model(self, state: State):
        m = state.model
        for attr in ("model", "module", "_model", "_original_model"):
            if hasattr(m, attr):
                return getattr(m, attr)
        return m

    def run_event(self, event: Event, state: State, logger: Logger) -> None:  # type: ignore[override]
        if self._done:
            return
        if event not in (Event.AFTER_LOSS,):
            return

        batch = state.batch
        if not isinstance(batch, dict) or ("input_ids" not in batch or "labels" not in batch):
            logger.log_metrics({"debug/trainprobe/error": 1})
            self._done = True
            return

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]
        pad_id = self.pad_id

        hf = self._get_hf_model(state)
        hf.eval()

        with torch.no_grad():
            # Composer-computed training loss
            try:
                wrapper_loss_train = float(state.loss) if state.loss is not None else float("nan")
            except Exception:
                wrapper_loss_train = float("nan")

            # Wrapper forward loss on the same batch
            try:
                out_wrap = state.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)  # type: ignore[attr-defined]
                wrapper_loss_forward = float(out_wrap.loss.detach().cpu().item()) if getattr(out_wrap, "loss", None) is not None else float("nan")
            except Exception:
                wrapper_loss_forward = float("nan")

            # HF forward loss on the same batch (no wrapper logic)
            try:
                out_hf = hf(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                hf_loss = float(out_hf.loss.detach().cpu().item()) if getattr(out_hf, "loss", None) is not None else float("nan")
            except Exception:
                hf_loss = float("nan")

            # Manual CE sanity checks
            try:
                out_logits_only = hf(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = out_logits_only.logits
                V = logits.size(-1)
                lg1 = logits[..., :-1, :].contiguous()
                lb1 = labels[..., 1:].contiguous()
                ce1 = F.cross_entropy(lg1.view(-1, V), lb1.view(-1), ignore_index=pad_id, reduction="mean")
                n1 = int((lb1 != pad_id).sum().item())
                if logits.size(1) >= 3:
                    lg2 = logits[..., :-2, :].contiguous()
                    lb2 = labels[..., 2:].contiguous()
                    ce2 = F.cross_entropy(lg2.view(-1, V), lb2.view(-1), ignore_index=pad_id, reduction="mean")
                    n2 = int((lb2 != pad_id).sum().item())
                else:
                    ce2 = torch.tensor(float("nan"), device=logits.device)
                    n2 = 0
                logits_dtype = str(logits.dtype)
            except Exception:
                ce1 = torch.tensor(float("nan"))
                ce2 = torch.tensor(float("nan"))
                n1 = n2 = 0
                logits_dtype = "unknown"

            labels_dtype = str(labels.dtype)

        # Log metrics
        logger.log_metrics({
            "debug/trainprobe/pad_id": pad_id,
            "debug/trainprobe/wrapper_loss_train": wrapper_loss_train,
            "debug/trainprobe/wrapper_loss_forward": wrapper_loss_forward,
            "debug/trainprobe/hf_loss": hf_loss,
            "debug/trainprobe/ce_shift1": float(ce1.item()),
            "debug/trainprobe/ce_shift2": float(ce2.item()),
            "debug/trainprobe/tokens_shift1": n1,
            "debug/trainprobe/tokens_shift2": n2,
            "debug/trainprobe/logits_dtype": logits_dtype,
            "debug/trainprobe/labels_dtype": labels_dtype,
        })

        # Stdout echo
        print("[TrainLossProbe] pad_id:", pad_id)
        print("[TrainLossProbe] wrapper_loss_train:", wrapper_loss_train)
        print("[TrainLossProbe] wrapper_loss_forward:", wrapper_loss_forward)
        print("[TrainLossProbe] hf_loss:", hf_loss)
        print("[TrainLossProbe] ce_shift1:", float(ce1.item()), "tokens:", n1)
        print("[TrainLossProbe] ce_shift2:", float(ce2.item()), "tokens:", n2)
        print("[TrainLossProbe] dtypes:", {"logits": logits_dtype, "labels": labels_dtype})

        self._done = True


@click.command()
@click.option("--config", "config_path", required=True, help="JSON or YML file based on schema.FinetuneAPI")
def main(config_path: str):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    raw_cfg = _read_config_file(config_path)
    cfg = FinetuneAPI(**raw_cfg)

    # Inject training probe via subclassed Trainer without touching core trainer
    import opencrispr_repro.trainer as trainer_mod
    try:
        from composer import Trainer as ComposerTrainer
    except Exception:
        from composer.trainer import Trainer as ComposerTrainer  # pragma: no cover (fallback)

    class PatchedTrainer(ComposerTrainer):
        def __init__(self, *args, callbacks=None, **kwargs):  # type: ignore[no-untyped-def]
            # Avoid FSDP to reduce memory footprint for the probe
            kwargs.pop("parallelism_config", None)
            cbs = list(callbacks) if callbacks is not None else []
            cbs.append(TrainLossProbe(pad_id=0))
            super().__init__(*args, callbacks=cbs, **kwargs)

    # Apply patch and build trainer
    trainer_mod.Trainer = PatchedTrainer  # type: ignore[attr-defined]
    # Tighten data settings to minimize memory
    try:
        cfg.data.token_bucketed_batches = False  # type: ignore[attr-defined]
        cfg.data.batch_size = 1  # type: ignore[attr-defined]
        cfg.data.num_workers = 0  # type: ignore[attr-defined]
    except Exception:
        pass

    trainer = trainer_mod.get_trainer(cfg)

    # One-batch training probe
    trainer.fit(duration="1ba")


if __name__ == "__main__":
    main()
