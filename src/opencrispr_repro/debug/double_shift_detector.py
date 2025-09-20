from __future__ import annotations
import torch
import torch.nn.functional as F

from composer.core import Callback, State, Event
from composer.loggers import Logger


class DoubleShiftDetector(Callback):
    """Detects if labels are being shifted twice during training-time eval.

    Logs (once, on the first eval batch):
      - debug/doubleshift/wrapper_loss       (Composer's state.loss)
      - debug/doubleshift/ce_shift1          (manual CE: logits[:-1] vs labels[1:])
      - debug/doubleshift/ce_shift2          (manual CE: logits[:-2] vs labels[2:])
      - debug/doubleshift/tokens_shift1
      - debug/doubleshift/tokens_shift2
      - debug/doubleshift/pad_id
    """

    def __init__(self, pad_id: int = 0):
        super().__init__()
        self._done = False
        self.pad_id = int(pad_id)

    def _get_hf_model(self, state: State):
        """Try to fetch underlying HF PreTrainedModel from the Composer HuggingFaceModel wrapper."""
        m = state.model
        for attr in ("model", "module", "_model", "_original_model"):
            if hasattr(m, attr):
                return getattr(m, attr)
        # Fall back to the object itself (may still be the wrapper)
        return m

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if self._done:
            return
        # Prefer to run after loss/forward is available; be tolerant to Composer versions
        valid_events = (
            Event.AFTER_LOSS,            # generic hook (sometimes fires during eval)
            Event.EVAL_AFTER_FORWARD,     # eval forward completed
            Event.EVAL_BATCH_END,         # end of eval batch
        )
        if event not in valid_events:
            return

        batch = state.batch
        if not isinstance(batch, (tuple, list, dict)) or ("input_ids" not in batch or "labels" not in batch):
            logger.log_metrics({"debug/doubleshift/error": 1})
            self._done = True
            return

        # Underlying HF model, NOT the wrapper (so no wrapper-side label gymnastics)
        hf = self._get_hf_model(state)
        hf.eval()

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch["labels"]
        pad_id = self.pad_id

        with torch.no_grad():
            # 0) Wrapper forward loss via Composer model (reflects wrapper behavior)
            try:
                out_wrap = state.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)  # type: ignore[attr-defined]
                wrapper_loss_forward = float(out_wrap.loss.detach().cpu().item()) if getattr(out_wrap, "loss", None) is not None else float("nan")
            except Exception:
                wrapper_loss_forward = float("nan")

            # 1) Native HF forward loss using underlying model (no wrapper shifting logic)
            try:
                out_hf = hf(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
                hf_loss = float(out_hf.loss.detach().cpu().item()) if getattr(out_hf, "loss", None) is not None else float("nan")
            except Exception:
                hf_loss = float("nan")

            # 2) Get logits WITHOUT passing labels (so the HF model does not compute/shift loss internally)
            out = hf(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = out.logits  # [B, T, V]
            V = logits.size(-1)

            # Manual CE with ONE shift (the correct CLM objective)
            lg1 = logits[..., :-1, :].contiguous()
            lb1 = labels[..., 1:].contiguous()
            ce1 = F.cross_entropy(
                lg1.view(-1, V), lb1.view(-1),
                ignore_index=pad_id, reduction="mean"
            )
            n1 = int((lb1 != pad_id).sum().item())

            # Manual CE with TWO shifts (what you'd see if labels were pre-shifted once before calling a model that also shifts)
            if logits.size(1) >= 3:
                lg2 = logits[..., :-2, :].contiguous()
                lb2 = labels[..., 2:].contiguous()
                ce2 = F.cross_entropy(
                    lg2.view(-1, V), lb2.view(-1),
                    ignore_index=pad_id, reduction="mean"
                )
                n2 = int((lb2 != pad_id).sum().item())
            else:
                ce2 = torch.tensor(float("nan"), device=logits.device)
                n2 = 0

            # Prefer Composer-provided state.loss if convertible; fallback to wrapper forward value
            try:
                wrapper_loss = float(state.loss) if state.loss is not None else float("nan")
            except Exception:
                wrapper_loss = wrapper_loss_forward

        # Log once
        logger.log_metrics({
            "debug/doubleshift/pad_id": pad_id,
            "debug/doubleshift/wrapper_loss": wrapper_loss,
            "debug/doubleshift/wrapper_loss_forward": wrapper_loss_forward,
            "debug/doubleshift/hf_loss": hf_loss,
            "debug/doubleshift/ce_shift1": float(ce1.item()),
            "debug/doubleshift/ce_shift2": float(ce2.item()),
            "debug/doubleshift/tokens_shift1": n1,
            "debug/doubleshift/tokens_shift2": n2,
        })

        # Also print to stdout for quick inspection
        print("[DoubleShiftDetector] pad_id:", pad_id)
        print("[DoubleShiftDetector] wrapper_loss:", wrapper_loss)
        print("[DoubleShiftDetector] wrapper_loss_forward:", wrapper_loss_forward)
        print("[DoubleShiftDetector] hf_loss:", hf_loss)
        print("[DoubleShiftDetector] ce_shift1:", float(ce1.item()), "tokens:", n1)
        print("[DoubleShiftDetector] ce_shift2:", float(ce2.item()), "tokens:", n2)

        self._done = True
