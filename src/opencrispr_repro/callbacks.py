import math
from composer.core import Callback, Event, State
from composer.loggers import Logger
from composer.utils import dist
import torch


class EvalLossLogger(Callback):
    """Accumulates eval loss and logs an averaged metric at EVAL_END.

    Logs under both 'loss/eval/total' and 'val/loss' for easy discovery in dashboards.
    """

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.EVAL_START:
            self._sum = 0.0
            self._count = 0
        elif event == Event.EVAL_BATCH_END:
            try:
                loss = float(state.loss) if state.loss is not None else math.nan
            except Exception:
                loss = math.nan
            if not math.isnan(loss) and math.isfinite(loss):
                self._sum += loss
                self._count += 1
        elif event == Event.EVAL_END:
            if self._count > 0:
                mean_loss = self._sum / self._count
                logger.log_metrics({
                    'loss/eval/total': mean_loss,
                    'val/loss': mean_loss,
                })


class TokenMemoryLogger(Callback):
    """Logs tokens-per-batch and peak GPU memory to help tune batch sizing.

    - Logs every `log_every` training steps on rank 0 to avoid duplication.
    - Uses `torch.cuda.max_memory_allocated()` as a proxy for peak usage since last reset.
    """

    def __init__(self, log_every: int = 50) -> None:
        self.log_every = max(1, int(log_every))

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if not torch.cuda.is_available() or not dist.get_world_size() or not dist.get_global_rank() == 0:
            # Only log on rank 0 when CUDA is available
            return

        if event == Event.INIT:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        if event == Event.BATCH_END and state.timestamp.batch.value % self.log_every == 0:
            try:
                batch = state.batch
                tokens = None
                if isinstance(batch, dict):
                    if 'attention_mask' in batch:
                        am = batch['attention_mask']
                        if isinstance(am, torch.Tensor):
                            tokens = int(am.sum().item())
                    elif 'input_ids' in batch:
                        ii = batch['input_ids']
                        if isinstance(ii, torch.Tensor):
                            # Fall back to non-padded token count (approx)
                            tokens = int(ii.numel())
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                tokens = None
                peak_gb = float('nan')

            metrics = {'cuda/peak_mem_gb': peak_gb}
            if tokens is not None:
                metrics['train/tokens_in_batch'] = tokens
            logger.log_metrics(metrics)
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
