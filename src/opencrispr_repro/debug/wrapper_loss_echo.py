from __future__ import annotations
from composer.core import Callback, Event, State
from composer.loggers import Logger


class WrapperLossEcho(Callback):
    """Logs Composer's state.loss at eval batch end as debug/doubleshift/wrapper_loss.

    Useful on Composer versions where accessing state.loss inside another callback is tricky.
    """

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        # Capture Composer's computed loss when available; support multiple hook names
        if event not in (Event.AFTER_LOSS, Event.EVAL_BATCH_END):
            return
        try:
            loss = float(state.loss) if state.loss is not None else float('nan')
        except Exception:
            loss = float('nan')
        logger.log_metrics({"debug/doubleshift/wrapper_loss": loss})
