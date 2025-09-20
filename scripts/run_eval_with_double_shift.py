import os
import logging
import json
from ruamel.yaml import YAML
import click

from opencrispr_repro.schema import FinetuneAPI
from opencrispr_repro.debug.double_shift_detector import DoubleShiftDetector
from opencrispr_repro.debug.wrapper_loss_echo import WrapperLossEcho


def _read_config_file(config_path: str) -> dict:
    with open(config_path) as f:
        if config_path.endswith((".yml", ".yaml")):
            return YAML(typ="safe").load(f)
        return json.load(f)


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

    # Monkeypatch: inject DoubleShiftDetector at Trainer construction time
    import opencrispr_repro.trainer as trainer_mod
    try:
        from composer import Trainer as ComposerTrainer
    except Exception:
        from composer.trainer import Trainer as ComposerTrainer  # fallback

    class PatchedTrainer(ComposerTrainer):
        def __init__(self, *args, callbacks=None, **kwargs):  # type: ignore[no-untyped-def]
            cbs = list(callbacks) if callbacks is not None else []
            cbs.append(DoubleShiftDetector(pad_id=0))
            cbs.append(WrapperLossEcho())
            super().__init__(*args, callbacks=cbs, **kwargs)

    # Apply patch
    trainer_mod.Trainer = PatchedTrainer  # type: ignore[attr-defined]

    # Build trainer via codebase helper (will use PatchedTrainer internally)
    trainer = trainer_mod.get_trainer(cfg)

    # Eval-only run to trigger callback on first eval batch
    trainer.eval()


if __name__ == "__main__":
    main()
