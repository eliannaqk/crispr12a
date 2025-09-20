import json
import logging
import os

import click
import torch
import torch.distributed as dist
from ruamel.yaml import YAML
from torch.distributed.elastic.multiprocessing.errors import record

from opencrispr_repro.schema import FinetuneAPI
from opencrispr_repro.trainer import get_trainer

logger = logging.getLogger()


def setup_dist() -> None:
    rank = int(os.environ.get("RANK", -1))
    if dist.is_available() and torch.cuda.is_available() and rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def read_config_file(config_path: str) -> dict:
    with open(config_path) as f:
        if config_path.endswith(".yml") or config_path.endswith(".yaml"):
            config = YAML(typ="safe").load(f)
        else:
            config = json.load(f)
    return config


@record
@click.command()
@click.option("--config", "config_path", required=True, help="JSON or YML file based on schema.FinetuneAPI")
@click.option("--eval-only", is_flag=True, default=False,
              help="Skip training; just run evaluation")
def main(config_path: str, eval_only: bool) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    setup_dist()
    raw_cfg = read_config_file(config_path)
    config = FinetuneAPI(**raw_cfg)
    trainer = get_trainer(config)
    # After WandBLogger initializes the run inside the Trainer, log config and upload YAML
    try:
        import wandb
        if wandb.run is not None:
            # Log both the raw YAML (as 'spec') and the resolved Pydantic model
            try:
                wandb.config.update({"spec": raw_cfg}, allow_val_change=True)
            except Exception:
                pass
            try:
                wandb.config.update(config.model_dump(), allow_val_change=True)
            except Exception:
                pass
            try:
                art = wandb.Artifact("training_config", type="config")
                art.add_file(config_path)
                wandb.run.log_artifact(art)
            except Exception:
                pass
    except Exception:
        pass
    done = trainer.state.max_duration <= trainer.state.timestamp.get(trainer.state.max_duration.unit)
    if not done and not eval_only:
        trainer.fit()
    else:
        trainer.eval()


if __name__ == "__main__":
    main()
