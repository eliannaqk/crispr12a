import logging
import os
import time

import composer.algorithms
from composer import Trainer
from composer.core import DataSpec
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.loggers import WandBLogger

from .checkpoint import HuggingFaceCheckpointer
from .callbacks import EvalLossLogger, TokenMemoryLogger
from .data import SeqDataset, get_dataloader
from .model import get_model, get_tokenizer
from .scheduler import InvSqrtWithWarmupScheduler
from .schema import FinetuneAPI

logger = logging.getLogger(__name__)


def get_trainer(config: FinetuneAPI):
    run_name = os.path.basename(config.save_folder) + time.strftime("%Y%m%d-%H%M%S")
    tokenizer = get_tokenizer(config.model)

    # Build base model (no LoRA support in this trainer)
    base_model = get_model(config.model)
    # Keep model vocab size as-is to avoid mismatches with lm_head during checkpointing

    model = HuggingFaceModel(
        model=base_model,
        tokenizer=tokenizer,
        shift_labels=config.model.name == "progen2",
    )
    logger.info("Initialized model")

    train_data = SeqDataset(
        csv_fname=config.data.train_data_path,
        sequence_col=config.data.sequence_col,
        label_col=config.data.label_col,
    )
    train_dataloader = get_dataloader(config, train_data, tokenizer)
    val_data = SeqDataset(
        csv_fname=config.data.val_data_path,
        sequence_col=config.data.sequence_col,
        label_col=config.data.label_col,
    )
    eval_dataloader = get_dataloader(config, val_data, tokenizer)
    logger.info("Initialized dataloaders")

    train_duration = f"{config.training.train_steps}ba"
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = InvSqrtWithWarmupScheduler(
        t_warmup=f"{config.training.warmup_steps}ba", t_max=train_duration,
    )
    logger.info("Initialized optimizer")

    algorithms = []
    clipping_threshold = config.training.gradient_clipping_threshold
    if clipping_threshold is not None:
        gradclip = composer.algorithms.GradientClipping(
            clipping_type="norm",
            clipping_threshold=float(clipping_threshold),
        )
        algorithms.append(gradclip)

    save_interval = f"{config.save_interval_steps}ba"
    half = "bf16"
    # When autoresume is False, save under a fresh subfolder to avoid colliding with prior runs
    save_root = config.save_folder
    if not config.autoresume:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_root = os.path.join(config.save_folder, f"run-{timestamp}")
        logger.info(f"Autoresume disabled; saving checkpoints under new subfolder: {save_root}")
    # Save Hugging Face-format checkpoints under the chosen folder
    checkpointer = HuggingFaceCheckpointer(save_folder=save_root, save_interval=save_interval, precision=half)
    wandb_logger = WandBLogger(project="crispr12a", entity="eqk3")

    # FSDP setup
    
    fsdp_config = dict(
        use_orig_params=False,
        limit_all_gathers=True,
        activation_checkpointing=True,
        activation_checkpointing_reentrant=False,
        sync_module_states=True,
        keep_low_precision_grads=False,
        mixed_precision=dict(param_dtype=half, reduce_dtype=half, buffer_dtype=half),
        forward_prefetch=False,
        backward_prefetch="BACKWARD_PRE",
        sharding_strategy="FULL_SHARD",
        state_dict_type="sharded",
        sharded_ckpt_prefix_dir="ba{batch}",
    )

    
    # No gradient accumulation logic in Trainer; YAML-only knobs control microbatch size

    # Eval cadence from YAML if provided (either 'eval_interval' as time string or 'eval_interval_steps' as int)
    eval_interval = None
    if getattr(config, "eval_interval", None):
        eval_interval = config.eval_interval
    elif getattr(config, "eval_interval_steps", None):
        eval_interval = f"{config.eval_interval_steps}ba"

    # Honor YAML setting for autoresume directly (no LoRA logic)
    composer_autoresume = config.autoresume

    # If using token-bucketed batches (variable batch sizes), prevent Composer from
    # further splitting batches by wrapping with a DataSpec that returns the batch
    # as a single microbatch. This ensures the packed token count (approx. target)
    # directly controls per-step memory usage.
    token_bucketed = getattr(config.data, "token_bucketed_batches", False)
    device_microbatch = 1 if token_bucketed else config.data.batch_size

    # Wrap loaders with DataSpec to disable Composer split when token-bucketed
    train_loader_for_trainer = train_dataloader
    eval_loader_for_trainer = eval_dataloader
    if token_bucketed:
        train_loader_for_trainer = DataSpec(
            dataloader=train_dataloader,
            split_batch=lambda batch, microbatch_size: [batch],
        )
        eval_loader_for_trainer = DataSpec(
            dataloader=eval_dataloader,
            split_batch=lambda batch, microbatch_size: [batch],
        )

    trainer = Trainer(
        run_name=run_name,
        model=model,
        train_dataloader=train_loader_for_trainer,
        eval_dataloader=eval_loader_for_trainer,
        eval_interval=eval_interval or "300ba",
        spin_dataloaders=False,
        # Disable progress bar to avoid autoresume requiring ProgressBarLogger state
        progress_bar=False,
        device_train_microbatch_size=device_microbatch,
        precision=f"amp_{half}",
        parallelism_config={"fsdp": fsdp_config},
        # Optimization
        max_duration=train_duration,
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=algorithms,
        step_schedulers_every_batch=True,
        loggers=[wandb_logger],
        # Save/load
        autoresume=composer_autoresume,
        callbacks=[checkpointer, EvalLossLogger(), TokenMemoryLogger(log_every=50)],
        save_folder=save_root,
        save_filename="ba{batch}-rank{rank}.pt",
        save_latest_filename="latest",
        save_overwrite=False,
        save_interval=save_interval,
    )
    logger.info("Initialized trainer")
    return trainer
