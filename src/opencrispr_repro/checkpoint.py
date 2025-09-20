# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/mosaicml/llm-foundry/blob/41e02d778c22185dbcd3fd4fedd64ff264883ee9/llmfoundry/callbacks/hf_checkpointer.py

import contextlib
import copy
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from composer.core import Callback, Event, Precision, State, Time, TimeUnit
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import (
    dist,
    format_name_with_dist_and_time,
    maybe_create_remote_uploader_downloader_from_uri,
    parse_uri,
)
from composer.utils.misc import create_interval_scheduler
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

try:
    import transformer_engine.pytorch as te

    is_te_imported = True
except (ImportError, ModuleNotFoundError):
    is_te_imported = False

log = logging.getLogger(__name__)

__all__ = ["HuggingFaceCheckpointer"]


class HuggingFaceCheckpointer(Callback):
    """Save a huggingface formatted checkpoint during training.

    Args:
        save_folder (str): Top level folder to save checkpoints to (can be a
            URI). It is likely that this would be the same as your save_folder.
        save_interval: Union[str, int, Time]: The interval describing how often
            checkpoints should be saved. If an integer, it will be assumed to be
            in :attr:`.TimeUnit.EPOCH`. Otherwise, the unit must be either
            :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
            :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
        huggingface_folder_name (str): Folder to save each checkpoint under (can
            be a format string). Default is ``ba{batch}``..
        precision: The precision to save the model in. Default is ``float32``.
            Options are ``bfloat16``, ``float16``, or ``float32``.
        overwrite (bool): Whether to overwrite previous checkpoints.
    """

    def __init__(
        self,
        save_folder: str,
        save_interval: Union[str, int, Time],
        huggingface_folder_name: str = "ba{batch}",
        precision: str = "float32",
        overwrite: bool = True,
    ):
        _, _, self.save_dir_format_str = parse_uri(save_folder)
        self.overwrite = overwrite
        self.precision = precision
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[precision]
        self.using_peft = False

        self.huggingface_folder_name_fstr = os.path.join(
            "huggingface",
            huggingface_folder_name,
        )

        self.save_interval: Time = Time.from_input(
            save_interval,
            TimeUnit.EPOCH,
        )
        self.check_interval = create_interval_scheduler(
            self.save_interval,
            include_end_of_training=True,
        )
        self.remote_ud = maybe_create_remote_uploader_downloader_from_uri(
            save_folder,
            loggers=[],
        )
        if self.remote_ud is not None:
            self.remote_ud._num_concurrent_uploads = 4

        self.last_checkpoint_batch: Optional[Time] = None

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        # The interval scheduler handles only returning True for the appropriate events
        if (
            state.get_elapsed_duration() is not None
            and self.check_interval(
                state,
                event,
            )
            and self.last_checkpoint_batch != state.timestamp.batch
        ):
            self._save_checkpoint(state, logger)
        elif event == Event.INIT:
            if not isinstance(state.model, HuggingFaceModel):
                raise ValueError(
                    "`HuggingFaceCheckpointer` is only compatible with `HuggingFaceModel`s. "
                    + f"Got {type(state.model)} instead.",
                )
            if self.remote_ud is not None:
                self.remote_ud.init(state, logger)
                state.callbacks.append(self.remote_ud)

            # Check if the model is using PEFT
            if state.is_model_ddp:
                composer_model = state.model.module
            elif isinstance(state.model.model, FSDP):
                composer_model = state.model
            else:
                composer_model = state.model
            self.using_peft = composer_model.using_peft

    def transform_model_and_tokenizer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Transform the model and tokenizer before saving.

        This allows a subclass to modify the model and tokenizer before saving. The base class implementation will
        make no modifications.

        Args:
            model (PreTrainedModel): The model to be transformed.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be transformed.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizerBase]: The transformed model and tokenizer.
        """
        return model, tokenizer

    def transform_config(
        self,
        original_config: PretrainedConfig,
    ) -> PretrainedConfig:
        """Transform the model config before saving.

        Args:
            original_config (Any): The original model config.

        Returns:
            The transformed model config.
        """
        copied_config = copy.deepcopy(original_config)
        return copied_config

    def transform_model_pre_registration(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        """Transform the model before registering with MLflow.

        This allows a subclass to modify the model before registering with MLflow. The base class implementation will
        make no modifications.

        Args:
            model (PreTrainedModel): The model to be transformed.

        Returns:
            PreTrainedModel: The transformed model.
        """
        return model

    def _save_checkpoint(
        self,
        state: State,
        logger: Logger,
    ) -> None:
        """Save a HuggingFace formatted checkpoint.

        Args:
            state (State): The training state.
            logger (Logger): The logger.
        """
        del logger  # unused

        self.last_checkpoint_batch = state.timestamp.batch

        log.info("Saving HuggingFace formatted checkpoint")

        save_dir = format_name_with_dist_and_time(
            str(
                Path(self.save_dir_format_str) / self.huggingface_folder_name_fstr,
            ),
            state.run_name,
            state.timestamp,
        )

        # Use a temporary directory if save_dir is remote.
        use_temp_dir = self.remote_ud is not None
        temp_save_dir = tempfile.mkdtemp() if use_temp_dir else save_dir

        log.debug("Gathering state dict")

        if state.is_model_ddp:
            original_model: PreTrainedModel = state.model.module.model
            state_dict_model = state.model.module.model
            original_tokenizer = state.model.module.tokenizer
        elif isinstance(state.model.model, FSDP):
            original_model = state.model.model.module
            state_dict_model = state.model.model
            original_tokenizer = state.model.tokenizer
        else:
            original_model = state.model.model
            state_dict_model = state.model.model
            original_tokenizer = state.model.tokenizer

        cpu_offload = True

        # Add hook to move tensors to cpu to avoid CUDA OOM
        def tensor_hook(
            module: nn.Module,
            state_dict: dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> dict[str, Any]:
            dtensor_fqns = []
            for fqn in state_dict.keys():
                tensor = state_dict[fqn]
                if isinstance(tensor, DTensor):
                    dtensor_fqns.append(fqn)
                    tensor = tensor.full_tensor()  # type: ignore
                    if dist.get_global_rank() == 0:
                        # Offload any DTensors to CPU
                        if cpu_offload:
                            tensor = tensor.cpu()
                        state_dict[fqn] = tensor
                    else:
                        state_dict[fqn] = None

                if isinstance(state_dict[fqn], torch.Tensor):
                    state_dict[fqn] = state_dict[fqn].to(dtype=self.dtype)
                del tensor
            if dist.get_global_rank() != 0:
                state_dict = {}
            return state_dict

        hooks = []
        for _, module in state_dict_model.named_modules():
            hooks.append(
                module._register_state_dict_hook(tensor_hook),
            )

        state_dict = get_model_state_dict(
            state_dict_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=cpu_offload,
            ),
        )
        for hook in hooks:
            hook.remove()

        new_model_instance = None  # Need this for pyright because variable could be unbound

        if dist.get_global_rank() == 0:
            log.debug("Saving Hugging Face checkpoint in global rank 0")

            # Transform HF config before building 2nd model copy
            new_config = self.transform_config(
                original_config=original_model.config,
            )

            log.debug("Creating new model instance")

            # First create the model instance on meta device to avoid the
            # initialization cost.
            with init_empty_weights():
                if self.using_peft:
                    active_adapter = original_model.active_adapter
                    base_model = original_model.get_base_model()
                    new_base_model_instance = type(base_model)(new_config)

                    new_model_instance = type(original_model)(
                        new_base_model_instance,
                        original_model.peft_config[active_adapter],
                    )
                    del new_base_model_instance
                else:
                    new_model_instance = type(original_model)(new_config)
                    if new_model_instance.generation_config is not None:
                        new_model_instance.generation_config.update(
                            **original_model.generation_config.to_dict(),
                        )

            # Then load the state dict in with "assign" so that the state dict
            # is loaded properly even though the model is initially on meta device.
            new_model_instance.load_state_dict(state_dict, assign=True)
            del state_dict

            # Transform the model and tokenizer before saving
            new_model_instance, original_tokenizer = self.transform_model_and_tokenizer(
                new_model_instance,
                original_tokenizer,
            )

            log.debug("Saving Hugging Face checkpoint to disk")

            # This context manager casts the TE extra state in io.BytesIO format to tensor format
            # Needed for proper hf ckpt saving.
            if is_te_imported and state.precision == Precision.AMP_FP8:
                context_manager = te.onnx_export(True)
            else:
                context_manager = contextlib.nullcontext()
            with context_manager:
                new_model_instance.save_pretrained(
                    temp_save_dir,
                    max_shard_size="5GB",
                )
            if original_tokenizer is not None:
                assert isinstance(
                    original_tokenizer,
                    PreTrainedTokenizerBase,
                )
                original_tokenizer.save_pretrained(temp_save_dir)

            if self.remote_ud is not None:
                for filename in os.listdir(temp_save_dir):
                    remote_file_name = os.path.join(save_dir, filename)
                    remote_file_uri = self.remote_ud.remote_backend.get_uri(
                        remote_file_name,
                    )
                    log.info(
                        f"Uploading HuggingFace formatted checkpoint to {remote_file_uri}",
                    )
                    self.remote_ud.upload_file(
                        state=state,
                        remote_file_name=remote_file_name,
                        file_path=Path(
                            os.path.join(temp_save_dir, filename),
                        ),
                        overwrite=self.overwrite,
                    )

        dist.barrier()

        if use_temp_dir and dist.get_global_rank() == 0:
            shutil.rmtree(temp_save_dir, ignore_errors=True)
        dist.barrier()
