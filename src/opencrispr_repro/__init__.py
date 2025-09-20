"""OpenCRISPR-Repro package root.

This module also registers the custom ProGen2 architecture with the
🤗 Transformers auto-classes so that `AutoConfig.from_pretrained` and
`AutoModelForCausalLM.from_pretrained` seamlessly recognise checkpoints whose
`config.json` has `"model_type": "progen"`.
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_progen2 import ProGenConfig
from .modeling_progen2 import ProGenForCausalLM

# ---------------------------------------------------------------------------
# Register the configuration and model with Transformers' auto factories.
# This allows loading via `AutoModelForCausalLM.from_pretrained(<folder>)` even
# when the folder was saved by Composer / HuggingFace without the original
# Python objects in scope.
# ---------------------------------------------------------------------------
try:
    # Configs are registered by *model_type* string.
    AutoConfig.register(ProGenConfig.model_type, ProGenConfig)
except ValueError:
    # Already registered in this interpreter session – safe to ignore.
    pass

try:
    # Models are registered by *config class*.
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)
except ValueError:
    pass

__all__ = [
    "ProGenConfig",
    "ProGenForCausalLM",
]

