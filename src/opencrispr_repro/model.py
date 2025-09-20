import os
import tarfile
import urllib
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from transformers.models.esm import EsmForSequenceClassification, EsmTokenizer

from .modeling_progen2 import ProGenForCausalLM
from .schema import ModelSchema


def get_progen2(path: str) -> ProGenForCausalLM:
    # Download pre-trained checkpoint if needed
    sizes = ["small", "medium", "oas", "base", "large", "BFD90", "xlarge"]
    if path in sizes:
        size = path
        path = str(Path.home() / f".cache/progen/progen2-{size}")
        url = f"https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-{size}.tar.gz"
        if not os.path.exists(path + ".success"):
            os.makedirs(path, exist_ok=True)
            urllib.request.urlretrieve(url, path + ".tar.gz")
            with tarfile.open(path + ".tar.gz") as f:
                f.extractall(path)
            Path(path + ".success").touch()

    # Create model and load actual checkpoint
    return ProGenForCausalLM.from_pretrained(path)


def get_esm2(path: str) -> EsmForSequenceClassification:
    sizes = [
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D",
    ]
    if path in sizes:
        path = f"facebook/{path}"
    return EsmForSequenceClassification.from_pretrained(
        path, problem_type="single_label_classification", num_labels=2
    )


def get_model(config: ModelSchema) -> PreTrainedModel:
    if config.name == "progen2":
        return get_progen2(config.path)
    if config.name == "esm2":
        return get_esm2(config.path)
    raise ValueError("config.model.name must be progen2 or esm2")


def get_tokenizer(config: ModelSchema) -> PreTrainedTokenizerBase:
    if config.name == "progen2":
        path = os.path.join(os.path.dirname(__file__), "tokenizer_progen2.json")
        tokenizer = Tokenizer.from_file(path)
        tokenizer.enable_padding(pad_id=0, pad_token="<|pad|>")  # nosec B106
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    if config.name == "esm2":
        return EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    raise ValueError("config.model.name must be progen2 or esm2")
