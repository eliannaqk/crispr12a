from typing import Any, Dict, List, Tuple, Union

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer

# This is the standard residue order when coding residues type as a number.
restypes = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
    "a",
    "c",
    "g",
    "t",
    "u",
    "x",
    "-",
    "_",
    "1",
    "2",
    "3",
    "4",
    "5",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}

restype_1to3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
    "a": "A",
    "c": "C",
    "g": "G",
    "t": "T",
    "u": "U",
    "x": "unk",
    "-": "GAP",
    "_": "PAD",
    "1": "SP1",
    "2": "SP2",
    "3": "SP3",
    "4": "SP4",
    "5": "SP5",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
restype_3to1.update(
    {
        "DA": "a",
        "DC": "c",
        "DG": "g",
        "DT": "t",
    }
)

protein_restypes = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
nucleic_restypes = ["a", "c", "g", "t", "u"]
special_restypes = ["-", "_", "1", "2", "3", "4", "5"]
unknown_protein_restype = "X"
unknown_nucleic_restype = "x"
gap_token = restypes.index("-")
pad_token = restypes.index("_")

alternative_restypes_map = {
    # Protein
    "MSE": "MET",
}
allowable_restypes = set(restypes + list(alternative_restypes_map.keys()))

vocab_dict = {r: i for i, r in enumerate(restypes)}
tokenizer = Tokenizer(
    BPE(
        vocab=vocab_dict,
        merges=[],
        unk_token="-",
        fuse_unk=False,
    ))
tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder(add_prefix_space=True)
tokenizer.enable_padding(pad_token="_", pad_id=pad_token)  # nosec



def tokenize_sequence(sequence: str, ) -> torch.Tensor:
    """Convert a sequence of characters into a sequence of numbers."""

    tokens = tokenizer.encode(sequence).ids

    return torch.tensor(tokens)


def decode_sequence(S: torch.Tensor) -> str:
    """Convert a sequence of numbers into a sequence of characters."""

    sequence = tokenizer.decode(S.tolist())

    return sequence


def is_protein_sequence(sequence: str) -> bool:
    """Check if a sequence is a protein sequence."""

    return all([s in protein_restypes for s in sequence])


def is_nucleic_sequence(sequence: str) -> bool:
    """Check if a sequence is a nucleic acid sequence."""

    return all([s in nucleic_restypes for s in sequence])
