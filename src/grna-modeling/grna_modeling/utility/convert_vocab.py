import torch

from grna_modeling.utility.vocabulary import model_restypes

standard_vocab = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y", "X", "a", "c", "g", "t", "u", "x", "-", "_", "1",
    "2", "3", "4", "5"
]


def vocab_to_vocab(vocab_in, vocab_out):
    """
    Create a tensor that maps from vocab_in to vocab_out
    """

    map_tensor = torch.zeros((len(vocab_in), len(vocab_out)),
                             dtype=torch.float32)
    for i, token in enumerate(vocab_in):
        if token in vocab_out:
            map_tensor[i, vocab_out.index(token)] = 1.0

    return map_tensor


def tokens_to_tokens(tokens, vocab_map):
    one_hot_in = torch.functional.F.one_hot(tokens.long(), vocab_map.size(0))
    one_hot_out = torch.matmul(one_hot_in.float(),
                               vocab_map.to(one_hot_in.device))
    return torch.argmax(one_hot_out, dim=-1)


# cache the conversion tensors
standard_to_model_map = vocab_to_vocab(standard_vocab, model_restypes)
model_to_standard_map = vocab_to_vocab(model_restypes, standard_vocab)


def standard_to_model_logits(logits):
    """
    Convert logits from model to standard
    """

    return torch.matmul(logits, standard_to_model_map.to(logits.device))


def model_to_standard_logits(logits):
    """
    Convert logits from model to standard
    """

    return torch.matmul(logits, model_to_standard_map.to(logits.device))


def standard_to_model_tokens(tokens):
    """
    Convert tokens from standard to model
    """

    return tokens_to_tokens(tokens, standard_to_model_map)


def model_to_standard_tokens(tokens):
    """
    Convert tokens from model to standard
    """

    return tokens_to_tokens(tokens, model_to_standard_map)
