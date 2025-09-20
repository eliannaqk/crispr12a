import torch

from profluent.utility.protein import vocabulary, NucleicAcidBatch

TRACR_START_SENT = "1"
TRACR_END_SENT = "2"
CRISPR_START_SENT = "3"
CRISPR_END_SENT = "4"

model_restypes = [
    "a",
    "c",
    "g",
    "t",
    TRACR_START_SENT,
    TRACR_END_SENT,
    CRISPR_START_SENT,
    CRISPR_END_SENT,
    "-",
    "_",
]
model_tokens = [vocabulary.restypes.index(aa) for aa in model_restypes]

gap_token = model_restypes.index("-")
pad_token = model_restypes.index("_")

tracr_start_token = model_restypes.index(TRACR_START_SENT)
tracr_end_token = model_restypes.index(TRACR_END_SENT)
crispr_start_token = model_restypes.index(CRISPR_START_SENT)
crispr_end_token = model_restypes.index(CRISPR_END_SENT)

standard_tracr_start_token = vocabulary.restypes.index(TRACR_START_SENT)
standard_tracr_end_token = vocabulary.restypes.index(TRACR_END_SENT)
standard_crispr_start_token = vocabulary.restypes.index(CRISPR_START_SENT)
standard_crispr_end_token = vocabulary.restypes.index(CRISPR_END_SENT)


def restrict_tokens(S: torch.Tensor, logits: torch.Tensor):
    token_mask = torch.ones(logits.shape[-1]).bool()
    token_mask[model_tokens] = False
    logits[..., token_mask] = -float('inf')

    tracr_started = S.eq(tracr_start_token).any(dim=1)
    tracr_ended = S.eq(tracr_end_token).any(dim=1)
    crispr_started = S.eq(crispr_start_token).any(dim=1)
    crispr_ended = S.eq(crispr_end_token).any(dim=1)

    tracr_start_token_allowed = ~tracr_started & crispr_started & crispr_ended
    tracr_end_token_allowed = tracr_started & ~tracr_ended
    crispr_start_token_allowed = tracr_started & tracr_ended & ~crispr_started
    crispr_end_token_allowed = crispr_started & ~crispr_ended

    for i in range(logits.shape[0]):
        if ~tracr_start_token_allowed[i]:
            logits[i, :, standard_tracr_start_token] = -float('inf')
        if ~tracr_end_token_allowed[i]:
            logits[i, :, standard_tracr_end_token] = -float('inf')
        if ~crispr_start_token_allowed[i]:
            logits[i, :, standard_crispr_start_token] = -float('inf')
        if ~crispr_end_token_allowed[i]:
            logits[i, :, standard_crispr_end_token] = -float('inf')


def add_tracr_sentinels(sequences):
    if isinstance(sequences, str):
        sequences = [sequences]

    return [TRACR_START_SENT + s + TRACR_END_SENT for s in sequences]


def add_crispr_sentinels(sequences):
    if isinstance(sequences, str):
        sequences = [sequences]

    return [CRISPR_START_SENT + s + CRISPR_END_SENT for s in sequences]


def concat_rna_sequences(tracr_sequences, crispr_sequences, tracr_first=True):
    tracr_sequences = add_tracr_sentinels(tracr_sequences)
    crispr_sequences = add_crispr_sentinels(crispr_sequences)

    if tracr_first:
        rna_sequences = [
            a + b for a, b in zip(tracr_sequences, crispr_sequences)
        ]
    else:
        rna_sequences = [
            a + b for a, b in zip(crispr_sequences, tracr_sequences)
        ]

    return rna_sequences


def check_rna_complete(rna_batch: NucleicAcidBatch):
    # check if rna_batch.S has all the sentinels

    has_tracr_start = rna_batch.S.eq(standard_tracr_start_token).any(dim=1)
    has_tracr_end = rna_batch.S.eq(standard_tracr_end_token).any(dim=1)
    has_crispr_start = rna_batch.S.eq(standard_crispr_start_token).any(dim=1)
    has_crispr_end = rna_batch.S.eq(standard_crispr_end_token).any(dim=1)

    has_all = has_tracr_start & has_tracr_end & has_crispr_start & has_crispr_end

    return has_all


def parse_rna_sequences(rna_batch: NucleicAcidBatch):
    tracr_sequences, crispr_sequences = [], []
    for rna in rna_batch:
        S = rna.S
        S = S[S != pad_token]
        S = S[S != gap_token]

        tracr_start_idx = torch.where(S == standard_tracr_start_token)[0][0]
        tracr_end_idx = torch.where(S == standard_tracr_end_token)[0][0]
        tracr_sequence = S[tracr_start_idx + 1:tracr_end_idx]
        tracr_sequence = vocabulary.decode_sequence(tracr_sequence)
        tracr_sequences.append(tracr_sequence)

        crispr_start_idx = torch.where(S == standard_crispr_start_token)[0][0]
        crispr_end_idx = torch.where(S == standard_crispr_end_token)[0][0]
        crispr_sequence = S[crispr_start_idx + 1:crispr_end_idx]
        crispr_sequence = vocabulary.decode_sequence(crispr_sequence)
        crispr_sequences.append(crispr_sequence)

    return tracr_sequences, crispr_sequences
