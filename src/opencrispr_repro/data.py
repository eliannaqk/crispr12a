import functools

import numpy as np
import pandas as pd
import torch
from composer.utils import dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import PreTrainedTokenizerBase

from .schema import FinetuneAPI


class SeqDataset(Dataset):
    def __init__(self, csv_fname: str, sequence_col: str, label_col: str | None = None):
        super().__init__()
        df = pd.read_csv(csv_fname, sep="\t" if csv_fname.endswith(".tsv") else ",")
        self.seqs = df[sequence_col].tolist()
        if label_col is not None:
            self.labels = df[label_col].tolist()
        else:
            self.labels = None
        # Precompute character lengths; token lengths computed lazily if needed
        self.char_lengths = [len(s) for s in self.seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        d = dict(sequence=self.seqs[idx])
        if self.labels is not None:
            d = dict(label=self.labels[idx])
        return d


def progen2_collate_fn(items: list[dict], tokenizer: PreTrainedTokenizerBase, random_reverse: bool = False):
    # Always append boundary tokens first, then optionally reverse the entire wrapped sequence
    wrapped = ["1" + it["sequence"] + "2" for it in items]
    if random_reverse:
        seqs = [w if np.random.random() < 0.5 else w[::-1] for w in wrapped]
    else:
        seqs = wrapped
    batch = tokenizer(seqs, return_tensors="pt", padding=True)

    # Labels = input_ids, but mask pads and the first position with 0 (pad_id)
    # so both HF model loss (ignore_index=0) and manual CE ignore them
    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = input_ids.clone()
    # Mask padded positions with 0 (pad token id)
    labels = labels.masked_fill(attn == 0, 0)
    # Also ignore the first position (BOS alignment for shifted loss) by setting to 0
    if labels.size(1) > 0:
        labels[:, 0] = 0

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
    }


def esm2_collate_fn(items: list[dict], tokenizer: PreTrainedTokenizerBase, random_reverse: bool = False):
    seqs = [item["sequence"] for item in items]
    if random_reverse:
        seqs = [seq if np.random.random() < 0.5 else seq[::-1] for seq in seqs]
    batch = tokenizer(seqs, return_tensors="pt", padding=True)
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": torch.tensor([item["label"] for item in items])
    }

def get_dataloader(config: FinetuneAPI, dataset: SeqDataset, tokenizer: PreTrainedTokenizerBase):
    model_name = config.model.name
    rand_rev = getattr(config.data, "random_reverse", False)
    if model_name == "progen2":
        collate_fn = functools.partial(progen2_collate_fn, tokenizer=tokenizer, random_reverse=rand_rev)
    elif model_name == "esm2":
        collate_fn = functools.partial(esm2_collate_fn, tokenizer=tokenizer, random_reverse=rand_rev)
    else:
        raise ValueError("config.model.name must be progen2 or esm2")

    # Optional token-bucketed batching
    if getattr(config.data, "token_bucketed_batches", False):
        batch_sampler = _TokenBucketedDistributedSampler(
            dataset=dataset,
            tokenizer=tokenizer,
            world_size=dist.get_world_size(),
            rank=dist.get_global_rank(),
            target_tokens=config.data.target_tokens_per_microbatch,
            sortish_multiplier=config.data.bucket_size_multiplier,
            seed=0,
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,  # yields lists of indices per microbatch
            collate_fn=collate_fn,
            num_workers=config.data.num_workers,
        )

    # Default fixed-size batches with distributed sampling
    return DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_global_rank(),
            seed=0,
            drop_last=False,
        ),
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
    )


class _TokenBucketedDistributedSampler(Sampler[list[int]]):
    """Length-bucketed sampler that yields microbatches of indices whose token-length sum
    approximates a target. Uses a sortish strategy per rank to keep batches similar in length.
    """

    def __init__(
        self,
        dataset: SeqDataset,
        tokenizer: PreTrainedTokenizerBase,
        world_size: int,
        rank: int,
        target_tokens: int,
        sortish_multiplier: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.world_size = max(1, world_size)
        self.rank = max(0, rank)
        self.target_tokens = int(max(1, target_tokens))
        self.sortish_multiplier = int(max(1, sortish_multiplier))
        self.seed = int(seed)

        # Compute token lengths once
        wrapped = ["1" + s + "2" for s in dataset.seqs]
        token_ids = tokenizer(wrapped, return_tensors=None, padding=False)["input_ids"]
        self.token_lengths = [len(x) for x in token_ids]

        # Sortish order: chunk by multiplier, sort within chunks by length desc, shuffle chunks
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(self.dataset))
        # initial sort by length desc
        idx = idx[np.argsort(-np.array(self.token_lengths))]
        # chunk and shuffle chunks
        chunk_size = max(1, self.sortish_multiplier * 1)  # factor of base batch; base=1 as we pack greedily
        chunks = [idx[i : i + chunk_size] for i in range(0, len(idx), chunk_size)]
        rng.shuffle(chunks)
        self.sorted_indices = np.concatenate(chunks).tolist()

        # Shard indices across ranks
        self.shard_indices = self.sorted_indices[self.rank :: self.world_size]

    def __iter__(self):
        batch: list[int] = []
        tokens_in_batch = 0
        for i in self.shard_indices:
            tl = self.token_lengths[i]
            if tokens_in_batch > 0 and tokens_in_batch + tl > self.target_tokens:
                # yield current microbatch
                yield batch
                batch = []
                tokens_in_batch = 0
            batch.append(i)
            tokens_in_batch += tl
        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate number of microbatches
        total_tokens = sum(self.token_lengths[i] for i in self.shard_indices)
        return max(1, int(np.ceil(total_tokens / self.target_tokens)))
