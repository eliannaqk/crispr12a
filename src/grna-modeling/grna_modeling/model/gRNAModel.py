from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl

from grna_modeling.utility.tensor import padcat
from grna_modeling.utility.protein import ProteinBatch, NucleicAcidBatch

from grna_modeling.model.transformer import *
from grna_modeling.utility import vocabulary, convert_vocab


@dataclass
class gRNAModelInput():
    """
    Input type of for gRNA model.
    """

    rna_batch: NucleicAcidBatch
    protein_batch: ProteinBatch
    protein_embs: Optional[torch.FloatTensor] = None
    S_label: Optional[torch.LongTensor] = None


@dataclass
class gRNAModelOutput():
    """
    Output type of for gRNA model.
    """

    logits: torch.Tensor
    S_pred: torch.Tensor
    loss: Optional[torch.Tensor] = None
    accuracy: Optional[torch.Tensor] = None
    perplexity: Optional[torch.Tensor] = None


class gRNAModel(pl.LightningModule):

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.save_hyperparameters()
        config = self.hparams.config

        d_s = config["d_s"]
        d_s_protein = config["d_s_protein"]
        enc_config = config["enc_self_attn"]
        dec_config = config["dec_self_attn"]

        vocab_size = len(vocabulary.model_restypes)
        self.protein_embed = nn.Linear(d_s_protein, d_s)
        self.rna_embed = nn.Embedding(vocab_size, d_s)

        self.enc_layers = nn.ModuleList([])
        for _ in range(config["n_enc_layers"]):
            n_heads = enc_config["n_heads"]
            d_hidden = d_s // n_heads
            l = EncoderLayer(
                d_s=d_s,
                d_hidden=d_hidden,
                n_heads=n_heads,
            )
            self.enc_layers.append(l)

        self.dec_layers = nn.ModuleList([])
        for _ in range(config["n_dec_layers"]):
            n_heads = dec_config["n_heads"]
            d_hidden = d_s // n_heads
            l = CrossDecoderLayer(
                d_s_self=d_s,
                d_s_cross=d_s,
                d_hidden=d_hidden,
                n_heads=n_heads,
                use_self_rope=True,
            )
            self.dec_layers.append(l)

        self.lm_head = nn.Linear(d_s, vocab_size)

    def forward(
        self,
        input: gRNAModelInput,
        sampling=False,
    ) -> gRNAModelOutput:
        rna_batch = input.rna_batch.to(self.device)
        protein_batch = input.protein_batch.to(self.device)
        protein_embs = input.protein_embs.to(self.device)
        S_label = input.S_label

        protein_mask = (protein_batch.C != 0).int()
        rna_mask = (rna_batch.C != 0).int()

        protein_s = self.protein_embed(protein_embs)
        rna_s = self.rna_embed(
            convert_vocab.standard_to_model_tokens(rna_batch.S))

        for layer in self.enc_layers:
            protein_s = layer(protein_s, protein_mask)

        for layer in self.dec_layers:
            rna_s = layer(rna_s, protein_s, rna_mask, protein_mask)

        logits = self.lm_head(rna_s)

        ### Calculate losses if given label
        if S_label is not None:
            S_label = S_label.to(self.device)
            S_label_model = convert_vocab.standard_to_model_tokens(S_label)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = S_label_model[..., 1:].contiguous()
            loss_mask = ((shift_labels != vocabulary.pad_token) &
                         (shift_labels != vocabulary.gap_token)).int()

            cce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            cce_loss = cce_loss_fct(shift_logits.transpose(1, 2), shift_labels)
            cce_loss = (cce_loss * loss_mask).sum(-1) / loss_mask.sum(-1)
            loss = cce_loss

            acc = (shift_logits.argmax(-1) == shift_labels).float() * loss_mask
            acc = acc.sum(-1) / loss_mask.sum(-1)

            ppl = cce_loss.exp()
        else:
            loss, acc, ppl = None, None, None

        return_logits = convert_vocab.model_to_standard_logits(logits)
        if sampling:
            _S = convert_vocab.standard_to_model_tokens(rna_batch.S)
            vocabulary.restrict_tokens(_S, return_logits)

        S_pred = return_logits.argmax(-1)

        output = gRNAModelOutput(
            logits=return_logits,
            S_pred=S_pred,
            loss=loss,
            accuracy=acc,
            perplexity=ppl,
        )

        return output
