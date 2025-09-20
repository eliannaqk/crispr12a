import os
from typing import List, Tuple, Union
from copy import deepcopy

from einops import repeat
import torch
from tqdm import tqdm

from profluent.models.esm import ESM2
from profluent.utility.common.basic import exists
from profluent.utility.common.tensor import padcat
from profluent.utility.protein import Protein, ProteinBatch, NucleicAcidBatch

from grna_modeling.model.interface import gRNAModelInput, gRNAModelOutput
from grna_modeling.model.gRNAModel import gRNAModel
from grna_modeling.model.DecoderOnlyModel import gRNAModel as DecoderOnlyModel
from grna_modeling.utility import vocabulary
from grna_modeling.utility.gcs import get_parameters


class gRNAModelRunner():

    def __init__(
        self,
        esm_model="esm8m",
        cluster_id=90,
        cache_dir: str = None,
        device: str = "cuda",
    ) -> None:
        ckpt_file = get_parameters(
            esm_model=esm_model,
            cluster_id=cluster_id,
            cache_dir=cache_dir,
        )
        self.model = gRNAModel.load_from_checkpoint(ckpt_file)
        self.esm = ESM2(
            model=self.model.hparams.config["esm_model"],
            device=device,
        )

        self.model.to(device)
        self.model.eval()
        self.device = device

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, input: gRNAModelInput) -> gRNAModelOutput:
        return self.model(input)

    def score(
        self,
        tracr_batch: NucleicAcidBatch,
        crispr_batch: NucleicAcidBatch,
        protein_batch: Union[Protein, ProteinBatch],
        tracr_first: bool = True,
        batch_size: int = 1,
    ):
        """
        Score sequences for a given protein

        Parameters
        ----------
        tracr_batch : NucleicAcidBatch
            tracrRNA sequences
        crispr_batch : NucleicAcidBatch
            crRNA sequences
        protein_batch : Union[Protein, ProteinBatch]
            Protein or ProteinBatch object
        tracr_first : bool, optional
            Whether to score with tracrRNA first, by default True
        batch_size : int, optional
            Batch size for scoring, by default 1

        Returns
        -------
        torch.Tensor
            Log-likelihood scores
        """
        assert tracr_batch.size == crispr_batch.size, \
            "tracr_batch and crispr_batch must have the same size"

        if isinstance(protein_batch, Protein):
            single_protein = True
            protein_batch = ProteinBatch.from_proteins([protein_batch] *
                                                       tracr_batch.size)
        else:
            single_protein = False
            assert tracr_batch.size == crispr_batch.size == protein_batch.size, \
            "tracr_batch, crispr_batch, and protein_batch must have the same size"

        tracr_sequences = [rna.get_sequence()[0] for rna in tracr_batch]
        crispr_sequences = [rna.get_sequence()[0] for rna in crispr_batch]
        rna_sequences = vocabulary.concat_rna_sequences(
            tracr_sequences,
            crispr_sequences,
            tracr_first=tracr_first,
        )

        rna_batch = NucleicAcidBatch.from_sequences(rna_sequences)
        protein_batch = protein_batch.to(self.device)

        ll = []
        for i in range(0, rna_batch.size, batch_size):
            rna_batch_i = rna_batch[i:i + batch_size]
            protein_batch_i = protein_batch[i:i + batch_size]

            protein_seqs_i = [
                protein.get_sequence()[0] for protein in protein_batch_i
            ]
            if not single_protein:
                protein_embs_i = [
                    self.esm.embed(sequence)[None].detach()
                    for sequence in protein_seqs_i
                ]
                protein_embs_i = padcat(protein_embs_i, val=0, dim=0)
            else:
                protein_embs_i = self.esm.embed(
                    protein_seqs_i[0])[None].detach()
                protein_embs_i = repeat(
                    protein_embs_i,
                    "1 n d -> b n d",
                    b=protein_batch_i.size,
                )

            model_in = gRNAModelInput(
                rna_batch=rna_batch_i,
                protein_batch=protein_batch_i,
                protein_embs=protein_embs_i,
                S_label=rna_batch_i.S,
            )
            output = self.model(model_in)

            ll.append(-output.loss.cpu().detach())

            del model_in
            del protein_embs_i
            del protein_batch_i
            del rna_batch_i
            del output

        ll = torch.cat(ll)

        return ll

    def sample(
        self,
        protein: Union[Protein, ProteinBatch],
        num_samples: int,
        tracr_first=True,
        batch_size: int = 1,
        temperature: float = None,
        max_len: int = 300,
        silent: bool = True,
    ):
        """
        Sample sequences for a given protein

        Parameters
        ----------
        protein : Union[Protein, ProteinBatch]
            Protein or ProteinBatch object
        num_samples : int
            Number of samples to generate
        tracr_first : bool, optional
            Whether to generate tracrRNA first, by default True
        batch_size : int, optional
            Batch size for sampling, by default 1
        temperature : float, optional
            Sampling temperature, by default None
        max_len : int, optional
            Maximum length of the generated sequences, by default 300
        silent : bool, optional
            Whether to show progress bar, by default True

        Returns
        -------
        List[Tuple[str, str]]
            List of tuples of tracrRNA and crRNA sequences
        """
        if isinstance(protein, ProteinBatch):
            assert protein.size == 1, "ProteinBatch must have size 1 to sample"
            protein = protein[0]

        protein_sequence = protein.get_sequence()[0]
        protein_embs = self.esm.embed(protein_sequence)

        rna_start_token = vocabulary.TRACR_START_SENT if tracr_first else vocabulary.CRISPR_START_SENT

        sequences = []
        sample_pbar = tqdm(total=num_samples,
                           desc="Sampling",
                           ncols=80,
                           disable=silent)
        while len(sequences) < num_samples:
            batch_size_i = min(batch_size, num_samples - len(sequences))
            protein_batch_i = ProteinBatch.from_proteins([protein] *
                                                         batch_size_i)
            protein_embs_i = repeat(protein_embs,
                                    "n d -> b n d",
                                    b=batch_size_i)
            rna_batch_i = NucleicAcidBatch.from_sequences([rna_start_token] *
                                                          batch_size_i)

            input_i = gRNAModelInput(
                rna_batch=rna_batch_i,
                protein_batch=protein_batch_i,
                protein_embs=protein_embs_i,
            )

            all_finished = False
            pos_i = 0
            while not all_finished:
                model_out = self.model(input_i, sampling=True)

                logits = model_out.logits
                logits_i = logits[:, pos_i]

                # apply temperature and sampling parameters
                if exists(temperature):
                    logits_i = logits_i / temperature
                    # logits_i = limit_logits(logits_i, top_p, top_k)
                    probs_i = torch.softmax(logits_i, dim=-1)
                    S_pred_i = torch.multinomial(probs_i, 1)
                else:
                    S_pred_i = torch.argmax(logits_i, dim=-1)[None]

                rna_batch_i.X = torch.cat(
                    [rna_batch_i.X,
                     torch.zeros_like(rna_batch_i.X[:, :1])],
                    dim=1)
                rna_batch_i.C = torch.cat([
                    rna_batch_i.C, -1 * torch.ones_like(rna_batch_i.C[:, :1])
                ],
                                          dim=1)
                rna_batch_i.S = torch.cat([rna_batch_i.S, S_pred_i], dim=1)

                pos_i += 1
                all_finished = vocabulary.check_rna_complete(rna_batch_i).all()

                if pos_i > max_len:
                    break

            # keep only unique combinations of tracr and crispr sequences

            sampled_sequences = []
            try:
                for tracr, crispr in zip(
                        *vocabulary.parse_rna_sequences(rna_batch_i)):
                    if len(tracr) < 0 or len(crispr) < 0:
                        continue
                    elif len(tracr) + len(crispr) > max_len:
                        continue

                    sampled_sequences.append((tracr, crispr))
            except:
                sampled_sequences = [("", "")]

            sequences.extend(sampled_sequences)
            sample_pbar.update(len(sequences) - sample_pbar.n)

        return sequences
