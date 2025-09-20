from typing import Dict, List, Optional

import torch
from Bio import SeqIO

from profluent.utility.common.basic import exists
from profluent.utility.protein import (
    MacromoleculeBatch,
    NucleicAcid,
    nucleotide_constants,
    vocabulary,
)


class NucleicAcidBatch(MacromoleculeBatch):

    def __init__(
        self,
        C: torch.Tensor,
        S: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__(NucleicAcidBatch, NucleicAcid)
        self.check_dims(C, S)

        self.C = C
        self.S = S

        if exists(metadata):
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(len(C))]

        self.size = C.shape[0]
        self.length = C.shape[1]

    @staticmethod
    def from_nucleic_acids(nucleic_acids: List[NucleicAcid]):
        C = torch.nn.utils.rnn.pad_sequence(
            [nucleic_acid.C for nucleic_acid in nucleic_acids],
            batch_first=True,
            padding_value=0,
        )
        S = torch.nn.utils.rnn.pad_sequence(
            [nucleic_acid.S for nucleic_acid in nucleic_acids],
            batch_first=True,
            padding_value=vocabulary.pad_token,
        )

        metadata = [nucleic_acid.metadata for nucleic_acid in nucleic_acids]

        return NucleicAcidBatch(C, S, metadata)

    @staticmethod
    def from_sequences(sequences: List[str], device: str = "cpu"):
        sequences_flat = []
        for sequence in sequences:
            if isinstance(sequence, str):
                sequences_flat.append(sequence)
            else:
                sequences_flat.append("".join(sequence))

        encodings = vocabulary.tokenizer.encode_batch(sequences_flat)
        S = torch.tensor([e.ids for e in encodings], device=device)

        C = torch.zeros_like(S, device=device).long()
        for batch_i in range(len(sequences)):
            sequence = sequences[batch_i]
            if isinstance(sequence, str):
                sequence = [sequence]
            last_seq_end = 0
            for seq_i, seq in enumerate(sequence):
                C[batch_i, last_seq_end:last_seq_end + len(seq)] = -(seq_i + 1)
                last_seq_end += len(seq)

        return NucleicAcidBatch(C, S)

    @staticmethod
    def from_fasta(
        fasta_file: str,
        record_id: str = None,
    ):
        """Convert a FASTA file to a NucleicAcidBatch object.

        Parameters
        ----------
        fasta_file : str
            Path to FASTA file.
        record_id : str, optional
            If specified, only the sequence with the given record ID will be
            included. If not specified, all sequences in the FASTA file will be
            included.

        Returns
        -------
        NucleicAcidBatch
            NucleicAcidBatch object.
        """

        nucleic_acids = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            if not exists(record_id) or record.id == record_id:
                nucleic_acids.append(NucleicAcid.from_sequence(str(
                    record.seq)))

        return NucleicAcidBatch.from_nucleic_acids(nucleic_acids)

    @staticmethod
    def from_fastas(
        fasta_files: List[str],
        record_ids: Optional[List[str]] = None,
    ):
        """Convert a list of FASTA files to a NucleicAcidBatch object.

        Parameters
        ----------
        fasta_files : list
            List of paths to FASTA files.
        record_ids : list, optional
            List of record IDs to extract from each FASTA file. If None, all records will be extracted.

        Returns
        -------
        NucleicAcidBatch
            NucleicAcidBatch object.
        """

        if not exists(record_ids):
            record_ids = [None for _ in fasta_files]
        else:
            assert len(fasta_files) == len(record_ids)

        nucleic_acids = []
        for fasta_file, record_id in zip(fasta_files, record_ids):
            nucleic_acids.append(NucleicAcid.from_fasta(fasta_file, record_id))

        return NucleicAcidBatch.from_nucleic_acids(nucleic_acids)
