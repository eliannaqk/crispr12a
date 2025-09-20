from typing import List, Optional, Union

from Bio import SeqIO
import torch

from profluent.utility.common.basic import exists
from profluent.utility.protein import Macromolecule, vocabulary


class NucleicAcid(Macromolecule):

    def __init__(
        self,
        C: torch.Tensor,
        S: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.check_dims(C, S)

        self.C = C
        self.S = S

        if exists(metadata):
            self.metadata = metadata
        else:
            self.metadata = {}

        self.length = len(C)

    @staticmethod
    def from_sequence(sequence: Union[str, List[str]], device: str = "cpu"):
        if isinstance(sequence, str):
            sequence = [sequence]

        encodings = vocabulary.tokenizer.encode("".join(sequence)).ids
        S = torch.tensor(encodings, device=device)

        C = torch.zeros_like(S, dtype=torch.long)
        last_seq_end = 0
        for seq_i, seq in enumerate(sequence):
            C[last_seq_end:last_seq_end + len(seq)] = -(seq_i + 1)
            last_seq_end += len(seq)

        return NucleicAcid(C, S).to(device)

    @staticmethod
    def from_fasta(
        fasta_file: str,
        record_id: str = None,
    ):
        """Convert a FASTA file to a NucleicAcid object.

        Paramters
        ---------
        fasta_file : str
            Path to FASTA file.
        record_id : str, optional
            If specified, only the sequence with the given record ID will be
            included. If not specified, all sequences in the FASTA file will be
            included.
        """

        sequences = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            if not exists(record_id) or record.id == record_id:
                sequences.append(str(record.seq))

        return NucleicAcid.from_sequence(sequences)
