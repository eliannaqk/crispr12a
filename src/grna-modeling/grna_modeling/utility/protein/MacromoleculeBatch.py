import torch

from profluent.utility.common.basic import exists
from profluent.utility.protein import vocabulary


class MacromoleculeBatch:

    def __init__(self, batch_class, singleton_class) -> None:
        self.batch_class = batch_class
        self.singleton_class = singleton_class

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        C = self.C[key]
        S = self.S[key]
        metadata = self.metadata[key]

        if isinstance(key, slice):
            return self.batch_class(X, C, S, metadata)
        elif isinstance(key, int):
            unpaded_mask = C != 0
            C = C[unpaded_mask]
            S = S[unpaded_mask]
            return self.singleton_class(C, S, metadata)

    def __add__(self, other):
        if self.length < other.length:
            pad_size = other.length - self.length
            C1 = torch.nn.functional.pad(
                self.C,
                (0, pad_size),
                value=0,
            )
            S1 = torch.nn.functional.pad(
                self.S,
                (0, pad_size),
                value=vocabulary.pad_token,
            )
            C2, S2 = other.get_CS()
        elif other.length < self.length:
            pad_size = self.length - other.length
            C2 = torch.nn.functional.pad(
                other.C,
                (0, pad_size),
                value=0,
            )
            S2 = torch.nn.functional.pad(
                other.S,
                (0, pad_size),
                value=vocabulary.pad_token,
            )
            C1, S1 = self.get_CS()
        else:
            C1, S1 = self.get_CS()
            C2, S2 = other.get_CS()

        C = torch.cat((C1, C2), dim=0)
        S = torch.cat((S1, S2), dim=0)

        metadata = self.metadata + other.metadata

        return self.batch_class(C, S, metadata)

    def check_dims(self, C, S):
        assert (C.shape[0] == S.shape[0]), "C and S must have the same length"

    def to(self, device):
        self.C = self.C.to(device)
        self.S = self.S.to(device)

        return self

    def get_CS(self, to_numpy=False, clone=False):
        if clone:
            C, S = self.C.clone(), self.S.clone()
        else:
            C, S = self.C, self.S

        if to_numpy:
            return C.numpy(), S.numpy()
        return C, S

    def get_metadata(self):
        return self.metadata

    def get_missing_residue_mask(self):
        return self.C < 0

    def set_XCS(self, X=None, C=None, S=None):
        if exists(C):
            self.check_dims(C, self.S)
            self.C = C
        if exists(S):
            self.check_dims(self.C, S)
            self.S = S

    def set_metadata(self, metadata):
        assert self.C.shape[0] == len(metadata)
        self.metadata = metadata

    def update_metadata(self, metadata):
        assert self.C.shape[0] == len(metadata)
        for i, new_metadata in enumerate(metadata):
            self.metadata[i].update(new_metadata)

    def to_fasta(
        self,
        fasta_file: str,
        delimiter: str = None,
    ) -> None:
        """Convert a MacromoleculeBatch object to a FASTA file.

        Parameters
        ----------
        fasta_file : str
            Path to FASTA file.
        delimiter : str, optional
            Delimiter to use between chain sequences. If None, all macromolecules must have a single chain ID.
        """

        chain_ids = [
            macromolecule.get_chain_ids(reduce=True) for macromolecule in self
        ]
        sequences = [macromolecule.get_sequence() for macromolecule in self]

        if not exists(delimiter):
            for chain_id in chain_ids:
                assert len(
                    chain_id
                ) == 1, "All macromolecules must have a single chain ID, or a delimiter must be provided."

            delimiter = ""

        with open(fasta_file, 'w') as f:
            for i, sequences_ in enumerate(sequences):
                sequence = delimiter.join(sequences_)
                f.write(f'>{i}\n{sequence}\n\n')
