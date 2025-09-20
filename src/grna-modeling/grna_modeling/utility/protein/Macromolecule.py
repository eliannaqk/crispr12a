import copy

import numpy as np
import torch

from profluent.utility.common.basic import exists
from profluent.utility.protein import vocabulary


class Macromolecule():

    def __init__(self) -> None:
        pass

    def __len__(self):
        return self.length

    @staticmethod
    def check_dims(C, S):
        if exists(C) and exists(S):
            assert C.shape[0] == S.shape[
                0], "C and S must have the same length"

    def to(self, device):
        self.C = self.C.to(device)
        self.S = self.S.to(device)

        return self

    def get_XCS(self, to_numpy=False, clone=False):
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

    def get_unpadded_mask(self):
        return self.C != 0

    def set_CS(self, C=None, S=None):
        if exists(C):
            if not len(C) == self.length:
                raise ValueError("C must have the same length as the protein")
            self.C = C
        if exists(S):
            if not len(S) == self.length:
                raise ValueError("S must have the same length as the protein")
            self.S = S

    def set_metadata(self, metadata):
        self.metadata = metadata

    def update_metadata(self, metadata):
        self.metadata.update(metadata)

    def get_sequence(self):
        sequence = []
        C_unique = self.C.abs().unique()
        for c in C_unique:
            if c == 0:
                continue
            sequence.append(
                vocabulary.decode_sequence(self.S[self.C.abs() == c]))

        return sequence

    def get_chain_ids(self, reduce=False):
        PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)

        if "chain_ids" in self.metadata:
            chain_ids = self.metadata["chain_ids"]
        else:
            if len(self.C.abs().unique()) > PDB_MAX_CHAINS:
                # extend PDB_CHAIN_IDS with numbers
                PDB_CHAIN_IDS = PDB_CHAIN_IDS + "".join([
                    str(i) for i in range(
                        len(self.C.abs().unique() - len(PDB_CHAIN_IDS)))
                ])

            C_abs = self.C.abs()
            C_unique = C_abs.unique().tolist()
            C_index = [C_unique.index(c) for c in self.C.abs()]
            chain_ids = [
                PDB_CHAIN_IDS[i] for i, c in zip(C_index, C_abs.tolist())
                if c > 0
            ]

        if len(chain_ids) < self.length:
            chain_ids = chain_ids + [
                None for _ in range(self.length - len(chain_ids))
            ]

        if reduce:
            chain_ids = [c for c in chain_ids if c is not None]
            # Reduce but keep order
            chain_ids = list(dict.fromkeys(chain_ids))

        return chain_ids

    def to_fasta(self, fasta_file: str):
        sequences = self.get_sequence()
        chain_ids = self.get_chain_ids(reduce=True)

        with open(fasta_file, 'w') as f:
            for chain_id, sequence in zip(chain_ids, sequences):
                f.write(f'>{chain_id}\n{sequence}\n\n')
