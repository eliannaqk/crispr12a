#!/usr/bin/env python
"""
split_cas12a_fasta.py
----------------------------------------------------------
Take a CD-HIT-deduplicated FASTA of Cas12a/Cpf1 sequences and
write 90 / 5 / 5 train-valid-test splits plus a meta.json
manifest into:

    /gpfs/gibbs/pi/gerstein/crispr/Cas12a/

Usage:
    python split_cas12a_fasta.py  cas12_id90.fasta
"""

import sys, random, json
from pathlib import Path
from Bio import SeqIO

# ------------ destination ----------------------------------------------------
# Write splits under a specific subfolder next to your newCrisprData root
DEST = Path("/home/eqk3/project_pi_mg269/eqk3/crisprData/newCrisprData/cas12a_expanded_cdhit99")
DEST.mkdir(parents=True, exist_ok=True)

# ------------ read input -----------------------------------------------------
if len(sys.argv) != 2:
    sys.exit("Usage: python split_cas12a_fasta.py <deduplicated_fasta>")

fasta_in = Path(sys.argv[1])
records  = list(SeqIO.parse(str(fasta_in), "fasta"))

# ------------ shuffle & slice ------------------------------------------------
random.seed(42)               # reproducible split
random.shuffle(records)

n   = len(records)
n_tr = int(0.90 * n)
n_va = int(0.05 * n)          # remainder goes to test

splits = {
    "train": records[:n_tr],
    "valid": records[n_tr:n_tr + n_va],
    "test":  records[n_tr + n_va:],
}

# ------------ write out ------------------------------------------------------
manifest = {}

for name, recs in splits.items():
    out_fa = DEST / f"{name}.fasta"
    SeqIO.write(recs, out_fa, "fasta")
    print(f"{name}: {len(recs)} sequences ➜ {out_fa}")

    for r in recs:
        manifest[r.id] = {"split": name}

with open(DEST / "meta.json", "w") as meta:
    json.dump(manifest, meta, indent=2)

print(f"✓ Finished: files written to {DEST}")