from tempfile import TemporaryDirectory
import subprocess as sp

import pandas as pd
import Bio.SeqIO
import Bio.Seq


def make_sgrnas(
    tracr_sequences,
    crispr_sequences,
    repeat_length=16,
    linker="GAAA",
):
    """
    data_dir: dir containing data/repeats.fna, data/tracrs.fna;
              ids should be are matched: e.g. ML-Cas9-0-0 in repeats.fna matches ML-Cas9-0-0 in tracrs.fna
    repeat_length: length of repeat after trimming (equiv to length of stem in the 1st stem loop)
    linker: used to link repeat and tracrRNA; GAAA tetraloop is the standard
    """

    # create a temporary directory to store the sequences
    tmp_dir = TemporaryDirectory()
    data_dir = tmp_dir.name

    unique_sgrna_dict = {
        f"{tracr_sequences[i]}_{crispr_sequences[i]}": i
        for i in range(len(tracr_sequences))
    }

    tracr_f = open(f"{data_dir}/tracrs.fna", "w")
    crispr_f = open(f"{data_dir}/repeats.fna", "w")
    for concat_sequence, sgrna_idx in unique_sgrna_dict.items():
        tracr_sequence, crispr_sequence = concat_sequence.split("_")
        tracr_f.write(f">sgrna{sgrna_idx}\n{tracr_sequence}\n")
        crispr_f.write(f">sgrna{sgrna_idx}\n{crispr_sequence}\n")
    tracr_f.close()
    crispr_f.close()

    print("blasting repeats to tracrs")
    cmd = f"""
    blastn -query {data_dir}/repeats.fna \
        -subject {data_dir}/tracrs.fna \
        -word_size 4 \
        -outfmt '6 std qlen slen'  \
        -out {data_dir}/repeat_tracr_alignments.tsv
    """
    proc = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()

    print("extracting self alignments")
    self_align = {}
    for l in open(f"{data_dir}/repeat_tracr_alignments.tsv"):
        r = l.split()
        if r[0] != r[1]: continue
        if r[0] not in self_align:
            self_align[r[0]] = r
        elif float(r[-3]) > float(self_align[r[0]][-3]):
            self_align[r[0]] = r

    print("writing self alignments")
    with open(f"{data_dir}/repeat_tracr_alignments_self.tsv", "w") as f:
        fields = ["qname", "tname", "aln", "qstart", "qend", "tstart", "tend"]
        f.write("\t".join(fields) + "\n")
        for row in self_align.values():
            x = [row[0], row[2], row[3], row[6], row[7], row[8], row[9]]
            f.write("\t".join(x) + "\n")

    print("making sgrnas")
    repeats = dict(
        [[_.id, str(_.seq)]
         for _ in Bio.SeqIO.parse(open(f"{data_dir}/repeats.fna"), "fasta")])
    tracrs = dict(
        [[_.id, str(_.seq)]
         for _ in Bio.SeqIO.parse(open(f"{data_dir}/tracrs.fna"), "fasta")])

    sgrnas = {}
    aligns = pd.read_csv(f"{data_dir}/repeat_tracr_alignments_self.tsv",
                         delimiter="\t")

    for i, r in aligns.iterrows():

        repeat = repeats[r["qname"]]
        tracr = tracrs[r["qname"]]
        offset = r["qend"] - repeat_length
        repeat_trimmed = repeat[:r["qend"] - offset]
        tracr_trimmed = tracr[r["tend"] - 1 + offset:]
        sgrna = repeat_trimmed + linker + tracr_trimmed

        pass_qc = r["qend"] > r["qstart"]
        pass_qc = r["tstart"] > r["tend"] and pass_qc
        pass_qc = len(repeat_trimmed) == repeat_length and pass_qc

        if pass_qc:
            sgrnas[r["qname"]] = sgrna
        else:
            sgrnas[r["qname"]] = None

    # get the sgrna sequences and return them in the same order as the input
    sgrna_sequences = []
    for tracr_sequence, crispr_sequence in zip(tracr_sequences,
                                               crispr_sequences):
        concat_sequence = f"{tracr_sequence}_{crispr_sequence}"
        sgrna_idx = unique_sgrna_dict[concat_sequence]
        sgrna_id = f"sgrna{sgrna_idx}"

        if sgrna_id not in sgrnas:
            sgrna_sequences.append(None)
            continue

        sgrna_sequence = sgrnas[sgrna_id]

        if sgrna_sequence is not None:
            sgrna_sequences.append(sgrna_sequence.lower())
        else:
            sgrna_sequences.append(None)

    return sgrna_sequences
