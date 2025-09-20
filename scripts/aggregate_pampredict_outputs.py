#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from typing import List, Dict, Tuple

# Reuse parsing helpers from atlas_make_pam_crispr if available
try:
    from atlas_make_pam_crispr import parse_pam_prediction_txt, parse_info_table, count_unique_protospacers
except Exception:
    # Fallback minimal implementations
    def parse_pam_prediction_txt(p: Path):
        res = {"pam_side": None, "pam_iupac": None, "inferred_spacer_orient": None}
        if not p.exists():
            return res
        for line in p.read_text().splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            k, v = [t.strip() for t in line.split(":", 1)]
            k = k.lower()
            if k.startswith("pam position"):
                res["pam_side"] = v.upper()
            elif k.startswith("predicted pam") or k == "pam":
                res["pam_iupac"] = v.strip().upper().split()[0]
            elif k.startswith("inferred"):
                res["inferred_spacer_orient"] = v.upper()
        return res

    def parse_info_table(p: Path) -> float:
        if not p.exists():
            return 0.0
        with open(p) as fh:
            rdr = csv.DictReader(fh, delimiter='\t')
            peak = 0.0
            for row in rdr:
                try:
                    vals = [float(row.get(b, "0")) for b in ("A", "C", "G", "T")]
                except Exception:
                    continue
                peak = max(peak, max(vals))
        return peak

    def count_unique_protospacers(filtered_dir: Path) -> int:
        uniq = set()
        files = sorted(filtered_dir.glob("*_filtered_matches_with_flanking_sequences.tsv"))
        for f in files:
            with open(f) as fh:
                rdr = csv.DictReader(fh, delimiter='\t')
                cols = {c.lower(): c for c in (rdr.fieldnames or [])}
                for row in rdr:
                    sseqid = None
                    for cand in ("sseqid", "subject", "ref", "reference"):
                        if cand in cols:
                            sseqid = row[cols[cand]]
                            break
                    sstart = row.get(cols.get("sstart", ""), "") if cols.get("sstart") else ""
                    send = row.get(cols.get("send", ""), "") if cols.get("send") else ""
                    sstrand = row.get(cols.get("sstrand", ""), "") if cols.get("sstrand") else ""
                    key = (sseqid, sstart, send, sstrand) if sseqid else ("LINE", hash("\t".join(row.get(c, "") for c in rdr.fieldnames or [])))
                    uniq.add(key)
        return len(uniq)


def count_fasta_records(p: Path) -> int:
    n = 0
    if not p.exists():
        return 0
    with open(p) as fh:
        for line in fh:
            if line.startswith('>'):
                n += 1
    return n


def scan_outdir(outdir: Path, min_unique: int, min_snr: float) -> List[Dict]:
    rows = []
    spacers = sorted((outdir / "spacers").glob("*.fna"))
    for fa in spacers:
        name = fa.stem  # <operon_safe>__arrN
        run_out = outdir / "pampredict_runs" / name
        pred = parse_pam_prediction_txt(run_out / "PAM_prediction.txt")
        peak_up = parse_info_table(run_out / "upstream_flanking_sequence_info.tsv")
        peak_dn = parse_info_table(run_out / "downstream_flanking_sequence_info.tsv")
        unique = count_unique_protospacers(run_out / "blastn")
        snr = (peak_up / max(peak_dn, 1e-9)) if (peak_up and peak_dn is not None) else 0.0
        pam_iupac = pred.get("pam_iupac") or ""
        if pam_iupac:
            if unique >= min_unique and snr > min_snr:
                label, reason = "high_confidence", "unique>=min && snr>min"
            else:
                label, reason = "low_confidence", "insufficient_evidence"
        else:
            label, reason = "no_call", "no_pam_prediction_txt"

        # Parse array idx
        op_id = name
        arr_idx = ""
        if "__arr" in name:
            op_id, arr = name.rsplit("__arr", 1)
            arr_idx = int(arr) if arr.isdigit() else arr

        rows.append({
            "operon_id": op_id,
            "array_idx": arr_idx,
            "n_spacers": count_fasta_records(fa),
            "pam_side": pred.get("pam_side") or "UPSTREAM",
            "pam_iupac": pam_iupac,
            "unique_hits": unique,
            "peak_bits_up": f"{peak_up:.3f}",
            "peak_bits_down": f"{peak_dn:.3f}",
            "snr": f"{snr:.3f}",
            "label": label,
            "reason": reason,
            "protein_sha1s": "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Aggregate PAMpredict outputs into TSVs (per-array + operon-best)")
    ap.add_argument("--outdirs", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-unique", type=int, default=10)
    ap.add_argument("--min-snr", type=float, default=2.0)
    args = ap.parse_args()

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for od in args.outdirs:
        all_rows.extend(scan_outdir(Path(od), args.min_unique, args.min_snr))

    # per-array TSV
    pa = outdir / "pam_crispr.per_array.tsv"
    hdr = [
        "operon_id","array_idx","n_spacers","pam_side","pam_iupac","unique_hits",
        "peak_bits_up","peak_bits_down","snr","label","reason","protein_sha1s"
    ]
    with open(pa, "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for r in all_rows:
            fh.write("\t".join(str(r[k]) for k in hdr) + "\n")

    # per-operon best (choose best row per operon_id)
    best_by_operon: Dict[str, Tuple[Tuple[int,int,float], Dict]] = {}
    for r in all_rows:
        # Score: (is_high, unique, snr)
        is_high = 1 if r["label"] == "high_confidence" else 0
        score = (is_high, int(r["unique_hits"]), float(r["snr"]))
        key = r["operon_id"]
        if (key not in best_by_operon) or (score > best_by_operon[key][0]):
            best_by_operon[key] = (score, r)

    pm = outdir / "pam_map.tsv"
    with open(pm, "w") as fh:
        fh.write("operon_id\tpam_iupac\tpam_side\tsource\n")
        for _k, (_s, r) in best_by_operon.items():
            fh.write(f"{r['operon_id']}\t{r['pam_iupac']}\t{r['pam_side']}\tPAMpredict\n")

    print(f"[aggregate] wrote:\n  {pa}\n  {pm}")


if __name__ == "__main__":
    main()

