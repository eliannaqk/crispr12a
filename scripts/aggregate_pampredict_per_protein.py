#!/usr/bin/env python3
import argparse, csv, json, gzip, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from atlas_make_pam_crispr import (
        parse_pam_prediction_txt,
        parse_info_table,
        count_unique_protospacers,
    )
except Exception:
    # Minimal fallbacks
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
                    sseqid = row.get(cols.get("sseqid",""), "") if cols.get("sseqid") else ""
                    sstart = row.get(cols.get("sstart",""), "") if cols.get("sstart") else ""
                    send   = row.get(cols.get("send",""),   "") if cols.get("send")   else ""
                    sstrand= row.get(cols.get("sstrand",""),"") if cols.get("sstrand") else ""
                    key = (sseqid, sstart, send, sstrand)
                    uniq.add(key)
        return len(uniq)

def fast_unique_hits(run_dir: Path) -> int:
    """
    Prefer fast parse of spacer_alignment_stats.tsv for 'Total # of unique spacers'.
    Fallback to expensive scan of blastn/ filtered matches if stats file is absent.
    """
    stats = run_dir / "spacer_alignment_stats.tsv"
    try:
        if stats.exists():
            with open(stats) as fh:
                for line in fh:
                    if line.lower().startswith("total # of unique spacers"):
                        parts = line.strip().split('\t')
                        if parts:
                            try:
                                return int(parts[-1])
                            except Exception:
                                pass
    except Exception:
        pass
    # fallback
    return count_unique_protospacers(run_dir / "blastn")


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def iter_atlas_records(path: str):
    def _yield_ndjson(fh):
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    def _yield_top_array(fh):
        buf=[]; in_str=False; esc=False; depth=0; saw_arr=False
        while True:
            ch = fh.read(8192)
            if not ch:
                break
            for c in ch:
                if not saw_arr:
                    if c.isspace():
                        continue
                    if c == '[':
                        saw_arr = True
                        continue
                if not in_str and depth == 0 and c == ']':
                    s = ''.join(buf).strip()
                    if s:
                        yield json.loads(s)
                    return
                if not in_str and depth == 0 and c == ',':
                    s=''.join(buf).strip()
                    if s:
                        yield json.loads(s)
                    buf=[]; continue
                buf.append(c)
                if in_str:
                    if esc:
                        esc=False
                    elif c == '\\':
                        esc=True
                    elif c == '"':
                        in_str=False
                else:
                    if c == '"':
                        in_str=True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
        s=''.join(buf).strip()
        if s:
            yield json.loads(s)
    with open_text(path) as fh:
        first = fh.read(1); fh.seek(0)
        if first == "[":
            yield from _yield_top_array(fh)
        else:
            yield from _yield_ndjson(fh)


def is_cas12a_gene(g: str, h: str) -> bool:
    s = f"{g} {h}".lower()
    return ("cas12a" in s) or ("cpf1" in s)


def scan_outdirs(outdirs: List[Path]) -> Dict[str, Dict]:
    """
    Produce best-per-operon rows keyed by op_slug using available outputs.
    """
    per_array: Dict[Tuple[str,int], Dict] = {}
    for od in outdirs:
        rundir = od / "pampredict_runs"
        if not rundir.exists():
            continue
        for arr_dir in sorted(rundir.glob("*__arr*")):
            if not arr_dir.is_dir():
                continue
            name = arr_dir.name
            if "__arr" not in name:
                continue
            op_slug, arr = name.rsplit("__arr", 1)
            try:
                idx = int(arr)
            except ValueError:
                continue
            up = arr_dir / "upstream_flanking_sequence_info.tsv"
            dn = arr_dir / "downstream_flanking_sequence_info.tsv"
            if not (up.exists() and dn.exists()):
                continue
            pred = parse_pam_prediction_txt(arr_dir / "PAM_prediction.txt")
            peak_up = parse_info_table(up)
            peak_dn = parse_info_table(dn)
            unique = fast_unique_hits(arr_dir)
            snr = (peak_up / max(peak_dn, 1e-9)) if (peak_up and peak_dn is not None) else 0.0
            pam = pred.get("pam_iupac") or ""
            if pam:
                label, reason = ("high_confidence","unique>=min && snr>min") if (unique >= 10 and snr > 2.0) else ("low_confidence","insufficient_evidence")
            else:
                label, reason = ("no_call","no_pam_prediction_txt")
            per_array[(op_slug, idx)] = {
                "op_slug": op_slug,
                "array_idx": idx,
                "pam_side": pred.get("pam_side") or "UPSTREAM",
                "pam_iupac": pam,
                "unique_hits": unique,
                "peak_bits_up": f"{peak_up:.3f}",
                "peak_bits_down": f"{peak_dn:.3f}",
                "snr": f"{snr:.3f}",
                "label": label,
                "reason": reason,
                "run_dir": str(arr_dir.resolve()),
            }
    # choose best per op_slug
    best: Dict[str, Dict] = {}
    for (slug, _idx), r in per_array.items():
        is_high = 1 if r["label"] == "high_confidence" else 0
        uniq = int(r["unique_hits"])
        snr = float(r["snr"])
        score = (is_high, uniq, snr)
        prev = best.get(slug)
        if (prev is None) or (score > (1 if prev["label"] == "high_confidence" else 0, int(prev["unique_hits"]), float(prev["snr"]))):
            best[slug] = r
    return best


def main():
    ap = argparse.ArgumentParser(description="Aggregate existing PAMPredict outputs into pam_crispr.per_protein.tsv using Atlas mapping")
    ap.add_argument("--atlas-json", required=True)
    ap.add_argument("--outdirs", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdirs = [Path(p).resolve() for p in args.outdirs]
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    best_by_slug = scan_outdirs(outdirs)

    # Stream atlas; emit proteins only for operons we have results for
    pp = outdir / "pam_crispr.per_protein.tsv"
    hdr = [
        "protein_sha1","operon_id","array_idx","pam_side","pam_iupac","unique_hits",
        "peak_bits_up","peak_bits_down","snr","label","reason","run_dir"
    ]
    total = 0
    with open(pp, "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for rec in iter_atlas_records(args.atlas_json):
            op = rec.get("operon_id")
            if not op:
                continue
            slug = safe_name(op)
            r = best_by_slug.get(slug)
            if not r:
                continue
            arr_idx = r.get("array_idx", "")
            for cas in (rec.get("cas") or []):
                g = cas.get("gene_name", ""); h = cas.get("hmm_name", "")
                if not is_cas12a_gene(g, h):
                    continue
                aa = (cas.get("protein") or "").strip()
                if not aa:
                    continue
                import hashlib
                sha = hashlib.sha1(aa.encode("utf-8")).hexdigest()
                row = {
                    "protein_sha1": sha,
                    "operon_id": op,
                    "array_idx": arr_idx,
                    "pam_side": r["pam_side"],
                    "pam_iupac": r["pam_iupac"],
                    "unique_hits": r["unique_hits"],
                    "peak_bits_up": r["peak_bits_up"],
                    "peak_bits_down": r["peak_bits_down"],
                    "snr": r["snr"],
                    "label": r["label"],
                    "reason": r["reason"],
                    "run_dir": r.get("run_dir",""),
                }
                fh.write("\t".join(str(row[k]) for k in hdr) + "\n")
                total += 1

    print(f"[aggregate_per_protein] wrote {pp} with {total} proteins")


if __name__ == "__main__":
    main()
