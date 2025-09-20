#!/usr/bin/env python3
"""
atlas_make_pam_crispr.py
Build a Cas12a (Type V) PAM database from your Atlas using PAMpredict.

What it does
------------
1) Reads your Atlas (JSON array or NDJSON lines).
2) For each operon that contains a Cas12a/Cpf1 protein, exports each CRISPR
   array's spacers to its own FASTA (arrays are already co-oriented).
3) Runs PAMpredict on each array FASTA against your IMG/VR+IMG/PR BLAST DB
   with: -p UPSTREAM (Cas12) and -l 10 (10-nt window).
4) Parses PAMpredict outputs to compute:
     - consensus PAM (IUPAC) and side,
     - #unique protospacers across DBs,
     - peak information bits (upstream vs downstream),
     - SNR = peak_upstream / peak_downstream,
     - confidence: HIGH if unique>=MIN_UNIQUE and SNR>MIN_SNR (paper).
5) Writes:
   - pam_crispr.per_array.tsv  (one row per (operon_id, array_idx))
   - pam_crispr.per_protein.tsv (best array assigned to each Cas12a protein)
   - pam_map.tsv (operon_id -> consensus PAM for easiest merging later)

References
----------
- PAMpredict usage and outputs (README): spacers must be in SAME orientation;
  use -p UPSTREAM for Cas12; outputs listed in README.  [GitHub]
- Confidence rule: >=10 unique protospacers AND SNR>2 (Type V upstream). [Paper]

"""

import argparse, json, gzip, os, re, subprocess, sys, csv
from pathlib import Path
from typing import Iterable, Optional


# ---------- helpers ----------

def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def iter_atlas_records(path: str) -> Iterable[dict]:
    """
    Stream Atlas records from either:
      - NDJSON (one JSON object per line), or
      - a single JSON array at the top level (without loading into RAM).

    This avoids json.load on the entire ~GB-scale Atlas when it's a top-level array.
    """
    def _yield_from_ndjson(fh):
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

    def _yield_from_top_array(fh):
        # Consume initial whitespace and '['
        # Then stream object by object using a simple state machine.
        buf = []
        in_string = False
        esc = False
        depth = 0  # depth of { } only; we are inside the top-level [ ... ]
        saw_array_start = False
        saw_any = False

        while True:
            ch = fh.read(8192)
            if not ch:
                break
            for c in ch:
                if not saw_array_start:
                    if c.isspace():
                        continue
                    if c == '[':
                        saw_array_start = True
                        continue
                    # If file is actually NDJSON without '[', fallback
                    # (shouldn't happen due to first char check)
                    buf.append(c)
                    return

                # Detect end of array when not inside an object
                if not in_string and depth == 0 and c == ']':
                    # Flush any pending (shouldn't be)
                    s = ''.join(buf).strip()
                    if s:
                        try:
                            yield json.loads(s)
                        except Exception:
                            pass
                    return

                # Skip commas between objects when not in string/object
                if not in_string and depth == 0 and c == ',':
                    s = ''.join(buf).strip()
                    if s:
                        yield json.loads(s)
                        saw_any = True
                    buf = []
                    continue

                buf.append(c)

                # Track JSON string state to ignore braces inside strings
                if in_string:
                    if esc:
                        esc = False
                    elif c == '\\':
                        esc = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1

        # If we exit loop without closing ']', try to parse last buffer
        s = ''.join(buf).strip()
        if s:
            try:
                yield json.loads(s)
            except Exception:
                return

    with open_text(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            # Stream objects from top-level array
            for rec in _yield_from_top_array(fh):
                yield rec
        else:
            # NDJSON
            for rec in _yield_from_ndjson(fh):
                yield rec


def sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def norm_nt(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip().upper().replace("U", "T")
    s = re.sub(r"[^ACGT]", "", s)
    return s or None


def is_cas12a(gene_name: str, hmm_name: str) -> bool:
    s = f"{gene_name} {hmm_name}".lower()
    return ("cas12a" in s) or ("cpf1" in s)


def write_fasta(spacers, out_fa: Path):
    with open(out_fa, "w") as out:
        for i, sp in enumerate(spacers, 1):
            out.write(f">sp{i}\n{sp}\n")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run(cmd, **kw):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

def safe_name(s: str) -> str:
    """Return a filesystem-safe slug for s (keep alnum, dot, dash, underscore)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


# ---------- PAMpredict parsers ----------

def parse_pam_prediction_txt(p: Path):
    # Expected lines like:
    # PAM position: UPSTREAM
    # PAM: TTTV
    # Inferred CRISPR spacers orientation: FORWARD/REVERSE
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
        elif k == "pam":
            res["pam_iupac"] = v.strip().upper()
        elif k.startswith("inferred"):
            res["inferred_spacer_orient"] = v.upper()
    return res


def parse_info_table(p: Path) -> float:
    """
    Return the peak information (max bits across positions) for this table.
    Table has columns like: pos, A, C, G, T  (values are information per base)
    """
    if not p.exists():
        return 0.0
    with open(p) as fh:
        rdr = csv.DictReader(fh, delimiter='\t')
        peak = 0.0
        for row in rdr:
            try:
                vals = [float(row.get(b, "0")) for b in ("A", "C", "G", "T")]
            except ValueError:
                # tolerate headers/odd rows
                continue
            peak = max(peak, max(vals))
    return peak


def count_unique_protospacers(filtered_dir: Path) -> int:
    """
    Count unique protospacer hits across all *filtered_matches_with_flanking_sequences.tsv files.
    We try to use sseqid + sstart + send + sstrand if present; else de-dup by full line.
    """
    uniq = set()
    files = sorted(filtered_dir.glob("*_filtered_matches_with_flanking_sequences.tsv"))
    for f in files:
        with open(f) as fh:
            rdr = csv.DictReader(fh, delimiter='\t')
            # normalize field names
            cols = {c.lower(): c for c in (rdr.fieldnames or [])}
            for row in rdr:
                # best-effort column mapping across PAMpredict versions
                sseqid = None
                for cand in ("sseqid", "subject", "ref", "reference"):
                    if cand in cols:
                        sseqid = row[cols[cand]]
                        break
                sstart = ""
                for cand in ("sstart", "start", "protospacer_start"):
                    if cand in cols:
                        sstart = row[cols[cand]]
                        break
                send = ""
                for cand in ("send", "end", "protospacer_end"):
                    if cand in cols:
                        send = row[cols[cand]]
                        break
                sstrand = ""
                for cand in ("sstrand", "strand", "orientation"):
                    if cand in cols:
                        sstrand = row[cols[cand]]
                        break
                if sseqid:
                    key = (sseqid, sstart, send, sstrand)
                else:
                    key = ("LINE", hash("\t".join(row.get(c, "") for c in rdr.fieldnames or [])))
                uniq.add(key)
    return len(uniq)


# ---------- main orchestrator ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-json", required=True, help="Atlas JSON or NDJSON")
    ap.add_argument("--pampredict-dir", required=True, help="Path to cloned PAMpredict repo")
    ap.add_argument("--dbdir", required=True, help="Folder with BLAST DB (viruses_plasmids.fna + *.nsq etc.)")
    ap.add_argument("--outdir", required=True, help="Output folder (will create)")
    ap.add_argument("--threads", type=int, default=8, help="Threads to pass to PAMpredict (-t)")
    ap.add_argument("--min-unique", type=int, default=10, help=">= this many unique protospacers for high-confidence (paper)")
    ap.add_argument("--min-snr", type=float, default=2.0, help="Upstream/Downstream peak bits ratio threshold (paper)")
    ap.add_argument("--min_spacers_per_array", type=int, default=3, help="Skip arrays with fewer spacers")
    ap.add_argument("--include-operon-ids", default=None,
                    help="Optional file with operon_ids to include (one per line).")
    ap.add_argument("--max-operons", type=int, default=0,
                    help="If >0, process at most this many matching operons (after filtering).")
    args = ap.parse_args()

    ppdir = Path(args.pampredict_dir).resolve()
    db = Path(args.dbdir).resolve()
    out = Path(args.outdir).resolve()
    safe_mkdir(out)
    spdir = out / "spacers"
    rundir = out / "pampredict_runs"
    safe_mkdir(spdir)
    safe_mkdir(rundir)

    per_array_rows = []
    protein_to_best = {}  # sha1 -> (score tuple, row dict)

    # Optional allow-list of operon_ids
    allow_ids = None
    if args.include_operon_ids:
        p = Path(args.include_operon_ids)
        if not p.exists():
            raise SystemExit(f"[error] include-operon-ids file not found: {p}")
        with open(p) as fh:
            allow_ids = {line.strip() for line in fh if line.strip()}

    n_ops = 0
    n_done = 0
    for rec in iter_atlas_records(args.atlas_json):
        n_ops += 1
        operon_id = rec.get("operon_id") or f"op{n_ops}"
        if allow_ids is not None and operon_id not in allow_ids:
            continue
        arrays = []
        for arr in (rec.get("crispr") or []):
            rep = norm_nt(arr.get("crispr_repeat"))
            spacers = [norm_nt(s) for s in (arr.get("crispr_spacers") or [])]
            spacers = [s for s in spacers if s and len(s) >= 15]
            if spacers:
                arrays.append((rep, spacers))
        if not arrays:
            continue
        cas12a_shas = set()
        for cas in (rec.get("cas") or []):
            g = (cas.get("gene_name") or "")
            h = (cas.get("hmm_name") or "")
            if not is_cas12a(g, h):
                continue
            aa = (cas.get("protein") or "").strip()
            if aa:
                cas12a_shas.add(sha1(aa))
        if not cas12a_shas:
            continue

        # We have a matching operon
        n_done += 1
        if args.max_operons and n_done > args.max_operons:
            break

        # Export and run for each array separately (keeps spacers co-oriented)
        for idx, (_rep, spacers) in enumerate(arrays):
            if len(spacers) < args.min_spacers_per_array:
                continue
            op_safe = safe_name(operon_id)
            fa = spdir / f"{op_safe}__arr{idx}.fna"
            write_fasta(spacers, fa)
            run_out = rundir / f"{op_safe}__arr{idx}"
            safe_mkdir(run_out)

            # Call PAMpredict (Type V => UPSTREAM PAM; 10-nt window)
            pampredict_py = ppdir / "PAMpredict.py"
            if not pampredict_py.exists():
                alt = ppdir / "PAMpredict" / "PAMpredict.py"
                pampredict_py = alt if alt.exists() else pampredict_py

            cmd = [
                sys.executable,
                str(pampredict_py),
                str(fa),
                str(db),
                str(run_out),
                "-p", "UPSTREAM",
                "-l", "10",
                "-t", str(args.threads),
                "--keep_tmp",
                "--force",
            ]
            try:
                run(cmd)
            except subprocess.CalledProcessError:
                # record a failed/no-call row
                per_array_rows.append({
                    "operon_id": operon_id,
                    "array_idx": idx,
                    "pam_side": "",
                    "pam_iupac": "",
                    "unique_hits": 0,
                    "peak_bits_up": 0.0,
                    "peak_bits_down": 0.0,
                    "snr": 0.0,
                    "label": "no_call",
                    "reason": "pampredict_failed",
                    "n_spacers": len(spacers),
                    "protein_sha1s": ",".join(sorted(cas12a_shas)),
                })
                continue

            # Parse outputs
            pred = parse_pam_prediction_txt(run_out / "PAM_prediction.txt")
            peak_up = parse_info_table(run_out / "upstream_flanking_sequence_info.tsv")
            peak_dn = parse_info_table(run_out / "downstream_flanking_sequence_info.tsv")
            filtered_dir = run_out / "blastn"
            unique = count_unique_protospacers(filtered_dir)
            snr = (peak_up / max(peak_dn, 1e-9)) if (peak_up and peak_dn is not None) else 0.0

            # Confidence (paper): >=10 unique + SNR>2 for Type V
            if pred.get("pam_iupac"):
                if unique >= args.min_unique and snr > args.min_snr:
                    label, reason = "high_confidence", "unique>=min && snr>min"
                else:
                    label, reason = "low_confidence", "insufficient_evidence"
            else:
                label, reason = "no_call", "no_pam_prediction_txt"

            row = {
                "operon_id": operon_id,
                "array_idx": idx,
                "pam_side": pred.get("pam_side") or "UPSTREAM",
                "pam_iupac": pred.get("pam_iupac") or "",
                "unique_hits": unique,
                "peak_bits_up": f"{peak_up:.3f}",
                "peak_bits_down": f"{peak_dn:.3f}",
                "snr": f"{snr:.3f}",
                "label": label,
                "reason": reason,
                "n_spacers": len(spacers),
                "protein_sha1s": ",".join(sorted(cas12a_shas)),
            }
            per_array_rows.append(row)

            # Assign a "best" call per protein (pick highest unique, then SNR)
            for sha in cas12a_shas:
                # score: (label_is_high, unique, snr)
                score = (1 if label == "high_confidence" else 0, unique, snr)
                prev = protein_to_best.get(sha)
                if (prev is None) or (score > prev[0]):
                    protein_to_best[sha] = (score, {
                        "protein_sha1": sha,
                        "operon_id": operon_id,
                        "array_idx": idx,
                        "pam_iupac": row["pam_iupac"],
                        "pam_side": row["pam_side"],
                        "unique_hits": unique,
                        "peak_bits_up": row["peak_bits_up"],
                        "peak_bits_down": row["peak_bits_down"],
                        "snr": row["snr"],
                        "label": label,
                        "reason": reason,
                    })

    # Write per-array TSV
    pa = out / "pam_crispr.per_array.tsv"
    with open(pa, "w") as fh:
        hdr = [
            "operon_id",
            "array_idx",
            "n_spacers",
            "pam_side",
            "pam_iupac",
            "unique_hits",
            "peak_bits_up",
            "peak_bits_down",
            "snr",
            "label",
            "reason",
            "protein_sha1s",
        ]
        fh.write("\t".join(hdr) + "\n")
        for r in per_array_rows:
            fh.write("\t".join(str(r[k]) for k in hdr) + "\n")

    # Write per-protein TSV
    pp = out / "pam_crispr.per_protein.tsv"
    with open(pp, "w") as fh:
        hdr = [
            "protein_sha1",
            "operon_id",
            "array_idx",
            "pam_side",
            "pam_iupac",
            "unique_hits",
            "peak_bits_up",
            "peak_bits_down",
            "snr",
            "label",
            "reason",
        ]
        fh.write("\t".join(hdr) + "\n")
        for _sha, (_score, r) in protein_to_best.items():
            fh.write("\t".join(str(r[k]) for k in hdr) + "\n")

    # Optional: operon_id -> PAM map (best high-conf array in operon)
    best_by_operon = {}
    for r in per_array_rows:
        if r["label"] != "high_confidence":
            continue
        key = r["operon_id"]
        score = (r["label"] == "high_confidence", r["unique_hits"], float(r["snr"]))
        if (key not in best_by_operon) or (score > best_by_operon[key][0]):
            best_by_operon[key] = (score, r)

    pm = out / "pam_map.tsv"
    with open(pm, "w") as fh:
        fh.write("operon_id\tpam_iupac\tpam_side\tsource\n")
        for _op, (_s, r) in best_by_operon.items():
            fh.write(f"{r['operon_id']}\t{r['pam_iupac']}\t{r['pam_side']}\tPAMpredict\n")

    sys.stderr.write(
        f"[atlas_make_pam_crispr] Wrote:\n  {pa}\n  {pp}\n  {pm}\n"
    )


if __name__ == "__main__":
    main()
