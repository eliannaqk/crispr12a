#!/usr/bin/env python3
import argparse, csv, gzip, json, os, re, sys, hashlib
from pathlib import Path

# ----- small utils -----
IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
    "R": set("AG"), "Y": set("CT"), "S": set("CG"), "W": set("AT"),
    "K": set("GT"), "M": set("AC"),
    "B": set("CGT"), "D": set("AGT"), "H": set("ACT"), "V": set("ACG"),
    "N": set("ACGT")
}
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def iter_atlas_records(atlas_json: str):
    def _yield_ndjson(fh):
        for line in fh:
            line = line.strip()
            if not line: continue
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
    with open_text(atlas_json) as fh:
        first = fh.read(1); fh.seek(0)
        if first == "[":
            yield from _yield_top_array(fh)
        else:
            yield from _yield_ndjson(fh)

def is_cas12a(g, h) -> bool:
    s = f"{g} {h}".lower()
    return ("cas12a" in s) or ("cpf1" in s)

def set_from_iupac(ch: str) -> set[str]:
    return IUPAC.get(ch.upper(), set())

def iupac_allows(ch: str, base: str) -> bool:
    return base in set_from_iupac(ch)

def consensus_includes_TTTV(pam_iupac: str,
                            mode: str = "inclusive") -> bool:
    """
    Return True if the consensus allows a 4-mer that is T,T,T,(A/C/G)
    somewhere in the motif.

    modes:
      - 'inclusive' (default): position 1..3 only need to ALLOW T (may allow others);
                              position 4 must ALLOW at least one of A/C/G.
        -> counts 'TTTN' as positive (it includes TTTV PAMs).
      - 'strict':       position 1..3 must REQUIRE T (i.e., letter's allowed set == {'T'});
                        position 4 must EXCLUDE T and ALLOW at least one of A/C/G.
        -> requires a true '...TTTV...' with V∈{A,C,G}, not N/W/etc.
    """
    s = pam_iupac.strip().upper()
    L = len(s)
    if L < 4:
        return False

    for i in range(0, L - 4 + 1):
        a,b,c,d = s[i], s[i+1], s[i+2], s[i+3]
        A = set_from_iupac(a); B = set_from_iupac(b)
        C = set_from_iupac(c); D = set_from_iupac(d)

        if mode == "inclusive":
            if "T" in A and "T" in B and "T" in C and len(D & set("ACG")) > 0:
                return True
        else:  # strict
            if A == {"T"} and B == {"T"} and C == {"T"} and ("T" not in D) and len(D & set("ACG")) > 0:
                return True
    return False

# ----- main -----
def main():
    ap = argparse.ArgumentParser(description="Build TTTV labels from pam_crispr + Atlas (Cas12a only).")
    ap.add_argument("--atlas-json", required=True, help="crispr-cas-atlas-v1.0.json (JSON or NDJSON)")
    ap.add_argument("--pam-per-protein", required=True, help="pam_crispr.per_protein.tsv from atlas_make_pam_crispr.py")
    ap.add_argument("--out-csv", required=True, help="Output CSV with columns: protein_seq,pam_tttv,...")
    ap.add_argument("--confidence", choices=["high_only","any"], default="high_only",
                   help="Use only high-confidence PAMpredict calls (>=10 unique hits & SNR>2)?")
    ap.add_argument("--tttv-mode", choices=["inclusive","strict"], default="inclusive",
                   help="Labeling regime for TTTV (see function docstring).")
    ap.add_argument("--min-unique", type=int, default=10, help="Unique protospacers required (for high_only).")
    ap.add_argument("--min-snr", type=float, default=2.0, help="SNR threshold required (for high_only).")
    args = ap.parse_args()

    # 1) Map protein_sha1 -> seq (Cas12a only)
    sha_to_seq = {}
    n_ops = 0
    for rec in iter_atlas_records(args.atlas_json):
        n_ops += 1
        for cas in (rec.get("cas") or []):
            g = cas.get("gene_name",""); h = cas.get("hmm_name","")
            if not is_cas12a(g,h): continue
            aa = (cas.get("protein") or "").strip()
            if not aa: continue
            sha = sha1(aa)
            sha_to_seq.setdefault(sha, aa)  # first wins (identical by sha)

    # 2) Read per_protein PAM calls
    keep_rows = []
    with open(args.pam_per_protein) as f:
        reader = csv.DictReader(f, delimiter="\t")
        need = {"protein_sha1","pam_iupac","pam_side","unique_hits","snr","label"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f"[error] {args.pam_per_protein} missing columns {need - set(reader.fieldnames or [])}")
        for r in reader:
            # Only Cas12a were written by atlas_make_pam_crispr.py, but keep side=UPSTREAM for safety
            if (r["pam_side"] or "").upper() != "UPSTREAM": 
                continue
            if args.confidence == "high_only":
                try:
                    uniq = int(r.get("unique_hits", "0"))
                    snr  = float(r.get("snr","0"))
                except ValueError:
                    uniq, snr = 0, 0.0
                if not (r.get("label") == "high_confidence" and uniq >= args.min_unique and snr > args.min_snr):
                    continue
            keep_rows.append(r)

    # 3) Build labeled examples
    out_rows = []
    missing_seq = 0
    for r in keep_rows:
        sha = r["protein_sha1"]
        seq = sha_to_seq.get(sha)
        if not seq:
            missing_seq += 1
            continue
        pam = (r.get("pam_iupac") or "").upper()
        label = 1 if consensus_includes_TTTV(pam, args.tttv_mode) else 0
        out_rows.append({
            "protein_sha1": sha,
            "protein_seq": seq,
            "pam_tttv": label,
            "pam_iupac": pam,
            "pam_side": r.get("pam_side",""),
            "unique_hits": r.get("unique_hits",""),
            "snr": r.get("snr",""),
            "pampredict_label": r.get("label",""),
            "operon_id": r.get("operon_id",""),
            "array_idx": r.get("array_idx",""),
        })
    if missing_seq:
        print(f"[warn] Skipped {missing_seq} proteins with missing seq (sha not found in Atlas).", file=sys.stderr)

    # 4) Write CSV
    Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    cols = ["protein_seq","pam_tttv","protein_sha1","pam_iupac","pam_side","unique_hits","snr","pampredict_label","operon_id","array_idx"]
    with open(args.out_csv, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[build_pam_tttv_dataset] wrote {len(out_rows)} rows to {args.out_csv}")
    print(f"[build_pam_tttv_dataset] positives={sum(r['pam_tttv'] for r in out_rows)} negatives={len(out_rows)-sum(r['pam_tttv'] for r in out_rows)}")

if __name__ == "__main__":
    main()
