#!/usr/bin/env python3
import argparse, csv, json, gzip, re, hashlib, sys
from pathlib import Path

IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
    "R": set("AG"), "Y": set("CT"), "S": set("CG"), "W": set("AT"),
    "K": set("GT"), "M": set("AC"),
    "B": set("CGT"), "D": set("AGT"), "H": set("ACT"), "V": set("ACG"),
    "N": set("ACGT")
}

def open_text(path: str):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'rt')

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def iter_atlas_records(path: str):
    def _yield_ndjson(fh):
        for line in fh:
            line=line.strip()
            if not line: continue
            yield json.loads(line)
    def _yield_top_array(fh):
        buf=[]; in_str=False; esc=False; depth=0; saw_arr=False
        while True:
            ch = fh.read(8192)
            if not ch: break
            for c in ch:
                if not saw_arr:
                    if c.isspace():
                        continue
                    if c=='[':
                        saw_arr=True; continue
                if not in_str and depth==0 and c==']':
                    s=''.join(buf).strip()
                    if s: yield json.loads(s)
                    return
                if not in_str and depth==0 and c==',':
                    s=''.join(buf).strip()
                    if s: yield json.loads(s)
                    buf=[]; continue
                buf.append(c)
                if in_str:
                    if esc: esc=False
                    elif c=='\\': esc=True
                    elif c=='"': in_str=False
                else:
                    if c=='"': in_str=True
                    elif c=='{': depth+=1
                    elif c=='}': depth-=1
        s=''.join(buf).strip()
        if s: yield json.loads(s)
    with open_text(path) as fh:
        first = fh.read(1); fh.seek(0)
        if first=='[':
            yield from _yield_top_array(fh)
        else:
            yield from _yield_ndjson(fh)

def is_cas12a(g, h) -> bool:
    s = f"{g} {h}".lower()
    return ('cas12a' in s) or ('cpf1' in s)

def main():
    ap = argparse.ArgumentParser(description='Export full Cas12a → PAM map (no TTTV filter).')
    ap.add_argument('--atlas-json', required=True)
    ap.add_argument('--pam-per-protein', required=True, help='pam_crispr.per_protein.tsv')
    ap.add_argument('--out-tsv', required=True)
    ap.add_argument('--include-seq', action='store_true', help='Join protein sequences (slower, larger output)')
    args = ap.parse_args()

    # Optional sha -> seq map
    sha_to_seq = {}
    if args.include_seq:
        for rec in iter_atlas_records(args.atlas_json):
            for cas in (rec.get('cas') or []):
                g=cas.get('gene_name',''); h=cas.get('hmm_name','')
                if not is_cas12a(g,h): continue
                aa=(cas.get('protein') or '').strip()
                if not aa: continue
                sha = sha1(aa)
                sha_to_seq.setdefault(sha, aa)

    need_cols = [
        'protein_sha1','operon_id','array_idx','pam_side','pam_iupac','label',
        'unique_hits','snr'
    ]
    out_cols = need_cols + (['protein_seq'] if args.include_seq else [])

    with open(args.pam_per_protein) as f, open(args.out_tsv, 'w', newline='') as out:
        rdr = csv.DictReader(f, delimiter='\t')
        missing = [c for c in need_cols if c not in (rdr.fieldnames or [])]
        if missing:
            print(f"[error] pam_per_protein missing columns: {missing}", file=sys.stderr)
            sys.exit(2)
        w = csv.DictWriter(out, fieldnames=out_cols, delimiter='\t')
        w.writeheader()
        for r in rdr:
            row = {k: r.get(k, '') for k in need_cols}
            if args.include_seq:
                row['protein_seq'] = sha_to_seq.get(r.get('protein_sha1',''), '')
            w.writerow(row)

    print(f"[export_cas12a_pam_map] wrote {args.out_tsv}")

if __name__ == '__main__':
    main()

