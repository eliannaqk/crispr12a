#!/usr/bin/env python3
import argparse, csv, json, gzip, random, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

IUPAC = {
    'A': set('A'), 'C': set('C'), 'G': set('G'), 'T': set('T'),
    'R': set('AG'), 'Y': set('CT'), 'S': set('CG'), 'W': set('AT'),
    'K': set('GT'), 'M': set('AC'),
    'B': set('CGT'), 'D': set('AGT'), 'H': set('ACT'), 'V': set('ACG'),
    'N': set('ACGT')
}

def set_from_iupac(ch: str):
    return IUPAC.get(ch.upper(), set())

def includes_tttv(pam: str, strict: bool) -> bool:
    if not pam:
        return False
    s = pam.strip().upper()
    L = len(s)
    if L < 4:
        return False
    for i in range(L - 3):
        a, b, c, d = s[i], s[i+1], s[i+2], s[i+3]
        A, B, C, D = set_from_iupac(a), set_from_iupac(b), set_from_iupac(c), set_from_iupac(d)
        if strict:
            if A == {'T'} and B == {'T'} and C == {'T'} and ('T' not in D) and len(D & set('ACG')) > 0:
                return True
        else:
            if 'T' in A and 'T' in B and 'T' in C and len(D & set('ACG')) > 0:
                return True
    return False

def open_text(path: str):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'rt')

def iter_atlas_records(path: str):
    def _yield_ndjson(fh):
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    def _yield_top_array(fh):
        buf = []
        in_str = False
        esc = False
        depth = 0
        saw_arr = False
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
                    s = ''.join(buf).strip()
                    if s:
                        yield json.loads(s)
                    buf = []
                    continue
                buf.append(c)
                if in_str:
                    if esc:
                        esc = False
                    elif c == '\\':
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
        s = ''.join(buf).strip()
        if s:
            yield json.loads(s)
    with open_text(path) as fh:
        first = fh.read(1); fh.seek(0)
        if first == '[':
            yield from _yield_top_array(fh)
        else:
            yield from _yield_ndjson(fh)

def build_sha_to_seq(atlas_path: str, wanted_shas: set) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not wanted_shas:
        return out
    for rec in iter_atlas_records(atlas_path):
        for cas in (rec.get('cas') or []):
            sha = (cas.get('sha1') or '').strip()
            if sha and sha in wanted_shas and sha not in out:
                seq = cas.get('sequence') or cas.get('protein') or ''
                if seq:
                    out[sha] = seq
        if len(out) == len(wanted_shas):
            break
    return out

def read_candidates(per_protein_tsv: str, high_only: bool, upstream_only: bool,
                    strict_tttv: bool) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with open(per_protein_tsv) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            if upstream_only and (r.get('pam_side','').upper() != 'UPSTREAM'):
                continue
            if high_only and (r.get('label') != 'high_confidence'):
                continue
            sha = r.get('protein_sha1','')
            pam = r.get('pam_iupac','')
            y = 1 if includes_tttv(pam, strict=strict_tttv) else 0
            rows.append((sha, y))
    return rows

def write_csv(path: str, items: List[Tuple[str, int]]):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protein_seq','pam_tttv'])
        for seq, y in items:
            w.writerow([seq, y])

def main():
    ap = argparse.ArgumentParser(description='Build TTTV dataset CSVs from per-protein aggregation + Atlas sequences')
    ap.add_argument('--atlas-json', required=True)
    ap.add_argument('--pam-per-protein', required=True)
    ap.add_argument('--out-train', required=True)
    ap.add_argument('--out-valid', required=True)
    ap.add_argument('--split', type=float, default=0.9, help='Train fraction')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--confidence', choices=['high_only','any'], default='high_only')
    ap.add_argument('--tttv', choices=['strict','inclusive'], default='strict')
    ap.add_argument('--upstream-only', action='store_true', default=True)
    args = ap.parse_args()

    high_only = (args.confidence == 'high_only')
    strict_tttv = (args.tttv == 'strict')

    pairs = read_candidates(args.pam_per_protein, high_only=high_only, upstream_only=args.upstream_only,
                            strict_tttv=strict_tttv)
    # Collect shas and labels
    shas = [sha for sha,_ in pairs]
    labels = [y for _,y in pairs]
    uniq = list(dict.fromkeys(shas))

    sha_to_seq = build_sha_to_seq(args.atlas_json, set(uniq))

    # Build dataset with available sequences
    items: List[Tuple[str,int]] = []
    dropped_missing = 0
    for sha, y in pairs:
        seq = sha_to_seq.get(sha)
        if not seq:
            dropped_missing += 1
            continue
        items.append((seq, y))

    random.Random(args.seed).shuffle(items)
    n = len(items)
    k = int(round(args.split * n))
    train = items[:k]
    valid = items[k:]

    # Stats
    def counts(rows):
        p = sum(1 for _,y in rows if y==1)
        n = sum(1 for _,y in rows if y==0)
        return p, n, len(rows)
    tr_p, tr_n, tr_tot = counts(train)
    va_p, va_n, va_tot = counts(valid)
    all_p, all_n, all_tot = counts(items)

    Path(Path(args.out_train).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(args.out_valid).parent).mkdir(parents=True, exist_ok=True)
    write_csv(args.out_train, train)
    write_csv(args.out_valid, valid)

    print(f"[dataset] built: confidence={args.confidence} tttv={args.tttv} upstream_only={args.upstream_only}")
    print(f"[dataset] total: n={all_tot} pos={all_p} neg={all_n} dropped_missing_seq={dropped_missing}")
    print(f"[dataset] train: n={tr_tot} pos={tr_p} neg={tr_n}")
    print(f"[dataset] valid: n={va_tot} pos={va_p} neg={va_n}")

if __name__ == '__main__':
    main()

