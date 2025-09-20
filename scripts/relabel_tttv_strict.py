#!/usr/bin/env python3
import argparse, csv

IUPAC = {
    'A': set('A'), 'C': set('C'), 'G': set('G'), 'T': set('T'),
    'R': set('AG'), 'Y': set('CT'), 'S': set('CG'), 'W': set('AT'),
    'K': set('GT'), 'M': set('AC'),
    'B': set('CGT'), 'D': set('AGT'), 'H': set('ACT'), 'V': set('ACG'),
    'N': set('ACGT')
}

def set_from_iupac(ch: str):
    return IUPAC.get(ch.upper(), set())

def strict_tttv(pam: str) -> bool:
    if not pam:
        return False
    s = pam.strip().upper()
    if len(s) < 4:
        return False
    for i in range(len(s)-3):
        a,b,c,d = s[i], s[i+1], s[i+2], s[i+3]
        if set_from_iupac(a)=={'T'} and set_from_iupac(b)=={'T'} and set_from_iupac(c)=={'T'}:
            D=set_from_iupac(d)
            if 'T' not in D and len(D & set('ACG'))>0:
                return True
    return False

def relabel(in_csv: str, out_csv: str) -> tuple[int,int,int]:
    pos=neg=total=0
    with open(in_csv, newline='') as f, open(out_csv, 'w', newline='') as g:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        if 'pam_tttv' not in cols or 'pam_iupac' not in cols or 'protein_seq' not in cols:
            raise SystemExit(f"{in_csv} missing required columns; saw: {cols}")
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for r in rdr:
            y = 1 if strict_tttv(r.get('pam_iupac','')) else 0
            r['pam_tttv'] = str(y)
            w.writerow(r)
            total += 1
            if y==1: pos += 1
            else: neg += 1
    return pos, neg, total

def main():
    ap = argparse.ArgumentParser(description='Relabel existing high-only TTTV CSVs to strict TTTV, preserving split')
    ap.add_argument('--train-in', required=True)
    ap.add_argument('--valid-in', required=True)
    ap.add_argument('--train-out', required=True)
    ap.add_argument('--valid-out', required=True)
    args = ap.parse_args()

    tr_p, tr_n, tr_tot = relabel(args.train_in, args.train_out)
    va_p, va_n, va_tot = relabel(args.valid_in, args.valid_out)
    print(f"[strict] train: n={tr_tot} pos={tr_p} neg={tr_n}")
    print(f"[strict] valid: n={va_tot} pos={va_p} neg={va_n}")
    print(f"[strict] total: n={tr_tot+va_tot} pos={tr_p+va_p} neg={tr_n+va_n}")

if __name__ == '__main__':
    main()

