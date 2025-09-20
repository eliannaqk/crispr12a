#!/usr/bin/env python3
import argparse, json, gzip, sys

def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

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
        if first == '[':
            yield from _yield_top_array(fh)
        else:
            yield from _yield_ndjson(fh)

def is_cas12a(gene_name: str, hmm_name: str) -> bool:
    s = f"{gene_name} {hmm_name}".lower()
    return ("cas12a" in s) or ("cpf1" in s)

def main():
    ap = argparse.ArgumentParser(description="List operon_ids that contain Cas12a proteins (optionally with >=K spacers).")
    ap.add_argument('--atlas-json', required=True, help='crispr-cas-atlas-v1.0.json (.gz OK)')
    ap.add_argument('--min-spacers-per-array', type=int, default=1, help='Require at least K spacers in any array')
    ap.add_argument('--out', default='-', help='Output file (default stdout)')
    args = ap.parse_args()

    out = sys.stdout if args.out == '-' else open(args.out, 'w')
    seen = set()
    n = 0
    for rec in iter_atlas_records(args.atlas_json):
        operon_id = rec.get('operon_id') or f'op{n+1}'
        has_cas12a = False
        for cas in (rec.get('cas') or []):
            g = cas.get('gene_name',''); h = cas.get('hmm_name','')
            if is_cas12a(g,h):
                has_cas12a = True; break
        if not has_cas12a:
            continue
        if args.min_spacers_per_array > 1:
            ok = False
            for arr in (rec.get('crispr') or []):
                sp = arr.get('crispr_spacers') or []
                if len([s for s in sp if s and len(s) >= 15]) >= args.min_spacers_per_array:
                    ok = True; break
            if not ok:
                continue
        if operon_id not in seen:
            seen.add(operon_id)
            out.write(operon_id + '\n')
        n += 1
    if out is not sys.stdout:
        out.close()

if __name__ == '__main__':
    main()

