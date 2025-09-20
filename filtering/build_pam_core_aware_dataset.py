#!/usr/bin/env python3
import argparse, csv, gzip, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

IUPAC_ENCODE = {
    frozenset({'A'}): 'A', frozenset({'C'}): 'C', frozenset({'G'}): 'G', frozenset({'T'}): 'T',
    frozenset({'A','G'}): 'R', frozenset({'C','T'}): 'Y', frozenset({'C','G'}): 'S', frozenset({'A','T'}): 'W',
    frozenset({'G','T'}): 'K', frozenset({'A','C'}): 'M',
    frozenset({'C','G','T'}): 'B', frozenset({'A','G','T'}): 'D', frozenset({'A','C','T'}): 'H',
    frozenset({'A','C','G'}): 'V', frozenset({'A','C','G','T'}): 'N'
}

BASES = ('A','C','G','T')

def open_text(path: str):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'rt')

def read_info_matrix(path: str) -> List[Dict[str, float]]:
    """Read upstream_flanking_sequence_info.tsv as a list of dicts per position with keys A,C,G,T.
    Tolerates headers/format variants; missing values become 0.0.
    """
    rows: List[Dict[str, float]] = []
    with open(path) as fh:
        rdr = csv.DictReader(fh, delimiter='\t')
        for row in rdr:
            try:
                vals = {b: float(row.get(b, '0') or '0') for b in BASES}
            except Exception:
                continue
            rows.append(vals)
    # Expect length 10, but be defensive
    if len(rows) < 10:
        # pad with zeros if short
        for _ in range(10 - len(rows)):
            rows.append({b: 0.0 for b in BASES})
    return rows[:10]

def pos_bits_sum(pos: Dict[str, float]) -> float:
    return sum(pos.get(b, 0.0) for b in BASES)

def pick_core_window(info: List[Dict[str, float]], k: int = 4) -> Tuple[int, float]:
    """Slide k window across 10 positions; pick start with max total bits.
    Tie-breaker: pick the rightmost (nearest protospacer) window.
    Return (start_index, bits_sum).
    """
    best_start, best_sum = 0, -1.0
    for s in range(0, max(0, len(info) - k + 1)):
        v = sum(pos_bits_sum(info[i]) for i in range(s, s + k))
        if v > best_sum or (abs(v - best_sum) < 1e-9 and s > best_start):
            best_start, best_sum = s, v
    return best_start, best_sum

def normalize_pos(pos: Dict[str, float]) -> Dict[str, float]:
    s = pos_bits_sum(pos)
    if s <= 0:
        return {b: 0.0 for b in BASES}
    return {b: max(0.0, pos.get(b, 0.0)) / s for b in BASES}

def iupac_from_weights(pos: Dict[str, float], thresh: float = 0.2) -> str:
    """Convert normalized weights into an IUPAC letter using a threshold per base.
    """
    allowed = {b for b, w in pos.items() if w >= thresh}
    if not allowed:
        # if nothing passes threshold, pick the argmax
        m = max(pos.items(), key=lambda t: t[1])[0]
        allowed = {m}
    return IUPAC_ENCODE.get(frozenset(allowed), 'N')

def core_iupac(info: List[Dict[str, float]], start: int, k: int = 4, thresh: float = 0.2) -> str:
    letters = []
    for i in range(start, start + k):
        w = normalize_pos(info[i])
        letters.append(iupac_from_weights(w, thresh))
    return ''.join(letters)

def tttv_labels_from_core(info: List[Dict[str, float]], start: int, eps_only: float = 0.05) -> Tuple[int, int]:
    """Return (strict_label, permissive_label) for TTTV using normalized weights in core window.
    strict: positions 1..3 T-only (others < eps_only), pos4 excludes T (w_T < eps_only) and has any of A/C/G.
    permissive: sum(A,C,G) >= w_T at pos4.
    """
    strict = 1
    for i in range(start, start + 3):
        w = normalize_pos(info[i])
        if not (w['T'] > 0.5 and all(w[b] < eps_only for b in ('A','C','G'))):
            strict = 0; break
    w4 = normalize_pos(info[start + 3])
    cond4_strict = (w4['T'] < eps_only) and ((w4['A'] + w4['C'] + w4['G']) > eps_only)
    if strict == 1 and not cond4_strict:
        strict = 0
    permissive = 1 if ((w4['A'] + w4['C'] + w4['G']) >= w4['T']) else 0
    return strict, permissive

def augmented_cosine_vec(info: List[Dict[str, float]], max_bits: float = 2.0) -> List[float]:
    """Build a 10×5 vector [A,C,G,T,N] per position; N captures low information as (max_bits - sum_4).
    """
    vec: List[float] = []
    for i in range(10):
        pos = info[i] if i < len(info) else {b: 0.0 for b in BASES}
        a = [pos.get('A',0.0), pos.get('C',0.0), pos.get('G',0.0), pos.get('T',0.0)]
        s = sum(a)
        n = max(0.0, max_bits - s)
        vec.extend(a + [n])
    return vec

def cosine(a: List[float], b: List[float]) -> float:
    import math
    da = sum(x*x for x in a)
    db = sum(y*y for y in b)
    if da <= 0 or db <= 0:
        return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    return dot / math.sqrt(da*db)

def iter_per_protein(path: str):
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            yield r

def load_fasta_sha_map(fasta_path: Optional[str]) -> Dict[str, str]:
    """Optional: read FASTA and index by sha1(sequence)."""
    out: Dict[str, str] = {}
    if not fasta_path:
        return out
    import hashlib
    def sha1(s: str) -> str:
        return hashlib.sha1(s.encode('utf-8')).hexdigest()
    seq = []
    try:
        opener = open
        if fasta_path.endswith('.gz'):
            import gzip as _gz
            opener = _gz.open
        with opener(fasta_path, 'rt') as fh:
            for line in fh:
                if line.startswith('>'):
                    if seq:
                        s = ''.join(seq).strip(); seq=[]
                        if s:
                            out.setdefault(sha1(s), s)
                else:
                    seq.append(line.strip())
            if seq:
                s=''.join(seq).strip()
                if s:
                    out.setdefault(sha1(s), s)
    except Exception:
        pass
    return out

def build_sha_to_seq(atlas_json: str, wanted_shas: set[str], extra_fasta: Optional[str] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not wanted_shas:
        return out
    # Optionally seed with extra FASTA
    extra = load_fasta_sha_map(extra_fasta)
    out.update({k:v for k,v in extra.items() if k in wanted_shas})
    def iter_atlas_records(p: str):
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
                        if c.isspace(): continue
                        if c=='[': saw_arr=True; continue
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
        with open_text(p) as fh:
            first = fh.read(1); fh.seek(0)
            if first == '[':
                yield from _yield_top_array(fh)
            else:
                yield from _yield_ndjson(fh)
    def enough():
        return len(set(out.keys()) & wanted_shas) == len(wanted_shas)
    for rec in iter_atlas_records(atlas_json):
        for cas in (rec.get('cas') or []):
            sha = (cas.get('sha1') or '').strip()
            if sha and sha in wanted_shas and sha not in out:
                seq = cas.get('sequence') or cas.get('protein') or ''
                if seq:
                    out[sha] = seq
        if enough():
            break
    return out

def main():
    ap = argparse.ArgumentParser(description='Core-aware PAM dataset builder (upstream 10×4, max-info core, TTTV labels).')
    ap.add_argument('--atlas-json', required=True)
    ap.add_argument('--pam-per-protein', required=True, help='per_protein TSV with run_dir column')
    ap.add_argument('--out-strict', required=True, help='CSV: protein_seq,pam_tttv (strict) + metadata')
    ap.add_argument('--out-permissive', required=True, help='CSV: protein_seq,pam_tttv (permissive) + metadata')
    ap.add_argument('--out-matrices', required=True, help='CSV: includes 10×4/10×5 vectors JSON + core info')
    ap.add_argument('--confidence', choices=['high_only','any'], default='high_only')
    ap.add_argument('--min-unique', type=int, default=10)
    ap.add_argument('--min-snr', type=float, default=2.0)
    ap.add_argument('--split', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--extra-fasta', default=None, help='Optional FASTA of Cas12a proteins to augment sha1->seq mapping')
    ap.add_argument('--seq-csv', default=None, help='Optional CSV with protein_sha1 and protein_seq columns to seed mapping')
    args = ap.parse_args()

    high_only = (args.confidence == 'high_only')

    # Gather candidates and locate upstream info tables
    rows = []
    for r in iter_per_protein(args.pam_per_protein):
        if (r.get('pam_side','').upper() != 'UPSTREAM'):
            continue
        if high_only:
            try:
                uniq = int(r.get('unique_hits','0'))
                snr = float(r.get('snr','0'))
            except Exception:
                uniq, snr = 0, 0.0
            if not (r.get('label') == 'high_confidence' and uniq >= args.min_unique and snr > args.min_snr):
                continue
        run_dir = r.get('run_dir','').strip()
        if not run_dir:
            # Skip if we cannot locate the info matrix
            continue
        up_tsv = os.path.join(run_dir, 'upstream_flanking_sequence_info.tsv')
        if not os.path.isfile(up_tsv):
            continue
        rows.append({
            'protein_sha1': r.get('protein_sha1',''),
            'operon_id': r.get('operon_id',''),
            'array_idx': r.get('array_idx',''),
            'run_dir': run_dir,
            'snr': r.get('snr',''),
            'unique_hits': r.get('unique_hits',''),
            'up_info': up_tsv,
        })

    # Map sha -> seq
    wanted = {r['protein_sha1'] for r in rows if r['protein_sha1']}
    seed: Dict[str,str] = {}
    if args.seq_csv and os.path.isfile(args.seq_csv):
        try:
            with open(args.seq_csv) as f:
                rdr = csv.DictReader(f)
                for rr in rdr:
                    sha = (rr.get('protein_sha1') or '').strip()
                    seq = (rr.get('protein_seq') or '').strip()
                    if sha and seq:
                        seed.setdefault(sha, seq)
        except Exception:
            pass
    # merge with atlas + optional FASTA
    sha_to_seq = {k:v for k,v in seed.items() if k in wanted}
    if len(set(sha_to_seq.keys()) & wanted) < len(wanted):
        rem = wanted - set(sha_to_seq.keys())
        atlas_map = build_sha_to_seq(args.atlas_json, rem, extra_fasta=args.extra_fasta)
        sha_to_seq.update(atlas_map)

    # Prepare outputs
    Path(os.path.dirname(args.out_strict) or '.').mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out_permissive) or '.').mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out_matrices) or '.').mkdir(parents=True, exist_ok=True)

    strict_items = []
    perm_items = []
    matrix_rows = []

    for r in rows:
        seq = sha_to_seq.get(r['protein_sha1'])
        if not seq:
            continue
        info = read_info_matrix(r['up_info'])
        start, bits_sum = pick_core_window(info, k=4)
        strict, permissive = tttv_labels_from_core(info, start)
        core_str = core_iupac(info, start, 4, thresh=0.2)
        vec10x4 = [[info[i].get(b,0.0) for b in BASES] for i in range(10)]
        vec10x5 = augmented_cosine_vec(info, max_bits=2.0)

        strict_items.append({
            'protein_seq': seq,
            'pam_tttv': strict,
            'protein_sha1': r['protein_sha1'],
            'operon_id': r['operon_id'],
            'array_idx': r['array_idx'],
            'core_start': start,
            'core_iupac': core_str,
            'snr': r['snr'],
            'unique_hits': r['unique_hits'],
        })
        perm_items.append({**strict_items[-1], 'pam_tttv': permissive})

        matrix_rows.append({
            'protein_sha1': r['protein_sha1'],
            'operon_id': r['operon_id'],
            'array_idx': r['array_idx'],
            'core_start': start,
            'core_iupac': core_str,
            'snr': r['snr'],
            'unique_hits': r['unique_hits'],
            'run_dir': r['run_dir'],
            'pam_up_10x4_json': json.dumps(vec10x4),
            'pam_up_10x5_json': json.dumps(vec10x5),
        })

    # Write CSVs
    with open(args.out_strict, 'w', newline='') as f:
        cols = ['protein_seq','pam_tttv','protein_sha1','operon_id','array_idx','core_start','core_iupac','snr','unique_hits']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(strict_items)
    with open(args.out_permissive, 'w', newline='') as f:
        cols = ['protein_seq','pam_tttv','protein_sha1','operon_id','array_idx','core_start','core_iupac','snr','unique_hits']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(perm_items)
    with open(args.out_matrices, 'w', newline='') as f:
        cols = ['protein_sha1','operon_id','array_idx','core_start','core_iupac','snr','unique_hits','run_dir','pam_up_10x4_json','pam_up_10x5_json']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(matrix_rows)

    # Stats
    def _cnt(items):
        p = sum(1 for r in items if int(r['pam_tttv']) == 1)
        return p, len(items)-p, len(items)
    sp, sn, st = _cnt(strict_items)
    pp, pn, pt = _cnt(perm_items)
    print(f"[core-aware] strict: n={st} pos={sp} neg={sn}")
    print(f"[core-aware] permissive: n={pt} pos={pp} neg={pn}")

if __name__ == '__main__':
    main()
