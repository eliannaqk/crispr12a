# prep_cas12a_training_data_from_atlas.py
#!/usr/bin/env python3
import argparse, json, re, sys, gzip, os
from collections import defaultdict

def load_dr(ref_fasta=None, dr_seq=None):
    if dr_seq:
        return dr_seq.strip().upper().replace('U', 'T')
    if ref_fasta:
        # parse tiny 1-seq FASTA of the DR (nucleotide); not the protein FASTA
        seq = []
        with open(ref_fasta) as f:
            for line in f:
                if line.startswith('>'): continue
                seq.append(line.strip())
        return ''.join(seq).upper().replace('U', 'T')
    raise SystemExit("Provide --lbcas12a_dr_seq or --lbcas12a_dr_fasta")

def walk(obj):
    """Yield dict-like nodes in a big nested JSON, schema-agnostic."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values(): yield from walk(v)
    elif isinstance(obj, list):
        for v in obj: yield from walk(v)

def canon_pam(pam):
    if not pam: return None
    s = pam.strip().upper().replace('N', 'N')
    s = s.replace(' ', '')
    return s

def is_tttv(pam):
    # strict TTTV: T T T [A/C/G]
    if not pam: return False
    pam = pam.upper()
    if 'TTTV' in pam: return True
    return bool(re.fullmatch(r'TTT[ACG]', pam))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas_json', required=True,
                    help='Path to crispr-cas-atlas-v1.0.json (gz OK)')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--lbcas12a_dr_seq', default=None,
                    help='Direct repeat sequence for LbCas12a (DNA letters).')
    ap.add_argument('--lbcas12a_dr_fasta', default=None,
                    help='FASTA with single DR sequence (alternative to --lbcas12a_dr_seq).')
    ap.add_argument('--min_len', type=int, default=800,
                    help='Min protein length to consider (avoid fragments).')
    ap.add_argument('--pam-map', dest='pam_map', default=None,
                    help='Optional TSV/CSV with two columns: operon_id and consensus_pam. If provided, overrides/augments summary PAMs.')
    args = ap.parse_args()

    LB_DR = load_dr(args.lbcas12a_dr_fasta, args.lbcas12a_dr_seq)

    # tolerant open (plain or gz)
    opener = gzip.open if args.atlas_json.endswith('.gz') else open
    with opener(args.atlas_json, 'rt') as f:
        atlas = json.load(f)

    # Optionally load external operon_id -> consensus_pam mapping
    pam_map = {}
    if args.pam_map:
        try:
            with open(args.pam_map, 'r') as pm:
                for line in pm:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # support both TSV and CSV (split on comma first, then tab)
                    parts = line.split('\t') if ('\t' in line) else line.split(',')
                    if len(parts) < 2:
                        continue
                    oid, p = parts[0].strip(), canon_pam(parts[1].strip())
                    if oid and p:
                        pam_map[oid] = p
        except FileNotFoundError:
            print(f"Warning: pam-map not found: {args.pam_map}", file=sys.stderr)

    # Heuristics to find Cas12a effector entries with (protein seq, PAM, DR)
    rows = []
    seen = 0

    # First, handle known Atlas v1.0 structure (top-level list with 'cas' per record)
    if isinstance(atlas, list):
        for rec in atlas:
            if not isinstance(rec, dict):
                continue
            # subtype/type hints
            subtype = ((rec.get('summary') or {}).get('subtype') or '').upper()
            # Attempt to discover a DR at the record level (array metadata)
            rec_dr = None
            arr = rec.get('array') or rec.get('crispr_array') or {}
            if isinstance(arr, dict):
                rec_dr = arr.get('repeat') or arr.get('repeat_seq')
                if isinstance(rec_dr, str):
                    rec_dr = rec_dr.upper().replace('U', 'T')
            # Fallback: some Atlas exports keep arrays under 'crispr' list with 'crispr_repeat'
            if not rec_dr and isinstance(rec.get('crispr'), list):
                for c in rec['crispr']:
                    if isinstance(c, dict) and isinstance(c.get('crispr_repeat'), str):
                        rec_dr = c['crispr_repeat'].upper().replace('U', 'T')
                        break
            # Try summary-level PAM if present
            rec_pam = None
            summ = rec.get('summary') or {}
            for k in ('pam', 'consensus_pam', 'pam_consensus', 'pam_seq'):
                v = summ.get(k)
                if isinstance(v, str):
                    rec_pam = canon_pam(v)
                    break
            # Override/augment using external pam_map if present (keyed by operon_id)
            oid = rec.get('operon_id') or rec.get('id')
            if oid and oid in pam_map:
                rec_pam = pam_map[oid]
            for cg in rec.get('cas', []) or []:
                name = ((cg.get('gene_name') or '') + ' ' + (cg.get('hmm_name') or '')).upper()
                if not any(tag in name for tag in ('CAS12A','CPF1')):
                    continue
                prot = cg.get('protein') or cg.get('protein_sequence') or ''
                if not (isinstance(prot, str) and set(prot.upper()) <= set('ACDEFGHIKLMNPQRSTVWY')):
                    continue
                if len(prot) < args.min_len:
                    continue
                pam = rec_pam  # default to record-level PAM if available
                pam_tttv = int(is_tttv(pam)) if pam else 0
                dr = rec_dr
                dr_exact = int(dr == LB_DR) if dr else None
                rows.append((prot, pam or '', pam_tttv, dr or '', dr_exact))
                seen += 1

    # Fallback: generic schema-agnostic walk (kept for robustness and smaller JSONs)
    if not rows:
        for node in walk(atlas):
            # a node is interesting if it looks like a Cas12a operon or contains an effector
            typ = (node.get('type') or node.get('crispr_type') or '').upper()
            fam = (node.get('family') or node.get('effector_family') or node.get('protein_family') or '').upper()
            # try to get protein sequence (support a few common keys)
            prot = node.get('protein') or node.get('protein_sequence') or node.get('aa_sequence') or node.get('sequence')
            if prot and isinstance(prot, str) and set(prot.upper()) <= set('ACDEFGHIKLMNPQRSTVWY'):
                if len(prot) < args.min_len:
                    continue
                # try to get PAM consensus
                pam = node.get('pam') or node.get('consensus_pam') or node.get('pam_consensus') or node.get('pam_seq')
                pam = canon_pam(pam) if isinstance(pam, str) else None
                # try to override with external pam_map if same node carries operon_id
                oid = node.get('operon_id') or node.get('id')
                if oid and oid in pam_map:
                    pam = pam_map[oid]
                # try to get a DR (direct repeat) sequence from an array field nearby
                dr = node.get('direct_repeat') or node.get('repeat') or node.get('repeat_seq') \
                     or node.get('crrna_repeat') or node.get('dr_sequence')
                if isinstance(dr, str):
                    dr = dr.upper().replace('U', 'T')
                # Some schemas keep arrays as lists of dicts, try to grab a repeat
                if not dr and isinstance(node.get('array') or node.get('crispr_array'), dict):
                    arr = node.get('array') or node.get('crispr_array')
                    dr = arr.get('repeat') or arr.get('repeat_seq')
                    if isinstance(dr, str): dr = dr.upper().replace('U', 'T')
                # Fallback for list-style 'crispr' entries
                if not dr and isinstance(node.get('crispr'), list):
                    for c in node['crispr']:
                        if isinstance(c, dict) and isinstance(c.get('crispr_repeat'), str):
                            dr = c['crispr_repeat'].upper().replace('U', 'T')
                            break
                # Labels
                pam_tttv = int(is_tttv(pam))
                dr_exact = int(dr == LB_DR) if dr else None
                # Family/name hints
                name = (node.get('name') or node.get('id') or '')
                ok_family = ('CAS12A' in fam) or ('CPF1' in fam) or ('V' in typ) or (pam and pam_tttv==1)
                if ok_family:
                    rows.append((prot,  pam if pam else '', pam_tttv, dr if dr else '', dr_exact))
                    seen += 1

    if not rows:
        raise SystemExit("No Cas12a-like rows found. Adjust key heuristics in the script for your JSON schema.")

    with open(args.out_csv, 'w') as out:
        out.write('protein_seq,consensus_pam,pam_tttv,dr_seq,dr_exact\n')
        for r in rows:
            out.write(','.join([
                r[0],
                r[1],
                str(r[2]),
                r[3],
                '' if r[4] is None else str(r[4])
            ]) + '\n')

    print(f"Wrote {len(rows)} labeled rows to {args.out_csv}")

if __name__ == '__main__':
    main()
