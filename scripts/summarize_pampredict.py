import sys, os, glob, csv, re
from pathlib import Path
from typing import Tuple

def parse_pam_prediction(txt_path: str):
    pam = None
    lines = []
    try:
        with open(txt_path) as f:
            lines = [l.strip() for l in f]
    except FileNotFoundError:
        return None
    for l in lines:
        m = re.search(r"Predicted PAM:\s*([A-ZN-]+)", l)
        if m:
            pam = m.group(1)
            break
    return pam

def count_unique_hits(filtered_tsv: str):
    if not filtered_tsv or not os.path.exists(filtered_tsv):
        return 0, 0
    qseqids = []
    with open(filtered_tsv, newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r, None)
        for row in r:
            if not row or row[0].startswith('#'):
                continue
            # col 2 (index 1) in our sample was qseqid prefixed with row index; robustly search by header if present
            if header and 'qseqid' in header:
                qidx = header.index('qseqid')
                qseqids.append(row[qidx])
            else:
                # fallback: second field looked like 'sp1', 'sp3' in our test
                if len(row) > 1:
                    qseqids.append(row[1])
    total = len(qseqids)
    uniq = len(set(qseqids))
    return uniq, total

def peak_bits_from_info_tsv(path: str) -> float:
    """Return the maximum per-base information bit across positions.
    Expects a TSV with header: pos\tA\tC\tG\tT and numeric values (bits) per base.
    """
    if not path or not os.path.exists(path):
        return 0.0
    peak = 0.0
    with open(path, newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r, None)
        # expect columns >= 5
        for row in r:
            if not row or row[0].startswith('#'):
                continue
            try:
                vals = [float(x) for x in row[1:5]]
            except Exception:
                continue
            m = max(vals) if vals else 0.0
            if m > peak:
                peak = m
    return peak

def compute_snr(up_info: str, down_info: str) -> Tuple[float, float, float]:
    """Compute peak bits for upstream and downstream, and SNR=up/down.
    Adds small epsilon to denominator to avoid division by zero.
    """
    up = peak_bits_from_info_tsv(up_info)
    down = peak_bits_from_info_tsv(down_info)
    eps = 1e-9
    snr = (up) / (down + eps) if (up > 0 or down > 0) else 0.0
    return up, down, snr

def main(root: str):
    runs = sorted([p for p in Path(root).glob('*') if p.is_dir()])
    out_path = Path(root)/'pam_crispr.per_array.tsv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['array_id','pam_side','pam_iupac','unique_hits','total_hits','peak_bits_up','peak_bits_down','snr','label'])
        for run in runs:
            filtered = glob.glob(str(run/'blastn'/'*_filtered_matches_with_flanking_sequences.tsv'))
            filtered_path = filtered[0] if filtered else ''
            uniq, total = count_unique_hits(filtered_path)
            pam = parse_pam_prediction(run/'PAM_prediction.txt')
            up_info = run/'upstream_flanking_sequence_info.tsv'
            down_info = run/'downstream_flanking_sequence_info.tsv'
            up_bits, down_bits, snr = compute_snr(str(up_info), str(down_info))
            # High-confidence per Type V heuristic: >=10 unique hits AND SNR > 2
            label = 'high_confidence' if (pam and uniq >= 10 and snr > 2.0) else ('low_confidence' if pam else 'no_call')
            w.writerow([run.name, 'UPSTREAM', pam or '', uniq, total, f"{up_bits:.3f}", f"{down_bits:.3f}", f"{snr:.3f}", label])
    print(f'[summary] wrote {out_path}')

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'eval_outputs/pampredict_runs'
    main(root)
