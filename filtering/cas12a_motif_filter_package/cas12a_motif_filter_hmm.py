#!/usr/bin/env python3
"""
Cas12a motif-only filter (HMMER-based, LbCas12a anchor)

What this does (high level)
---------------------------
- Uses an HMM (built from references or provided) to align sequences with **hmmalign**.
- Maps *LbCas12a anchor residues* to **HMM match-state indices** by aligning the anchor to the HMM.
- For each design sequence, aligns to the HMM (together with the anchor in a 2-seq alignment)
  so the columns are consistent. Reads residues at motif match states and applies strict checks.
- Optionally *masks* any motif that falls inside a user-provided "seed" segment in the design.

Why this meets your requirement
-------------------------------
- No "exact positions" are assumed for designs. Every design is aligned via **HMMER**.
- Motifs are evaluated by **HMM match states**, not by raw query indices.
- Only the **anchor motif coordinates** are in LbCas12a numbering, which are converted to
  HMM match-state indices automatically from an anchor-vs-HMM alignment.

Dependencies (CLI tools must be on PATH)
----------------------------------------
- HMMER 3.x: hmmalign (required), hmmbuild (if you build HMM), hmmsearch (optional)
- MAFFT (only if you choose to build the HMM from a references FASTA)

Python deps
-----------
pip install biopython pyyaml

Usage examples
--------------
# Case A: You already have a prebuilt HMM (full-length LbCas12a family/profile)
python cas12a_motif_filter_hmm.py \
  --designs my_designs.faa \
  --config cas12a_motifs_lb_hmm.yaml \
  --hmm lbcas12a.hmm \
  --anchor_fasta lb_anchor.faa --anchor_id_substring LbCas12a \
  --outdir out_hmm

# Case B: Build an HMM from reference FASTA (must include the Lb anchor)
python cas12a_motif_filter_hmm.py \
  --designs my_designs.faa \
  --config cas12a_motifs_lb_hmm.yaml \
  --refs_fasta cas12a_refs.faa \
  --anchor_fasta lb_anchor.faa --anchor_id_substring LbCas12a \
  --outdir out_hmm --threads 8
"""

import os, sys, json, argparse, subprocess, tempfile, re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ---- Optional deps ----
try:
    import yaml
except Exception:
    yaml = None

try:
    from Bio import SeqIO  # type: ignore
    _HAVE_BIO = True
except Exception:
    SeqIO = None  # type: ignore
    _HAVE_BIO = False

# ----------------------
# Utilities
# ----------------------

def log(msg: str):
    print(f"[motif-HMM] {msg}", flush=True)

def which(x: str) -> Optional[str]:
    import shutil
    return shutil.which(x)

def run_cmd(cmd: List[str], cwd: Optional[str]=None, capture: bool=False) -> str:
    log("RUN: " + " ".join(cmd))
    if capture:
        r = subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.stderr:
            # HMMER writes some status to stderr; keep quiet unless needed
            pass
        return r.stdout
    else:
        subprocess.run(cmd, cwd=cwd, check=True)
        return ""

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_fasta(path: str) -> Dict[str,str]:
    recs = OrderedDict()
    if _HAVE_BIO:
        with open(path, "r") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                recs[rec.id] = str(rec.seq).upper().replace("*","")
        return recs
    # Fallback: minimal FASTA parser
    cur = None
    buf = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur is not None:
                    recs[cur] = ''.join(buf).upper().replace('*','')
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(''.join(ch for ch in line if ch.isalpha()))
    if cur is not None:
        recs[cur] = ''.join(buf).upper().replace('*','')
    return recs

def write_fasta(recs: Dict[str,str], path: str):
    with open(path, "w") as fh:
        for rid, seq in recs.items():
            fh.write(f">{rid}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")

def read_config(path: str) -> dict:
    with open(path, "r") as fh:
        txt = fh.read()
    if yaml is not None:
        try:
            return yaml.safe_load(txt)
        except Exception:
            pass
    try:
        import json
        return json.loads(txt)
    except Exception:
        raise RuntimeError("Could not parse config as YAML or JSON")

# ----------------------
# HMM building (optional)
# ----------------------

def build_hmm_from_refs(refs_fasta: str, out_hmm: str, threads: int=4, mafft_path: Optional[str]=None, hmmbuild_path: Optional[str]=None) -> str:
    mafft = mafft_path or which("mafft")
    hmmbuild = hmmbuild_path or which("hmmbuild")
    if mafft is None:
        raise RuntimeError("mafft not found in PATH. Install MAFFT or provide a prebuilt HMM via --hmm.")
    if hmmbuild is None:
        raise RuntimeError("hmmbuild not found in PATH (HMMER). Install HMMER or provide --hmm.")
    with tempfile.TemporaryDirectory() as td:
        aln = os.path.join(td, "refs.aln.fasta")
        cmd = [mafft, "--thread", str(threads), "--maxiterate", "1000", "--localpair", refs_fasta]
        with open(aln, "w") as out:
            log("RUN: " + " ".join(cmd) + f" > {aln}")
            p = subprocess.run(cmd, stdout=out, text=True)
            if p.returncode != 0:
                raise RuntimeError("MAFFT failed.")
        run_cmd([hmmbuild, "--amino", out_hmm, aln])
    return out_hmm

# ----------------------
# hmmalign (A2M) parsing
# ----------------------

AA_UPPER = set(list("ACDEFGHIKLMNPQRSTVWYBJZXO"))  # include ambiguity
AA_LOWER = set([c.lower() for c in AA_UPPER])
GAP_CHARS = set(".-")

def hmmalign_a2m(hmm_path: str, seqs: Dict[str,str]) -> Dict[str,str]:
    """
    Align provided sequences to HMM using A2M format.
    Returns dict: id -> A2M alignment string (one line per id; linebreaks removed).
    """
    hmmalign = which("hmmalign")
    if hmmalign is None:
        raise RuntimeError("hmmalign not found in PATH (HMMER needed).")
    with tempfile.TemporaryDirectory() as td:
        infa = os.path.join(td, "in.faa")
        with open(infa, "w") as fh:
            for rid, seq in seqs.items():
                fh.write(f">{rid}\n{seq}\n")
        # hmmalign prints to stdout by default
        out = run_cmd([hmmalign, "--outformat", "A2M", hmm_path, infa], capture=True)
    # Parse A2M (FASTA-like)
    out = out.strip().splitlines()
    a2m: Dict[str,str] = {}
    cur = None
    buf = []
    for line in out:
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                a2m[cur] = "".join(buf)
                buf = []
            cur = line[1:].strip().split()[0]
        else:
            buf.append(line.strip())
    if cur is not None:
        a2m[cur] = "".join(buf)
    if not a2m:
        raise RuntimeError("Failed to parse hmmalign A2M output.")
    return a2m

def map_anchor_to_mstates(anchor_a2m: str) -> Tuple[Dict[int, int], int]:
    """
    Map: anchor residue index (1-based along anchor sequence) -> HMM match-state index (1-based).
    Counting rule (A2M):
      - Uppercase letter  = residue at a match state   -> advance anchor_pos and mstate; map anchor_pos -> mstate
      - Lowercase letter  = residue at an insert state -> advance anchor_pos only
      - '.' or '-'        = deletion at a match state  -> advance mstate only
    Returns (anchorpos_to_mstate, num_mstates_in_alignment).
    """
    anchorpos_to_mstate: Dict[int, int] = {}
    anchor_pos = 0
    mstate = 0
    for ch in anchor_a2m:
        if ch in AA_UPPER:
            anchor_pos += 1
            mstate += 1
            anchorpos_to_mstate[anchor_pos] = mstate
        elif ch in AA_LOWER:
            anchor_pos += 1
        elif ch in GAP_CHARS:
            mstate += 1
        else:
            # whitespace etc.
            pass
    return anchorpos_to_mstate, mstate

def map_query_by_mstates(query_a2m: str) -> Dict[int, Tuple[Optional[int], Optional[str]]]:
    """
    Build mapping: HMM match-state index (1-based) -> (query residue index (1-based), AA letter)
    or (None, None) if the query has a gap at that match-state).

    Rule (A2M):
      - Lowercase (insert)   -> advance qpos only
      - Uppercase (match)    -> advance qpos and mstate; record (qpos, AA)
      - '.' or '-' (deletion)-> advance mstate; record (None, None)
    """
    m: Dict[int, Tuple[Optional[int], Optional[str]]] = {}
    qpos = 0     # 1-based query residue coordinate (letters only)
    mstate = 0   # 1-based model match-state coordinate
    for ch in query_a2m:
        if ch in AA_LOWER:
            qpos += 1
        elif ch in AA_UPPER:
            qpos += 1
            mstate += 1
            m[mstate] = (qpos, ch.upper())
        elif ch in GAP_CHARS:
            mstate += 1
            m[mstate] = (None, None)
        else:
            # whitespace, etc.
            pass
    return m

# ----------------------
# Seeds
# ----------------------

def clean_seed(s: str) -> str:
    return "".join(ch for ch in s if ch.isalpha()).upper()

def find_seed_hit(qseq: str, seeds: List[Dict], mode: str="prefix") -> Optional[Tuple[str,int,int]]:
    """
    Returns (seed_name, qstart1, qend1) if found.
    mode: "prefix" (fast) or "any" (substring search)
    """
    best = None
    for ent in seeds:
        name = ent.get("name","seed")
        seq = clean_seed(ent.get("seq",""))
        if not seq:
            continue
        if mode == "prefix":
            if qseq.startswith(seq):
                cand = (name, 1, len(seq))
            else:
                continue
        else:
            idx = qseq.find(seq)
            if idx == -1: continue
            cand = (name, idx+1, idx+len(seq))
        if best is None or (cand[2]-cand[1]) > (best[2]-best[1]):
            best = cand
    return best

# ----------------------
# Motif evaluation
# ----------------------

def check_wed(anchorpos_to_m, query_m_to_q, motifs, masked_mstates):
    H = int(motifs["WED"].get("H", 0))
    K1 = int(motifs["WED"].get("K1", 0))
    K2 = int(motifs["WED"].get("K2", 0))
    K_allowed = set([x.upper() for x in motifs["WED"].get("K_allowed", ["K"])])
    req = [("H", H, {"H"}), ("K1", K1, K_allowed), ("K2", K2, K_allowed)]
    found = {}
    ok = True
    for label, apos, allowed in req:
        if apos <= 0 or apos not in anchorpos_to_m:
            found[label] = None; ok = False; continue
        mstate = anchorpos_to_m[apos]
        if mstate in masked_mstates:
            found[label] = "MASKED"; continue
        qpos, aa = query_m_to_q.get(mstate, (None, None))
        found[label] = aa
        ok = ok and (aa in allowed)
    return {"checked": True, "pass": ok, "found": found}

def check_pi(anchorpos_to_m, query_m_to_q, motifs, masked_mstates):
    kpos = [int(p) for p in motifs["PI"].get("K_positions", [])]
    allow_basic = not bool(motifs["PI"].get("require_exact_lys", True))
    found = {}
    oks = []
    for apos in kpos:
        if apos <= 0 or apos not in anchorpos_to_m:
            found[apos] = None; oks.append(False); continue
        mstate = anchorpos_to_m[apos]
        if mstate in masked_mstates:
            found[apos] = "MASKED"; continue
        qpos, aa = query_m_to_q.get(mstate, (None, None))
        found[apos] = aa
        if aa is None:
            oks.append(False)
        else:
            oks.append(aa == "K" or (allow_basic and aa in ("K","R")))
    ok = all(oks) if kpos else False
    return {"checked": True, "pass": ok, "found": found, "allow_basic": allow_basic}

def check_nuc(anchorpos_to_m, query_m_to_q, motifs, masked_mstates):
    pos = int(motifs["NUC"].get("routing_Arg", 0))
    if pos <= 0 or pos not in anchorpos_to_m:
        return {"checked": False, "reason": "no_anchor_pos"}
    mstate = anchorpos_to_m[pos]
    if mstate in masked_mstates:
        return {"checked": True, "pass": True, "found": {pos: "MASKED"}, "note": "masked"}
    qpos, aa = query_m_to_q.get(mstate, (None, None))
    ok = (aa == "R")
    return {"checked": True, "pass": ok, "found": {pos: aa}}

def check_bh(anchorpos_to_m, query_m_to_q, motifs, masked_mstates):
    pos = int(motifs["BH"].get("aromatic_pos", 0))
    allowed = set([x.upper() for x in motifs["BH"].get("allowed", ["W"])])
    if pos <= 0 or pos not in anchorpos_to_m:
        return {"checked": False, "reason": "no_anchor_pos"}
    mstate = anchorpos_to_m[pos]
    if mstate in masked_mstates:
        return {"checked": True, "pass": True, "found": {pos: "MASKED"}, "allowed": sorted(allowed)}
    qpos, aa = query_m_to_q.get(mstate, (None, None))
    ok = (aa in allowed)
    return {"checked": True, "pass": ok, "found": {pos: aa}, "allowed": sorted(allowed)}

def check_lid(anchorpos_to_m, query_m_to_q, motifs, masked_mstates):
    pos_list = [int(p) for p in motifs.get("LID",{}).get("basic_positions", [])]
    if not pos_list:
        return {"checked": False}
    min_basic = int(motifs["LID"].get("min_basic", max(1, len(pos_list))))
    found = {}
    basic = 0
    for apos in pos_list:
        if apos <= 0 or apos not in anchorpos_to_m:
            found[apos] = None; continue
        mstate = anchorpos_to_m[apos]
        if mstate in masked_mstates:
            found[apos] = "MASKED"; continue
        qpos, aa = query_m_to_q.get(mstate, (None, None))
        found[apos] = aa
        if aa in ("K","R"):
            basic += 1
    ok = basic >= min_basic
    return {"checked": True, "pass": ok, "found": found, "basic_count": basic, "min_basic": min_basic}

# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--designs", required=True, help="Design protein FASTA")
    ap.add_argument("--config", required=True, help="YAML/JSON config (motifs in Lb numbering)")
    ap.add_argument("--outdir", default="out_hmm", help="Output directory")
    ap.add_argument("--hmm", help="Prebuilt HMM file (full-length Cas12a family/profile)")
    ap.add_argument("--refs_fasta", help="Reference FASTA to build HMM (must include the Lb anchor)")
    ap.add_argument("--threads", type=int, default=4, help="Threads for MAFFT if building HMM")
    ap.add_argument("--anchor_fasta", help="FASTA containing the LbCas12a anchor (if not given inline in config)")
    ap.add_argument("--anchor_id_substring", help="Substring to select the anchor record in --anchor_fasta or --refs_fasta")
    args = ap.parse_args()

    cfg = read_config(args.config)
    ensure_dir(args.outdir)
    res_dir = os.path.join(args.outdir, "results"); ensure_dir(res_dir)
    det_dir = os.path.join(res_dir, "details"); ensure_dir(det_dir)
    sets_dir = os.path.join(args.outdir, "sets"); ensure_dir(sets_dir)
    bymotif_dir = os.path.join(sets_dir, "by_motif"); ensure_dir(bymotif_dir)

    # Obtain anchor sequence: from config.anchor.seq or --anchor_fasta (or refs_fasta if used to build HMM)
    anchor_seq = None; anchor_id = None
    anc = cfg.get("anchor", {}) or {}
    if anc.get("seq"):
        anchor_seq = re.sub(r"\s+","", anc["seq"].upper()).replace("*","")
        anchor_id = anc.get("id","LbCas12a_anchor")
    else:
        # find in anchor_fasta or refs_fasta
        src = args.anchor_fasta or args.refs_fasta or anc.get("fasta")
        if not src:
            raise RuntimeError("Provide anchor seq via config.anchor.seq or pass --anchor_fasta/--refs_fasta.")
        ref = parse_fasta(src)
        hint = args.anchor_id_substring or anc.get("id_substring")
        if hint:
            for rid, seq in ref.items():
                if hint in rid:
                    anchor_id, anchor_seq = rid, seq
                    break
            if anchor_seq is None:
                raise RuntimeError(f"Could not find anchor id_substring '{hint}' in {src}")
        else:
            # Take first
            anchor_id, anchor_seq = next(iter(ref.items()))
    if anchor_seq is None:
        raise RuntimeError("Failed to obtain anchor sequence.")
    log(f"Anchor: {anchor_id} (len={len(anchor_seq)})")

    # Prepare HMM: use provided or build from refs
    hmm_path = args.hmm
    if hmm_path is None:
        refs = args.refs_fasta or (cfg.get("hmm",{}).get("build",{}).get("refs_fasta"))
        if not refs:
            raise RuntimeError("Provide --hmm or --refs_fasta to build the HMM.")
        hmm_path = os.path.join(args.outdir, "lbcas12a_from_refs.hmm")
        build_hmm_from_refs(refs, hmm_path, threads=args.threads,
                            mafft_path=which("mafft"), hmmbuild_path=which("hmmbuild"))
        log(f"Built HMM: {hmm_path}")
    else:
        if not os.path.exists(hmm_path):
            raise RuntimeError(f"HMM not found: {hmm_path}")
        log(f"Using provided HMM: {hmm_path}")

    # Motifs (Lb numbering): default to Lb positions; allow overrides in config
    defaults = {
        "WED": {"H": 759, "K1": 768, "K2": 785},
        "PI":  {"K_positions": [538, 595], "require_exact_lys": True},
        "NUC": {"routing_Arg": 1138},
        "BH":  {"aromatic_pos": 890, "allowed": ["W"]},
        # "LID": {"basic_positions": [...], "min_basic": 3}  # optional
    }
    motifs = defaults
    user_motifs = cfg.get("motifs") or {}
    for k,v in user_motifs.items():
        if isinstance(v, dict) and isinstance(motifs.get(k, {}), dict):
            tmp = motifs.get(k, {}).copy(); tmp.update(v); motifs[k] = tmp
        else:
            motifs[k] = v

    # Seeds
    seed_cfg = cfg.get("seeds", {}) or {}
    seeds = seed_cfg.get("list", []) or []
    seed_search_mode = seed_cfg.get("search_in", "prefix")
    enable_masking = bool(seed_cfg.get("enable_masking", True))

    # Load designs
    designs = parse_fasta(args.designs)
    if not designs:
        raise RuntimeError("No sequences found in --designs FASTA")

    # 1) Align anchor to HMM to sanity-check match-state parsing
    a2m = hmmalign_a2m(hmm_path, {anchor_id: anchor_seq})
    if anchor_id not in a2m:
        raise RuntimeError("hmmalign failed to produce alignment for anchor")
    anchor_a2m = a2m[anchor_id]
    _anchorpos_to_m_solo, _num_mstates_solo = map_anchor_to_mstates(anchor_a2m)
    if _num_mstates_solo <= 0:
        raise RuntimeError("Could not determine HMM match-state indices from anchor A2M.")

    # Optional tail suffix fast-path config
    def collect_motif_anchor_positions(motifs: dict) -> List[int]:
        pos: List[int] = []
        if "WED" in motifs:
            for k in ("H","K1","K2"):
                try:
                    v = int(motifs["WED"].get(k, 0) or 0)
                    if v > 0:
                        pos.append(v)
                except Exception:
                    pass
        if "PI" in motifs:
            try:
                pos += [int(p) for p in motifs["PI"].get("K_positions", [])]
            except Exception:
                pass
        if "NUC" in motifs:
            try:
                v = int(motifs["NUC"].get("routing_Arg", 0) or 0)
                if v > 0:
                    pos.append(v)
            except Exception:
                pass
        if "BH" in motifs:
            try:
                v = int(motifs["BH"].get("aromatic_pos", 0) or 0)
                if v > 0:
                    pos.append(v)
            except Exception:
                pass
        if "LID" in motifs:
            try:
                pos += [int(p) for p in motifs["LID"].get("basic_positions", [])]
            except Exception:
                pass
        return pos

    def all_in_range(positions: List[int], start: int, end: int) -> bool:
        return all(start <= p <= end for p in positions)

    tail_cfg = ((cfg.get("anchor") or {}).get("tail_check") or {})
    tail_enabled = bool(tail_cfg.get("enabled", False))
    tail_start = int(tail_cfg.get("start", 514))
    tail_end   = int(tail_cfg.get("end", 1228))
    tail_mode  = str(tail_cfg.get("mode", "endswith")).lower()
    tail_require_all = bool(tail_cfg.get("require_all_motifs_in_range", True))
    tail_seq = anchor_seq[tail_start-1:tail_end] if tail_enabled else ""
    motif_positions = collect_motif_anchor_positions(motifs)

    # Output summary file
    tsv = os.path.join(res_dir, "motif_summary.tsv")
    with open(tsv, "w") as out:
        header = ["seq_id","WED_pass","PI_pass","NUC_pass","BH_pass","LID_pass","overall_pass","seed_masked","fail_reasons"]
        out.write("\t".join(header) + "\n")

        # Output per-motif FASTAs
        per_motif_pass = {"WED":{}, "PI":{}, "NUC":{}, "BH":{}, "LID":{}}
        pass_all = {}
        fail_any = {}

        for sid, qseq in designs.items():
            used_tail_fastpath = False
            seed_hit: Optional[Tuple[str,int,int]] = None
            masked_mstates: set = set()

            def do_tail_fastpath(qseq_local: str) -> Optional[Tuple[Dict[int,int], Dict[int, Tuple[int,str]]]]:
                if not (tail_enabled and tail_seq):
                    return None
                if tail_require_all and not all_in_range(motif_positions, tail_start, tail_end):
                    return None
                hits = (qseq_local.endswith(tail_seq)) if (tail_mode == "endswith") else (tail_seq in qseq_local)
                if not hits:
                    return None
                # Map anchor positions directly to query positions via suffix offset
                if tail_mode == "endswith":
                    suffix_start1 = len(qseq_local) - len(tail_seq) + 1
                else:
                    idx0 = qseq_local.find(tail_seq)
                    if idx0 < 0:
                        return None
                    suffix_start1 = idx0 + 1
                anchorpos_to_m_direct: Dict[int, int] = {}
                q_m_to_q_direct: Dict[int, Tuple[int, str]] = {}
                for apos in motif_positions:
                    if not (tail_start <= apos <= tail_end):
                        continue
                    offset = apos - tail_start
                    qpos1 = suffix_start1 + offset
                    if 1 <= qpos1 <= len(qseq_local):
                        aa = qseq_local[qpos1-1].upper()
                        anchorpos_to_m_direct[apos] = apos
                        q_m_to_q_direct[apos] = (qpos1, aa)
                if tail_require_all and any(apos not in anchorpos_to_m_direct for apos in motif_positions):
                    return None
                return anchorpos_to_m_direct, q_m_to_q_direct

            fast = do_tail_fastpath(qseq)
            if fast is not None:
                used_tail_fastpath = True
                anchorpos_to_m_use, q_m_to_q_use = fast
            else:
                # 2) For each query, align [anchor + query] together so columns are shared
                a2m_pair = hmmalign_a2m(hmm_path, {anchor_id: anchor_seq, sid: qseq})
                if anchor_id not in a2m_pair or sid not in a2m_pair:
                    raise RuntimeError(f"hmmalign did not return both anchor and {sid}")
                anchor_pair = a2m_pair[anchor_id]
                query_pair  = a2m_pair[sid]

                # Map anchor positions to mstate indices for this pair, and query residues by mstates
                anchorpos_to_m_use, _ = map_anchor_to_mstates(anchor_pair)
                q_m_to_q_use = map_query_by_mstates(query_pair)

                # Seed masking if enabled
                seed_hit = find_seed_hit(qseq, seeds, mode=seed_search_mode) if (enable_masking and seeds) else None
                if seed_hit:
                    s_name, s_start, s_end = seed_hit
                    for mi, (qpos, aa) in q_m_to_q_use.items():
                        if qpos is not None and s_start <= qpos <= s_end:
                            masked_mstates.add(mi)

            # Evaluate motifs using the chosen mapping
            wed = check_wed(anchorpos_to_m_use, q_m_to_q_use, motifs, masked_mstates) if "WED" in motifs else {"checked": False}
            pi  = check_pi (anchorpos_to_m_use, q_m_to_q_use, motifs, masked_mstates) if "PI"  in motifs else {"checked": False}
            nuc = check_nuc(anchorpos_to_m_use, q_m_to_q_use, motifs, masked_mstates) if "NUC" in motifs else {"checked": False}
            bh  = check_bh (anchorpos_to_m_use, q_m_to_q_use, motifs, masked_mstates) if "BH"  in motifs else {"checked": False}
            lid = check_lid(anchorpos_to_m_use, q_m_to_q_use, motifs, masked_mstates) if "LID" in motifs else {"checked": False}

            def pf(x): return (x.get("checked", False) and x.get("pass", False))
            fails = []
            if wed.get("checked") and not wed.get("pass"): fails.append("WED")
            if pi.get("checked")  and not pi.get("pass"):  fails.append("PI")
            if nuc.get("checked") and not nuc.get("pass"): fails.append("NUC")
            if bh.get("checked")  and not bh.get("pass"):  fails.append("BH")
            if lid.get("checked") and not lid.get("pass"): fails.append("LID")
            overall = (len(fails) == 0)

            # write TSV row
            out.write("\t".join([sid, str(pf(wed)), str(pf(pi)), str(pf(nuc)), str(pf(bh)), str(pf(lid)),
                                 str(overall), (seed_hit[0] if seed_hit else ""), ";".join(fails)]) + "\n")

            # details JSON
            det = {
                "id": sid,
                "motifs": {"WED": wed, "PI": pi, "NUC": nuc, "BH": bh, "LID": lid},
                "overall_pass": overall,
                "seed_hit": {"name": seed_hit[0], "qstart": seed_hit[1], "qend": seed_hit[2]} if seed_hit else None,
                "fast_path": {"used_tail": used_tail_fastpath,
                               "tail_window": {"start": tail_start, "end": tail_end} if tail_enabled else None}
            }
            with open(os.path.join(det_dir, f"{sid}.json"), "w") as fh:
                json.dump(det, fh, indent=2)

            # collect FASTA sets
            if pf(wed): per_motif_pass["WED"][sid] = qseq
            if pf(pi):  per_motif_pass["PI"][sid]  = qseq
            if pf(nuc): per_motif_pass["NUC"][sid] = qseq
            if pf(bh):  per_motif_pass["BH"][sid]  = qseq
            if pf(lid): per_motif_pass["LID"][sid] = qseq
            if overall: pass_all[sid] = qseq
            else: fail_any[sid] = qseq

    # write FASTAs
    def write_fasta_dict(d: Dict[str,str], path: str):
        with open(path, "w") as fh:
            for rid, seq in d.items():
                fh.write(f">{rid}\n")
                for i in range(0, len(seq), 80):
                    fh.write(seq[i:i+80] + "\n")

    write_fasta_dict(pass_all, os.path.join(sets_dir, "motif_pass.faa"))
    write_fasta_dict(fail_any, os.path.join(sets_dir, "motif_fail.faa"))
    for name, d in per_motif_pass.items():
        if d:
            write_fasta_dict(d, os.path.join(bymotif_dir, f"{name}_pass.faa"))

    log(f"Wrote summary: {tsv}")
    log("Done.")

if __name__ == "__main__":
    main()
