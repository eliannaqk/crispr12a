#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Per-residue language-model NLL/Perplexity filter.

- Filtering:
  - By default, keeps sequences with NLL < --nll_threshold.
  - Alternatively, pass --ppl_threshold to keep sequences with PPL < --ppl_threshold.

- Stats mode:
  - With --stats-only, prints average perplexity for the input and, if a threshold
    is provided (either --ppl_threshold or --nll_threshold), prints the percentage
    of sequences above that threshold (interpreted in the same metric as the filter).

Notes:
- By default this script computed average token NLL excluding special tokens.
- You can now enable --align_training_collator to mirror training exactly:
  prepend literal '1' and append '2', add BOS/EOS, build labels like training
  (mask PAD and position 0), and compute NLL over labels != 0 (includes EOS).
"""

import argparse, math, csv, torch, os, sys
from pathlib import Path
from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer

AA_SET = set(list("ACDEFGHIKLMNPQRSTVWYXBZJUO"))  # allow uncommon tokens; prior step should remove non-standard AAs

def clean_seq(seq: str) -> str:
    s = "".join([ch for ch in str(seq).strip().upper() if ch != "*" and ch != " " and ch != "\n" and ch != "\r"])
    return s

def seq_ok(seq: str) -> bool:
    return all((ch in AA_SET) for ch in seq)

def iter_fasta(path):
    rid = None
    seq_parts = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if rid is not None:
                    yield rid, clean_seq("".join(seq_parts))
                rid = line[1:].strip().split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if rid is not None:
            yield rid, clean_seq("".join(seq_parts))

def compute_nll(
    model,
    tokenizer,
    seqs,
    device="cpu",
    batch_size=8,
    use_bf16=True,
    align_training_collator=True,
):
    """
    Compute per-sequence average NLL (and PPL).

    If align_training_collator=True, mirror training-time collator:
      - Wrap text as '1' + seq + '2'
      - Tokenize with add_special_tokens=True (adds BOS/EOS)
      - labels = input_ids with PAD=0 masked and labels[:, 0] = 0
      - NLL over positions where labels != 0 (includes EOS)

    Else, tokenize with add_special_tokens=True and exclude BOS/EOS and PAD
    from the average via special_tokens_mask and attention_mask.

    Returns list of dicts: {id, n_tokens, nll, ppl}
    """
    results = []
    model.eval()
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if (use_bf16 and device.startswith("cuda") and torch.cuda.is_available())
        else torch.cuda.amp.autocast(enabled=False)
    )
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        ids = [x[0] for x in batch]
        txts_raw = [x[1] for x in batch]

        if align_training_collator:
            # Mirror training collator wrapping: '1' + seq + '2'
            txts = ["1" + t + "2" for t in txts_raw]
            enc = tokenizer(
                txts,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Labels as in training: mask PAD (attn==0) and the first position
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, 0)
            if labels.size(1) > 0:
                labels[:, 0] = 0  # ignore BOS alignment

            with torch.no_grad(), autocast_ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Shift for next-token prediction (same as model's internal loss)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Valid positions where labels != 0 (ignore_index)
            valid_mask = (shift_labels != 0)

            # Compute per-token NLL
            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            token_nll = -token_lp  # (B, T-1)

        else:
            # Legacy mode: no '1'/'2' wrapping; exclude BOS/EOS and PAD from averages
            enc = tokenizer(
                txts_raw,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                return_special_tokens_mask=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            special_tokens_mask = enc.get("special_tokens_mask", None)
            if special_tokens_mask is None:
                sm_list = []
                for t in txts_raw:
                    sm_list.append(
                        tokenizer.get_special_tokens_mask(
                            tokenizer(t, add_special_tokens=True)["input_ids"], already_has_special_tokens=True
                        )
                    )
                special_tokens_mask = torch.zeros_like(input_ids)
                for bi, mask_list in enumerate(sm_list):
                    for ti, flag in enumerate(mask_list):
                        if flag == 1:
                            special_tokens_mask[bi, ti] = 1
            else:
                special_tokens_mask = special_tokens_mask.to(device)

            with torch.no_grad(), autocast_ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            pad_mask = (attention_mask[:, 1:] == 0)
            spec_mask = (special_tokens_mask[:, 1:] == 1)
            valid_mask = ~(pad_mask | spec_mask)

            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            token_nll = -token_lp

        # Per-sequence averages
        B = input_ids.size(0)
        for bi in range(B):
            m = valid_mask[bi]
            n_tok = int(m.sum().item())
            if n_tok > 0:
                nll_bi = float(token_nll[bi][m].mean().item())
                ppl_bi = math.exp(nll_bi)
            else:
                nll_bi = float("inf")
                ppl_bi = float("inf")
            results.append({"id": ids[bi], "n_tokens": n_tok, "nll": nll_bi, "ppl": ppl_bi})

    return results

def main():
    ap = argparse.ArgumentParser(description="Per-residue LM NLL/Perplexity filter for protein FASTA.")
    ap.add_argument("--model", required=True, help="Path to local HuggingFace model dir (e.g., ba8000).")
    ap.add_argument("--in_fasta", required=True, help="Input FASTA (from coverage step).")
    ap.add_argument("--out_fasta", required=False, help="Output FASTA with sequences passing threshold.")
    ap.add_argument("--out_tsv", required=False, help="TSV report with NLL and perplexity per sequence.")
    ap.add_argument("--nll_threshold", type=float, default=None, help="Keep if NLL < threshold.")
    ap.add_argument("--ppl_threshold", type=float, default=None, help="Keep if PPL < threshold (takes precedence over NLL threshold).")
    ap.add_argument("--stats-only", action="store_true", help="Only print stats (avg PPL and %% above threshold); do not write outputs.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument(
        "--align_training_collator",
        action="store_true",
        help="Mirror training collator: wrap '1'+seq+'2', add BOS/EOS, mask PAD & BOS in loss (includes EOS).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[info] Loading model from {args.model} on device={args.device}")
    tokenizer = get_tokenizer(ModelSchema(name="progen2", path=args.model))
    model = get_model(ModelSchema(name="progen2", path=args.model)).to(args.device)

    # Read sequences
    seqs = [(rid, s) for rid, s in iter_fasta(args.in_fasta) if seq_ok(s)]
    if not seqs:
        print("[warn] No sequences found after cleaning.")
        if args.out_fasta:
            open(args.out_fasta, "w").close()
        if args.out_tsv:
            with open(args.out_tsv, "w") as f:
                f.write("id\tn_tokens\tnll\tppl\tkeep\n")
        return

    # Score (bf16 autocast on GPU for speed)
    results = compute_nll(
        model,
        tokenizer,
        seqs,
        device=args.device,
        batch_size=args.batch_size,
        use_bf16=True,
        align_training_collator=args.align_training_collator,
    )

    # Stats
    avg_ppl = float(sum(r["ppl"] for r in results) / max(1, len(results)))
    print(f"[info] Average perplexity over {len(results)} sequences: {avg_ppl:.6f}")

    # Determine filter metric
    use_ppl = args.ppl_threshold is not None
    keep_ids = set()
    if use_ppl:
        thr = float(args.ppl_threshold)
        keep_ids = set([r["id"] for r in results if (r["ppl"] < thr and r["n_tokens"] > 0)])
        n_above = sum(1 for r in results if r["ppl"] > thr)
        pct_above = 100.0 * n_above / max(1, len(results))
        print(f"[ok] PPL filter kept {len(keep_ids)} / {len(results)} (threshold={thr}); % above: {pct_above:.2f}%")
    elif args.nll_threshold is not None:
        thr = float(args.nll_threshold)
        keep_ids = set([r["id"] for r in results if (r["nll"] < thr and r["n_tokens"] > 0)])
        n_above = sum(1 for r in results if r["nll"] > thr)
        pct_above = 100.0 * n_above / max(1, len(results))
        print(f"[ok] NLL filter kept {len(keep_ids)} / {len(results)} (threshold={thr}); % above: {pct_above:.2f}%")
    else:
        print("[warn] No threshold provided; stats-only behavior assumed.")

    if args.stats_only:
        return

    # Write FASTA
    if args.out_fasta:
        id2seq = {rid: s for rid, s in seqs}
        with open(args.out_fasta, "w") as fout:
            for rid in keep_ids:
                fout.write(f">{rid}\n{id2seq[rid]}\n")

    # Report
    if args.out_tsv:
        with open(args.out_tsv, "w", newline="") as ftsv:
            w = csv.writer(ftsv, delimiter="\t")
            w.writerow(["id", "n_tokens", "nll", "ppl", "keep"])
            for r in results:
                w.writerow([r["id"], r["n_tokens"], f"{r['nll']:.6f}", f"{r['ppl']:.6f}", int(r["id"] in keep_ids)])

    if args.out_fasta:
        print(f"[ok] Wrote: {args.out_fasta}")
    if args.out_tsv:
        print(f"[ok] Report: {args.out_tsv}")

if __name__ == "__main__":
    main()
