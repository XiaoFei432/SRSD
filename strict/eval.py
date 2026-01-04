#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute sampling-quality metrics with bootstrap CI over prompts:
- Translation: sacreBLEU (sentence-level, averaged over seeds then prompts)
- Summarization: ROUGE-1/2/L (F1), averaged over seeds then prompts

Inputs:
  --ar   AR outputs JSONL:   {id, seed, output}
  --srsd SRSD outputs JSONL: {id, seed, output}
  --ref  references JSONL:   {id, reference}

Reports:
  AR mean + 95% bootstrap CI
  SRSD mean + 95% bootstrap CI
  Delta(SRSD-AR) mean + 95% bootstrap CI (paired bootstrap over ids)

Dependencies:
  pip install sacrebleu rouge-score numpy

python eval.py \
  --task summarization \
  --ar /root/autodl-tmp/strict/result/ar_summarizaiton_sampling.jsonl \
  --srsd /root/autodl-tmp/strict/result/srsd_summarizaiton_sampling.jsonl \
  --ref /root/autodl-tmp/strict/eval_data/refs_summarization.jsonl \
  --bootstrap 5000 --seed 123 --use_common_seeds
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np

# deps:
#   pip install sacrebleu rouge-score numpy
from rouge_score import rouge_scorer
import sacrebleu


# ---------------- IO ----------------

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_refs(ref_path):
    """
    Returns:
      ref_id2text: dict[id] = reference
      ref_ids_in_order: list of ids in file order
    """
    ref_id2text = {}
    ref_ids_in_order = []
    for r in read_jsonl(ref_path):
        _id = str(r["id"])
        ref = r.get("reference", "")
        ref_id2text[_id] = ref
        ref_ids_in_order.append(_id)
    return ref_id2text, ref_ids_in_order


# -------------- ID Alignment (sample_idx -> ref order) --------------

def detect_sample_idx_base(rows):
    """Return 0 if seems 0-based else 1-based."""
    idxs = [int(r["sample_idx"]) for r in rows if "sample_idx" in r]
    if not idxs:
        return 0
    if min(idxs) == 0:
        return 0
    # if no zero appears, assume 1-based
    return 1


def build_outputs_map(rows, ref_ids_in_order, require_task=None, use_common_seeds=False):
    """
    Build: out_map[id][seed] = output
    Supports rows without "id": use sample_idx mapped to ref_ids_in_order.
    """
    # Pre-scan for base detection if needed
    base = detect_sample_idx_base(rows)

    out_map = defaultdict(dict)
    dropped = 0
    for r in rows:
        if require_task is not None and r.get("task") not in (None, require_task):
            # some logs may not have task; don't over-filter
            pass

        seed = int(r.get("seed", 0))
        out = r.get("output", "")

        if "id" in r:
            _id = str(r["id"])
        else:
            if "sample_idx" not in r:
                dropped += 1
                continue
            sidx = int(r["sample_idx"]) - base  # convert to 0-based
            if sidx < 0 or sidx >= len(ref_ids_in_order):
                dropped += 1
                continue
            _id = ref_ids_in_order[sidx]

        out_map[_id][seed] = out

    return out_map, dropped, base


# ---------------- Metrics ----------------

def rouge_per_prompt(outputs_by_seed, reference, scorer):
    """
    outputs_by_seed: dict[seed] -> hyp
    Return averaged ROUGE (r1,r2,rl) F1 over seeds for this prompt.
    """
    r1s, r2s, rls = [], [], []
    for _, hyp in outputs_by_seed.items():
        s = scorer.score(reference, hyp)
        r1s.append(s["rouge1"].fmeasure)
        r2s.append(s["rouge2"].fmeasure)
        rls.append(s["rougeL"].fmeasure)
    if not r1s:
        return None
    return float(np.mean(r1s)), float(np.mean(r2s)), float(np.mean(rls))


def bleu_per_prompt(outputs_by_seed, reference):
    """
    Return averaged sentence-BLEU over seeds for this prompt.
    """
    vals = []
    for _, hyp in outputs_by_seed.items():
        # sacrebleu sentence_bleu returns score in [0,100]
        vals.append(sacrebleu.sentence_bleu(hyp, [reference]).score)
    if not vals:
        return None
    return float(np.mean(vals))


def bootstrap_ci(values, n_bootstrap=5000, seed=123):
    """
    values: list[float] per prompt
    Return (mean, lo, hi)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(arr.mean())
    n = arr.size
    boots = []
    for _ in range(int(n_bootstrap)):
        sample = arr[rng.integers(0, n, size=n)]
        boots.append(sample.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["summarization", "translation"])
    ap.add_argument("--ar", required=True)
    ap.add_argument("--srsd", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--use_common_seeds", action="store_true",
                    help="Only evaluate seeds that exist in BOTH AR and SRSD for each prompt id.")
    args = ap.parse_args()

    ref_id2text, ref_ids_in_order = load_refs(args.ref)

    ar_rows = list(read_jsonl(args.ar))
    srsd_rows = list(read_jsonl(args.srsd))

    ar_map, ar_dropped, ar_base = build_outputs_map(ar_rows, ref_ids_in_order, require_task=args.task)
    srsd_map, srsd_dropped, srsd_base = build_outputs_map(srsd_rows, ref_ids_in_order, require_task=args.task)

    # overlapping ids
    common_ids = sorted(set(ar_map.keys()) & set(srsd_map.keys()) & set(ref_id2text.keys()))
    if not common_ids:
        raise ValueError("No overlapping ids among ar/srsd/ref. "
                         f"(ar_ids={len(ar_map)}, srsd_ids={len(srsd_map)}, ref_ids={len(ref_id2text)})")

    # metrics per prompt (averaged across seeds)
    if args.task == "summarization":
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        ar_r1, ar_r2, ar_rl = [], [], []
        s_r1, s_r2, s_rl = [], [], []

        kept = 0
        for _id in common_ids:
            ref = ref_id2text[_id]
            ar_seeds = set(ar_map[_id].keys())
            srsd_seeds = set(srsd_map[_id].keys())
            if args.use_common_seeds:
                inter = ar_seeds & srsd_seeds
                if not inter:
                    continue
                ar_outs = {k: ar_map[_id][k] for k in inter}
                srsd_outs = {k: srsd_map[_id][k] for k in inter}
            else:
                ar_outs = ar_map[_id]
                srsd_outs = srsd_map[_id]

            ar_scores = rouge_per_prompt(ar_outs, ref, scorer)
            s_scores = rouge_per_prompt(srsd_outs, ref, scorer)
            if ar_scores is None or s_scores is None:
                continue

            kept += 1
            ar_r1.append(ar_scores[0]); ar_r2.append(ar_scores[1]); ar_rl.append(ar_scores[2])
            s_r1.append(s_scores[0]);  s_r2.append(s_scores[1]);  s_rl.append(s_scores[2])

        ar1 = bootstrap_ci(ar_r1, args.bootstrap, args.seed)
        ar2 = bootstrap_ci(ar_r2, args.bootstrap, args.seed + 1)
        arl = bootstrap_ci(ar_rl, args.bootstrap, args.seed + 2)

        sr1 = bootstrap_ci(s_r1, args.bootstrap, args.seed + 3)
        sr2 = bootstrap_ci(s_r2, args.bootstrap, args.seed + 4)
        srl = bootstrap_ci(s_rl, args.bootstrap, args.seed + 5)

        print("==== Alignment Info ====")
        print(f"ref_lines={len(ref_ids_in_order)} | ar_rows={len(ar_rows)} (dropped={ar_dropped}, base={ar_base}) "
              f"| srsd_rows={len(srsd_rows)} (dropped={srsd_dropped}, base={srsd_base})")
        print(f"common_ids={len(common_ids)} | used_prompts={kept} | use_common_seeds={args.use_common_seeds}")

        print("\n==== ROUGE (F1) mean ± 95% bootstrap CI over prompts ====")
        print(f"AR    ROUGE-1: {ar1[0]:.4f}  [{ar1[1]:.4f}, {ar1[2]:.4f}]")
        print(f"AR    ROUGE-2: {ar2[0]:.4f}  [{ar2[1]:.4f}, {ar2[2]:.4f}]")
        print(f"AR    ROUGE-L: {arl[0]:.4f}  [{arl[1]:.4f}, {arl[2]:.4f}]")
        print(f"SRSD  ROUGE-1: {sr1[0]:.4f}  [{sr1[1]:.4f}, {sr1[2]:.4f}]")
        print(f"SRSD  ROUGE-2: {sr2[0]:.4f}  [{sr2[1]:.4f}, {sr2[2]:.4f}]")
        print(f"SRSD  ROUGE-L: {srl[0]:.4f}  [{srl[1]:.4f}, {srl[2]:.4f}]")

    else:  # translation
        ar_bleu, s_bleu = [], []
        kept = 0
        for _id in common_ids:
            ref = ref_id2text[_id]
            ar_seeds = set(ar_map[_id].keys())
            srsd_seeds = set(srsd_map[_id].keys())
            if args.use_common_seeds:
                inter = ar_seeds & srsd_seeds
                if not inter:
                    continue
                ar_outs = {k: ar_map[_id][k] for k in inter}
                srsd_outs = {k: srsd_map[_id][k] for k in inter}
            else:
                ar_outs = ar_map[_id]
                srsd_outs = srsd_map[_id]

            a = bleu_per_prompt(ar_outs, ref)
            b = bleu_per_prompt(srsd_outs, ref)
            if a is None or b is None:
                continue
            kept += 1
            ar_bleu.append(a)
            s_bleu.append(b)

        ar_ci = bootstrap_ci(ar_bleu, args.bootstrap, args.seed)
        s_ci = bootstrap_ci(s_bleu, args.bootstrap, args.seed + 1)

        print("==== Alignment Info ====")
        print(f"ref_lines={len(ref_ids_in_order)} | ar_rows={len(ar_rows)} (dropped={ar_dropped}, base={ar_base}) "
              f"| srsd_rows={len(srsd_rows)} (dropped={srsd_dropped}, base={srsd_base})")
        print(f"common_ids={len(common_ids)} | used_prompts={kept} | use_common_seeds={args.use_common_seeds}")

        print("\n==== sacreBLEU (sentence BLEU avg) mean ± 95% bootstrap CI over prompts ====")
        print(f"AR    BLEU: {ar_ci[0]:.2f}  [{ar_ci[1]:.2f}, {ar_ci[2]:.2f}]")
        print(f"SRSD  BLEU: {s_ci[0]:.2f}  [{s_ci[1]:.2f}, {s_ci[2]:.2f}]")

if __name__ == "__main__":
    main()