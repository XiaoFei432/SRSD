#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_eval_sets.py

Build evaluation prompt/reference JSONLs for:
- translation: WMT14 de-en
- summarization: CNN/DailyMail (3.0.0)
- math_reasoning: GSM8K (main)

Outputs per task:
  prompts_<task>.jsonl : {id, task, prompt}
  refs_<task>.jsonl    : {id, reference, ...}

Example:
  python build_eval_sets.py --tasks translation summarization math_reasoning \
    --n 200 --split test --seed 123 --out-dir eval_sets

Notes:
- This script downloads datasets via HuggingFace `datasets`.
- For greedy mode, quality metrics are redundant if you already verify token-by-token equality.
- For sampling mode, this gives you the needed references for sacreBLEU/ROUGE (and GSM8K accuracy).
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

from datasets import load_dataset


def _write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sample_indices(n_total: int, n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    n = min(n, n_total)
    return rng.sample(range(n_total), n)


def _clean_text(s: str) -> str:
    # keep it conservative; do not over-normalize
    return re.sub(r"[ \t]+", " ", s).strip()


# -------------------- Translation: WMT14 de->en --------------------

def load_wmt14_de_en(split: str):
    # Robust-ish loader: HF configs sometimes vary; try common ones.
    attempts = [
        ("wmt14", "de-en"),
        ("wmt14", "de-en"),
    ]
    last_err = None
    for name, config in attempts:
        try:
            return load_dataset(name, config, split=split)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load WMT14 de-en split={split}. Last error: {last_err}")


def build_translation(n: int, split: str, seed: int) -> Tuple[List[Dict], List[Dict]]:
    ds = load_wmt14_de_en(split)
    idxs = _sample_indices(len(ds), n, seed)
    prompts, refs = [], []
    for i, idx in enumerate(idxs):
        ex = ds[int(idx)]
        # WMT14 format: ex["translation"] is dict: {"de":..., "en":...}
        trans = ex.get("translation", {})
        src = _clean_text(trans.get("de", ""))
        ref = _clean_text(trans.get("en", ""))

        uid = f"translation::{split}::{idx}"
        prompt = f"Translate German to English: {src}"

        prompts.append({"id": uid, "task": "translation", "prompt": prompt})
        refs.append({"id": uid, "reference": ref})
    return prompts, refs


# -------------------- Summarization: CNN/DailyMail --------------------

def load_cnn_dm(split: str):
    # CNN/DailyMail usually uses config "3.0.0"
    try:
        return load_dataset("cnn_dailymail", "3.0.0", split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load cnn_dailymail 3.0.0 split={split}: {e}")


def build_summarization(n: int, split: str, seed: int) -> Tuple[List[Dict], List[Dict]]:
    ds = load_cnn_dm(split)
    idxs = _sample_indices(len(ds), n, seed)
    prompts, refs = [], []
    for idx in idxs:
        ex = ds[int(idx)]
        article = _clean_text(ex.get("article", ""))
        highlights = _clean_text(ex.get("highlights", ""))

        uid = f"summarization::{split}::{idx}"
        # You can swap this to match Spec-Bench prompt style if you want consistency.
        prompt = (
            "Summarize the following article in a concise manner.\n\n"
            f"Article:\n{article}\n\nSummary:"
        )

        prompts.append({"id": uid, "task": "summarization", "prompt": prompt})
        refs.append({"id": uid, "reference": highlights})
    return prompts, refs


# -------------------- Math Reasoning: GSM8K --------------------

def load_gsm8k(split: str):
    try:
        return load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to load gsm8k main split={split}: {e}")


_ANS_RE = re.compile(r"####\s*(.+)\s*$")


def extract_gsm8k_final(answer_field: str) -> str:
    """
    GSM8K answer format contains rationale + '#### <final>'.
    We extract the final part for exact-match accuracy.
    """
    m = _ANS_RE.search(answer_field.strip())
    if m:
        return _clean_text(m.group(1))
    # fallback: return whole string (still useful for qualitative checks)
    return _clean_text(answer_field)


def build_math_reasoning(n: int, split: str, seed: int) -> Tuple[List[Dict], List[Dict]]:
    ds = load_gsm8k(split)
    idxs = _sample_indices(len(ds), n, seed)
    prompts, refs = [], []
    for idx in idxs:
        ex = ds[int(idx)]
        q = _clean_text(ex.get("question", ""))
        a = ex.get("answer", "")
        final = extract_gsm8k_final(a)

        uid = f"math_reasoning::{split}::{idx}"
        prompt = (
            "Solve the following math problem step by step. "
            "At the end, output only the final numeric answer.\n\n"
            f"Problem: {q}\n\nAnswer:"
        )

        prompts.append({"id": uid, "task": "math_reasoning", "prompt": prompt})
        refs.append({"id": uid, "reference": _clean_text(a), "final_answer": final})
    return prompts, refs


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    choices=["translation", "summarization", "math_reasoning"],
                    default=["translation", "summarization","math_reasoning"])
    ap.add_argument("--n", type=int, default=10, help="samples per task")
    ap.add_argument("--split", type=str, default="test", help="dataset split (e.g., test/validation)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", type=str, default="eval_sets")
    args = ap.parse_args()

    builders = {
        "translation": build_translation,
        "summarization": build_summarization,
        "math_reasoning": build_math_reasoning,
    }

    for task in args.tasks:
        prompts, refs = builders[task](args.n, args.split, args.seed)
        prompt_path = os.path.join(args.out_dir, f"prompts_{task}.jsonl")
        ref_path = os.path.join(args.out_dir, f"refs_{task}.jsonl")
        _write_jsonl(prompt_path, prompts)
        _write_jsonl(ref_path, refs)
        print(f"[OK] {task}: prompts -> {prompt_path} ({len(prompts)}), refs -> {ref_path} ({len(refs)})")


if __name__ == "__main__":
    main()
