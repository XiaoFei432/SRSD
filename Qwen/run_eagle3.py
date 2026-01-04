#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

python run_eagle3.py \
  --tasks project \
  --samples-per-task 50 \
  --max-new-tokens 2048 \
  --max-length 3200 \
  --truncate-mode headtail \
  --trunc-head-tokens 256 \
  --base-model-path /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --ea-model-path /root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B \
  --question-path ../spec_bench/long_text.jsonl \
  --baseline-prefill-json advantage/la31/pld_greedy.jsonl \
  --output advantage/la31/eagle3.jsonl \
  --seed 42

"""

import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Any, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

# Try import EAGLE
try:
    from eagle.model.ea_model import EaModel
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "EAGLE"))
    try:
        from eagle.model.ea_model import EaModel
    except ImportError as e:
        print("[Error] Cannot import EaModel. Please clone https://github.com/SafeAILab/EAGLE.git")
        raise e


# ======================
# 0. Default config
# ======================

DEFAULT_BASE_MODEL_PATH = "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
DEFAULT_EA_MODEL_PATH = "/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B"

SPEC_BENCH_QUESTION_PATH = "../spec_bench/long_text.jsonl"

FINE_TASKS: List[str] = [
    "writing", "roleplay", "reasoning", "math", "project",
    "coding", "extraction", "stem", "humanities",
    "translation", "summarization", "text_edit", "math_reasoning", "code_edit",
]

DEFAULT_EAGLE_MAX_LENGTH = 8192
DEFAULT_TRUNCATE_MODE = "headtail"
DEFAULT_TRUNC_HEAD_TOKENS = 256


# ======================
# 1. Utils
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_stop_token_ids(model, tokenizer) -> List[int]:
    """
    Robustly fetch stop token IDs from model.generation_config and tokenizer.
    Adapted from PLD script.
    """
    eos_ids: List[int] = []

    # 1. Check generation_config (The Source of Truth)
    eos = None
    if getattr(model, "generation_config", None) is not None:
        eos = getattr(model.generation_config, "eos_token_id", None)

    if isinstance(eos, int):
        eos_ids.append(int(eos))
    elif isinstance(eos, (list, tuple)):
        for x in eos:
            if isinstance(x, int) and x >= 0:
                eos_ids.append(int(x))

    # 2. Check Tokenizer (Fallback)
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    # 3. Extra Safety for Llama-3 Instruct (Explicitly check for <|eot_id|>)
    # Some older HF versions might not auto-load eot_id into generation_config
    if "<|eot_id|>" in tokenizer.get_vocab():
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids.append(eot_id)

    # 4. Dedup
    dedup: List[int] = []
    seen = set()
    for x in eos_ids:
        if x not in seen:
            dedup.append(x)
            seen.add(x)

    return dedup

def load_eagle3_and_tokenizer(base_path: str, ea_path: str) -> Tuple[AutoTokenizer, EaModel, torch.device]:
    print(f"[EAGLE] Loading base model from : {base_path}")
    print(f"[EAGLE] Loading EAGLE-3 weights : {ea_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ea_model = EaModel.from_pretrained(
        ea_model_path=ea_path,
        base_model_path=base_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        use_eagle3=True,
    )
    ea_model.eval()

    main_device = next(ea_model.parameters()).device
    print(f"[EAGLE] Main device: {main_device}")
    
    # ========================================================
    # ✅ FIX: Robust Stop Token Injection
    # ========================================================
    # Use the robust logic to get all stop tokens
    stop_ids = get_stop_token_ids(ea_model.base_model, tokenizer)
    
    print(f"[Config] Detected Stop Token IDs: {stop_ids}")
    
    # Inject into the base model's configuration
    # EAGLE uses the base model's config to determine when to stop
    ea_model.base_model.generation_config.eos_token_id = stop_ids
    ea_model.base_model.config.eos_token_id = stop_ids

    return tokenizer, ea_model, main_device

def encode_prompt(tokenizer: AutoTokenizer, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(enc, torch.Tensor):
            input_ids = enc
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    else:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }

def iter_question_files(question_path: str) -> List[str]:
    if os.path.isfile(question_path):
        return [question_path]
    if os.path.isdir(question_path):
        out = []
        for root, _, files in os.walk(question_path):
            for fn in files:
                if fn.endswith(".jsonl"):
                    out.append(os.path.join(root, fn))
        out.sort()
        return out
    raise FileNotFoundError(f"question_path not found: {question_path}")

def get_dataset_prompts(question_path: str, task: str, samples_per_task: int) -> List[str]:
    prompts: List[str] = []
    files = iter_question_files(question_path)

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError: continue
                if obj.get("category") != task: continue
                
                turns = obj.get("turns", [])
                prompt = "\n".join(turns) if isinstance(turns, list) else str(turns)
                prompts.append(prompt)
                
                if 0 < samples_per_task <= len(prompts):
                    print(f"[Data] Task={task}, loaded {len(prompts)} prompts from {len(files)} files.")
                    return prompts
    print(f"[Data] Task={task}, loaded {len(prompts)} prompts from {len(files)} files.")
    return prompts

def load_baseline_prefill_map(path: str) -> Dict[Tuple[str, int], float]:
    mp: Dict[Tuple[str, int], float] = {}
    if not path or not os.path.exists(path):
        return mp
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    task = obj.get("task")
                    idx = obj.get("sample_idx")
                    prefill = float(obj.get("prefill_sec") or obj.get("prefill_time") or 0.0)
                    if task is not None and idx is not None:
                        mp[(str(task), int(idx))] = prefill
                except: continue
    except Exception as e:
        print(f"[Warn] Failed to load baseline prefill map: {e}")
    return mp

def truncate_input_ids(
    input_ids: torch.LongTensor,
    max_prompt_tokens: int,
    mode: str = "headtail",
    head_tokens: int = 256,
) -> Tuple[torch.LongTensor, bool, int]:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    orig_len = int(input_ids.shape[1])
    if orig_len <= max_prompt_tokens:
        return input_ids, False, orig_len

    if mode == "skip":
        return input_ids, True, orig_len
    if max_prompt_tokens <= 8:
        return input_ids[:, -max_prompt_tokens:], True, orig_len
    if mode == "left":
        return input_ids[:, -max_prompt_tokens:], True, orig_len

    ht = min(head_tokens, max_prompt_tokens // 2)
    tail_len = max_prompt_tokens - ht
    new_ids = torch.cat([input_ids[:, :ht], input_ids[:, -tail_len:]], dim=1)
    return new_ids, True, orig_len


# ======================
# 2. Core Logic
# ======================

def eagle3_forward_one_prompt(
    ea_model: EaModel,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    max_length: int,
) -> Tuple[torch.LongTensor, Dict[str, Any], int]:
    
    device = next(ea_model.parameters()).device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    prompt_len = int(input_ids.shape[1])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.time()

    with torch.no_grad():
        # ea_model will now pick up the eos_token_id from base_model.generation_config
        seq = ea_model.eagenerate(
            input_ids.to(device),
            temperature=0.0,
            max_new_tokens=int(max_new_tokens),
            max_length=int(max_length),
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = time.time()

    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    elif seq.dim() > 2:
        seq = seq[0].unsqueeze(0)

    full_len = int(seq.shape[1])
    raw_new_tokens = max(full_len - prompt_len, 0)
    decode_tokens = max(raw_new_tokens - 1, 0)
    total_time = max(t_end - t_start, 1e-6)

    stats = {
        "prefill_time": 0.0,
        "decode_time": float(total_time),
        "raw_new_tokens": int(raw_new_tokens),
        "decode_tokens": int(decode_tokens),
    }
    return seq, stats, prompt_len


# ======================
# 3. Main Loop
# ======================

def run_eagle3(args):
    set_seed(args.seed)

    # Terminators are handled inside here
    tokenizer, ea_model, main_device = load_eagle3_and_tokenizer(args.base_model_path, args.ea_model_path)

    baseline_prefill = load_baseline_prefill_map(args.baseline_prefill_json)
    if not baseline_prefill:
        print("[Config] No baseline prefill map. Will use Total Time as Decode Time (Conservative).")

    tasks = FINE_TASKS if ("all" in args.tasks) else args.tasks

    # Warmup
    try:
        print("[Warmup] EAGLE-3 ...")
        w_enc = encode_prompt(tokenizer, "Warmup", main_device)
        _ = ea_model.eagenerate(
            w_enc["input_ids"], 
            max_new_tokens=8, 
            temperature=0.0, 
            max_length=int(args.max_length)
        )
    except Exception as e:
        print(f"[Warmup] Warning: Warmup failed ({e}), continuing...")

    slack = 16 
    if args.max_length <= args.max_new_tokens + slack:
        raise ValueError(f"--max-length({args.max_length}) too small for --max-new-tokens({args.max_new_tokens})")
    
    max_prompt_tokens = int(args.max_length) - int(args.max_new_tokens) - slack

    with open(args.output, "w", encoding="utf-8") as fout:
        for task in tasks:
            print(f"\n=== [EAGLE-3] Task: {task} ===")
            prompts = get_dataset_prompts(args.question_path, task, args.samples_per_task)

            for idx, prompt in enumerate(prompts, start=1):
                print(f"  - [{task}] sample {idx}/{len(prompts)}", end="")

                enc = encode_prompt(tokenizer, prompt, main_device)
                input_ids = enc["input_ids"]
                input_ids2, truncated, orig_len = truncate_input_ids(
                    input_ids,
                    max_prompt_tokens=max_prompt_tokens,
                    mode=args.truncate_mode,
                    head_tokens=args.trunc_head_tokens,
                )

                if args.truncate_mode == "skip" and truncated and orig_len > max_prompt_tokens:
                    print(f" (orig_prompt={orig_len}, truncated=True) -> SKIP")
                    continue

                used_len = int(input_ids2.shape[1])
                print(f" (orig={orig_len}, used={used_len})")

                # forward
                seq, stats, prompt_len = eagle3_forward_one_prompt(
                    ea_model=ea_model,
                    input_ids=input_ids2,
                    max_new_tokens=args.max_new_tokens,
                    max_length=args.max_length
                )

                total_time = stats["decode_time"]
                approx_prefill = 0.0
                if (not truncated) and baseline_prefill:
                    approx_prefill = baseline_prefill.get((task, idx), 0.0)

                real_decode_time = max(total_time - approx_prefill, 1e-6)
                stats["prefill_time"] = approx_prefill
                stats["decode_time"] = real_decode_time

                decode_tokens = stats["decode_tokens"]
                throughput = decode_tokens / real_decode_time if decode_tokens > 0 else 0.0

                try:
                    decoded_text = tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True)
                except Exception:
                    decoded_text = ""

                rec = {
                    "task": task,
                    "sample_idx": idx,
                    "prompt": prompt,
                    "output": decoded_text,
                    "prompt_tokens": int(prompt_len),
                    "prompt_tokens_orig": int(orig_len),
                    "prompt_truncated": bool(truncated),
                    "new_tokens": int(decode_tokens),
                    "raw_new_tokens": int(stats["raw_new_tokens"]),
                    "prefill_sec": float(approx_prefill),
                    "decode_sec": float(real_decode_time),
                    "throughput": float(throughput),
                    "engine": "eagle3_official",
                }

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(f"    Dec: {real_decode_time:.2f}s | Tok: {decode_tokens} | TP: {throughput:.2f} t/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--ea-model-path", type=str, default=DEFAULT_EA_MODEL_PATH)
    parser.add_argument("--baseline-prefill-json", type=str, default="")
    parser.add_argument("--question-path", type=str, default=SPEC_BENCH_QUESTION_PATH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_EAGLE_MAX_LENGTH)
    parser.add_argument("--truncate-mode", type=str, default=DEFAULT_TRUNCATE_MODE)
    parser.add_argument("--trunc-head-tokens", type=int, default=DEFAULT_TRUNC_HEAD_TOKENS)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_eagle3(args)
