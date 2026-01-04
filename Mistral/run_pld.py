#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

python run_pld.py \
  --model-path /root/autodl-tmp/models/Mistral-Small-3.1-24B-Instruct-2503 \
  --tasks coding \
  --samples-per-task 80 \
  --max-new-tokens 2048 \
  --n-gram 4 \
  --K 16 \
  --output pld_greedy.jsonl


python run_pld.py \
  --model-path /root/autodl-tmp/models/Mistral-Small-3.1-24B-Instruct-2503 \
  --tasks math \
  --samples-per-task 80 \
  --max-new-tokens 2048 \
  --n-gram 4 \
  --K 8 \
  --do-sample \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k 0 \
  --output pld.jsonl

"""

import argparse
import json
import time
import random
import threading
import os
from typing import Tuple, List, Dict, Any, Optional, Set

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    TextIteratorStreamer,
)

from data_utils import TASK_CHOICES, get_dataset_prompts


# ======================
# Defaults
# ======================
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_GPU_ID = 0


# ======================
# Utils
# ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _needs_mistral_regex_fix(model_path: str) -> bool:
    name = os.path.basename(model_path).lower()
    # covers: mistral, mistral-small-3.1, ministral, etc.
    return "mistral" in name


def _safe_from_pretrained(cls, model_path: str, **kwargs):
    """
    Make from_pretrained robust across transformers versions:
    - try as-is
    - if dtype unsupported, fallback to torch_dtype
    - if fix_mistral_regex unsupported, retry without it
    """
    try:
        return cls.from_pretrained(model_path, **kwargs)
    except TypeError as e:
        msg = str(e)
        # handle dtype vs torch_dtype
        if "dtype" in msg and "unexpected keyword argument" in msg and "dtype" in kwargs:
            kw = dict(kwargs)
            kw["torch_dtype"] = kw.pop("dtype")
            return cls.from_pretrained(model_path, **kw)
        # handle fix_mistral_regex unsupported
        if "fix_mistral_regex" in msg and "unexpected keyword argument" in msg:
            kw = dict(kwargs)
            kw.pop("fix_mistral_regex", None)
            return cls.from_pretrained(model_path, **kw)
        raise


def get_stop_token_ids(model, tokenizer) -> List[int]:
    """
    Safe EOS ids:
    - trust generation_config.eos_token_id + tokenizer.eos_token_id
    - DO NOT filter pad_token_id (pad may equal eos)
    - filter BOS only
    """
    ids: List[int] = []

    def add(x):
        if isinstance(x, int) and x >= 0:
            ids.append(int(x))

    eos = None
    if getattr(model, "generation_config", None) is not None:
        eos = getattr(model.generation_config, "eos_token_id", None)

    if isinstance(eos, int):
        add(eos)
    elif isinstance(eos, (list, tuple)):
        for t in eos:
            add(t)

    add(getattr(tokenizer, "eos_token_id", None))

    if not ids:
        # final fallback
        unk = getattr(tokenizer, "unk_token_id", None)
        for s in [getattr(tokenizer, "eos_token", None), "</s>"]:
            if isinstance(s, str):
                tid = tokenizer.convert_tokens_to_ids(s)
                if isinstance(tid, int) and tid >= 0 and (unk is None or tid != unk):
                    add(tid)
        if not ids:
            add(2)

    bos = getattr(tokenizer, "bos_token_id", None)
    out: List[int] = []
    seen = set()
    for x in ids:
        if x in seen:
            continue
        if bos is not None and x == int(bos):
            continue
        out.append(int(x))
        seen.add(int(x))
    return out


class ChatApplier:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        raise AttributeError("No apply_chat_template found on processor/tokenizer.")


def load_model_tokenizer_processor(model_path: str, gpu_id: int):
    print(f"[Model] Loading {model_path} (HF PLD + Streamer) ...")

    fix_regex = _needs_mistral_regex_fix(model_path)

    # Load processor first (for Mistral3 chat template). If it has tokenizer, reuse it -> avoid double warnings.
    processor = None
    try:
        processor = _safe_from_pretrained(
            AutoProcessor,
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True if fix_regex else False,
        )
    except Exception:
        processor = None

    tokenizer = None
    if processor is not None and hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        tokenizer = processor.tokenizer

    if tokenizer is None:
        tokenizer = _safe_from_pretrained(
            AutoTokenizer,
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True if fix_regex else False,
        )

    # ensure pad exists
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # device + dtype
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = {"": gpu_id}
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        device_map = None

    # Prefer Mistral3 loader; fallback to causal LM
    model = None
    if device_map is not None:
        try:
            model = _safe_from_pretrained(
                AutoModelForImageTextToText,
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map,
                dtype=dtype,
            )
        except Exception as e:
            print(f"[WARN] AutoModelForImageTextToText failed -> fallback AutoModelForCausalLM: {e}")
            model = _safe_from_pretrained(
                AutoModelForCausalLM,
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map,
                dtype=dtype,
            )
    else:
        # CPU
        try:
            model = _safe_from_pretrained(
                AutoModelForImageTextToText,
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                dtype=dtype,
            ).to(device)
        except Exception:
            model = _safe_from_pretrained(
                AutoModelForCausalLM,
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                dtype=dtype,
            ).to(device)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    chat = ChatApplier(tokenizer, processor)
    return tokenizer, processor, model, device, chat


def encode_prompt(tokenizer, chat: ChatApplier, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Mistral3-correct:
    chat_template -> string -> tokenizer()
    """
    messages = [{"role": "user", "content": text}]
    try:
        prompt_text = chat.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(prompt_text, str) and len(prompt_text) > 0:
            enc = tokenizer(prompt_text, return_tensors="pt")
        else:
            enc = tokenizer(text, return_tensors="pt")
    except Exception:
        enc = tokenizer(text, return_tensors="pt")

    enc = {k: v.to(device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    return enc


# ======================
# Core: PLD with Streamer Timing
# ======================
def pld_forward_one_prompt(
    model,
    tokenizer,
    enc: Dict[str, torch.Tensor],
    n_gram: int,
    K: int,
    max_new_tokens: int,
    stop_token_ids: List[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[torch.LongTensor, Dict[str, Any], int]:

    device = enc["input_ids"].device
    prompt_len = int(enc["input_ids"].shape[1])

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    eos_arg: Any = stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids

    gen_kwargs = dict(
        **enc,
        max_new_tokens=int(max_new_tokens),
        streamer=streamer,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=eos_arg,
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),  # 0 disables
        )
    else:
        gen_kwargs.update(do_sample=False, num_beams=1)

    if K > 0 and n_gram > 0:
        gen_kwargs["prompt_lookup_num_tokens"] = int(K)
        gen_kwargs["max_matching_ngram_size"] = int(n_gram)

    result_container: Dict[str, Any] = {}

    def _generate_thread():
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        # some models return GenerateOutput with .sequences
        seqs = out.sequences if hasattr(out, "sequences") else out
        result_container["sequences"] = seqs

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    thread = threading.Thread(target=_generate_thread)
    thread.start()

    t_first_token: Optional[float] = None
    for _chunk in streamer:
        if t_first_token is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_first_token = time.time()

    thread.join()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.time()

    if t_first_token is None:
        t_first_token = t_end

    prefill_time = float(t_first_token - t_start)
    decode_time = float(max(t_end - t_first_token, 1e-6))

    seq = result_container["sequences"][0].to(device)
    full_len = int(seq.shape[0])
    raw_new_tokens = max(full_len - prompt_len, 0)
    decode_tokens = max(raw_new_tokens - 1, 0)

    stats: Dict[str, Any] = {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "raw_new_tokens": int(raw_new_tokens),
        "decode_tokens": int(decode_tokens),
        "N_forward_total": 0,
        "N_forward_decode": 0,
        "N_tokens_decode": int(decode_tokens),
        "draft_attempts": 0,
        "draft_accepted_tokens": 0,
    }

    return seq.unsqueeze(0), stats, prompt_len


# ======================
# Main Loop
# ======================
def run_pld_specbench(args):
    set_seed(args.seed)

    tokenizer, processor, model, device, chat = load_model_tokenizer_processor(args.model_path, args.gpu_id)
    stop_ids = get_stop_token_ids(model, tokenizer)

    print(f"\n[Config] Stop token IDs: {stop_ids}")
    print(f"[Config] PLD: n_gram={args.n_gram}, K={args.K}")
    print(f"[Config] Sampling: do_sample={args.do_sample}")
    if args.do_sample:
        print(f"[Config] temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")

    # Warmup
    print("\n[Warmup] HF PLD ...")
    warmup_enc = encode_prompt(tokenizer, chat, "Warmup prompt.", device)
    eos_arg: Any = stop_ids[0] if len(stop_ids) == 1 else stop_ids

    warmup_kwargs = dict(
        **warmup_enc,
        max_new_tokens=8,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=eos_arg,
    )
    if args.do_sample:
        warmup_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    else:
        warmup_kwargs.update(do_sample=False)

    if args.K > 0 and args.n_gram > 0:
        warmup_kwargs["prompt_lookup_num_tokens"] = min(4, int(args.K))
        warmup_kwargs["max_matching_ngram_size"] = min(2, int(args.n_gram))

    with torch.no_grad():
        _ = model.generate(**warmup_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[Warmup done]\n")

    tasks: List[str] = TASK_CHOICES if "all" in args.tasks else args.tasks

    total_raw_new_tokens = 0
    total_decode_tokens = 0
    total_decode_time = 0.0

    with open(args.output, "w", encoding="utf-8") as fout:
        for task in tasks:
            print(f"\n=== [PLD-HF] Task: {task} ===")
            prompts = get_dataset_prompts(task, args.samples_per_task)

            for idx, prompt in enumerate(prompts, start=1):
                print(f"  - [{task}] sample {idx}/{len(prompts)}")
                enc = encode_prompt(tokenizer, chat, prompt, device)

                seq_batch, stats, prompt_len = pld_forward_one_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    enc=enc,
                    n_gram=args.n_gram,
                    K=args.K,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_ids,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )

                seq = seq_batch[0]
                raw_new_tokens = stats["raw_new_tokens"]
                decode_tokens = stats["decode_tokens"]
                decode_time = stats["decode_time"]
                prefill_time = stats["prefill_time"]

                throughput = (decode_tokens / decode_time) if (decode_time > 0 and decode_tokens > 0) else 0.0

                total_raw_new_tokens += raw_new_tokens
                total_decode_tokens += decode_tokens
                total_decode_time += decode_time

                decoded_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)

                rec = {
                    "task": task,
                    "sample_idx": idx,
                    "prompt": prompt,
                    "output": decoded_text,
                    "prompt_tokens": int(prompt_len),
                    "new_tokens": int(decode_tokens),
                    "raw_new_tokens": int(raw_new_tokens),
                    "prefill_sec": float(prefill_time),
                    "decode_sec": float(decode_time),
                    "elapsed_sec": float(decode_time),
                    "throughput": float(throughput),
                    "ideal_speedup": 0.0,
                    "K_max": int(args.K),
                    "pld_n_gram": int(args.n_gram),
                    "do_sample": bool(args.do_sample),
                    "temperature": float(args.temperature) if args.do_sample else None,
                    "top_p": float(args.top_p) if args.do_sample else None,
                    "top_k": int(args.top_k) if args.do_sample else None,
                    "model_path": args.model_path,
                    "engine": "hf_pld_streamer",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(
                    f"    Prefill: {prefill_time:.4f}s | Decode: {decode_time:.4f}s | "
                    f"RawTok: {raw_new_tokens} | EffTok(N-1): {decode_tokens} | TP: {throughput:.2f} tok/s"
                )

    overall_tp = (total_decode_tokens / total_decode_time) if (total_decode_time > 0 and total_decode_tokens > 0) else 0.0

    print("\n========== HF PLD (Benchmark Summary) ==========")
    print(f"Model                  : {args.model_path}")
    print(f"Sampling Mode          : {'Yes' if args.do_sample else 'No (Greedy)'}")
    if args.do_sample:
        print(f"  Temperature          : {args.temperature}")
        print(f"  Top-p                : {args.top_p}")
        print(f"  Top-k                : {args.top_k}")
    print(f"Stop token IDs         : {stop_ids}")
    print(f"Total Raw Tokens       : {total_raw_new_tokens}")
    print(f"Total Eff Tokens (N-1) : {total_decode_tokens}")
    print(f"Total Decode Time      : {total_decode_time:.3f} s")
    print(f"Overall Throughput     : {overall_tp:.2f} tok/s")
    print("================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True, help="HF model path")
    parser.add_argument("--tasks", type=str, nargs="+", required=True, help="Task names or 'all'")
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n-gram", type=int, required=True, help="PLD ngram size")
    parser.add_argument("--K", type=int, required=True, help="PLD lookup window")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")

    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)

    parser.add_argument("--do-sample", action="store_true", default=False, help="Enable sampling (default: greedy)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_pld_specbench(args)
