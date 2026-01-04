#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

python run_pld.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --tasks code_edit \
  --samples-per-task 80 \
  --max-new-tokens 2048 \
  --n-gram 0 \
  --K 0 \
  --output pld_greedy.jsonl 


python run_pld.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --tasks all \
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
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from data_utils import TASK_CHOICES, get_dataset_prompts

# ======================
# 0. Global Configuration
# ======================

MODEL_PATH = "/root/autodl-tmp/models/gemma-2-27b-it"

# ✅ 新增：默认采样参数（基于你的实验结论）
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0  # 0 表示不使用 top-k


def print_runtime_diag(model):
    import torch

    print("\n========== [Runtime Diag] ==========")
    print("[Diag] torch:", torch.__version__, " torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("[Diag] GPU:", torch.cuda.get_device_name(0),
              "cap:", torch.cuda.get_device_capability(0))

    p = next(model.parameters())
    print("[Diag] main param device/dtype:", p.device, p.dtype)

    hf_map = getattr(model, "hf_device_map", None)
    if hf_map is None:
        print("[Diag] hf_device_map: <None>")
    else:
        kinds = sorted(set(hf_map.values()), key=str)
        print("[Diag] device map kinds:", kinds)
        if any(str(k).startswith("cpu") or str(k) == "disk" for k in kinds):
            print("[Diag][WARN] CPU/DISK offload detected!")

    backend = torch.backends.cuda
    flash = getattr(backend, "flash_sdp_enabled", lambda: None)()
    memef = getattr(backend, "mem_efficient_sdp_enabled", lambda: None)()
    math  = getattr(backend, "math_sdp_enabled", lambda: None)()
    print("[Diag] flash_sdp:", flash)
    print("[Diag] mem_efficient_sdp:", memef)
    print("[Diag] math_sdp:", math)
    print("====================================\n")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_stop_token_ids(model, tokenizer) -> List[int]:
    """
    Robustly fetch stop token IDs from model.generation_config.eos_token_id.
    """
    eos_ids: List[int] = []

    eos = None
    if getattr(model, "generation_config", None) is not None:
        eos = getattr(model.generation_config, "eos_token_id", None)

    if isinstance(eos, int):
        eos_ids.append(int(eos))
    elif isinstance(eos, (list, tuple)):
        for x in eos:
            if isinstance(x, int) and x >= 0:
                eos_ids.append(int(x))

    if not eos_ids and tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    dedup: List[int] = []
    seen = set()
    for x in eos_ids:
        if x not in seen:
            dedup.append(x)
            seen.add(x)

    return dedup


GPU_ID = 0


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    print(f"[Model] Loading {model_path} (HF PLD + KV + Streamer) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = torch.device(f"cuda:{GPU_ID}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return tokenizer, model, device


def encode_prompt(tokenizer: AutoTokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": text}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(enc, torch.Tensor):
        enc_dict = {"input_ids": enc}
    else:
        enc_dict = dict(enc)
    
    enc_dict = {k: v.to(device) for k, v in enc_dict.items()}
    if "attention_mask" not in enc_dict:
        enc_dict["attention_mask"] = torch.ones_like(enc_dict["input_ids"], device=device)
    return enc_dict


# ======================
# 2. Core Logic: PLD with Streamer Timing (Sampling Version)
# ======================

def pld_forward_one_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    enc: Dict[str, torch.Tensor],
    n_gram: int,
    K: int,
    max_new_tokens: int,
    stop_token_ids: List[int],
    # ✅ 新增采样参数
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Tuple[torch.LongTensor, Dict[str, Any], int]:
    
    device = enc["input_ids"].device
    prompt_len = int(enc["input_ids"].shape[1])

    # 1. Initialize Streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # 2. Prepare Generation Arguments
    gen_kwargs = dict(
        **enc,
        max_new_tokens=int(max_new_tokens),
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_token_ids,
    )

    # ✅ 修复：采样 vs 贪心
    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),  # ✅ 关键修复：直接传 0，HF 会正确禁用 top-k
        )
    else:
        gen_kwargs.update(
            do_sample=False,
            num_beams=1,
        )

    # PLD 参数（仅当 K > 0 时启用）
    if K > 0 and n_gram > 0:
        gen_kwargs["prompt_lookup_num_tokens"] = int(K)
        gen_kwargs["max_matching_ngram_size"] = int(n_gram)

    result_container: Dict[str, Any] = {}

    def _generate_thread():
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        result_container["sequences"] = out

    # 3. Execution & Timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_start = time.time()

    thread = threading.Thread(target=_generate_thread)
    thread.start()

    t_first_token = None
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

    # 4. Calculate Durations
    prefill_time = t_first_token - t_start
    decode_time = max(t_end - t_first_token, 1e-6)

    # 5. Process Output
    seq = result_container["sequences"][0].to(device)
    full_len = int(seq.shape[0])
    raw_new_tokens = max(full_len - prompt_len, 0)
    decode_tokens = max(raw_new_tokens - 1, 0)

    stats: Dict[str, Any] = {
        "prefill_time": float(prefill_time),
        "decode_time": float(decode_time),
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
# 3. Main Loop
# ======================

def run_pld_specbench(args):
    set_seed(args.seed)
    tokenizer, model, main_device = load_model_and_tokenizer(args.model_path)
    
    stop_ids = get_stop_token_ids(model, tokenizer)
    
    # ✅ 打印采样配置
    print(f"\n[Config] Sampling: do_sample={args.do_sample}")
    if args.do_sample:
        print(f"[Config] temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"[Config] Stop token IDs: {stop_ids}")
    print(f"[Config] PLD: n_gram={args.n_gram}, K={args.K}")

    # --- Warmup ---
    print("\n[Warmup] HF PLD + KV ...")
    warmup_enc = encode_prompt(tokenizer, "Warmup prompt.", main_device)
    
    warmup_kwargs = dict(
        **warmup_enc,
        max_new_tokens=8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
    )
    # ✅ 修复：Warmup 也要用相同的采样参数
    if args.do_sample:
        warmup_kwargs.update(
            do_sample=True, 
            temperature=args.temperature, 
            top_p=args.top_p,
            top_k=args.top_k  # ✅ 关键修复：显式传 top_k
        )
    else:
        warmup_kwargs.update(do_sample=False)
    
    if args.K > 0:
        warmup_kwargs["prompt_lookup_num_tokens"] = 4
        warmup_kwargs["max_matching_ngram_size"] = 2
    
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
                enc = encode_prompt(tokenizer, prompt, main_device)

                seq_batch, stats, prompt_len = pld_forward_one_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    enc=enc,
                    n_gram=args.n_gram,
                    K=args.K,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_ids,
                    # ✅ 传递采样参数
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

                throughput = (
                    decode_tokens / decode_time if decode_time > 0 and decode_tokens > 0 else 0.0
                )

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
                    "N_forward_total": int(stats["N_forward_total"]),
                    "N_forward_decode": int(stats["N_forward_decode"]),
                    "draft_attempts": int(stats["draft_attempts"]),
                    "draft_accepted_tokens": int(stats["draft_accepted_tokens"]),
                    "K_max": int(args.K),
                    "sim_threshold": 0.0,
                    "layer_idx": -1,
                    "pld_n_gram": int(args.n_gram),
                    # ✅ 记录采样参数
                    "do_sample": args.do_sample,
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                    "top_k": args.top_k if args.do_sample else None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(
                    f"    Prefill: {prefill_time:.4f}s | "
                    f"Decode: {decode_time:.4f}s | "
                    f"RawTok: {raw_new_tokens} | "
                    f"EffTok(N-1): {decode_tokens} | "
                    f"TP: {throughput:.2f} tok/s"
                )

    overall_tp = (
        total_decode_tokens / total_decode_time
        if total_decode_time > 0 and total_decode_tokens > 0
        else 0.0
    )

    print("\n========== HF PLD (Benchmark Summary) ==========")
    print(f"Sampling Mode           : {'Yes' if args.do_sample else 'No (Greedy)'}")
    if args.do_sample:
        print(f"  Temperature           : {args.temperature}")
        print(f"  Top-p                 : {args.top_p}")
        print(f"  Top-k                 : {args.top_k}")
    print(f"Total Raw Tokens        : {total_raw_new_tokens}")
    print(f"Total Eff Tokens (N-1)  : {total_decode_tokens}")
    print(f"Total Decode Time       : {total_decode_time:.3f} s")
    print(f"Overall Throughput      : {overall_tp:.2f} tok/s")
    print("================================================")


# ======================
# 4. Entry Point
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"HF model path (default: {MODEL_PATH})",
    )
    parser.add_argument("--tasks", type=str, nargs="+", required=True, help="Task names or 'all'")
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n-gram", type=int, required=True, help="PLD ngram size")
    parser.add_argument("--K", type=int, required=True, help="PLD lookup window")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    
    parser.add_argument("--do-sample", action="store_true", default=False,
                        help="Enable sampling (default: greedy)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,  # ✅ 改成下划线
                        help=f"Top-p (nucleus) sampling (default: {DEFAULT_TOP_P})")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,    # ✅ 改成下划线
                        help=f"Top-k sampling, 0=disabled (default: {DEFAULT_TOP_K})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    run_pld_specbench(args)
