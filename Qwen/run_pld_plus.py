#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PLD+ (h) runner (paper-faithful core) + your benchmarking scaffold:
- data loading: data_utils.TASK_CHOICES / get_dataset_prompts
- timing: prefill_time + decode_time (with cuda sync)
- throughput / ideal_speedup computed like your script style
- output: jsonl per sample

Core algorithm follows PLD+ (h) exactly as your provided snippet:
  - P = { j | x_j = x_t, j < t }
  - j* = argmax_{j in P} cos(H_{j-1}^(l), H_{t-1}^(l))
  - draft: copy x_{j*+1..j*+K} from context
  - verification: single forward on [last_token] + draft => K+1 dists; accept until mismatch; discard rest
  - keep last_token outside prefix cache; crop cache to prefix_len after accepting
"""

import argparse
import json
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import TASK_CHOICES, get_dataset_prompts


# ======================
# Defaults
# ======================

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"
DEFAULT_GPU_ID = 0

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9


# ======================
# Repro
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# Stop tokens
# ======================

def get_stop_token_ids(model, tokenizer) -> List[int]:
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

    if not eos_ids and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    # Try adding some chat/end tokens if present
    try:
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_str in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(tid, int) and tid >= 0 and (unk_id is None or tid != unk_id):
                eos_ids.append(int(tid))
    except Exception:
        pass

    # unique preserve order
    return list(dict.fromkeys(eos_ids))


# ======================
# Loading (keep your style)
# ======================

def load_model_and_tokenizer(
    model_path: str,
    gpu_id: int
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    print(f"[Model] Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": gpu_id},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return tokenizer, model, device


def encode_prompt(
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
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
    else:
        enc_dict = tokenizer(text, return_tensors="pt")

    enc_dict = {k: v.to(device) for k, v in enc_dict.items()}
    if "attention_mask" not in enc_dict:
        enc_dict["attention_mask"] = torch.ones_like(enc_dict["input_ids"], device=device)
    return enc_dict


# ======================
# Sampling helpers (paper-core version)
# ======================

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering on logits. logits: (vocab,)
    """
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    cutoff = cumprobs > top_p
    cutoff[..., 0] = False

    sorted_logits[cutoff] = -float("inf")
    filtered = torch.empty_like(logits)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def pick_token(
    logits_1d: torch.Tensor,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> int:
    """
    logits_1d: (vocab,)
    """
    if mode == "greedy":
        return int(torch.argmax(logits_1d).item())

    if temperature is None or temperature <= 0:
        temperature = 1.0
    logits = logits_1d / float(temperature)
    logits = top_p_filtering(logits, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ======================
# Past KV cache cropping (same as your PLD+ snippet)
# ======================

def _crop_legacy_past(past_key_values, new_len: int):
    cropped = []
    for layer in past_key_values:
        if not (isinstance(layer, (tuple, list)) and len(layer) == 2):
            cropped.append(layer)
            continue
        k, v = layer
        if k is None or v is None:
            cropped.append((k, v))
            continue

        if k.dim() == 4:
            if k.size(2) >= new_len and v.size(2) >= new_len:
                k2 = k[:, :, :new_len, :]
                v2 = v[:, :, :new_len, :]
            elif k.size(1) >= new_len and v.size(1) >= new_len:
                k2 = k[:, :new_len, :, :]
                v2 = v[:, :new_len, :, :]
            else:
                k2 = k[..., :new_len, :]
                v2 = v[..., :new_len, :]
        else:
            k2 = k[..., :new_len, :]
            v2 = v[..., :new_len, :]
        cropped.append((k2, v2))
    return tuple(cropped)


def crop_past(past_key_values, new_len: int):
    if past_key_values is None:
        return None

    # Newer cache objects
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        legacy = _crop_legacy_past(legacy, new_len)
        try:
            from transformers.cache_utils import DynamicCache  # type: ignore
            return DynamicCache.from_legacy_cache(legacy)
        except Exception:
            return legacy

    # Legacy tuple
    if isinstance(past_key_values, tuple):
        return _crop_legacy_past(past_key_values, new_len)

    return past_key_values


# ======================
# PLD+ (h) core (paper-faithful)
# ======================

@dataclass
class PLDPlusConfig:
    max_draft_tokens: int = 32          # K
    layer_index: int = 9               # l in Eq(3)
    cos_threshold: Optional[float] = None  # optional threshold variant
    mode: str = "greedy"               # "greedy" or "sample"
    temperature: float = 1.0
    top_p: Optional[float] = None


@torch.no_grad()
def pld_plus_generate_ids(
    model: AutoModelForCausalLM,
    input_ids_1xT: torch.Tensor,
    max_new_tokens: int,
    stop_ids_set: set,
    cfg: PLDPlusConfig,
) -> Tuple[List[int], Dict[str, Any], int]:
    """
    Returns:
      - full token ids list (prompt + generated)
      - stats dict (timing + forward counts + draft stats)
      - prompt_len
    """
    device = input_ids_1xT.device
    prompt_ids_list = input_ids_1xT[0].tolist()
    prompt_len = len(prompt_ids_list)

    if prompt_len < 1:
        raise ValueError("Prompt is empty after tokenization.")

    # output token list (includes prompt + generated)
    out_ids: List[int] = prompt_ids_list[:]
    target_total_len = prompt_len + int(max_new_tokens)

    # state:
    prefix_ids: List[int] = out_ids[:-1]
    last_token: int = out_ids[-1]

    past_prefix = None
    h_prefix: List[torch.Tensor] = []  # CPU tensors

    N_forward_prefill = 0
    N_forward_decode = 0
    draft_attempts = 0
    draft_accepted_tokens = 0

    # ----------------------
    # Prefill: build cache + hidden vectors for prefix (paper core)
    # ----------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_start = time.time()

    if len(prefix_ids) > 0:
        prefix_tensor = torch.tensor([prefix_ids], device=device, dtype=torch.long)
        out = model(
            input_ids=prefix_tensor,
            use_cache=True,
            output_hidden_states=True,
        )
        N_forward_prefill += 1
        past_prefix = out.past_key_values

        l = cfg.layer_index
        if l < 0 or l >= len(out.hidden_states):
            raise ValueError(f"layer_index={l} out of range for hidden_states len={len(out.hidden_states)}")

        hs_l = out.hidden_states[l][0]  # (seq_len, hidden_dim)
        for i in range(hs_l.size(0)):
            h_prefix.append(hs_l[i].detach().cpu())
    else:
        # empty prefix cache; still OK
        past_prefix = None
        h_prefix = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.time() - t_prefill_start

    # ----------------------
    # Decode loop
    # ----------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode_start = time.time()

    while len(out_ids) < target_total_len and (last_token not in stop_ids_set):
        # -------- Drafting step (paper) --------
        # P = {j | x_j = x_t, j < t}, with j>0 to have j-1
        positions = [j for j, tok in enumerate(prefix_ids) if tok == last_token and j > 0]

        draft_tokens: List[int] = []

        if positions and len(prefix_ids) > 0 and len(h_prefix) == len(prefix_ids):
            # Compare H_{j-1}^(l) with H_{t-1}^(l)
            anchor = h_prefix[-1]
            anchor_norm = anchor / (anchor.norm(p=2) + 1e-8)

            best_j = None
            best_sim = -1e9

            for j in positions:
                cand = h_prefix[j - 1]
                cand_norm = cand / (cand.norm(p=2) + 1e-8)
                sim = float(torch.dot(cand_norm, anchor_norm).item())
                if cfg.cos_threshold is not None and sim < float(cfg.cos_threshold):
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_j = j

            if best_j is not None:
                start = best_j + 1
                end = min(len(prefix_ids), start + int(cfg.max_draft_tokens))
                draft_tokens = prefix_ids[start:end]

        if len(draft_tokens) > 0:
            draft_attempts += 1

        # -------- Verification step (paper) --------
        tokens_to_feed = [last_token] + draft_tokens
        feed = torch.tensor([tokens_to_feed], device=device, dtype=torch.long)

        out = model(
            input_ids=feed,
            past_key_values=past_prefix,
            use_cache=True,
            output_hidden_states=True,
        )
        N_forward_decode += 1

        logits = out.logits[0]  # (1+K, vocab)
        past_all = out.past_key_values

        l = cfg.layer_index
        hs_l = out.hidden_states[l][0]  # (1+K, hidden_dim)

        K = len(draft_tokens)

        # K+1 predictions
        preds: List[int] = []
        for i in range(K + 1):
            preds.append(
                pick_token(
                    logits_1d=logits[i],
                    mode=cfg.mode,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                )
            )

        # accept until first mismatch; discard rest
        accept_len = 0
        for i in range(K):
            if preds[i] == draft_tokens[i]:
                accept_len += 1
            else:
                break

        # length cap (not part of algorithm; just to respect max_new_tokens)
        remaining_to_target = target_total_len - len(out_ids)
        if remaining_to_target <= 0:
            break
        # we will add accepted_drafts + next_token (at least 1)
        max_accept = max(0, remaining_to_target - 1)
        if accept_len > max_accept:
            accept_len = max_accept

        accepted_drafts = draft_tokens[:accept_len]
        draft_accepted_tokens += int(accept_len)

        # next token:
        next_token = preds[accept_len] if accept_len < K else preds[K]

        # -------- State update (paper) --------
        prefix_extension = [last_token] + accepted_drafts

        # append hidden vectors for accepted input tokens (hs_l[0..accept_len])
        for i in range(1 + accept_len):
            h_prefix.append(hs_l[i].detach().cpu())

        prefix_ids.extend(prefix_extension)
        out_ids.extend(accepted_drafts)
        out_ids.append(next_token)

        new_prefix_len = len(prefix_ids)
        past_prefix = crop_past(past_all, new_prefix_len)

        last_token = next_token

        if last_token in stop_ids_set:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_time = time.time() - t_decode_start

    raw_new = max(len(out_ids) - prompt_len, 0)

    stats = {
        "prefill_time": float(prefill_time),
        "decode_time": float(decode_time),
        "N_forward_total": int(N_forward_prefill + N_forward_decode),
        "N_forward_decode": int(N_forward_decode),
        "N_tokens_decode": int(raw_new),
        "draft_attempts": int(draft_attempts),
        "draft_accepted_tokens": int(draft_accepted_tokens),
    }
    return out_ids, stats, prompt_len


# ======================
# Main runner (keep your scaffold)
# ======================

def run_pld_plus(args):
    set_seed(args.seed)
    tokenizer, model, device = load_model_and_tokenizer(args.model_path, args.gpu_id)

    stop_ids = get_stop_token_ids(model, tokenizer)
    stop_ids_set = set(stop_ids)

    print(f"\n[Config] Stop token IDs: {stop_ids}")
    print(f"[Config] mode={'sample' if args.do_sample else 'greedy'}")
    if args.do_sample:
        print(f"[Config] temperature={args.temperature}, top_p={args.top_p}")
    print(f"[Config] layer_idx={args.layer_idx}, K={args.K}, cos_threshold={args.cos_threshold}")

    print("\n[Warmup] PLD+ ...")
    warmup_enc = encode_prompt(tokenizer, "Warmup.", device)
    with torch.no_grad():
        _ = model(input_ids=warmup_enc["input_ids"], use_cache=True, output_hidden_states=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[Warmup done]\n")

    tasks: List[str] = TASK_CHOICES if "all" in args.tasks else args.tasks

    total_raw = 0
    total_decode_time = 0.0
    total_fwd_decode = 0
    total_attempts = 0
    total_accepted = 0

    cfg = PLDPlusConfig(
        max_draft_tokens=args.K,
        layer_index=args.layer_idx,
        cos_threshold=args.cos_threshold,
        mode="sample" if args.do_sample else "greedy",
        temperature=args.temperature,
        top_p=args.top_p,
    )

    with open(args.output, "w", encoding="utf-8") as fout:
        for task in tasks:
            print(f"\n=== [PLD+] Task: {task} ===")
            prompts = get_dataset_prompts(task, args.samples_per_task)

            for idx, prompt in enumerate(prompts, start=1):
                print(f"  - [{task}] sample {idx}/{len(prompts)}")
                enc = encode_prompt(tokenizer, prompt, device)

                out_ids_list, stats, prompt_len = pld_plus_generate_ids(
                    model=model,
                    input_ids_1xT=enc["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    stop_ids_set=stop_ids_set,
                    cfg=cfg,
                )

                seq = torch.tensor(out_ids_list, device="cpu", dtype=torch.long)
                raw_new = max(int(seq.shape[0]) - prompt_len, 0)

                decode_time = stats["decode_time"]
                prefill_time = stats["prefill_time"]

                # throughput: tokens / decode_time  (same style; here we don't subtract "first token")
                throughput = (raw_new / decode_time) if (decode_time > 0 and raw_new > 0) else 0.0
                ideal_speedup = (raw_new / stats["N_forward_decode"]) if stats["N_forward_decode"] > 0 else 0.0

                total_raw += raw_new
                total_decode_time += decode_time
                total_fwd_decode += stats["N_forward_decode"]
                total_attempts += stats["draft_attempts"]
                total_accepted += stats["draft_accepted_tokens"]

                decoded = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)

                rec = {
                    "task": task,
                    "sample_idx": idx,
                    "prompt": prompt,
                    "output": decoded,
                    "prompt_tokens": prompt_len,
                    "raw_new_tokens": raw_new,
                    "prefill_sec": prefill_time,
                    "decode_sec": decode_time,
                    "throughput": throughput,
                    "ideal_speedup": ideal_speedup,
                    "N_forward_total": stats["N_forward_total"],
                    "N_forward_decode": stats["N_forward_decode"],
                    "draft_attempts": stats["draft_attempts"],
                    "draft_accepted_tokens": stats["draft_accepted_tokens"],
                    "K": args.K,
                    "layer_idx": args.layer_idx,
                    "cos_threshold": args.cos_threshold,
                    "mode": "sample" if args.do_sample else "greedy",
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                    "model_path": args.model_path,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(
                    f"    Prefill: {prefill_time:.4f}s | Decode: {decode_time:.4f}s | "
                    f"RawNew: {raw_new} | TP: {throughput:.2f} tok/s | "
                    f"Ideal: {ideal_speedup:.2f} tok/step | "
                    f"Attempts: {stats['draft_attempts']} | DraftAcc: {stats['draft_accepted_tokens']}"
                )

    overall_tp = (total_raw / total_decode_time) if total_decode_time > 0 else 0.0
    overall_speedup = (total_raw / total_fwd_decode) if total_fwd_decode > 0 else 0.0
    acc_ratio = (total_accepted / (total_attempts * args.K)) if (total_attempts > 0 and args.K > 0) else 0.0

    print("\n" + "=" * 60)
    print("PLD+ Summary")
    print("=" * 60)
    print(f"Mode                : {'Sample' if args.do_sample else 'Greedy'}")
    if args.do_sample:
        print(f"  Temperature       : {args.temperature}")
        print(f"  Top-p             : {args.top_p}")
    print(f"K                   : {args.K}")
    print(f"Layer idx           : {args.layer_idx}")
    print(f"Cos threshold       : {args.cos_threshold}")
    print(f"Total Raw Tokens    : {total_raw}")
    print(f"Total Decode Time   : {total_decode_time:.3f} s")
    print(f"Overall Throughput  : {overall_tp:.2f} tok/s")
    print(f"Overall Speedup     : {overall_speedup:.2f} tok/step")
    print(f"Draft Acc Ratio     : {acc_ratio:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--layer-idx", type=int, default=9)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--cos-threshold", type=float, default=None)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)

    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_pld_plus(args)
