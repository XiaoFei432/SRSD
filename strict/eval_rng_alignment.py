#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_rng_alignment.py

Implements:
(a) RNG-aligned sampling trajectory equality between AR and SRSD.
(b) Per-position distribution distance between AR next-token distribution and SRSD "actually used" distribution.

Usage example:
python eval_rng_alignment.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --tasks summarization \
  --samples-per-task 10 \
  --max-new-tokens 512 \
  --layer-idx 22 \
  --K 16 \
  --retrieval-topk 10 \
  --sim-threshold 0.0 \
  --global-seed 42 \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k 0 \
  --gpu-id 0 \
  --output alignment_eval.jsonl

"""

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from data_utils import TASK_CHOICES, get_dataset_prompts

from aligned_sampling import aligned_sample_token, sparse_dist_metrics, DistMetrics


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ---------------------------
# Helpers: stop tokens, prompt encoding, cache crop
# ---------------------------

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

    try:
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_str in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(tid, int) and tid >= 0 and (unk_id is None or tid != unk_id):
                eos_ids.append(int(tid))
    except Exception:
        pass

    # dedup
    return list(dict.fromkeys(eos_ids))


def encode_prompt(tokenizer: AutoTokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": text}]
        enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        enc_dict = {"input_ids": enc} if isinstance(enc, torch.Tensor) else dict(enc)
    else:
        enc_dict = tokenizer(text, return_tensors="pt")

    enc_dict = {k: v.to(device) for k, v in enc_dict.items()}
    if "attention_mask" not in enc_dict:
        enc_dict["attention_mask"] = torch.ones_like(enc_dict["input_ids"], device=device)
    return enc_dict


def crop_past_key_values(past_kv, max_length: int):
    if past_kv is None:
        return None
    if isinstance(past_kv, DynamicCache):
        if hasattr(past_kv, "crop"):
            past_kv.crop(max_length)
        return past_kv
    if isinstance(past_kv, tuple):
        new_cache = []
        for layer_past in past_kv:
            new_layer = []
            for kv in layer_past:
                if kv is None:
                    new_layer.append(None)
                    continue
                seq_dim = -2 if kv.dim() >= 3 else -1
                if kv.size(seq_dim) > max_length:
                    idx = [slice(None)] * kv.dim()
                    idx[seq_dim] = slice(0, max_length)
                    new_kv = kv[tuple(idx)].contiguous()
                else:
                    new_kv = kv
                new_layer.append(new_kv)
            new_cache.append(tuple(new_layer))
        return tuple(new_cache)
    return past_kv


def stop_hit(token_id: int, stop_ids: List[int]) -> bool:
    return int(token_id) in set(stop_ids)


# ---------------------------
# One-layer hidden capture (same idea as your runner)
# ---------------------------

def _get_transformer_layers(model):
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("model.decoder.layers", lambda m: m.model.decoder.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and hasattr(layers, "__len__"):
                return layers, name
        except Exception:
            continue
    raise RuntimeError("Cannot locate transformer layers list. Please extend _get_transformer_layers().")


class LayerCapture:
    def __init__(self):
        self.hidden: Optional[torch.Tensor] = None

    def hook_fn(self, module, inputs, outputs):
        h = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        self.hidden = h


def attach_layer_capture(model, layer_idx: int):
    layers, path_name = _get_transformer_layers(model)
    n_layers = len(layers)
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx={layer_idx} out of range for {path_name} with n_layers={n_layers}.")
    cap = LayerCapture()
    handle = layers[layer_idx].register_forward_hook(cap.hook_fn)
    return cap, handle


# ---------------------------
# SRSD hidden buffer (normalized)
# ---------------------------

class NormHiddenBuffer:
    def __init__(self, max_capacity: int, hidden_dim: int, device: torch.device, dtype: torch.dtype):
        self.max_capacity = max_capacity
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype
        self.eps = 1e-8
        self.buf = torch.empty((max_capacity, hidden_dim), device=device, dtype=dtype)
        self.length = 0

    def _norm(self, x2d: torch.Tensor) -> torch.Tensor:
        xf = x2d.float()
        n = torch.sqrt((xf * xf).sum(dim=-1, keepdim=True) + self.eps)
        y = xf / n
        return y.to(self.dtype)

    def init_from_prompt(self, hidden_2d: torch.Tensor):
        n = int(hidden_2d.shape[0])
        self.buf[:n] = self._norm(hidden_2d)
        self.length = n

    def append(self, hidden_2d: torch.Tensor):
        n = int(hidden_2d.shape[0])
        s = self.length
        e = s + n
        self.buf[s:e] = self._norm(hidden_2d)
        self.length = e

    def sims(self, q: torch.Tensor, upto: int) -> torch.Tensor:
        return torch.mv(self.buf[:upto], q)


# ---------------------------
# Trace structures
# ---------------------------

@dataclass
class TokenTraceItem:
    pos: int
    token_id: int
    source: str  # "base" | "draft" | "mismatch" | "ar"
    support_idx: torch.Tensor
    support_probs: torch.Tensor


# ---------------------------
# AR aligned decoding
# ---------------------------

@torch.no_grad()
def decode_ar_aligned(
    model,
    tokenizer,
    enc: Dict[str, torch.Tensor],
    stop_ids: List[int],
    global_seed: int,
    sample_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[torch.Tensor, List[TokenTraceItem], int]:
    device = enc["input_ids"].device
    prompt_len = int(enc["input_ids"].shape[1])

    out = model(**enc, use_cache=True, output_hidden_states=False)
    past_kv = out.past_key_values
    if isinstance(past_kv, tuple):
        past_kv = DynamicCache.from_legacy_cache(past_kv)

    last_logits = out.logits[:, -1, :].squeeze(0)  # [V]
    seq = [int(x) for x in enc["input_ids"][0].tolist()]

    trace: List[TokenTraceItem] = []
    for pos in range(max_new_tokens):
        tok, idx, probs = aligned_sample_token(
            last_logits,
            global_seed=global_seed,
            sample_id=sample_id,
            pos=pos,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_dist=True,
        )
        tok_id = int(tok.item())
        trace.append(TokenTraceItem(pos=pos, token_id=tok_id, source="ar", support_idx=idx, support_probs=probs))
        seq.append(tok_id)

        if stop_hit(tok_id, stop_ids):
            break

        inp = torch.tensor([[tok_id]], device=device, dtype=enc["input_ids"].dtype)
        out = model(inp, past_key_values=past_kv, use_cache=True, output_hidden_states=False)
        past_kv = crop_past_key_values(out.past_key_values, len(seq))
        last_logits = out.logits[:, -1, :].squeeze(0)

    return torch.tensor([seq], device=device, dtype=enc["input_ids"].dtype), trace, prompt_len


# ---------------------------
# SRSD aligned decoding (sampling-correct + RNG aligned)
# ---------------------------

@torch.no_grad()
def decode_srsd_aligned(
    model,
    tokenizer,
    cap: LayerCapture,
    enc: Dict[str, torch.Tensor],
    stop_ids: List[int],
    global_seed: int,
    sample_id: int,
    layer_idx: int,
    K_max: int,
    retrieval_topk: int,
    sim_threshold: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[torch.Tensor, List[TokenTraceItem], int]:

    device = enc["input_ids"].device
    prompt_len = int(enc["input_ids"].shape[1])

    out = model(**enc, use_cache=True, output_hidden_states=False)
    past_kv = out.past_key_values
    if isinstance(past_kv, tuple):
        past_kv = DynamicCache.from_legacy_cache(past_kv)

    if cap.hidden is None:
        raise RuntimeError("LayerCapture did not capture hidden during prefill.")

    h_prompt = cap.hidden.squeeze(0)  # [T,H]
    hidden_dim = int(h_prompt.size(-1))
    dtype_store = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    hbuf = NormHiddenBuffer(max_capacity=prompt_len + max_new_tokens + K_max + 32,
                            hidden_dim=hidden_dim, device=device, dtype=dtype_store)
    hbuf.init_from_prompt(h_prompt)

    last_logits = out.logits[:, -1, :].squeeze(0)  # [V]
    seq = [int(x) for x in enc["input_ids"][0].tolist()]

    trace: List[TokenTraceItem] = []

    # absolute generated position after prompt
    gen_pos = 0

    while gen_pos < max_new_tokens:
        # 1) sample base token with aligned RNG at position gen_pos
        base_tok, idx0, probs0 = aligned_sample_token(
            last_logits,
            global_seed=global_seed,
            sample_id=sample_id,
            pos=gen_pos,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_dist=True,
        )
        base_id = int(base_tok.item())
        trace.append(TokenTraceItem(pos=gen_pos, token_id=base_id, source="base", support_idx=idx0, support_probs=probs0))

        # stop early
        if stop_hit(base_id, stop_ids):
            seq.append(base_id)
            break

        # 2) retrieval for draft
        draft_tokens: Optional[torch.Tensor] = None
        K_cand = 0

        remaining = max_new_tokens - gen_pos
        # need at least 1 draft token
        if hbuf.length > 1 and remaining > 1:
            q = hbuf.buf[hbuf.length - 1]  # [H]
            sims = hbuf.sims(q, upto=hbuf.length - 1)  # [T-1]
            k = min(int(retrieval_topk), int(sims.numel()))
            topv, topi = torch.topk(sims, k=k, largest=True, sorted=True)
            # gating: sim threshold + next token equals base
            # anchor 'a' = topi; need x_{a+1} == x_t => token at (a+1) equals base_id
            # current prefix length is len(seq); positions in seq include prompt + committed
            cur_len = len(seq)
            cand_next = topi + 1  # [k]
            # require cand_next+1 < cur_len to have tail to copy
            has_tail = (cand_next + 1) < cur_len
            in_range = cand_next < cur_len
            cand_next_clamped = cand_next.clamp(min=0, max=max(cur_len - 1, 0))
            hist_tok = torch.tensor(seq, device=device, dtype=torch.long)[cand_next_clamped]
            tok_ok = hist_tok == base_id
            sim_ok = topv >= float(sim_threshold)
            mask = sim_ok & in_range & has_tail & tok_ok
            pos_ok = torch.nonzero(mask, as_tuple=False)
            if pos_ok.numel() > 0:
                j = int(pos_ok[0, 0].item())
                hist_next_idx = int(cand_next[j].item())
                start = hist_next_idx + 1
                avail = cur_len - start
                K_limit = min(K_max, remaining - 1, avail)
                if K_limit > 0:
                    draft_tokens = torch.tensor(seq, device=device, dtype=torch.long)[start:start + K_limit].view(1, -1)
                    K_cand = int(K_limit)

        # 3) merged forward: base + draft (if any)
        base_tensor = torch.tensor([[base_id]], device=device, dtype=enc["input_ids"].dtype)
        if K_cand > 0 and draft_tokens is not None:
            inp = torch.cat([base_tensor, draft_tokens.to(base_tensor.dtype)], dim=1)
        else:
            inp = base_tensor
            K_cand = 0

        out_m = model(inp, past_key_values=past_kv, use_cache=True, output_hidden_states=False)
        past_kv_new = out_m.past_key_values
        logits_new = out_m.logits  # [1, 1+K_cand, V] but verification uses first K_cand positions after base
        hidden_new = cap.hidden.squeeze(0) if cap.hidden is not None else None
        if hidden_new is None:
            raise RuntimeError("LayerCapture did not capture hidden in merged forward.")

        # 4) verify draft positions with aligned RNG at absolute positions gen_pos+1 ... gen_pos+K_cand
        n_acc = 0
        mismatch = False
        mismatch_id: Optional[int] = None

        if K_cand > 0 and draft_tokens is not None:
            # verify logits for draft tokens are logits_new[:, :K_cand, :]
            # (matching your runner's convention: first K correspond to drafted positions)
            verify_logits = logits_new[:, :K_cand, :].squeeze(0)  # [K,V]
            drafted = draft_tokens.squeeze(0).to(torch.long)      # [K]

            # sample each position using aligned RNG
            pred_ids = []
            pred_dists = []
            for r in range(K_cand):
                pos_abs = gen_pos + 1 + r
                tok_r, idx_r, prob_r = aligned_sample_token(
                    verify_logits[r],
                    global_seed=global_seed,
                    sample_id=sample_id,
                    pos=pos_abs,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    return_dist=True,
                )
                pred_ids.append(int(tok_r.item()))
                pred_dists.append((idx_r, prob_r))

            # compute lcp
            for r in range(K_cand):
                if pred_ids[r] == int(drafted[r].item()):
                    n_acc += 1
                else:
                    mismatch = True
                    mismatch_id = pred_ids[r]
                    break

            # record distributions for accepted draft positions (and also the mismatch position if occurs)
            for r in range(n_acc):
                pos_abs = gen_pos + 1 + r
                idx_r, prob_r = pred_dists[r]
                trace.append(TokenTraceItem(pos=pos_abs, token_id=int(drafted[r].item()),
                                            source="draft", support_idx=idx_r, support_probs=prob_r))

            if mismatch and mismatch_id is not None:
                pos_abs = gen_pos + 1 + n_acc
                idx_r, prob_r = pred_dists[n_acc]
                trace.append(TokenTraceItem(pos=pos_abs, token_id=mismatch_id,
                                            source="mismatch", support_idx=idx_r, support_probs=prob_r))

        # 5) commit: base + accepted draft; if mismatch => commit mismatch token and do KV repair
        seq.append(base_id)
        gen_pos += 1

        # append hidden for base + accepted draft tokens from merged forward
        # merged forward produced hidden for (base + draft tokens); we need base + accepted draft only
        take = 1 + n_acc
        hbuf.append(hidden_new[:take])

        # crop cache to committed prefix length
        past_kv = crop_past_key_values(past_kv_new, len(seq) + n_acc)
        # update logits for next base step:
        if K_cand > 0:
            # logits_new at index (take-1) corresponds to last committed token among (base + accepted draft)
            last_logits = logits_new[:, take - 1, :].squeeze(0)
        else:
            last_logits = logits_new[:, 0, :].squeeze(0)

        # commit accepted draft tokens in seq
        if n_acc > 0 and draft_tokens is not None:
            for r in range(n_acc):
                tok_id = int(draft_tokens[0, r].item())
                seq.append(tok_id)
                gen_pos += 1
                if stop_hit(tok_id, stop_ids):
                    return torch.tensor([seq], device=device, dtype=enc["input_ids"].dtype), trace, prompt_len

        # mismatch handling: commit mismatch token and KV repair forward
        if mismatch and mismatch_id is not None:
            if stop_hit(mismatch_id, stop_ids):
                seq.append(mismatch_id)
                return torch.tensor([seq], device=device, dtype=enc["input_ids"].dtype), trace, prompt_len

            # truncate cache to committed prefix (already cropped), then run one step on mismatch token
            inp_fix = torch.tensor([[mismatch_id]], device=device, dtype=enc["input_ids"].dtype)
            out_fix = model(inp_fix, past_key_values=past_kv, use_cache=True, output_hidden_states=False)
            past_kv = crop_past_key_values(out_fix.past_key_values, len(seq) + 1)
            last_logits = out_fix.logits[:, -1, :].squeeze(0)

            # append mismatch token to seq and hidden buffer
            seq.append(mismatch_id)
            gen_pos += 1
            if cap.hidden is None:
                raise RuntimeError("LayerCapture did not capture hidden in mismatch repair.")
            hbuf.append(cap.hidden.squeeze(0))  # [1,H]

    return torch.tensor([seq], device=device, dtype=enc["input_ids"].dtype), trace, prompt_len


# ---------------------------
# Evaluation loop
# ---------------------------

def summarize_dist_list(dlist: List[DistMetrics]) -> Dict[str, Any]:
    if not dlist:
        return {}
    tv = np.array([d.tv for d in dlist], dtype=np.float64)
    js = np.array([d.js for d in dlist], dtype=np.float64)
    klpq = np.array([d.kl_pq for d in dlist], dtype=np.float64)
    return {
        "count": int(len(dlist)),
        "tv_mean": float(tv.mean()),
        "tv_p50": float(np.quantile(tv, 0.5)),
        "tv_p90": float(np.quantile(tv, 0.9)),
        "tv_max": float(tv.max()),
        "js_mean": float(js.mean()),
        "js_p90": float(np.quantile(js, 0.9)),
        "js_max": float(js.max()),
        "kl_pq_mean": float(klpq.mean()),
        "kl_pq_max": float(klpq.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="+", required=True)
    ap.add_argument("--samples-per-task", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument("--layer-idx", type=int, default=30)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--retrieval-topk", type=int, default=10)
    ap.add_argument("--sim-threshold", type=float, default=0.0)

    ap.add_argument("--global-seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)

    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        device_map={"": args.gpu_id} if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    cap, handle = attach_layer_capture(model, args.layer_idx)

    stop_ids = get_stop_token_ids(model, tokenizer)
    print(f"[Stop IDs] {stop_ids}")

    tasks: List[str] = TASK_CHOICES if "all" in args.tasks else args.tasks

    agg_div_pos: List[int] = []
    agg_identical = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        global_sample_id = 0
        for task in tasks:
            prompts = get_dataset_prompts(task, args.samples_per_task)
            for i, prompt in enumerate(prompts):
                global_sample_id += 1
                sid = global_sample_id  # stable id for hashing

                enc = encode_prompt(tokenizer, prompt, device)

                # SRSD aligned
                srsd_ids, srsd_trace, prompt_len = decode_srsd_aligned(
                    model=model, tokenizer=tokenizer, cap=cap, enc=enc, stop_ids=stop_ids,
                    global_seed=args.global_seed, sample_id=sid,
                    layer_idx=args.layer_idx, K_max=args.K, retrieval_topk=args.retrieval_topk,
                    sim_threshold=args.sim_threshold, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                )

                # AR aligned
                ar_ids, ar_trace, _ = decode_ar_aligned(
                    model=model, tokenizer=tokenizer, enc=enc, stop_ids=stop_ids,
                    global_seed=args.global_seed, sample_id=sid,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                )

                # compare trajectories (only generated part)
                s_gen = srsd_ids[0][prompt_len:].tolist()
                a_gen = ar_ids[0][prompt_len:].tolist()
                L = min(len(s_gen), len(a_gen))

                div_pos = None
                for p in range(L):
                    if int(s_gen[p]) != int(a_gen[p]):
                        div_pos = p
                        break
                identical = (div_pos is None) and (len(s_gen) == len(a_gen))

                if identical:
                    agg_identical += 1
                else:
                    if div_pos is None:
                        div_pos = L
                    agg_div_pos.append(int(div_pos))

                # per-position distribution distances until divergence (or full)
                upto = div_pos if div_pos is not None else L
                dists: List[DistMetrics] = []
                # traces are aligned by absolute pos (0..)
                # build dict pos -> item for quick lookup
                ar_map = {it.pos: it for it in ar_trace}
                srsd_map = {it.pos: it for it in srsd_trace}

                for pos in range(upto):
                    if pos in ar_map and pos in srsd_map:
                        da = ar_map[pos]
                        ds = srsd_map[pos]
                        dm = sparse_dist_metrics(da.support_idx, da.support_probs, ds.support_idx, ds.support_probs)
                        dists.append(dm)

                dist_summary = summarize_dist_list(dists)

                # if divergence exists, compute metrics at divergence position if both available
                div_detail = {}
                if div_pos is not None and div_pos in ar_map and div_pos in srsd_map:
                    da = ar_map[div_pos]
                    ds = srsd_map[div_pos]
                    dm = sparse_dist_metrics(da.support_idx, da.support_probs, ds.support_idx, ds.support_probs)
                    div_detail = {
                        "div_pos": int(div_pos),
                        "ar_token": int(da.token_id),
                        "srsd_token": int(ds.token_id),
                        "srsd_source": ds.source,
                        "tv": dm.tv,
                        "js": dm.js,
                        "kl_pq": dm.kl_pq,
                        "support_ar": dm.support_p,
                        "support_srsd": dm.support_q,
                        "jaccard": dm.jaccard,
                    }
                else:
                    div_detail = {"div_pos": None} if identical else {"div_pos": int(div_pos)}

                rec = {
                    "task": task,
                    "sample_idx": i + 1,
                    "sample_id": sid,
                    "prompt_tokens": int(prompt_len),
                    "max_new_tokens": int(args.max_new_tokens),
                    "identical": bool(identical),
                    "divergence": div_detail,
                    "dist_summary_before_divergence": dist_summary,
                    "config": {
                        "global_seed": int(args.global_seed),
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                        "top_k": int(args.top_k),
                        "layer_idx": int(args.layer_idx),
                        "K": int(args.K),
                        "retrieval_topk": int(args.retrieval_topk),
                        "sim_threshold": float(args.sim_threshold),
                    },
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(f"[{task} #{i+1}] identical={identical} div_pos={div_detail.get('div_pos')}")

    # aggregate summary (print only)
    print("\n===== Alignment Summary =====")
    print(f"Total prompts: {global_sample_id}")
    print(f"Identical trajectories: {agg_identical}")
    if agg_div_pos:
        arr = np.array(agg_div_pos, dtype=np.int32)
        print(f"Divergence count: {len(agg_div_pos)}")
        print(f"Divergence pos mean: {arr.mean():.2f}, p50: {np.quantile(arr,0.5):.0f}, p90: {np.quantile(arr,0.9):.0f}, min: {arr.min()}, max: {arr.max()}")
    else:
        print("No divergences.")

    try:
        handle.remove()
    except Exception:
        pass


if __name__ == "__main__":
    main()
