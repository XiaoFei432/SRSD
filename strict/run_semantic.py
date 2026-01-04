#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Semantic Speculative Decoding (Merged Forward + KV Cache + Norm Cache)
with Sampling Support and *Sampling-Correct Verification* when --do-sample.

Optimizations in this version:
1) Token buffer pre-allocation (no torch.cat O(T^2))
2) Forward hook to capture only ONE layer hidden (no output_hidden_states=True)
3) Top-K retrieval filtering fully on GPU (no .tolist() + Python loop over candidates)
4) Verification:
   - Greedy: batched argmax + batched compare on GPU (1 sync max)
   - Sampling: batched sampling + batched compare on GPU (few syncs, no K-sync loop)

Change in this patch:
- Only store normalized hidden (no raw buffer + buffer_norm).
  * normalize in fp32, store in bf16/fp16 (CPU stores fp32)
  * retrieval uses buf_norm only

NOTE:

python run_semantic.py \
  --model-path /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --tasks summarization \
  --samples-per-task 10 \
  --max-new-tokens 2048 \
  --layer-idx 30 \
  --K 16 \
  --sim-threshold 0.0 \
  --retrieval-topk 10 \
  --do-sample \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k 0 \
  --output result/srsd_summarizaiton_sampling.jsonl

"""

import argparse
import json
import time
import random
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from data_utils import TASK_CHOICES, get_dataset_prompts
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# ======================
# Configuration
# ======================

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"
DEFAULT_GPU_ID = 0

# Sampling defaults (you said future experiments use top_k=0)
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 0

# Retrieval defaults (Top-K similarity candidates)
DEFAULT_RETRIEVAL_TOPK = 10


# ======================
# Utils
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    return list(dict.fromkeys(eos_ids))


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
            torch_dtype=torch.float32,
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


# ======================
# Sampling Utils (GPU-friendly)
# ======================

def _safe_softmax_2d(logits_2d: torch.Tensor) -> torch.Tensor:
    all_inf = torch.isinf(logits_2d).all(dim=-1)  # [N]
    probs = torch.softmax(logits_2d, dim=-1)
    nan_rows = torch.isnan(probs).any(dim=-1)
    bad = all_inf | nan_rows
    if bad.any():
        V = logits_2d.size(-1)
        probs[bad] = 1.0 / float(V)
    return probs


def _top_k_filter_2d(logits_2d: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits_2d
    k = min(int(top_k), logits_2d.size(-1))
    kth = torch.topk(logits_2d, k, dim=-1).values[..., -1]  # [N]
    return logits_2d.masked_fill(logits_2d < kth.unsqueeze(-1), float("-inf"))


def _top_p_filter_2d(logits_2d: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or not (0.0 < float(top_p) < 1.0):
        return logits_2d

    sorted_logits, sorted_indices = torch.sort(logits_2d, descending=True, dim=-1)  # [N,V]
    sorted_probs = _safe_softmax_2d(sorted_logits)
    cdf = torch.cumsum(sorted_probs, dim=-1)

    remove = cdf > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    remove_orig = torch.zeros_like(remove).scatter(dim=-1, index=sorted_indices, src=remove)
    return logits_2d.masked_fill(remove_orig, float("-inf"))


def sample_from_logits_1d(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # [1,V]

    if not do_sample:
        return torch.argmax(logits, dim=-1)  # [1]

    x = logits
    if temperature is not None and temperature > 0 and float(temperature) != 1.0:
        x = x / float(temperature)

    x = _top_k_filter_2d(x, top_k)
    x = _top_p_filter_2d(x, top_p)

    all_inf = torch.isinf(x).all(dim=-1)
    if all_inf.any():
        return torch.argmax(logits, dim=-1)

    probs = _safe_softmax_2d(x)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)  # [1]


def sample_from_logits_batch(
    logits_3d: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if logits_3d.dim() != 3 or logits_3d.size(0) != 1:
        raise ValueError(f"logits_3d must be [1,K,V], got {tuple(logits_3d.shape)}")
    if logits_3d.size(1) == 0:
        return torch.empty((1, 0), dtype=torch.long, device=logits_3d.device)

    if not do_sample:
        return torch.argmax(logits_3d, dim=-1)  # [1,K]

    x = logits_3d.squeeze(0)  # [K,V]
    if temperature is not None and temperature > 0 and float(temperature) != 1.0:
        x = x / float(temperature)

    x = _top_k_filter_2d(x, top_k)
    x = _top_p_filter_2d(x, top_p)

    all_inf = torch.isinf(x).all(dim=-1)  # [K]
    if all_inf.any():
        fallback = torch.argmax(logits_3d.squeeze(0), dim=-1)  # [K]
        probs = _safe_softmax_2d(x)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [K]
        sampled[all_inf] = fallback[all_inf]
        return sampled.unsqueeze(0)

    probs = _safe_softmax_2d(x)  # [K,V]
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [K]
    return sampled.unsqueeze(0)  # [1,K]


def stop_mask_1d(tokens_1d: torch.Tensor, stop_ids_tensor: torch.Tensor) -> torch.Tensor:
    if stop_ids_tensor.numel() == 0 or tokens_1d.numel() == 0:
        return torch.zeros_like(tokens_1d, dtype=torch.bool)
    return (tokens_1d.unsqueeze(-1) == stop_ids_tensor.view(1, -1)).any(dim=-1)


# ======================
# One-layer hidden capture
# ======================

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
    raise RuntimeError("Cannot locate transformer layers list. Please add your model's path in _get_transformer_layers().")


class LayerCapture:
    def __init__(self):
        self.hidden: Optional[torch.Tensor] = None

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, (tuple, list)):
            h = outputs[0]
        else:
            h = outputs
        self.hidden = h


def attach_layer_capture(model, layer_idx: int):
    layers, path_name = _get_transformer_layers(model)
    n_layers = len(layers)
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx={layer_idx} out of range for {path_name} with n_layers={n_layers}.")
    cap = LayerCapture()
    handle = layers[layer_idx].register_forward_hook(cap.hook_fn)
    return cap, handle


# ======================
# Normalized Buffer (ONLY normalized hidden)
# ======================

class NormHiddenBuffer:
    """
    Stores ONLY normalized hidden vectors for retrieval.

    - Normalize in fp32 (stable)
    - Store in norm_dtype (bf16/fp16 on GPU, fp32 on CPU)
    """
    def __init__(self, max_capacity: int, hidden_dim: int, device: torch.device, norm_dtype: torch.dtype):
        self.max_capacity = max_capacity
        self.hidden_dim = hidden_dim
        self.device = device
        self.eps = 1e-8

        self.buf_norm = torch.empty((max_capacity, hidden_dim), dtype=norm_dtype, device=device)
        self.length = 0

    def _normalize_to_store_dtype(self, x_2d: torch.Tensor) -> torch.Tensor:
        # fp32 normalize, then cast to storage dtype
        xf = x_2d.float()
        norm = torch.sqrt((xf * xf).sum(dim=-1, keepdim=True) + self.eps)
        y = xf / norm
        return y.to(self.buf_norm.dtype)

    def init_from_prompt(self, hidden_2d: torch.Tensor):
        n = int(hidden_2d.shape[0])
        self.buf_norm[:n] = self._normalize_to_store_dtype(hidden_2d)
        self.length = n

    def append(self, hidden_2d: torch.Tensor):
        n = int(hidden_2d.shape[0])
        start = self.length
        end = start + n
        self.buf_norm[start:end] = self._normalize_to_store_dtype(hidden_2d)
        self.length = end

    def compute_similarities(self, query_norm_1d: torch.Tensor, upto_exclusive: int) -> torch.Tensor:
        # query_norm_1d should already be normalized and same dtype as buf_norm
        keys = self.buf_norm[:upto_exclusive]  # [T,H]
        # GEMV
        return torch.mv(keys, query_norm_1d)   # [T]

    def has_capacity(self, needed: int) -> bool:
        return self.length + needed <= self.max_capacity


# ======================
# Core: Semantic SD (optimized)
# ======================

@torch.no_grad()
def semantic_sd_forward_one_prompt(
    model,
    tokenizer,
    cap: LayerCapture,
    enc: Dict[str, torch.Tensor],
    stop_ids_tensor: torch.Tensor,
    layer_idx: int = 1,
    K_max: int = 16,
    sim_threshold: float = -1.0,
    retrieval_topk: int = 1,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
):

    device = enc["input_ids"].device
    prompt_len = int(enc["input_ids"].shape[1])

    # -------- token buffer pre-alloc --------
    max_capacity_tokens = prompt_len + max_new_tokens + K_max + 32
    ids_buf = torch.empty((1, max_capacity_tokens), dtype=enc["input_ids"].dtype, device=device)
    ids_buf[:, :prompt_len] = enc["input_ids"]
    cur_len = prompt_len

    if max_new_tokens <= 0:
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "N_forward_total": 0, "N_forward_decode": 0,
            "N_tokens_decode": 0, "draft_attempts": 0, "draft_accepted_tokens": 0,
        }
        return ids_buf[:, :cur_len], stats, prompt_len

    # ----------------------
    # Prefill (hook captures one layer)
    # ----------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_start = time.time()

    out = model(**enc, use_cache=True, output_hidden_states=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.time() - t_prefill_start

    if cap.hidden is None:
        raise RuntimeError("LayerCapture did not capture hidden in prefill. Check layer hook path.")

    h_prompt = cap.hidden.squeeze(0)  # [T,H]
    hidden_dim = int(h_prompt.size(-1))
    buf_dtype = h_prompt.dtype

    past_kv = out.past_key_values
    if isinstance(past_kv, tuple):
        past_kv = DynamicCache.from_legacy_cache(past_kv)

    # choose storage dtype for normalized hidden
    if device.type == "cpu":
        norm_dtype = torch.float32
    else:
        # store bf16/fp16 (match model hidden dtype if possible)
        norm_dtype = buf_dtype if buf_dtype in (torch.float16, torch.bfloat16) else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    # Hidden buffer: ONLY normalized
    hbuf = NormHiddenBuffer(max_capacity_tokens, hidden_dim, device, norm_dtype)
    hbuf.init_from_prompt(h_prompt)

    last_logits = out.logits[:, -1, :]  # [1,V]

    N_forward_prefill = 1
    N_forward_decode = 0
    draft_attempts = 0
    draft_accepted_tokens = 0

    # ----------------------
    # First token (included in prefill timing)
    # ----------------------
    base_token_t = sample_from_logits_1d(last_logits, do_sample, temperature, top_p, top_k)  # [1]
    base_tensor = base_token_t.view(1, 1)  # [1,1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_first_start = time.time()

    out_first = model(
        base_tensor,
        past_key_values=past_kv,
        use_cache=True,
        output_hidden_states=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time += time.time() - t_first_start
    N_forward_prefill += 1

    if cap.hidden is None:
        raise RuntimeError("LayerCapture did not capture hidden in first token forward.")
    h_first = cap.hidden.squeeze(0)  # [1,H]

    # append hidden(normed)
    hbuf.append(h_first)

    # append token
    ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
    cur_len += 1

    past_kv = crop_past_key_values(out_first.past_key_values, cur_len)
    last_logits = out_first.logits[:, -1, :]

    hit_stop = bool(stop_mask_1d(base_tensor.view(-1).long(), stop_ids_tensor).any().item())
    if hit_stop:
        stats = {
            "prefill_time": float(prefill_time), "decode_time": 0.0,
            "N_forward_total": int(N_forward_prefill), "N_forward_decode": 0,
            "N_tokens_decode": 0, "draft_attempts": 0, "draft_accepted_tokens": 0,
        }
        return ids_buf[:, :cur_len], stats, prompt_len

    generated = 1

    # ----------------------
    # Decode loop
    # ----------------------
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode_start = time.time()

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        if remaining <= 0:
            break

        if not hbuf.has_capacity(1 + K_max) or (cur_len + 1 + K_max >= ids_buf.size(1)):
            # fallback AR
            while generated < max_new_tokens:
                base_token_t = sample_from_logits_1d(last_logits, do_sample, temperature, top_p, top_k)  # [1]
                base_tensor = base_token_t.view(1, 1)

                out_g = model(
                    base_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=False,
                )
                N_forward_decode += 1

                if cap.hidden is None:
                    raise RuntimeError("LayerCapture did not capture hidden in AR fallback forward.")
                h_g = cap.hidden.squeeze(0)  # [1,H]

                ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
                cur_len += 1
                hbuf.append(h_g)

                past_kv = crop_past_key_values(out_g.past_key_values, cur_len)
                last_logits = out_g.logits[:, -1, :]
                generated += 1

                if bool(stop_mask_1d(base_tensor.view(-1).long(), stop_ids_tensor).any().item()):
                    break
            break

        # Step 1: base token
        base_token_t = sample_from_logits_1d(last_logits, do_sample, temperature, top_p, top_k)  # [1]
        base_tensor = base_token_t.view(1, 1)  # [1,1]

        # Step 2: Top-K retrieval
        draft_tokens_tensor = None
        K_candidate = 0

        if hbuf.length > 1 and remaining > 1:
            # last normalized vector as query
            q_norm = hbuf.buf_norm[hbuf.length - 1]  # [H], already normalized & stored dtype

            sims = hbuf.compute_similarities(q_norm, hbuf.length - 1)  # [T-1]
            rK = max(1, int(retrieval_topk))
            k = min(rK, int(sims.numel()))
            top_vals, top_idxs = torch.topk(sims, k=k, largest=True, sorted=True)  # [k], [k]

            sim_ok = top_vals >= float(sim_threshold)

            hist_next = top_idxs + 1  # [k]
            cur_len_t = torch.tensor(cur_len, device=device, dtype=hist_next.dtype)
            has_tail = (hist_next + 1) < cur_len_t
            in_range = hist_next < cur_len_t

            hist_next_clamped = hist_next.clamp(min=0, max=max(cur_len - 1, 0))
            hist_tokens = ids_buf[0, hist_next_clamped]  # [k]
            tok_ok = hist_tokens == base_token_t.view(-1)[0]

            mask = sim_ok & in_range & has_tail & tok_ok
            pos = torch.nonzero(mask, as_tuple=False)
            if pos.numel() > 0:
                j = pos[0, 0]  # scalar tensor
                hist_next_idx = hist_next[j]
                start_draft = int((hist_next_idx + 1).item())  # (kept as-is)
                avail_len = cur_len - start_draft
                if avail_len > 0:
                    K_limit = min(K_max, remaining - 1, avail_len)
                    if K_limit > 0:
                        draft_tokens_tensor = ids_buf[:, start_draft:start_draft + K_limit]
                        K_candidate = K_limit

        # Step 3: merged forward
        if K_candidate > 0 and draft_tokens_tensor is not None:
            input_tokens = torch.cat([base_tensor, draft_tokens_tensor], dim=1)
            draft_attempts += 1
        else:
            input_tokens = base_tensor
            K_candidate = 0

        out_merged = model(
            input_tokens,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=False,
        )
        N_forward_decode += 1

        if cap.hidden is None:
            raise RuntimeError("LayerCapture did not capture hidden in merged forward.")
        hidden_new = cap.hidden.squeeze(0)  # [L,H]
        logits_new = out_merged.logits      # [1,L,V]
        past_kv_new = out_merged.past_key_values

        # Step 4: Verification (batched)
        base_is_stop = bool(stop_mask_1d(base_tensor.view(-1).long(), stop_ids_tensor).any().item())
        if base_is_stop:
            ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
            cur_len += 1
            hbuf.append(hidden_new[:1])
            past_kv = crop_past_key_values(past_kv_new, cur_len)
            generated += 1
            break

        n_draft_acc = 0
        mismatch = False
        mismatch_token_t: Optional[torch.Tensor] = None  # [1,1]

        if K_candidate > 0 and draft_tokens_tensor is not None:
            verify_logits = logits_new[:, :K_candidate, :]  # [1,K,V]
            if not do_sample:
                preds = torch.argmax(verify_logits, dim=-1)  # [1,K]
            else:
                preds = sample_from_logits_batch(
                    verify_logits,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

            matches = (preds == draft_tokens_tensor)
            mismatch_pos = torch.nonzero(~matches[0], as_tuple=False)

            if mismatch_pos.numel() == 0:
                n_draft_acc = K_candidate
            else:
                first_false = int(mismatch_pos[0, 0].item())  # kept
                n_draft_acc = first_false
                if do_sample:
                    mismatch = True
                    mismatch_token_t = preds[:, first_false:first_false + 1]  # [1,1]

            # stop truncation on matched prefix
            if n_draft_acc > 0:
                prefix = draft_tokens_tensor[:, :n_draft_acc].view(-1).long()
                sm = stop_mask_1d(prefix, stop_ids_tensor)
                sp = torch.nonzero(sm, as_tuple=False)
                if sp.numel() > 0:
                    first_stop = int(sp[0, 0].item())
                    n_draft_acc = first_stop + 1
                    mismatch = False
                    mismatch_token_t = None
                    stop_after_update = True
                else:
                    stop_after_update = False
            else:
                stop_after_update = False

            mismatch_is_stop = False
            if mismatch and mismatch_token_t is not None:
                mismatch_is_stop = bool(stop_mask_1d(mismatch_token_t.view(-1).long(), stop_ids_tensor).any().item())
        else:
            stop_after_update = False
            mismatch_is_stop = False

        draft_accepted_tokens += int(n_draft_acc)

        # Step 5: Update state
        if mismatch and mismatch_token_t is not None:
            prefix_without_mismatch = 1 + n_draft_acc
            target_len_prefix = cur_len + prefix_without_mismatch

            ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
            cur_len += 1
            if n_draft_acc > 0:
                ids_buf[0, cur_len:cur_len + n_draft_acc] = draft_tokens_tensor[0, :n_draft_acc]
                cur_len += n_draft_acc

            # append hidden for base + matched prefix
            hbuf.append(hidden_new[:prefix_without_mismatch])

            past_kv_prefix = crop_past_key_values(past_kv_new, target_len_prefix)
            generated += prefix_without_mismatch

            out_fix = model(
                mismatch_token_t,
                past_key_values=past_kv_prefix,
                use_cache=True,
                output_hidden_states=False,
            )
            N_forward_decode += 1

            if cap.hidden is None:
                raise RuntimeError("LayerCapture did not capture hidden in mismatch-fix forward.")
            h_fix = cap.hidden.squeeze(0)  # [1,H]

            ids_buf[0, cur_len:cur_len + 1] = mismatch_token_t[0]
            cur_len += 1
            hbuf.append(h_fix)

            past_kv = crop_past_key_values(out_fix.past_key_values, cur_len)
            last_logits = out_fix.logits[:, -1, :]
            generated += 1

            if mismatch_is_stop:
                break

        else:
            total_accepted = 1 + n_draft_acc
            target_len = cur_len + total_accepted

            ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
            cur_len += 1
            if n_draft_acc > 0 and draft_tokens_tensor is not None:
                ids_buf[0, cur_len:cur_len + n_draft_acc] = draft_tokens_tensor[0, :n_draft_acc]
                cur_len += n_draft_acc

            hbuf.append(hidden_new[:total_accepted])

            past_kv = crop_past_key_values(past_kv_new, target_len)
            last_logits = logits_new[:, total_accepted - 1, :]
            generated += total_accepted

            if stop_after_update:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_time = time.time() - t_decode_start

    stats: Dict[str, Any] = {
        "prefill_time": float(prefill_time),
        "decode_time": float(decode_time),
        "N_forward_total": int(N_forward_prefill + N_forward_decode),
        "N_forward_decode": int(N_forward_decode),
        "N_tokens_decode": int(max(generated - 1, 0)),
        "draft_attempts": int(draft_attempts),
        "draft_accepted_tokens": int(draft_accepted_tokens),
    }
    return ids_buf[:, :cur_len], stats, prompt_len


# ======================
# Main
# ======================

def run_semantic_sd(args):
    set_seed(args.seed)
    tokenizer, model, device = load_model_and_tokenizer(args.model_path, args.gpu_id)

    cap, handle = attach_layer_capture(model, args.layer_idx)

    stop_ids = get_stop_token_ids(model, tokenizer)
    stop_ids_tensor = torch.tensor(stop_ids, device=device, dtype=torch.long)

    print(f"\n[Config] Stop token IDs: {stop_ids}")
    print(f"[Config] Sampling: do_sample={args.do_sample}")
    if args.do_sample:
        print(f"[Config] temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
        print("[Config] Verification: Sampling-Correct (batched sample-then-compare + KV fix)")
    else:
        print("[Config] Verification: Strict Greedy (batched argmax)")
    print(f"[Config] layer_idx={args.layer_idx}, K={args.K}, sim_threshold={args.sim_threshold}")
    print(f"[Config] retrieval_topk={args.retrieval_topk} (Top-K similarity candidates; NOT sampling top_k)")

    print("\n[Warmup] Semantic SD ...")
    warmup_enc = encode_prompt(tokenizer, "Warmup.", device)
    with torch.no_grad():
        _ = model(**warmup_enc, use_cache=True, output_hidden_states=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[Warmup done]\n")

    tasks: List[str] = TASK_CHOICES if "all" in args.tasks else args.tasks

    total_raw = 0
    total_eff = 0
    total_decode_time = 0.0
    total_fwd_decode = 0
    total_attempts = 0
    total_accepted = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for task in tasks:
            print(f"\n=== [Semantic SD] Task: {task} ===")
            prompts = get_dataset_prompts(task, args.samples_per_task)

            for idx, prompt in enumerate(prompts, start=1):
                print(f"  - [{task}] sample {idx}/{len(prompts)}")
                enc = encode_prompt(tokenizer, prompt, device)

                out_ids, stats, prompt_len = semantic_sd_forward_one_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    cap=cap,
                    enc=enc,
                    stop_ids_tensor=stop_ids_tensor,
                    layer_idx=args.layer_idx,
                    K_max=args.K,
                    sim_threshold=args.sim_threshold,
                    retrieval_topk=args.retrieval_topk,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )

                seq = out_ids[0]
                raw_new = max(int(seq.shape[0]) - prompt_len, 0)
                eff_new = max(raw_new - 1, 0)

                decode_time = stats["decode_time"]
                prefill_time = stats["prefill_time"]

                throughput = (eff_new / decode_time) if (decode_time > 0 and eff_new > 0) else 0.0
                ideal_speedup = (raw_new / stats["N_forward_decode"]) if stats["N_forward_decode"] > 0 else 0.0

                total_raw += raw_new
                total_eff += eff_new
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
                    "new_tokens": eff_new,
                    "raw_new_tokens": raw_new,
                    "prefill_sec": prefill_time,
                    "decode_sec": decode_time,
                    "throughput": throughput,
                    "ideal_speedup": ideal_speedup,
                    "N_forward_total": stats["N_forward_total"],
                    "N_forward_decode": stats["N_forward_decode"],
                    "draft_attempts": stats["draft_attempts"],
                    "draft_accepted_tokens": stats["draft_accepted_tokens"],
                    "K_max": args.K,
                    "sim_threshold": args.sim_threshold,
                    "retrieval_topk": args.retrieval_topk,
                    "layer_idx": args.layer_idx,
                    "verification": "sample_then_compare" if args.do_sample else "greedy",
                    "do_sample": args.do_sample,
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                    "top_k": args.top_k if args.do_sample else None,
                    "model_path": args.model_path,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(
                    f"    Prefill: {prefill_time:.4f}s | Decode: {decode_time:.4f}s | "
                    f"Raw: {raw_new} | Eff: {eff_new} | TP: {throughput:.2f} | "
                    f"Ideal: {ideal_speedup:.2f} | "
                    f"Attempts: {stats['draft_attempts']} | DraftAcc: {stats['draft_accepted_tokens']}"
                )

    overall_tp = (total_eff / total_decode_time) if total_decode_time > 0 else 0.0
    overall_speedup = (total_raw / total_fwd_decode) if total_fwd_decode > 0 else 0.0
    acc_ratio = (total_accepted / (total_attempts * args.K)) if (total_attempts > 0 and args.K > 0) else 0.0

    print("\n" + "=" * 60)
    print("Semantic SD Summary")
    print("=" * 60)
    print(f"Verification        : {'Sampling-Correct' if args.do_sample else 'Greedy'}")
    print(f"Sampling            : {'Yes' if args.do_sample else 'No (Greedy)'}")
    if args.do_sample:
        print(f"  Temperature       : {args.temperature}")
        print(f"  Top-p             : {args.top_p}")
        print(f"  Top-k             : {args.top_k}")
    print(f"Retrieval Top-K     : {args.retrieval_topk}")
    print(f"Total Raw Tokens    : {total_raw}")
    print(f"Total Eff Tokens    : {total_eff}")
    print(f"Total Decode Time   : {total_decode_time:.3f} s")
    print(f"Overall Throughput  : {overall_tp:.2f} tok/s")
    print(f"Overall Speedup     : {overall_speedup:.2f} tok/step")
    print(f"Draft Acc Ratio     : {acc_ratio:.1%}")
    print("=" * 60)

    try:
        handle.remove()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--sim-threshold", type=float, default=0.0)
    parser.add_argument("--retrieval-topk", type=int, default=DEFAULT_RETRIEVAL_TOPK)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)

    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_semantic_sd(args)
