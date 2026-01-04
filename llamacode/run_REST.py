#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# build datastore
python run_REST.py --build-datastore \
    --model-path /root/autodl-tmp/models/CodeLlama-7b-Instruct-hf \
    --datastore-source the_stack \
    --datastore-path ./datastore_thestack.pt \
    --suffix-len 6 \
    --datastore-max-samples 200000

# decode
python run_REST.py \
    --model-path /root/autodl-tmp/models/CodeLlama-7b-Instruct-hf \
    --datastore-path ./datastore_thestack8.pt \
    --tasks project \
    --samples-per-task 40 \
    --suffix-len 6 \
    --K 16 \
    --output rest8.jsonl
"""

import argparse
import json
import time
import random
from typing import Tuple, List, Dict, Any, Optional
from collections import defaultdict, deque

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

try:
    from data_utils import TASK_CHOICES, get_dataset_prompts
except ImportError:
    TASK_CHOICES = ["code_edit", "translation", "summarization", "math", "multi_turn", "text_edit"]

    def get_dataset_prompts(task, n):
        return [f"Sample prompt {i} for {task}" for i in range(n)]


# ======================
# Configuration
# ======================

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"
DEFAULT_GPU_ID = 0
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 0  # future experiments: top_k=0

DEFAULT_SUFFIX_LEN = 8
DEFAULT_MAX_DRAFT_LEN = 16

# Rolling hash parameters
HASH_BASE = 31
HASH_MOD = (1 << 63) - 1


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
    # 检查是否有 chat_template 并且不为空
    has_chat_template = (
        hasattr(tokenizer, "apply_chat_template") 
        and hasattr(tokenizer, "chat_template") 
        and tokenizer.chat_template is not None
    )
    
    if has_chat_template:
        try:
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
        except Exception:
            # fallback
            enc_dict = tokenizer(text, return_tensors="pt")
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
# Sampling Utils
# ======================

def _safe_softmax_2d(logits_2d: torch.Tensor) -> torch.Tensor:
    all_inf = torch.isinf(logits_2d).all(dim=-1)
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
    kth = torch.topk(logits_2d, k, dim=-1).values[..., -1]
    return logits_2d.masked_fill(logits_2d < kth.unsqueeze(-1), float("-inf"))


def _top_p_filter_2d(logits_2d: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or not (0.0 < float(top_p) < 1.0):
        return logits_2d

    sorted_logits, sorted_indices = torch.sort(logits_2d, descending=True, dim=-1)
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
        logits = logits.unsqueeze(0)

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    x = logits
    if temperature is not None and temperature > 0 and float(temperature) != 1.0:
        x = x / float(temperature)

    x = _top_k_filter_2d(x, top_k)
    x = _top_p_filter_2d(x, top_p)

    all_inf = torch.isinf(x).all(dim=-1)
    if all_inf.any():
        return torch.argmax(logits, dim=-1)

    probs = _safe_softmax_2d(x)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


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
        return torch.argmax(logits_3d, dim=-1)

    x = logits_3d.squeeze(0)
    if temperature is not None and temperature > 0 and float(temperature) != 1.0:
        x = x / float(temperature)

    x = _top_k_filter_2d(x, top_k)
    x = _top_p_filter_2d(x, top_p)

    all_inf = torch.isinf(x).all(dim=-1)
    if all_inf.any():
        fallback = torch.argmax(logits_3d.squeeze(0), dim=-1)
        probs = _safe_softmax_2d(x)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled[all_inf] = fallback[all_inf]
        return sampled.unsqueeze(0)

    probs = _safe_softmax_2d(x)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampled.unsqueeze(0)


def stop_mask_1d(tokens_1d: torch.Tensor, stop_ids_tensor: torch.Tensor) -> torch.Tensor:
    if stop_ids_tensor.numel() == 0 or tokens_1d.numel() == 0:
        return torch.zeros_like(tokens_1d, dtype=torch.bool)
    return (tokens_1d.unsqueeze(-1) == stop_ids_tensor.view(1, -1)).any(dim=-1)


# ======================
# CPU Sliding Window + Rolling Hash
# ======================

class CPUSlidingWindow:
    def __init__(self, n: int, base: int = HASH_BASE, mod: int = HASH_MOD):
        self.n = n
        self.base = base
        self.mod = mod
        self.base_pow_n = pow(base, n, mod)
        self.window: deque = deque(maxlen=n)
        self.current_hash: int = 0
        self.is_full: bool = False

    def reset(self):
        self.window.clear()
        self.current_hash = 0
        self.is_full = False

    def init_from_prompt(self, prompt_tokens: List[int]):
        self.reset()
        init_tokens = prompt_tokens[-self.n:] if len(prompt_tokens) >= self.n else prompt_tokens
        for tok in init_tokens:
            self._push_token(tok)

    def _push_token(self, token: int):
        """
        Correct rolling update (MUST match datastore build):
            h_new = h*base - oldest*base^n + token  (mod M)
        """
        token = int(token)
        if self.is_full:
            oldest = self.window[0]
            self.current_hash = (
                self.current_hash * self.base
                - oldest * self.base_pow_n
                + token
            ) % self.mod
        else:
            self.current_hash = (self.current_hash * self.base + token) % self.mod

        self.window.append(token)
        if len(self.window) == self.n:
            self.is_full = True

    def push_tokens(self, tokens: List[int]):
        for tok in tokens:
            self._push_token(tok)

    def get_hash(self) -> Optional[int]:
        return self.current_hash if self.is_full else None

    def get_window_tokens(self) -> List[int]:
        return list(self.window)


# ======================
# REST Datastore V3 (with draft_start return)
# ======================

class RESTDatastoreV3:
    def __init__(self, suffix_len: int = 8, device: torch.device = None):
        self.suffix_len = suffix_len
        self.device = device or torch.device("cpu")

        self.base = HASH_BASE
        self.mod = HASH_MOD
        self.base_pow_n = pow(self.base, suffix_len, self.mod)

        self.tokens: Optional[np.ndarray] = None
        self.tokens_tensor: Optional[torch.Tensor] = None

        self.doc_starts: Optional[np.ndarray] = None
        self.doc_ends: Optional[np.ndarray] = None

        self.hash_index: Dict[int, np.ndarray] = {}

        self.total_tokens = 0
        self.num_docs = 0

    def build_from_tokens(self, token_list: List[List[int]]):
        print(f"[REST V3] Building datastore with suffix_len={self.suffix_len}...")

        all_tokens = []
        doc_starts = []
        doc_ends = []

        cur = 0
        for doc in token_list:
            doc_starts.append(cur)
            all_tokens.extend(doc)
            cur += len(doc)
            doc_ends.append(cur)

        self.tokens = np.array(all_tokens, dtype=np.int32)
        self.doc_starts = np.array(doc_starts, dtype=np.int64)
        self.doc_ends = np.array(doc_ends, dtype=np.int64)
        self.total_tokens = int(self.tokens.shape[0])
        self.num_docs = len(token_list)

        print(f"[REST V3] Total tokens: {self.total_tokens}, Docs: {self.num_docs}")
        print(f"[REST V3] Building per-document hash index...")

        n = self.suffix_len
        hash_to_positions = defaultdict(list)

        for d in range(self.num_docs):
            start = int(self.doc_starts[d])
            end = int(self.doc_ends[d])
            doc_tokens = self.tokens[start:end]
            if doc_tokens.shape[0] < n:
                continue

            # first n-gram
            h = 0
            for i in range(n):
                h = (h * self.base + int(doc_tokens[i])) % self.mod
            hash_to_positions[h].append(start)

            # roll within doc
            for i in range(1, doc_tokens.shape[0] - n + 1):
                h = (
                    h * self.base
                    - int(doc_tokens[i - 1]) * self.base_pow_n
                    + int(doc_tokens[i + n - 1])
                ) % self.mod
                hash_to_positions[h].append(start + i)

        self.hash_index = {h: np.array(pos, dtype=np.int64) for h, pos in hash_to_positions.items()}
        print(f"[REST V3] Index built: {len(self.hash_index)} unique {n}-grams (per-doc)")

        if self.device.type == "cuda":
            self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long, device=self.device)
        else:
            self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)

    def save(self, path: str):
        torch.save({
            "suffix_len": self.suffix_len,
            "tokens": self.tokens,
            "doc_starts": self.doc_starts,
            "doc_ends": self.doc_ends,
            "hash_index": self.hash_index,
            "total_tokens": self.total_tokens,
            "num_docs": self.num_docs,
        }, path)
        print(f"[REST V3] Saved to {path}")

    def load(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.suffix_len = int(data["suffix_len"])
        self.tokens = data["tokens"]
        self.doc_starts = data["doc_starts"]
        self.doc_ends = data["doc_ends"]
        self.hash_index = data["hash_index"]
        self.total_tokens = int(data["total_tokens"])
        self.num_docs = int(data["num_docs"])

        self.base_pow_n = pow(self.base, self.suffix_len, self.mod)

        if self.device.type == "cuda":
            self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long, device=self.device)
        else:
            self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)

        print(f"[REST V3] Loaded: {self.total_tokens} tokens, {self.num_docs} docs, suffix_len={self.suffix_len}")

    def _get_doc_end_for_pos(self, pos: int) -> int:
        idx = int(np.searchsorted(self.doc_ends, pos, side="right"))
        return int(self.doc_ends[idx]) if idx < len(self.doc_ends) else self.total_tokens

    def lookup(
        self,
        query_hash: int,
        query_tokens: List[int],
        base_token: int,
        K_max: int
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Returns:
            (draft_tokens_tensor_1d, draft_start_pos_in_global_tokens)
        where draft_start_pos is the first token AFTER base_token.
        """
        if self.tokens is None or query_hash is None:
            return None
        if query_hash not in self.hash_index:
            return None

        positions = self.hash_index[query_hash]
        n = self.suffix_len
        q = np.asarray(query_tokens, dtype=np.int32)

        for pos in positions:
            pos = int(pos)

            # collision check
            if not np.array_equal(self.tokens[pos:pos + n], q):
                continue

            cont_start = pos + n
            if cont_start >= self.total_tokens:
                continue

            if int(self.tokens[cont_start]) != int(base_token):
                continue

            doc_end = self._get_doc_end_for_pos(cont_start)
            available = doc_end - cont_start - 1
            if available < 1:
                continue

            draft_len = min(int(K_max), int(available))
            draft_start = cont_start + 1
            draft_end = draft_start + draft_len

            if self.tokens_tensor is not None and self.device.type == "cuda":
                return self.tokens_tensor[draft_start:draft_end], draft_start
            else:
                return torch.tensor(self.tokens[draft_start:draft_end], dtype=torch.long, device=self.device), draft_start

        return None


def build_datastore_v3(
    tokenizer: AutoTokenizer,
    source: str,
    suffix_len: int = 8,
    max_samples: int = 10000,
    max_tokens_per_sample: int = 2048,
    device: torch.device = torch.device("cpu")
) -> RESTDatastoreV3:
    print(f"[REST V3] Building from {source}...")

    token_list: List[List[int]] = []

    if source == "the_stack":
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)

            cnt = 0
            for sample in ds:
                if cnt >= max_samples:
                    break
                content = sample.get("content", "")
                if len(content) > 100:
                    tokens = tokenizer.encode(content, add_special_tokens=False)[:max_tokens_per_sample]
                    if len(tokens) > suffix_len + 10:
                        token_list.append(tokens)
                        cnt += 1
                if cnt % 1000 == 0 and cnt > 0:
                    print(f"  Processed {cnt} samples...")
        except Exception as e:
            print(f"  Warning: Could not load the_stack: {e}")
            source = "synthetic"

    if source == "sharegpt":
        try:
            from datasets import load_dataset
            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
            cnt = 0
            for sample in ds:
                if cnt >= max_samples:
                    break
                conversations = sample.get("conversations", [])
                text = " ".join([c.get("value", "") for c in conversations])
                if len(text) > 100:
                    tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens_per_sample]
                    if len(tokens) > suffix_len + 10:
                        token_list.append(tokens)
                        cnt += 1
        except Exception as e:
            print(f"  Warning: Could not load ShareGPT: {e}")
            source = "synthetic"

    if source == "synthetic" or len(token_list) == 0:
        print("  Using synthetic data...")
        for i in range(min(1000, max_samples)):
            text = f"def function_{i}(x, y):\n    result = x + y\n    return result\n\n"
            token_list.append(tokenizer.encode(text, add_special_tokens=False))

    ds = RESTDatastoreV3(suffix_len=suffix_len, device=device)
    ds.build_from_tokens(token_list)
    return ds


# ======================
# REST SD Core
# ======================

@torch.no_grad()
def rest_sd_forward_one_prompt_v3(
    model,
    tokenizer,
    datastore: RESTDatastoreV3,
    sliding_window: CPUSlidingWindow,
    enc: Dict[str, torch.Tensor],
    stop_ids_tensor: torch.Tensor,
    K_max: int = 16,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
):
    device = enc["input_ids"].device
    prompt_len = int(enc["input_ids"].shape[1])

    # one-time prompt init (allowed gpu->cpu once)
    prompt_tokens_list = enc["input_ids"][0].cpu().tolist()
    sliding_window.init_from_prompt(prompt_tokens_list)

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

    # Prefill timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = model(**enc, use_cache=True, output_hidden_states=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.time() - t0

    past_kv = out.past_key_values
    if isinstance(past_kv, tuple):
        past_kv = DynamicCache.from_legacy_cache(past_kv)

    last_logits = out.logits[:, -1, :]

    N_forward_prefill = 1
    N_forward_decode = 0
    draft_attempts = 0
    draft_accepted_tokens = 0

    # First token
    base_token_t = sample_from_logits_1d(last_logits, do_sample, temperature, top_p, top_k)
    base_tensor = base_token_t.view(1, 1)
    base_token_int = int(base_token_t.item())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    out_first = model(base_tensor, past_key_values=past_kv, use_cache=True, output_hidden_states=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time += time.time() - t1
    N_forward_prefill += 1

    ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
    cur_len += 1
    sliding_window.push_tokens([base_token_int])

    past_kv = crop_past_key_values(out_first.past_key_values, cur_len)
    last_logits = out_first.logits[:, -1, :]

    if bool(stop_mask_1d(base_tensor.view(-1).long(), stop_ids_tensor).any().item()):
        stats = {
            "prefill_time": float(prefill_time), "decode_time": 0.0,
            "N_forward_total": int(N_forward_prefill), "N_forward_decode": 0,
            "N_tokens_decode": 0, "draft_attempts": 0, "draft_accepted_tokens": 0,
        }
        return ids_buf[:, :cur_len], stats, prompt_len

    generated = 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode = time.time()

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        if remaining <= 0:
            break

        base_token_t = sample_from_logits_1d(last_logits, do_sample, temperature, top_p, top_k)
        base_tensor = base_token_t.view(1, 1)
        base_token_int = int(base_token_t.item())  # unavoidable scalar sync

        draft_tokens_tensor = None
        draft_start_pos: Optional[int] = None
        K_candidate = 0

        if remaining > 1 and sliding_window.is_full:
            qh = sliding_window.get_hash()
            qt = sliding_window.get_window_tokens()
            pack = datastore.lookup(qh, qt, base_token_int, min(K_max, remaining - 1))
            if pack is not None:
                draft_1d, draft_start_pos = pack
                if draft_1d.numel() > 0:
                    draft_tokens_tensor = draft_1d.unsqueeze(0)  # [1,K]
                    K_candidate = int(draft_1d.numel())

        if K_candidate > 0 and draft_tokens_tensor is not None:
            input_tokens = torch.cat([base_tensor, draft_tokens_tensor], dim=1)
            draft_attempts += 1
        else:
            input_tokens = base_tensor
            K_candidate = 0
            draft_start_pos = None

        out_merged = model(input_tokens, past_key_values=past_kv, use_cache=True, output_hidden_states=False)
        N_forward_decode += 1

        logits_new = out_merged.logits
        past_kv_new = out_merged.past_key_values

        base_is_stop = bool(stop_mask_1d(base_tensor.view(-1).long(), stop_ids_tensor).any().item())
        if base_is_stop:
            ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
            cur_len += 1
            sliding_window.push_tokens([base_token_int])
            past_kv = crop_past_key_values(past_kv_new, cur_len)
            generated += 1
            break

        n_draft_acc = 0
        mismatch = False
        mismatch_token_t: Optional[torch.Tensor] = None
        mismatch_token_int: Optional[int] = None

        if K_candidate > 0 and draft_tokens_tensor is not None:
            verify_logits = logits_new[:, :K_candidate, :]
            if not do_sample:
                preds = torch.argmax(verify_logits, dim=-1)
            else:
                preds = sample_from_logits_batch(
                    verify_logits, do_sample=True,
                    temperature=temperature, top_p=top_p, top_k=top_k
                )

            matches = (preds == draft_tokens_tensor)
            mismatch_pos = torch.nonzero(~matches[0], as_tuple=False)

            if mismatch_pos.numel() == 0:
                n_draft_acc = K_candidate
            else:
                first_false = int(mismatch_pos[0, 0].item())
                n_draft_acc = first_false
                if do_sample:
                    mismatch = True
                    mismatch_token_t = preds[:, first_false:first_false + 1]
                    mismatch_token_int = int(mismatch_token_t.item())

            # stop truncation within accepted draft
            stop_after_update = False
            if n_draft_acc > 0:
                prefix = draft_tokens_tensor[:, :n_draft_acc].view(-1).long()
                sm = stop_mask_1d(prefix, stop_ids_tensor)
                sp = torch.nonzero(sm, as_tuple=False)
                if sp.numel() > 0:
                    first_stop = int(sp[0, 0].item())
                    n_draft_acc = first_stop + 1
                    mismatch = False
                    mismatch_token_t = None
                    mismatch_token_int = None
                    stop_after_update = True
        else:
            stop_after_update = False

        draft_accepted_tokens += int(n_draft_acc)

        # helper: update sliding window WITHOUT GPU->CPU
        def get_accepted_draft_ints(n_acc: int) -> List[int]:
            if n_acc <= 0 or draft_start_pos is None:
                return []
            # take from CPU numpy tokens (no cuda sync)
            return [int(x) for x in datastore.tokens[draft_start_pos:draft_start_pos + n_acc]]

        if mismatch and mismatch_token_t is not None:
            prefix_without_mismatch = 1 + n_draft_acc
            target_len_prefix = cur_len + prefix_without_mismatch

            ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
            cur_len += 1

            committed = [base_token_int]
            if n_draft_acc > 0:
                ids_buf[0, cur_len:cur_len + n_draft_acc] = draft_tokens_tensor[0, :n_draft_acc]
                cur_len += n_draft_acc
                committed.extend(get_accepted_draft_ints(n_draft_acc))

            sliding_window.push_tokens(committed)

            past_kv_prefix = crop_past_key_values(past_kv_new, target_len_prefix)
            generated += prefix_without_mismatch

            out_fix = model(mismatch_token_t, past_key_values=past_kv_prefix, use_cache=True, output_hidden_states=False)
            N_forward_decode += 1

            ids_buf[0, cur_len:cur_len + 1] = mismatch_token_t[0]
            cur_len += 1
            sliding_window.push_tokens([mismatch_token_int])

            past_kv = crop_past_key_values(out_fix.past_key_values, cur_len)
            last_logits = out_fix.logits[:, -1, :]
            generated += 1

            if bool(stop_mask_1d(mismatch_token_t.view(-1).long(), stop_ids_tensor).any().item()):
                break

        else:
            total_accepted = 1 + n_draft_acc
            target_len = cur_len + total_accepted

            ids_buf[0, cur_len:cur_len + 1] = base_tensor[0]
            cur_len += 1

            committed = [base_token_int]
            if n_draft_acc > 0 and draft_tokens_tensor is not None:
                ids_buf[0, cur_len:cur_len + n_draft_acc] = draft_tokens_tensor[0, :n_draft_acc]
                cur_len += n_draft_acc
                committed.extend(get_accepted_draft_ints(n_draft_acc))

            sliding_window.push_tokens(committed)

            past_kv = crop_past_key_values(past_kv_new, target_len)
            last_logits = logits_new[:, total_accepted - 1, :]
            generated += total_accepted

            if stop_after_update:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_time = time.time() - t_decode

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

def run_rest_sd_v3(args):
    set_seed(args.seed)
    tokenizer, model, device = load_model_and_tokenizer(args.model_path, args.gpu_id)

    if args.build_datastore:
        print("\n[REST V3] Building datastore...")
        datastore = build_datastore_v3(
            tokenizer,
            args.datastore_source,
            suffix_len=args.suffix_len,
            max_samples=args.datastore_max_samples,
            device=device,
        )
        datastore.save(args.datastore_path)
        print("[REST V3] Datastore built. Exiting.")
        return

    datastore = RESTDatastoreV3(suffix_len=args.suffix_len, device=device)
    if args.datastore_path:
        datastore.load(args.datastore_path)
    else:
        print("[REST V3] Warning: No datastore. Building minimal synthetic...")
        datastore = build_datastore_v3(tokenizer, "synthetic", suffix_len=args.suffix_len, max_samples=100, device=device)

    sliding_window = CPUSlidingWindow(n=datastore.suffix_len)

    stop_ids = get_stop_token_ids(model, tokenizer)
    stop_ids_tensor = torch.tensor(stop_ids, device=device, dtype=torch.long)

    print(f"\n[Config] Method: REST V3.1 (Production-Grade)")
    print(f"[Config] suffix_len={datastore.suffix_len}, K={args.K}")
    print(f"[Config] Datastore: {datastore.total_tokens} tokens, {datastore.num_docs} docs")
    print(f"[Config] Sampling: do_sample={args.do_sample}, top_k={args.top_k}")

    print("\n[Warmup]...")
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
            print(f"\n=== [REST V3.1] Task: {task} ===")
            prompts = get_dataset_prompts(task, args.samples_per_task)

            for idx, prompt in enumerate(prompts, start=1):
                print(f"  - [{task}] sample {idx}/{len(prompts)}")
                enc = encode_prompt(tokenizer, prompt, device)

                out_ids, stats, prompt_len = rest_sd_forward_one_prompt_v3(
                    model=model,
                    tokenizer=tokenizer,
                    datastore=datastore,
                    sliding_window=sliding_window,
                    enc=enc,
                    stop_ids_tensor=stop_ids_tensor,
                    K_max=args.K,
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
                    "suffix_len": datastore.suffix_len,
                    "verification": "sample_then_compare" if args.do_sample else "greedy",
                    "do_sample": args.do_sample,
                    "temperature": args.temperature if args.do_sample else None,
                    "top_p": args.top_p if args.do_sample else None,
                    "top_k": args.top_k if args.do_sample else None,
                    "model_path": args.model_path,
                    "method": "REST_V3_1",
                    "datastore_tokens": datastore.total_tokens,
                    "datastore_docs": datastore.num_docs,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(
                    f"    Prefill: {prefill_time:.4f}s | Decode: {decode_time:.4f}s | "
                    f"Raw: {raw_new} | Eff: {eff_new} | TP: {throughput:.2f} | "
                    f"Ideal: {ideal_speedup:.2f} | Attempts: {stats['draft_attempts']} | DraftAcc: {stats['draft_accepted_tokens']}"
                )

    overall_tp = (total_eff / total_decode_time) if total_decode_time > 0 else 0.0
    overall_speedup = (total_raw / total_fwd_decode) if total_fwd_decode > 0 else 0.0
    acc_ratio = (total_accepted / (total_attempts * args.K)) if (total_attempts > 0 and args.K > 0) else 0.0

    print("\n" + "=" * 60)
    print("REST V3.1 Summary")
    print("=" * 60)
    print(f"Suffix Length       : {datastore.suffix_len}")
    print(f"Max Draft K         : {args.K}")
    print(f"Datastore           : {datastore.total_tokens} tokens, {datastore.num_docs} docs")
    print(f"Total Raw Tokens    : {total_raw}")
    print(f"Total Eff Tokens    : {total_eff}")
    print(f"Total Decode Time   : {total_decode_time:.3f}s")
    print(f"Overall Throughput  : {overall_tp:.2f} tok/s")
    print(f"Ideal Speedup       : {overall_speedup:.2f} tok/step")
    print(f"Draft Accept Ratio  : {acc_ratio:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REST V3.1 (Production-Grade)")

    parser.add_argument("--tasks", type=str, nargs="+", default=["code_edit"])
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, default="rest_v3_results.jsonl")

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)

    parser.add_argument("--suffix-len", type=int, default=DEFAULT_SUFFIX_LEN)
    parser.add_argument("--K", type=int, default=DEFAULT_MAX_DRAFT_LEN)

    parser.add_argument("--datastore-path", type=str, default=None)
    parser.add_argument("--build-datastore", action="store_true")
    parser.add_argument("--datastore-source", type=str, default="the_stack",
                        choices=["the_stack", "sharegpt", "ultrachat", "synthetic"])
    parser.add_argument("--datastore-max-samples", type=int, default=10000)

    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_rest_sd_v3(args)
