#!/usr/bin/env python
# -*- coding: utf-8 -*-




import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Mistral-3.1 robustness (as in your file) ----
try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None  # type: ignore

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None  # type: ignore


# ===========================
# Spec-Bench categories
# ===========================

MT_BENCH_SUBCATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]

SINGLE_TURN_CATEGORIES = [
    "translation",
    "summarization",
    "text_edit",
    "math_reasoning",
    "code_edit",
]

ALL_SPEC_BENCH_CATEGORIES = MT_BENCH_SUBCATEGORIES + SINGLE_TURN_CATEGORIES


# ===========================
# Defaults
# ===========================

MAX_N_GRAM_TOKEN = 4  # PLD: try 4→3→2→1
MIN_N_GRAM_TOKEN = 1
N_GRAM_SEMANTIC = 1   # keep 1-gram semantic

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_MAX_COPY_TOKENS = 32
DEFAULT_SEED = 42
DEFAULT_GPU_ID = 0
DEFAULT_RETRIEVAL_TOPK = 10

# --- sampling-aware offline approx knobs ---
DEFAULT_PMATCH_TOPN = 512   # top-n approximation for nucleus probability (set 0 for exact full softmax; slow)
DEFAULT_PMATCH_EPS = 1e-12

# ---- Mistral-3.1 robustness: minimal chat templates ----
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)
MISTRAL_INST_SINGLE_TURN = "<s>[INST] {user} [/INST]"


# ===========================
# Utils
# ===========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_and_dtype(gpu_id: int) -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


# ===========================
# Mistral-3.1 Robust Loader + Chat Template
# ===========================

def _is_mistral_like(model_path: str, model=None, tokenizer=None) -> bool:
    s = (model_path or "").lower()
    if "mistral" in s or "mixtral" in s or "mistral3" in s:
        return True
    mt = getattr(getattr(model, "config", None), "model_type", "") if model is not None else ""
    if isinstance(mt, str) and ("mistral" in mt.lower() or "mixtral" in mt.lower()):
        return True
    name = getattr(tokenizer, "name_or_path", "") if tokenizer is not None else ""
    if isinstance(name, str) and ("mistral" in name.lower() or "mixtral" in name.lower()):
        return True
    return False


def _load_tokenizer_mistral_fix(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _load_processor_use_fast_false(model_path: str):
    if AutoProcessor is None:
        return None
    try:
        return AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
    except TypeError:
        try:
            return AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        except Exception:
            return None
    except Exception:
        return None


def _from_pretrained_dtype_compat(cls, model_path: str, dtype: torch.dtype, **kwargs):
    try:
        return cls.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def _get_num_layers(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "num_hidden_layers"):
            return int(cfg.num_hidden_layers)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            return int(cfg.text_config.num_hidden_layers)

    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        cfg2 = model.language_model.config
        if hasattr(cfg2, "num_hidden_layers"):
            return int(cfg2.num_hidden_layers)

    raise AttributeError("Cannot find num_hidden_layers on this model/config.")


class ChatApplier:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
        if hasattr(self.tokenizer, "apply_chat_template") and callable(getattr(self.tokenizer, "apply_chat_template")):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
        raise AttributeError("No apply_chat_template found on processor/tokenizer.")


def _ensure_chat_template(tokenizer, model_path: str, model=None):
    if getattr(tokenizer, "chat_template", None):
        return
    if _is_mistral_like(model_path, model=model, tokenizer=tokenizer):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<s>[INST] {{ message['content'] }} [/INST]"
            "{% elif message['role'] == 'assistant' %}"
            " {{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            ""
            "{% endif %}"
        )
    else:
        tokenizer.chat_template = CHATML_TEMPLATE


def encode_chat_prompt(
    chat: ChatApplier,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    model_path: str,
    model=None,
) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": text}]
    prompt_text: Optional[str] = None

    try:
        prompt_text = chat.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(prompt_text, str) or len(prompt_text) == 0:
            prompt_text = None
    except Exception:
        prompt_text = None

    if prompt_text is None:
        _ensure_chat_template(tokenizer, model_path=model_path, model=model)
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if not isinstance(prompt_text, str) or len(prompt_text) == 0:
                    prompt_text = None
        except Exception:
            prompt_text = None

    if prompt_text is None:
        if _is_mistral_like(model_path, model=model, tokenizer=tokenizer):
            prompt_text = MISTRAL_INST_SINGLE_TURN.format(user=text)
        else:
            prompt_text = f"User: {text}\nAssistant:"

    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"], device=device)
    return enc


def _extract_hidden_states(out: Any):
    hs = getattr(out, "hidden_states", None)
    if hs is None and hasattr(out, "language_model_outputs"):
        hs = getattr(out.language_model_outputs, "hidden_states", None)
    if hs is None:
        raise RuntimeError("Output has no hidden_states (need output_hidden_states=True).")
    return hs


def load_model_tokenizer_processor(model_path: str, gpu_id: int, dtype: torch.dtype):
    print(f"[Model] Loading from {model_path} ...")

    processor = _load_processor_use_fast_false(model_path)
    tokenizer = _load_tokenizer_mistral_fix(model_path)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map = {"": gpu_id} if torch.cuda.is_available() else None

    if AutoModelForImageTextToText is not None:
        try:
            model = _from_pretrained_dtype_compat(
                AutoModelForImageTextToText,
                model_path,
                dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            model.eval()
            return tokenizer, processor, model
        except Exception as e:
            print(f"[WARN] AutoModelForImageTextToText failed, fallback to AutoModelForCausalLM: {e}")

    model = _from_pretrained_dtype_compat(
        AutoModelForCausalLM,
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, processor, model


def build_terminators(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Optional[List[int]]:
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

    out: List[int] = []
    seen = set()
    for x in eos_ids:
        if x not in seen:
            out.append(x)
            seen.add(x)

    return out if out else None


def load_spec_bench_prompts(question_path: str, category: str, max_samples: Optional[int]) -> List[str]:
    prompts: List[str] = []
    try:
        with open(question_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("category") != category:
                    continue
                turns = obj.get("turns", [])
                prompt_text = "\n".join(turns) if isinstance(turns, list) else str(turns)
                prompts.append(prompt_text)
                if max_samples is not None and len(prompts) >= max_samples:
                    break
    except FileNotFoundError:
        print(f"[Error] Question file not found: {question_path}")
        return []

    if len(prompts) == 0:
        print(f"[Warn] No samples found: category={category}")
    print(f"[Data] Loaded {len(prompts)} samples for category={category}")
    return prompts


# ===========================
# Vectorized LCP
# ===========================

def calc_lcp_vectorized(
    seq_ids: torch.Tensor,
    source_indices: torch.Tensor,
    query_indices: torch.Tensor,
    max_copy: int,
) -> torch.Tensor:
    """
    seq_ids: [T] long
    source_indices: [L, Q] long (start positions in history, -1 means invalid)
    query_indices:  [Q] long (start positions for target)
    returns: [L, Q] LCP lengths in [0..max_copy]
    """
    padded_seq = F.pad(seq_ids, (0, max_copy), value=-1)
    windows = padded_seq.unfold(0, max_copy, 1)  # [T+1, max_copy]

    q_windows = windows[query_indices]  # [Q, max_copy]

    safe_sources = source_indices.clamp(min=0, max=windows.size(0) - 1)
    s_windows = windows[safe_sources]  # [L, Q, max_copy]

    matches = (s_windows == q_windows.unsqueeze(0))  # [L, Q, max_copy]
    lcp_lens = matches.cumprod(dim=-1).sum(dim=-1)

    valid_mask = (source_indices >= 0)
    lcp_lens = lcp_lens * valid_mask
    return lcp_lens


# ===========================
# Online simulation helpers
# ===========================

def simulate_nonoverlap_semantic(lcps_after_base_1d: np.ndarray) -> Tuple[int, int]:
    """Greedy *online* walk for semantic retrieval SD.

    Here each entry m means **draft length AFTER the base token**.
    Online greedy step: accepted = 1 + m; and jump by accepted.

    Returns:
        total_tokens: total accepted tokens
        total_steps:  number of (target) forward passes
    """
    Q = int(lcps_after_base_1d.shape[0])
    cursor = 0
    total_tokens = 0
    total_steps = 0
    while cursor < Q:
        m = int(lcps_after_base_1d[cursor])
        accepted = 1 + max(0, m)
        if cursor + accepted > Q:
            accepted = Q - cursor
            if accepted <= 0:
                break
        total_tokens += accepted
        total_steps += 1
        cursor += accepted
    return total_tokens, total_steps


def simulate_sampling_expected_dp(
    lcp_after_base_1d: np.ndarray,
    kcand_1d: np.ndarray,
    base_pos_1d: np.ndarray,
    p_match_1d: np.ndarray,
    max_copy: int,
) -> Tuple[float, float]:
    """
    Sampling-aware offline approximation with *expected* costs, while staying on the same AR trajectory.

    Model:
      - propose K = kcand[t]
      - deterministic LCP = lcp[t] (<=K)
      - within the first M=min(lcp,K), the token equals AR token; it matches with prob p_match[pos]
      - if accept < K => sampling mismatch branch:
            +1 mismatch token, +1 extra forward (KV repair)
            advance tokens = 2 + accept
            forward cost   = 2
        else:
            full accept:
            advance tokens = 1 + K
            forward cost   = 1

    DP computes expected total advanced tokens and expected total forward passes from cursor=0.
    """
    Q = int(lcp_after_base_1d.shape[0])
    if Q <= 0:
        return 0.0, 0.0

    E_tok = np.zeros(Q + 1, dtype=np.float64)
    E_fwd = np.zeros(Q + 1, dtype=np.float64)

    T = int(p_match_1d.shape[0])

    for i in range(Q - 1, -1, -1):
        K = int(kcand_1d[i])
        if K <= 0:
            jump = 1
            nxt = min(i + jump, Q)
            E_tok[i] = float(nxt - i) + E_tok[nxt]
            E_fwd[i] = 1.0 + E_fwd[nxt]
            continue

        lcp = int(lcp_after_base_1d[i])
        M = min(int(max_copy), K, max(0, lcp))
        start_abs = int(base_pos_1d[i]) + 1  # token position of t+1 in full_ids

        # probabilities over accept=a (a in [0..M])
        probs: List[float] = []
        prefix = 1.0
        for a in range(0, M):
            pos = start_abs + a
            p = float(p_match_1d[pos]) if (0 <= pos < T) else 0.0
            p = min(max(p, 0.0), 1.0)
            probs.append(prefix * (1.0 - p))  # fail at a+1
            prefix *= p
        probs.append(prefix)  # accept M (no fail in first M)

        # normalize (numerical)
        s = sum(probs)
        if s <= 0:
            probs = [1.0] + [0.0] * M
        else:
            probs = [x / s for x in probs]

        exp_tok = 0.0
        exp_fwd = 0.0

        # outcome accept=a
        for a, pr in enumerate(probs):
            if pr <= 0:
                continue

            full_accept = (a == K)  # only possible when K<=lcp and no fail
            if full_accept:
                jump = 1 + K
                cost = 1.0
            else:
                jump = 2 + a   # base + a accepted + mismatch token
                cost = 2.0     # merged + repair

            nxt = min(i + jump, Q)
            adv = float(nxt - i)

            exp_tok += pr * (adv + E_tok[nxt])
            exp_fwd += pr * (cost + E_fwd[nxt])

        E_tok[i] = exp_tok
        E_fwd[i] = exp_fwd

    return float(E_tok[0]), float(E_fwd[0])


# ===========================
# Sampling probability (teacher-forcing) helpers
# ===========================

def _compute_p_match_topn(
    logits_2d: torch.Tensor,      # [N, V] predicting next tokens
    next_ids: torch.Tensor,       # [N]
    temperature: float,
    top_p: float,
    top_k: int,
    topn: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Approximate P(sample == next_ids) under (temperature, top_k, top_p) by only considering top-n logits.

    - If next_id is not in top-n (or filtered out by top_k/top_p), prob=0.
    - Good enough for offline layer selection; fast and memory-safe.
    """
    assert logits_2d.dim() == 2
    N, V = logits_2d.shape
    if N <= 0:
        return torch.empty((0,), device=logits_2d.device, dtype=torch.float32)

    temp = max(float(temperature), 1e-6)
    top_k = int(top_k) if top_k is not None else 0
    top_p = float(top_p) if top_p is not None else 1.0

    k_eff = max(1, int(topn))
    if top_k > 0:
        k_eff = max(k_eff, top_k)
    k_eff = min(k_eff, V)

    # top-k over scaled logits (scale doesn't change ordering, but keeps consistent)
    vals, idxs = torch.topk(logits_2d, k=k_eff, dim=-1, largest=True, sorted=True)
    vals = vals / temp

    probs = torch.softmax(vals, dim=-1)  # [N, k_eff]

    # apply top_k (mask beyond top_k)
    if top_k > 0 and top_k < k_eff:
        mask_k = (torch.arange(k_eff, device=probs.device).view(1, -1) < top_k)
        probs = probs * mask_k
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        probs = probs / denom

    # nucleus cutoff on this truncated distribution
    cdf = probs.cumsum(dim=-1)  # [N, k_eff]
    if 0.0 < top_p < 1.0:
        exceed = cdf > top_p
        any_exceed = exceed.any(dim=-1)
        first_exceed = torch.where(
            any_exceed,
            exceed.to(torch.int32).argmax(dim=-1),
            torch.full((N,), k_eff - 1, device=probs.device, dtype=torch.int32),
        )
        cutoff = first_exceed.to(torch.int64)  # inclusive
    else:
        cutoff = torch.full((N,), k_eff - 1, device=probs.device, dtype=torch.int64)

    mass_kept = cdf.gather(1, cutoff.view(-1, 1)).squeeze(1).clamp_min(eps)  # [N]

    # locate next_ids within idxs
    match = idxs.eq(next_ids.view(-1, 1))
    match_any = match.any(dim=-1)
    pos = match.to(torch.int32).argmax(dim=-1).to(torch.int64)  # valid only if match_any

    within = match_any & (pos <= cutoff)
    p_at_pos = probs.gather(1, pos.view(-1, 1)).squeeze(1)

    out = torch.zeros((N,), device=probs.device, dtype=torch.float32)
    out = torch.where(within, p_at_pos / mass_kept, out)
    return out


def compute_p_match_from_teacher_forcing(
    out_logits: torch.Tensor,     # [1, T, V] or [T, V]
    full_ids: torch.Tensor,       # [T]
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    p_match_topn: int,
    eps: float = 1e-12,
) -> Optional[np.ndarray]:
    """
    Return p_match[pos] = P(sample == full_ids[pos]) under sampling policy, for pos>=1.
    p_match[0] is set to 1.0 (unused).
    """
    if not do_sample:
        return None

    if out_logits.dim() == 3:
        logits = out_logits[0]
    else:
        logits = out_logits
    assert logits.dim() == 2, "logits must be [T, V]"
    T = int(full_ids.shape[0])
    if T <= 1:
        return np.ones((T,), dtype=np.float32)

    logits_pred = logits[:-1, :]          # predicts token at pos=1..T-1
    next_ids = full_ids[1:].to(logits_pred.device)

    # exact full softmax (slow) if p_match_topn <= 0 and no nucleus/topk; otherwise still expensive
    if p_match_topn <= 0 and (top_k is None or int(top_k) <= 0) and not (0.0 < float(top_p) < 1.0):
        temp = max(float(temperature), 1e-6)
        lp = torch.log_softmax(logits_pred / temp, dim=-1)
        p = torch.exp(lp.gather(1, next_ids.view(-1, 1)).squeeze(1)).to(torch.float32)
        p_full = torch.cat([torch.ones((1,), device=p.device), p], dim=0)
        return p_full.detach().cpu().numpy().astype(np.float32)

    # top-n approximation (default)
    p_next = _compute_p_match_topn(
        logits_2d=logits_pred,
        next_ids=next_ids,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        topn=int(p_match_topn),
        eps=eps,
    )  # [T-1]
    p_full = torch.cat([torch.ones((1,), device=p_next.device), p_next], dim=0)
    return p_full.detach().cpu().numpy().astype(np.float32)


# ===========================
# Token baseline: PLD-style (4→3→2→1)
# ===========================

class PLDMatcher:
    def __init__(self, max_n: int = 4, min_n: int = 1):
        self.max_n = max_n
        self.min_n = min_n
        self.history_maps: Dict[int, Dict[Tuple[int, ...], int]] = {
            n: {} for n in range(min_n, max_n + 1)
        }

    def reset(self):
        for n in self.history_maps:
            self.history_maps[n].clear()

    def update_history(self, ids_list: List[int], pos: int):
        for n in range(self.min_n, self.max_n + 1):
            start = pos - n + 1
            if start >= 0:
                ngram = tuple(ids_list[start:pos + 1])
                self.history_maps[n][ngram] = start

    def find_match(self, ids_list: List[int], current_pos: int) -> Tuple[int, int]:
        for n in range(self.max_n, self.min_n - 1, -1):
            start = current_pos - n
            if start < 0:
                continue
            ngram = tuple(ids_list[start:current_pos])
            matched_start = self.history_maps[n].get(ngram, -1)
            if matched_start != -1:
                matched_end = matched_start + n - 1
                return matched_end, n
        return -1, 0


def run_pld_baseline_with_kcand(
    ids_list: List[int],
    max_n: int,
    min_n: int,
    max_copy: int,
) -> Tuple[List[int], List[int]]:
    """
    Returns:
      - draft_lens_after_base[t] aligned with positions t from max_n..T-1
      - kcand[t] = proposed draft length (<= max_copy) under online buffer constraint
    """
    T = len(ids_list)
    if T <= max_n:
        return [], []

    matcher = PLDMatcher(max_n=max_n, min_n=min_n)
    draft_lens: List[int] = []
    kcands: List[int] = []

    for t in range(max_n, T):
        matched_end, matched_n = matcher.find_match(ids_list, t)

        draft_lcp = 0
        kcand = 0

        if matched_end >= 0:
            base_pos_hist = matched_end + 1
            draft_start_hist = matched_end + 2
            draft_start_tgt = t + 1

            if base_pos_hist < T and ids_list[base_pos_hist] == ids_list[t]:
                if draft_start_hist < t and draft_start_tgt < T:
                    max_avail = t - draft_start_hist
                    max_tgt = T - draft_start_tgt
                    kcand = min(int(max_copy), int(max_avail), int(max_tgt))
                    if kcand > 0:
                        l = 0
                        while (
                            l < kcand
                            and draft_start_hist + l < T
                            and draft_start_tgt + l < T
                            and ids_list[draft_start_hist + l] == ids_list[draft_start_tgt + l]
                        ):
                            l += 1
                        draft_lcp = l

        draft_lens.append(draft_lcp)
        kcands.append(kcand)

        matcher.update_history(ids_list, t - 1)

    return draft_lens, kcands


# ===========================
# Main analysis
# ===========================

def analyze_model_layers_fast(
    model_path: str,
    question_path: str,
    gpu_id: int,
    max_new_tokens: int,
    max_copy_tokens: int,
    max_samples_per_category: Optional[int],
    retrieval_topk: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    p_match_topn: int,
):
    device, dtype = get_device_and_dtype(gpu_id)

    tokenizer, processor, model = load_model_tokenizer_processor(model_path, gpu_id, dtype)
    chat = ChatApplier(tokenizer, processor)
    num_layers = _get_num_layers(model)

    terminators = build_terminators(model, tokenizer)
    print(f"[Info] Using terminators (eos_token_id): {terminators}")
    print(f"[Gen ] do_sample={do_sample}, temperature={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"[Sem ] retrieval_topk={retrieval_topk} (top-k similar candidates, pick FIRST valid)")
    print(f"[PLD ] n-gram fallback: {MAX_N_GRAM_TOKEN}→{MIN_N_GRAM_TOKEN}")
    if do_sample:
        print(f"[Approx] sampling-aware offline DP enabled; p_match_topn={p_match_topn} (0=exact full softmax, slow)")

    results_semantic_avg: Dict[str, np.ndarray] = {}
    results_pld_avg: Dict[str, float] = {}
    detailed_stats: Dict[str, dict] = {}

    for task in ALL_SPEC_BENCH_CATEGORIES:
        prompts = load_spec_bench_prompts(question_path, task, max_samples=max_samples_per_category)
        if len(prompts) == 0:
            continue

        layer_tok_sum = torch.zeros(num_layers, device=device, dtype=torch.float32)
        layer_step_sum = torch.zeros(num_layers, device=device, dtype=torch.float32)

        pld_tok_sum = 0.0
        pld_step_sum = 0.0

        print(f"\n  > Analyzing category={task} ...")
        t0_task = time.time()

        for prompt in prompts:
            enc = encode_chat_prompt(chat, tokenizer, prompt, device, model_path=model_path, model=model)
            prompt_len = int(enc["input_ids"].shape[1])

            gen_kwargs = dict(
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            if terminators is not None:
                gen_kwargs["eos_token_id"] = terminators

            if do_sample:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )
            else:
                gen_kwargs.update(do_sample=False)

            with torch.no_grad():
                outputs = model.generate(**enc, **gen_kwargs)

            full_ids = outputs[0]  # [T]
            seq_len = int(full_ids.shape[0])

            min_context = max(N_GRAM_SEMANTIC, MAX_N_GRAM_TOKEN)
            start_pos = prompt_len + min_context
            if seq_len <= start_pos + 1:
                continue

            # Full forward once (offline analysis): hidden states + logits
            with torch.no_grad():
                out = model(
                    input_ids=full_ids.unsqueeze(0),
                    output_hidden_states=True,
                    use_cache=False,
                )
            hs = _extract_hidden_states(out)

            # sampling-aware: compute p_match once per sample (based on teacher-forcing logits)
            p_match = compute_p_match_from_teacher_forcing(
                out_logits=out.logits,
                full_ids=full_ids,
                do_sample=do_sample,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                p_match_topn=int(p_match_topn),
                eps=DEFAULT_PMATCH_EPS,
            )

            # [L, T, D]
            all_hiddens = torch.stack(hs[1:], dim=0)
            if all_hiddens.dim() == 4:
                all_hiddens = all_hiddens.squeeze(1)  # [L, T, D]

            keys_norm = all_hiddens / (all_hiddens.norm(dim=-1, keepdim=True) + 1e-8)

            # sim_matrix: [L, T, T]
            sim_matrix = torch.matmul(keys_norm, keys_norm.transpose(1, 2))

            # Online-faithful causal mask
            causal_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril(diagonal=-1)
            sim_matrix = sim_matrix.masked_fill(~causal_mask, -1e9)

            # base positions t, and query hidden row is (t-1)
            base_pos = torch.arange(start_pos, seq_len - 1, device=device, dtype=torch.long)  # [Q]
            q_row_indices = base_pos - 1  # [Q]
            Q = int(base_pos.numel())
            if Q <= 0:
                continue

            # [L, Q, T]
            relevant_sims = sim_matrix[:, q_row_indices, :]

            # top-k similar candidates (sorted)
            k = int(max(1, retrieval_topk))
            k = min(k, seq_len)
            top_vals, top_idxs = torch.topk(relevant_sims, k=k, dim=-1, largest=True, sorted=True)  # [L,Q,k]

            # Token-alignment filter: seq[j+1] must equal base_token seq[t]
            base_tokens = full_ids[base_pos]  # [Q]
            hist_next = top_idxs + 1          # [L,Q,k]
            draft_start = top_idxs + 2        # [L,Q,k]

            # bounds under online buffer length (= current base_pos t)
            t_buf = base_pos.view(1, Q, 1)  # [1,Q,1]
            hist_next_in = hist_next < t_buf
            has_tail = draft_start < t_buf

            hist_next_clamped = hist_next.clamp(min=0, max=seq_len - 1)
            gathered = torch.take(full_ids, hist_next_clamped.reshape(-1)).view(hist_next.shape)  # [L,Q,k]
            tok_ok = (gathered == base_tokens.view(1, Q, 1))

            sim_ok = top_vals > -1e8
            valid = sim_ok & hist_next_in & has_tail & tok_ok  # [L,Q,k]

            # pick FIRST valid candidate along k
            ranks = torch.arange(k, device=device).view(1, 1, k)
            big = torch.full_like(ranks, k)
            rank_mat = torch.where(valid, ranks, big)
            best_rank = rank_mat.min(dim=-1).values
            has_any = best_rank < k

            best_rank_clamped = best_rank.clamp(max=k - 1)
            chosen_j = top_idxs.gather(-1, best_rank_clamped.unsqueeze(-1)).squeeze(-1)
            chosen_source = torch.where(
                has_any,
                chosen_j + 2,                         # draft_start_hist
                torch.full_like(chosen_j, -1),
            )  # [L,Q] start pos in history for draft

            # semantic proposed Kcand per (layer, q): min(K_max, avail_history)
            avail_hist = (base_pos.view(1, Q) - chosen_source).clamp(min=0)  # [L,Q]
            sem_kcand = torch.where(
                chosen_source >= 0,
                torch.minimum(avail_hist, torch.full_like(avail_hist, int(max_copy_tokens))),
                torch.zeros_like(avail_hist),
            ).to(torch.int32)  # [L,Q]

            # LCP between seq[draft_start:] and seq[t+1:]
            target_start = base_pos + 1
            sem_lcps = calc_lcp_vectorized(
                seq_ids=full_ids,
                source_indices=chosen_source,
                query_indices=target_start,
                max_copy=int(max_copy_tokens),
            )  # [L,Q]

            # Cap by available history tail length (online buffer)
            avail_cap = avail_hist.clamp(max=int(max_copy_tokens)).to(sem_lcps.dtype)
            sem_lcps = torch.minimum(sem_lcps, avail_cap)

            # Online simulation
            base_pos_cpu = base_pos.detach().cpu().numpy().astype(np.int32)

            sem_lcps_cpu = sem_lcps.to(torch.int16).cpu().numpy()      # [L,Q]
            sem_kcand_cpu = sem_kcand.cpu().numpy().astype(np.int16)   # [L,Q]

            for li in range(num_layers):
                if not do_sample:
                    tok_n, step_n = simulate_nonoverlap_semantic(sem_lcps_cpu[li])
                else:
                    # sampling-aware DP (expected tokens / expected forward passes)
                    tok_n, step_n = simulate_sampling_expected_dp(
                        lcp_after_base_1d=sem_lcps_cpu[li].astype(np.int32),
                        kcand_1d=sem_kcand_cpu[li].astype(np.int32),
                        base_pos_1d=base_pos_cpu,
                        p_match_1d=p_match,
                        max_copy=int(max_copy_tokens),
                    )
                layer_tok_sum[li] += float(tok_n)
                layer_step_sum[li] += float(step_n)

            # ======================
            # PLD baseline
            # ======================
            ids_list = full_ids.tolist()
            pld_draft_lens, pld_kcands = run_pld_baseline_with_kcand(
                ids_list,
                max_n=MAX_N_GRAM_TOKEN,
                min_n=MIN_N_GRAM_TOKEN,
                max_copy=int(max_copy_tokens)
            )

            offset = start_pos - MAX_N_GRAM_TOKEN
            if offset >= 0 and offset < len(pld_draft_lens):
                pld_valid_lcp = pld_draft_lens[offset:][:Q]
                pld_valid_k   = pld_kcands[offset:][:Q]

                pld_lcp_arr = np.asarray(pld_valid_lcp, dtype=np.int32)
                pld_k_arr   = np.asarray(pld_valid_k, dtype=np.int32)

                if not do_sample:
                    tok_n, step_n = simulate_nonoverlap_semantic(pld_lcp_arr)
                else:
                    tok_n, step_n = simulate_sampling_expected_dp(
                        lcp_after_base_1d=pld_lcp_arr,
                        kcand_1d=pld_k_arr,
                        base_pos_1d=base_pos_cpu,
                        p_match_1d=p_match,
                        max_copy=int(max_copy_tokens),
                    )
                pld_tok_sum += float(tok_n)
                pld_step_sum += float(step_n)

        print(f"    Category {task} done in {time.time() - t0_task:.2f}s")

        l_sum = layer_tok_sum.detach().float().cpu().numpy()
        l_cnt = layer_step_sum.detach().float().cpu().numpy()
        safe_cnt = np.where(l_cnt == 0, 1, l_cnt)

        sem_avg_curve = l_sum / safe_cnt
        results_semantic_avg[task] = sem_avg_curve

        best_l = int(np.argmax(sem_avg_curve))
        best_val = float(sem_avg_curve[best_l])

        pld_avg = (pld_tok_sum / pld_step_sum) if pld_step_sum > 0 else 1.0
        results_pld_avg[task] = pld_avg

        detailed_stats[task] = {
            "pld_avg_accepted": float(pld_avg),
            "semantic_avg_curve": sem_avg_curve,
            "best_layer": int(best_l),
            "best_layer_avg_accepted": float(best_val),
            "retrieval_topk": int(retrieval_topk),
        }

        print(f"    [PLD-{MAX_N_GRAM_TOKEN}→{MIN_N_GRAM_TOKEN}] Avg accepted/step: {pld_avg:.2f}")
        print(f"    [Semantic] best_layer=L{best_l} | Avg accepted/step={best_val:.2f}")

    return results_semantic_avg, results_pld_avg, detailed_stats, num_layers


# ===========================
# Save mapping + plot
# ===========================

def save_best_layer_mapping(detailed_stats: Dict[str, dict], out_path: str):
    best_map: Dict[str, int] = {}
    for task, stats in detailed_stats.items():
        if "best_layer" in stats:
            best_map[task] = int(stats["best_layer"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_map, f, ensure_ascii=False, indent=2)

    print(f"\n[Save] Best-layer mapping saved to {out_path}")
    for t in sorted(best_map.keys()):
        print(f"  - {t:15s} -> L{best_map[t]}")


def plot_semantic_curves_beautiful(
    res_sem_avg: Dict[str, np.ndarray],
    res_pld_avg: Dict[str, float],
    detailed_stats: Dict[str, dict],
    num_layers: int,
    out_path: str,
    categories: Optional[List[str]] = None,
    smooth_sigma: float = 1.5,
):
    if categories is None:
        categories = SINGLE_TURN_CATEGORIES

    tasks = [t for t in categories if t in res_sem_avg]
    if not tasks:
        print("[Plot] No tasks found in results; skip plotting.")
        return

    colors = [
        '#2E86AB', '#E94F37', '#4CAF50', '#FF9800',
        '#9C27B0', '#00BCD4', '#795548', '#607D8B',
    ]
    markers = ['v', 's', '^', 'D', 'o', 'p', 'h', '*']

    task_names = {
        'translation': 'Translation',
        'summarization': 'Summarization',
        'text_edit': 'Text Editing',
        'math_reasoning': 'Math Reasoning',
        'code_edit': 'Code Editing',
        'writing': 'Writing',
        'roleplay': 'Roleplay',
        'reasoning': 'Reasoning',
        'math': 'Math',
        'coding': 'Coding',
        'extraction': 'Extraction',
        'stem': 'STEM',
        'humanities': 'Humanities',
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    layers = np.arange(num_layers)
    legend_elements = []

    for idx, task in enumerate(tasks):
        curve = res_sem_avg[task]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        display_name = task_names.get(task, task.title())

        if smooth_sigma > 0:
            smooth_curve = gaussian_filter1d(curve, sigma=smooth_sigma)
        else:
            smooth_curve = curve

        ax.plot(layers, smooth_curve, color=color, linewidth=2.5, alpha=0.9, zorder=3)
        ax.fill_between(layers, 0, smooth_curve, color=color, alpha=0.15, zorder=1)

        best_layer = int(detailed_stats[task]["best_layer"])
        best_val = float(curve[best_layer])

        ax.scatter(
            [best_layer], [best_val],
            color=color, marker=marker, s=150,
            edgecolors='white', linewidths=2, zorder=5,
        )

        offset_y = 0.1 if idx % 2 == 0 else -0.15
        ax.annotate(
            f'L{best_layer}',
            xy=(best_layer, best_val),
            xytext=(best_layer + 1, best_val + offset_y),
            fontsize=9, color=color, fontweight='bold',
            ha='left', va='center',
        )

        pld_val = res_pld_avg.get(task, 1.0)

        legend_elements.append(
            Line2D(
                [0], [0],
                color=color, linewidth=2.5,
                marker=marker, markersize=8,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=1.5,
                label=f'{display_name} (Best: L{best_layer}, {best_val:.2f} | PLD: {pld_val:.2f})'
            )
        )

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='AR Baseline (1.0)')

    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Avg. Accepted Tokens per Step', fontsize=14, fontweight='bold')
    ax.set_title('Layer-wise Semantic Speculative Decoding Performance', fontsize=16, fontweight='bold', pad=15)

    ax.set_xlim(-0.5, num_layers - 0.5)
    y_max = max([res_sem_avg[t].max() for t in tasks]) * 1.1
    ax.set_ylim(0.8, max(y_max, 2.0))

    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
    )
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"[Plot] Saved beautiful plot to {out_path}")


def plot_all_categories(
    res_sem_avg: Dict[str, np.ndarray],
    res_pld_avg: Dict[str, float],
    detailed_stats: Dict[str, dict],
    num_layers: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    plot_semantic_curves_beautiful(
        res_sem_avg=res_sem_avg,
        res_pld_avg=res_pld_avg,
        detailed_stats=detailed_stats,
        num_layers=num_layers,
        out_path=os.path.join(out_dir, "single_turn_layers.png"),
        categories=SINGLE_TURN_CATEGORIES,
        smooth_sigma=1.5,
    )

    plot_semantic_curves_beautiful(
        res_sem_avg=res_sem_avg,
        res_pld_avg=res_pld_avg,
        detailed_stats=detailed_stats,
        num_layers=num_layers,
        out_path=os.path.join(out_dir, "mt_bench_layers.png"),
        categories=MT_BENCH_SUBCATEGORIES,
        smooth_sigma=1.5,
    )

    plot_semantic_curves_beautiful(
        res_sem_avg=res_sem_avg,
        res_pld_avg=res_pld_avg,
        detailed_stats=detailed_stats,
        num_layers=num_layers,
        out_path=os.path.join(out_dir, "all_categories_layers.png"),
        categories=ALL_SPEC_BENCH_CATEGORIES,
        smooth_sigma=1.5,
    )


def print_summary_table(
    res_sem_avg: Dict[str, np.ndarray],
    res_pld_avg: Dict[str, float],
    detailed_stats: Dict[str, dict],
):
    print("\n" + "=" * 80)
    print("SUMMARY: Semantic SD vs PLD Baseline (Accepted Tokens per Step)")
    print("=" * 80)
    print(f"{'Category':<20} {'Best Layer':<12} {'Semantic':<12} {'PLD (4→1)':<12} {'Improvement':<12}")
    print("-" * 80)

    total_sem = 0.0
    total_pld = 0.0
    count = 0

    for task in sorted(detailed_stats.keys()):
        stats = detailed_stats[task]
        best_layer = stats["best_layer"]
        sem_val = stats["best_layer_avg_accepted"]
        pld_val = res_pld_avg.get(task, 1.0)
        improvement = ((sem_val - pld_val) / pld_val) * 100 if pld_val > 0 else 0.0

        print(f"{task:<20} L{best_layer:<10} {sem_val:<12.3f} {pld_val:<12.3f} {improvement:>+10.1f}%")

        total_sem += sem_val
        total_pld += pld_val
        count += 1

    if count > 0:
        avg_sem = total_sem / count
        avg_pld = total_pld / count
        avg_imp = ((avg_sem - avg_pld) / avg_pld) * 100 if avg_pld > 0 else 0.0
        print("-" * 80)
        print(f"{'AVERAGE':<20} {'':<12} {avg_sem:<12.3f} {avg_pld:<12.3f} {avg_imp:>+10.1f}%")

    print("=" * 80 + "\n")


# ===========================
# Entry
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Analyze layer-wise semantic copyability with retrieval_topk.")

    parser.add_argument("--model-path", type=str, required=True, help="HF model path")
    parser.add_argument("--question-path", type=str, required=True, help="Spec-Bench question.jsonl path")
    parser.add_argument("--output-json", type=str, default="best_layer_mapping.json")
    parser.add_argument("--output-plot", type=str, default="single.png")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for all plots")

    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-copy-tokens", type=int, default=DEFAULT_MAX_COPY_TOKENS)
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=-1,
        help="Cap samples per category. -1 means use all.",
    )

    parser.add_argument(
        "--retrieval-topk",
        type=int,
        default=DEFAULT_RETRIEVAL_TOPK,
        help="Top-k similarity candidates; pick FIRST valid (token-aligned) candidate.",
    )

    # sampling
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling generation")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)

    # NEW: p_match approximation
    parser.add_argument(
        "--p-match-topn",
        type=int,
        default=DEFAULT_PMATCH_TOPN,
        help="Top-n approximation for nucleus probability. 0 = exact full softmax (slow).",
    )

    args = parser.parse_args()
    max_samples = None if args.max_samples_per_category < 0 else int(args.max_samples_per_category)

    set_seed(args.seed)

    res_sem, res_pld, stats, n_layers = analyze_model_layers_fast(
        model_path=args.model_path,
        question_path=args.question_path,
        gpu_id=args.gpu_id,
        max_new_tokens=args.max_new_tokens,
        max_copy_tokens=args.max_copy_tokens,
        max_samples_per_category=max_samples,
        retrieval_topk=int(args.retrieval_topk),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        p_match_topn=int(args.p_match_topn),
    )

    print_summary_table(res_sem, res_pld, stats)
    save_best_layer_mapping(stats, out_path=args.output_json)

    if args.output_dir:
        plot_all_categories(res_sem, res_pld, stats, n_layers, out_dir=args.output_dir)
    else:
        plot_semantic_curves_beautiful(
            res_sem_avg=res_sem,
            res_pld_avg=res_pld,
            detailed_stats=stats,
            num_layers=n_layers,
            out_path=args.output_plot,
            categories=SINGLE_TURN_CATEGORIES,
        )


if __name__ == "__main__":
    main()
