#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Example:
python layer_code.py \
  --model-path /root/autodl-tmp/models/CodeLlama-7b-Instruct-hf \
  --question-path ../spec_bench/long_text.jsonl \
  --output-json best_layer1.json \
  --output-plot single_last.png \
  --max-new-tokens 2048 \
  --max-copy-tokens 32 \
  --retrieval-topk 10 \
  --gpu-id 0
"""

import argparse
import json
import random
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    "project",
    "translation",
    "summarization",
    "text_edit",
    "math_reasoning",
    "code_edit",
]

ALL_SPEC_BENCH_CATEGORIES = SINGLE_TURN_CATEGORIES


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


def encode_chat_prompt(tokenizer: AutoTokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
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
            enc_dict = tokenizer(text, return_tensors="pt")
    else:
        enc_dict = tokenizer(text, return_tensors="pt")

    enc_dict = {k: v.to(device) for k, v in enc_dict.items()}
    if "attention_mask" not in enc_dict:
        enc_dict["attention_mask"] = torch.ones_like(enc_dict["input_ids"], device=device)
    return enc_dict


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

    Here each entry m means **draft length AFTER the base token** (i.e., LCP on target[t+1:],
    not including the base token itself). In an online SD step, accepted tokens per verification are:

        accepted = 1 (base) + m (draft)

    We then jump the cursor by `accepted` to avoid double-counting overlapping drafts.

    Returns:
        total_tokens: total accepted tokens
        total_steps:  number of verifications
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


# ===========================
# Token baseline: PLD-style (4→3→2→1)
# ===========================

class PLDMatcher:
    """
    PLD-style n-gram matcher with fallback: tries 4-gram first, then 3, 2, 1.
    Maintains separate hash maps for each n-gram level.
    
    For consistency with semantic SD, we compute:
      - base token: seq[matched_pos + n] (the token right after the n-gram)
      - draft LCP: match starting from (matched_pos + n + 1) vs (current_pos + 1)
      
    Final accepted = 1 (base) + draft_lcp
    """
    
    def __init__(self, max_n: int = 4, min_n: int = 1):
        self.max_n = max_n
        self.min_n = min_n
        # history_maps[n] stores {ngram_tuple: last_position}
        self.history_maps: Dict[int, Dict[Tuple[int, ...], int]] = {
            n: {} for n in range(min_n, max_n + 1)
        }
    
    def reset(self):
        for n in self.history_maps:
            self.history_maps[n].clear()
    
    def update_history(self, ids_list: List[int], pos: int):
        """Update all n-gram maps with position pos as the END of the n-gram."""
        for n in range(self.min_n, self.max_n + 1):
            start = pos - n + 1
            if start >= 0:
                ngram = tuple(ids_list[start:pos + 1])
                self.history_maps[n][ngram] = start
    
    def find_match(self, ids_list: List[int], current_pos: int) -> Tuple[int, int]:
        """
        Try to find a match using n-gram (4→3→2→1).
        
        Args:
            ids_list: full token sequence
            current_pos: current position (we want to predict ids_list[current_pos])
        
        Returns:
            (matched_end_pos, n): position where n-gram ends, and n value
                                  (-1, 0) if no match found
        """
        for n in range(self.max_n, self.min_n - 1, -1):
            start = current_pos - n
            if start < 0:
                continue
            ngram = tuple(ids_list[start:current_pos])
            matched_start = self.history_maps[n].get(ngram, -1)
            if matched_start != -1:
                matched_end = matched_start + n - 1  # end position of matched n-gram
                return matched_end, n
        return -1, 0


def run_pld_baseline(ids_list: List[int], max_n: int, min_n: int, max_copy: int) -> List[int]:
    """
    PLD-style token matching with fallback (max_n → ... → min_n).
    
    CRITICAL FIXES:
    1. Check base token alignment: ids_list[matched_end+1] must == ids_list[t]
    2. Cap LCP by online buffer length: lcp <= (t - draft_start_hist)
    
    For each position t (starting from max_n), we:
    1. Try to find an n-gram match in history (4→3→2→1)
    2. If found at position j (end of matched n-gram):
       - base token is at j+1, MUST equal ids_list[t] ← FIX 1
       - draft starts at j+2, target starts at t+1
       - compute LCP between seq[j+2:] and seq[t+1:]
       - cap LCP by (t - draft_start_hist) ← FIX 2
       - accepted = 1 (base) + lcp (draft)
    3. If no match: accepted = 1 (just base, no draft)
    
    Returns:
        match_lens: list of "draft LCP after base" for each position from max_n to T-1
    """
    T = len(ids_list)
    if T <= max_n:
        return []
    
    matcher = PLDMatcher(max_n=max_n, min_n=min_n)
    draft_lens: List[int] = []
    
    for t in range(max_n, T):
        # Find match using fallback n-gram
        matched_end, matched_n = matcher.find_match(ids_list, t)
        
        draft_lcp = 0
        if matched_end >= 0:
            base_pos_hist = matched_end + 1
            draft_start_hist = matched_end + 2
            draft_start_tgt = t + 1
            
            # FIX 1: Check base token alignment (CRITICAL!)
            # Online decoder would reject if base tokens don't match
            if base_pos_hist < T and ids_list[base_pos_hist] == ids_list[t]:
                # FIX 2: Cap by online buffer length
                # At position t, we can only read from buffer [0, t)
                # So draft_start_hist must be < t, and lcp <= (t - draft_start_hist)
                if draft_start_hist < t and draft_start_tgt < T:
                    # Maximum draftable length from history
                    max_avail = t - draft_start_hist
                    
                    # Compute LCP with online constraint
                    src = draft_start_hist
                    tgt = draft_start_tgt
                    l = 0
                    while (
                        l < max_copy
                        and l < max_avail  # ← FIX 2: can't exceed online buffer
                        and src + l < T
                        and tgt + l < T
                        and ids_list[src + l] == ids_list[tgt + l]
                    ):
                        l += 1
                    draft_lcp = l
        
        draft_lens.append(draft_lcp)
        
        # Update history with current position (add all n-grams ending at t-1)
        matcher.update_history(ids_list, t - 1)
    
    return draft_lens


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
):
    print(f"[Model] Loading from {model_path} ...")
    device, dtype = get_device_and_dtype(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map={"": gpu_id},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)

    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    terminators = build_terminators(model, tokenizer)
    print(f"[Info] Using terminators (eos_token_id): {terminators}")
    print(f"[Gen ] do_sample={do_sample}, temperature={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"[Sem ] retrieval_topk={retrieval_topk} (top-k similar candidates, pick FIRST valid)")
    print(f"[PLD ] n-gram fallback: {MAX_N_GRAM_TOKEN}→{MIN_N_GRAM_TOKEN}")

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
            enc = encode_chat_prompt(tokenizer, prompt, device)
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

            # We need: t (base token position) and (t+1) exists for LCP after base
            min_context = max(N_GRAM_SEMANTIC, MAX_N_GRAM_TOKEN)
            start_pos = prompt_len + min_context  # base position starts here
            if seq_len <= start_pos + 1:
                continue

            # Full forward to get all hidden states for all layers (offline analysis)
            with torch.no_grad():
                out = model(
                    input_ids=full_ids.unsqueeze(0),
                    output_hidden_states=True,
                    use_cache=False,
                )

            # [L, T, D]
            all_hiddens = torch.stack(out.hidden_states[1:]).squeeze(1)
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

            # gather seq[hist_next]
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
                chosen_j + 2,
                torch.full_like(chosen_j, -1),
            )

            # LCP between seq[draft_start:] and seq[t+1:]
            target_start = base_pos + 1
            sem_lcps = calc_lcp_vectorized(
                seq_ids=full_ids,
                source_indices=chosen_source,
                query_indices=target_start,
                max_copy=int(max_copy_tokens),
            )

            # Cap by available history tail length
            avail = (base_pos.view(1, Q) - chosen_source).clamp(min=0, max=int(max_copy_tokens)).to(sem_lcps.dtype)
            sem_lcps = torch.minimum(sem_lcps, avail)

            # Online simulation for semantic
            sem_lcps_cpu = sem_lcps.to(torch.int16).cpu().numpy()
            for li in range(num_layers):
                tok_n, step_n = simulate_nonoverlap_semantic(sem_lcps_cpu[li])
                layer_tok_sum[li] += float(tok_n)
                layer_step_sum[li] += float(step_n)

            # PLD baseline (CPU)
            ids_list = full_ids.tolist()
            pld_draft_lens = run_pld_baseline(
                ids_list, 
                max_n=MAX_N_GRAM_TOKEN, 
                min_n=MIN_N_GRAM_TOKEN, 
                max_copy=int(max_copy_tokens)
            )

            offset = start_pos - MAX_N_GRAM_TOKEN
            if offset >= 0 and offset < len(pld_draft_lens):
                pld_valid = pld_draft_lens[offset:]
                pld_valid = pld_valid[:Q]
                pld_arr = np.asarray(pld_valid, dtype=np.int32)
                tok_n, step_n = simulate_nonoverlap_semantic(pld_arr)  # same logic: 1 + draft
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
    smooth_sigma: float = 1.2,
):
    """
    Generate a beautiful ACL-style plot.
    
    Features:
    - Clean color palette with better contrast
    - Smart annotation positioning to avoid overlap
    - Professional typography (Times New Roman)
    - Minimal fill for cleaner look
    - Best layer markers with annotations
    """
    if categories is None:
        categories = SINGLE_TURN_CATEGORIES
    
    tasks = [t for t in categories if t in res_sem_avg]
    if not tasks:
        print("[Plot] No tasks found in results; skip plotting.")
        return

    # ===== ACL Style Settings =====
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 25,   # 原本10，现在2.5倍
        'ytick.labelsize': 25,   # 原本10，现在2.5倍
        'legend.fontsize': 9,
        'axes.linewidth': 1.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # ===== Professional Color Palette (Colorblind-friendly) =====
    colors = [
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion/Orange
        '#009E73',  # Bluish Green
        '#CC79A7',  # Reddish Purple
        '#E69F00',  # Orange/Yellow
        '#56B4E9',  # Sky Blue
        '#F0E442',  # Yellow
        '#000000',  # Black
    ]
    
    # Marker styles
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    # Task display names
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
    
    # ===== Create Figure =====
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    
    layers = np.arange(num_layers)
    
    # Store best layer info for smart annotation
    best_layer_info = []
    
    for idx, task in enumerate(tasks):
        curve = res_sem_avg[task]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        display_name = task_names.get(task, task.replace('_', ' ').title())
        
        # Smooth the curve
        if smooth_sigma > 0 and len(curve) > 3:
            smooth_curve = gaussian_filter1d(curve, sigma=smooth_sigma)
        else:
            smooth_curve = curve
        
        # Plot the main curve
        ax.plot(
            layers, 
            smooth_curve, 
            color=color,
            linewidth=2.2,
            alpha=0.9,
            zorder=3,
            label=display_name,
        )
        
        # Light fill under curve
        ax.fill_between(
            layers, 
            np.minimum(smooth_curve, 1.0),  # fill only above baseline
            smooth_curve, 
            color=color, 
            alpha=0.08,
            zorder=1,
        )
        
        # Mark the best layer
        best_layer = int(detailed_stats[task]["best_layer"])
        best_val = float(curve[best_layer])
        
        ax.scatter(
            [best_layer], 
            [best_val], 
            color=color,
            marker=marker,
            s=120,
            edgecolors='white',
            linewidths=1.5,
            zorder=5,
        )
        
        # Store for annotation
        pld_val = res_pld_avg.get(task, 1.0)
        best_layer_info.append({
            'task': display_name,
            'layer': best_layer,
            'value': best_val,
            'pld': pld_val,
            'color': color,
            'idx': idx,
        })
    
    # ===== Smart Annotation (avoid overlap) =====
    # Sort by layer, then by value
    best_layer_info.sort(key=lambda x: (x['layer'], -x['value']))
    
    # Group by layer
    layer_groups = {}
    for info in best_layer_info:
        layer = info['layer']
        if layer not in layer_groups:
            layer_groups[layer] = []
        layer_groups[layer].append(info)
    
    # Add annotations
    for layer, group in layer_groups.items():
        n = len(group)
        for i, info in enumerate(group):
            # Vertical offset to spread overlapping annotations
            if n == 1:
                y_off = 0.08
            else:
                y_off = 0.05 + 0.12 * (i - (n - 1) / 2)
            
            # Horizontal direction based on layer position
            if layer > num_layers * 0.75:
                x_off, ha = -1.5, 'right'
            else:
                x_off, ha = 1.2, 'left'
            
            ax.annotate(
                f'L{info["layer"]}',
                xy=(info['layer'], info['value']),
                xytext=(info['layer'] + x_off, info['value'] + y_off),
                fontsize=8,
                color=info['color'],
                fontweight='bold',
                ha=ha,
                va='center',
            )
    
    # ===== Reference Line (AR baseline) =====
    ax.axhline(
        y=1.0, 
        color='#888888', 
        linestyle=':', 
        linewidth=1.5, 
        alpha=0.7, 
        zorder=1,
    )
    ax.text(
        num_layers - 1, 1.02, 
        'AR Baseline', 
        fontsize=9, 
        color='#666666',
        ha='right',
        va='bottom',
    )
    
    # ===== Axis Labels (去掉标题，标签字体放大2倍) =====
    ax.set_xlabel('Layer Index', fontsize=26, fontweight='bold', labelpad=8)  # 原本13，现在2倍
    ax.set_ylabel('Avg. Accepted Tokens per Step', fontsize=26, fontweight='bold', labelpad=8)  # 原本13，现在2倍
    # 删除标题
    
    # ===== Axis Limits =====
    ax.set_xlim(-0.5, num_layers - 0.5)
    y_vals = [res_sem_avg[t] for t in tasks]
    y_min = min([v.min() for v in y_vals])
    y_max = max([v.max() for v in y_vals])
    y_range = y_max - y_min
    ax.set_ylim(max(0.85, y_min - 0.1 * y_range), y_max + 0.15 * y_range)
    
    # ===== Grid =====
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ===== Legend with Stats (去掉+XX%部分) =====
    # Build custom legend entries WITHOUT PLD comparison percentage
    legend_handles = []
    for info in sorted(best_layer_info, key=lambda x: x['idx']):
        # 去掉 +{imp:.0f}% 部分
        label = f"{info['task']} (L{info['layer']}: {info['value']:.2f})"
        handle = Line2D(
            [0], [0],
            color=info['color'],
            linewidth=2.2,
            marker=markers[info['idx'] % len(markers)],
            markersize=7,
            markerfacecolor=info['color'],
            markeredgecolor='white',
            markeredgewidth=1,
            label=label,
        )
        legend_handles.append(handle)
    
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=9,
        framealpha=0.95,
        edgecolor='#cccccc',
        fancybox=False,
        borderpad=0.8,
        labelspacing=0.5,
    )
    
    # ===== Background =====
    ax.set_facecolor('#fefefe')
    fig.patch.set_facecolor('white')
    
    # ===== Save =====
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save PDF
    pdf_path = out_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"[Plot] Saved: {out_path}")
    print(f"[Plot] Saved: {pdf_path}")



def plot_all_categories(
    res_sem_avg: Dict[str, np.ndarray],
    res_pld_avg: Dict[str, float],
    detailed_stats: Dict[str, dict],
    num_layers: int,
    out_dir: str,
):
    """Generate separate plots for single-turn and MT-bench categories."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # Single-turn categories
    plot_semantic_curves_beautiful(
        res_sem_avg=res_sem_avg,
        res_pld_avg=res_pld_avg,
        detailed_stats=detailed_stats,
        num_layers=num_layers,
        out_path=os.path.join(out_dir, "single_turn_layers.png"),
        categories=SINGLE_TURN_CATEGORIES,
        smooth_sigma=1.5,
    )
    
    # MT-bench categories
    plot_semantic_curves_beautiful(
        res_sem_avg=res_sem_avg,
        res_pld_avg=res_pld_avg,
        detailed_stats=detailed_stats,
        num_layers=num_layers,
        out_path=os.path.join(out_dir, "mt_bench_layers.png"),
        categories=MT_BENCH_SUBCATEGORIES,
        smooth_sigma=1.5,
    )
    
    # All categories combined
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
    """Print a nice summary table comparing semantic SD vs PLD baseline."""
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

    # retrieval_topk (semantic)
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
    )

    # Print summary table
    print_summary_table(res_sem, res_pld, stats)

    # Save best layer mapping
    save_best_layer_mapping(stats, out_path=args.output_json)

    # Generate plots
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
