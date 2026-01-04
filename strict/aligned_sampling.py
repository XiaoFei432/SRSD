# aligned_sampling.py
# -*- coding: utf-8 -*-
"""
Counter-based / position-indexed RNG aligned sampling utilities.

Key idea:
- For absolute generated position pos (0-based after prompt), define a deterministic uniform u in (0,1):
    u = U64_to_uniform(splitmix64(hash64(global_seed, sample_id, pos)))
- Sample from the *same truncated distribution* (temperature + top_k + top_p) using this u.
This makes AR and SRSD consume the same random variable at the same position.

Also provides sparse distribution extraction (support indices + probs) after truncation,
and exact TV/KL/JS on the union support (with epsilon for KL stability).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch


# ---------------------------
# 64-bit counter-based hashing
# ---------------------------

# ---------------------------
# 64-bit counter-based hashing (CPU / Python int)
# ---------------------------

_MASK64 = (1 << 64) - 1

def _splitmix64_py(x: int) -> int:
    """SplitMix64 in pure Python (uint64 arithmetic)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = (z ^ (z >> 30)) & _MASK64
    z = (z * 0xBF58476D1CE4E5B9) & _MASK64
    z = (z ^ (z >> 27)) & _MASK64
    z = (z * 0x94D049BB133111EB) & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return z

def _hash64_triplet_py(global_seed: int, sample_id: int, pos: int) -> int:
    """Combine (global_seed, sample_id, pos) into a uint64 state (Python int)."""
    gs = global_seed & _MASK64
    sid = sample_id & _MASK64
    p = pos & _MASK64
    x = gs ^ ((sid << 32) & _MASK64) ^ p
    return x & _MASK64

def aligned_uniform(global_seed: int, sample_id: int, pos: int, device: torch.device) -> torch.Tensor:
    """
    Deterministic u in [0,1) as a scalar tensor on `device`.
    Computed on CPU via Python uint64 ops to avoid CUDA uint64 bit-op limitations.
    """
    x = _hash64_triplet_py(global_seed, sample_id, pos)
    z = _splitmix64_py(x)
    # Convert to float in [0,1): take top 53 bits
    u = ((z >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    if u >= 1.0:
        u = 1.0 - 1e-15
    if u < 0.0:
        u = 0.0
    return torch.tensor(u, device=device, dtype=torch.float64)



# ---------------------------
# Truncation & sampling
# ---------------------------

def _safe_softmax_1d(logits_1d: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits_1d, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(logits_1d).all():
        V = logits_1d.numel()
        return torch.full((V,), 1.0 / float(V), device=logits_1d.device, dtype=torch.float32)
    return probs


def truncate_to_distribution(
    logits_1d: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply temperature + top_k + top_p to logits, return (support_idx, support_probs).
    Returned distribution is exact under this truncation pipeline (outside support prob = 0).
    """
    x = logits_1d
    if temperature is not None and temperature > 0 and float(temperature) != 1.0:
        x = x / float(temperature)

    V = x.numel()

    # Top-k: keep only k tokens (by logits). If top_k<=0 => no top-k.
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), V)
        topv, topi = torch.topk(x, k=k, dim=-1)
        # we will apply top_p inside this top-k set if needed
        x_sel = topv
        idx_sel = topi
    else:
        # treat as full vocab
        x_sel = x
        idx_sel = None

    # Top-p: nucleus sampling
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        if idx_sel is None:
            sorted_logits, sorted_idx = torch.sort(x_sel, descending=True)
        else:
            sorted_logits, order = torch.sort(x_sel, descending=True)
            sorted_idx = idx_sel[order]

        sorted_probs = _safe_softmax_1d(sorted_logits.float())
        cdf = torch.cumsum(sorted_probs, dim=-1)

        remove = cdf > float(top_p)
        remove[1:] = remove[:-1].clone()
        remove[0] = False

        keep = ~remove
        keep_idx = sorted_idx[keep]
        keep_probs_unnorm = sorted_probs[keep]
        keep_probs = keep_probs_unnorm / (keep_probs_unnorm.sum() + 1e-12)
        return keep_idx.to(torch.long), keep_probs.to(torch.float32)

    # No top-p: just softmax on selected set
    if idx_sel is None:
        probs = _safe_softmax_1d(x_sel.float())
        idx = torch.arange(V, device=x.device, dtype=torch.long)
        return idx, probs.to(torch.float32)
    else:
        probs = _safe_softmax_1d(x_sel.float())
        probs = probs / (probs.sum() + 1e-12)
        return idx_sel.to(torch.long), probs.to(torch.float32)


def sample_from_truncated(
    support_idx: torch.Tensor,
    support_probs: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """
    Deterministic sampling using u via CDF over support_probs (support order is arbitrary but fixed).
    Returns token_id scalar (long).
    """
    cdf = torch.cumsum(support_probs, dim=-1)
    # Ensure u is on same device/dtype
    u = u.to(device=cdf.device, dtype=cdf.dtype)
    # searchsorted expects 1D cdf and scalar u
    j = torch.searchsorted(cdf, u, right=False).clamp(min=0, max=cdf.numel() - 1)
    return support_idx[j]


def aligned_sample_token(
    logits_1d: torch.Tensor,
    global_seed: int,
    sample_id: int,
    pos: int,
    temperature: float,
    top_p: float,
    top_k: int,
    return_dist: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Return (token_id, support_idx, support_probs).
    """
    device = logits_1d.device
    u = aligned_uniform(global_seed, sample_id, pos, device=device)
    idx, probs = truncate_to_distribution(logits_1d, temperature, top_p, top_k)
    tok = sample_from_truncated(idx, probs, u).to(torch.long)
    if return_dist:
        return tok, idx, probs
    return tok, None, None


# ---------------------------
# Sparse distance metrics (exact on union support)
# ---------------------------

@dataclass
class DistMetrics:
    tv: float
    js: float
    kl_pq: float
    kl_qp: float
    support_p: int
    support_q: int
    support_union: int
    support_intersect: int
    jaccard: float


def _scatter_to_union(union: torch.Tensor, idx: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    # union is sorted; idx may be unsorted; use searchsorted
    pos = torch.searchsorted(union, idx)
    out = torch.zeros((union.numel(),), device=union.device, dtype=torch.float64)
    out[pos] = probs.to(torch.float64)
    return out


def sparse_dist_metrics(
    idx_p: torch.Tensor,
    p: torch.Tensor,
    idx_q: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-12,
) -> DistMetrics:
    """
    Compute TV / JS / KL(p||q) / KL(q||p) on union support of two sparse distributions.
    Missing prob treated as 0. KL uses eps smoothing to avoid inf.
    """
    device = idx_p.device
    union = torch.unique(torch.cat([idx_p, idx_q], dim=0)).to(device=device)
    pp = _scatter_to_union(union, idx_p, p)
    qq = _scatter_to_union(union, idx_q, q)

    # Support stats
    supp_p = int(idx_p.numel())
    supp_q = int(idx_q.numel())
    supp_union = int(union.numel())
    # intersection count: tokens present in both supports
    inter = torch.intersect1d(idx_p, idx_q).numel() if hasattr(torch, "intersect1d") else int(
        torch.unique(torch.cat([idx_p, idx_q])).numel() - union.numel()
    )
    inter = int(inter)
    jacc = float(inter / max(1, (supp_p + supp_q - inter)))

    # TV
    tv = 0.5 * torch.sum(torch.abs(pp - qq)).item()

    # JS
    m = 0.5 * (pp + qq)
    # KL with eps smoothing
    pp_s = torch.clamp(pp, min=0.0)
    qq_s = torch.clamp(qq, min=0.0)
    m_s = torch.clamp(m, min=eps)

    kl_pq = torch.sum(pp_s * (torch.log(pp_s + eps) - torch.log(qq_s + eps))).item()
    kl_qp = torch.sum(qq_s * (torch.log(qq_s + eps) - torch.log(pp_s + eps))).item()

    kl_pm = torch.sum(pp_s * (torch.log(pp_s + eps) - torch.log(m_s))).item()
    kl_qm = torch.sum(qq_s * (torch.log(qq_s + eps) - torch.log(m_s))).item()
    js = 0.5 * (kl_pm + kl_qm)

    return DistMetrics(
        tv=float(tv),
        js=float(js),
        kl_pq=float(kl_pq),
        kl_qp=float(kl_qp),
        support_p=supp_p,
        support_q=supp_q,
        support_union=supp_union,
        support_intersect=inter,
        jaccard=jacc,
    )
