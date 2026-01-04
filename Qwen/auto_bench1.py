#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

python auto_bench1.py \
  --tasks all \
  --samples-per-task 200 \
  --max-new-tokens 2048 \
  --out-dir result/Qw_7B_greedy/result \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --sem-layer-mapping map/best_layer_Qw_7B_greedy.json \
  --sem-retrieval-topk 10 

python auto_bench1.py \
  --tasks all \
  --samples-per-task 200 \
  --max-new-tokens 2048 \
  --out-dir result/Qw_7B_sample/result \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --sem-layer-mapping map/best_layer_Qw_7B_sample.json \
  --sem-retrieval-topk 10 \
  --do-sample --temperature 0.8 --top_p 0.9 --top_k 0 



"""

import argparse
import json
import os
import subprocess
from typing import Dict, List, Any, Callable


# =========================
# Spec-Bench Task Config
# =========================

FINE_TASKS: List[str] = [
    "writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities",
    "translation", "summarization", "text_edit", "math_reasoning", "code_edit",
]

COARSE_GROUPS: Dict[str, List[str]] = {
    "multi_turn": ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"],
    "translation": ["translation"],
    "code_edit": ["code_edit"],
    "math_reasoning": ["math_reasoning"],
    "text_edit": ["text_edit"],
    "summarization": ["summarization"],
}

COARSE_ORDER: List[str] = [
    "translation", "multi_turn", "code_edit", "math_reasoning", "text_edit", "summarization",
]

COARSE_LABELS: Dict[str, str] = {
    "translation": "Translation",
    "multi_turn": "Multi-turn Conversation",
    "code_edit": "Code Editing",
    "math_reasoning": "Mathematical Reasoning",
    "text_edit": "Text Editing",
    "summarization": "Summarization",
}

# Default configs
PLD_CONFIG = {"ngram": 4, "K": 16}
SEM_CONFIG = {"K": 16, "threshold": 0.0, "retrieval_topk": 10}

# Fallback mapping (will be overridden by --sem-layer-mapping if provided)
SEM_BEST_LAYER_PER_TASK: Dict[str, int] = {
    "writing": 61,
    "roleplay": 62,
    "reasoning": 61,
    "math": 61,
    "coding": 61,
    "extraction": 12,
    "stem": 62,
    "humanities": 62,
    "translation": 16,
    "summarization": 5,
    "text_edit": 2,
    "math_reasoning": 61,
    "code_edit": 17,
}


# =========================
# Utils
# =========================

def run_subprocess(cmd: List[str], log_prefix: str = "", cwd: str = None) -> None:
    print(f"[CMD] ({log_prefix})", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run(cmd, env=env, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed code {proc.returncode}: {' '.join(cmd)}")


def parse_stats_from_file(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Aggregate a JSONL by task:
      tokens: use new_tokens (preferred) else raw_new_tokens-1
      time:   use decode_sec (preferred) else elapsed_sec
    """
    stats: Dict[str, Dict[str, float]] = {}
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return stats

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task = obj.get("task")
            if not task:
                continue

            new_tokens = obj.get("new_tokens", None)
            raw_new_tokens = obj.get("raw_new_tokens", None)
            if new_tokens is None and raw_new_tokens is None:
                continue

            tokens = float(new_tokens) if new_tokens is not None else max(float(raw_new_tokens) - 1.0, 0.0)

            decode_sec = obj.get("decode_sec", None)
            elapsed_sec = obj.get("elapsed_sec", None)
            t = float(decode_sec) if decode_sec is not None else (float(elapsed_sec) if elapsed_sec is not None else None)
            if t is None or t <= 0:
                continue

            if task not in stats:
                stats[task] = {"tokens": 0.0, "time": 0.0}
            stats[task]["tokens"] += tokens
            stats[task]["time"] += t

    return stats


def merge_stats_dict(dst: Dict[str, Dict[str, float]], src: Dict[str, Dict[str, float]]) -> None:
    for task, v in src.items():
        if task not in dst:
            dst[task] = {"tokens": 0.0, "time": 0.0}
        dst[task]["tokens"] += float(v.get("tokens", 0.0))
        dst[task]["time"] += float(v.get("time", 0.0))


def load_sem_layer_mapping(path: str) -> Dict[str, int]:
    mapping = SEM_BEST_LAYER_PER_TASK.copy()
    if not path:
        return mapping
    if not os.path.exists(path):
        print(f"[Semantic] mapping '{path}' not found, use default.")
        return mapping
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    mapping[k] = int(v)
                except Exception:
                    pass
        print(f"[Semantic] loaded mapping from {path} (n={len(data) if isinstance(data, dict) else 0}).")
    except Exception as e:
        print(f"[Semantic] failed to load mapping from {path}: {e}. use default.")
    return mapping


# =========================
# Sampling args passthrough
# =========================

def build_sampling_args(args: argparse.Namespace) -> List[str]:
    """
    If --do-sample is enabled, pass sampling args to run_pld.py / run_semantic.py.
    Note: you要求 top_k=0 也要显式传，所以 top_k 不做 >0 判断。
    """
    out: List[str] = []
    if args.do_sample:
        out.append("--do-sample")
        if args.temperature is not None:
            out += ["--temperature", str(args.temperature)]
        if args.top_p is not None:
            out += ["--top_p", str(args.top_p)]
        if args.top_k is not None:
            out += ["--top_k", str(args.top_k)]
    return out


# =========================
# Runner
# =========================

def run_algo_per_task(
    algo_name: str,
    script_name: str,
    arg_builder: Callable[[str], List[str]],
    fine_tasks: List[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    algo_dir = os.path.join(args.out_dir, algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    script_path = os.path.join(args.script_dir, script_name)
    print(f"\n>>> [Algo={algo_name}] script={script_path}")

    stats_all: Dict[str, Dict[str, float]] = {}
    for task in fine_tasks:
        out_path = os.path.join(algo_dir, f"{task}.jsonl")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not args.force_rerun:
            print(f"  -> Skip {algo_name}:{task} (exists) {out_path}")
        else:
            cmd = [
                "python", script_path,
                "--tasks", task,
                "--samples-per-task", str(args.samples_per_task),
                "--max-new-tokens", str(args.max_new_tokens),
                "--output", out_path,
            ]

            # both run_pld.py and run_semantic.py accept --model-path
            if args.model_path:
                cmd += ["--model-path", str(args.model_path)]

            cmd += arg_builder(task)

            sampling = build_sampling_args(args)
            if sampling:
                cmd += sampling

            run_subprocess(cmd, log_prefix=f"{algo_name}:{task}")

        merge_stats_dict(stats_all, parse_stats_from_file(out_path))

    return stats_all


# =========================
# Speedup computation
# =========================

def calc_group_tp(stats: Dict[str, Dict[str, float]], tasks_in_group: List[str]) -> float:
    tok = 0.0
    t = 0.0
    for tt in tasks_in_group:
        if tt in stats:
            tok += stats[tt]["tokens"]
            t += stats[tt]["time"]
    return tok / t if t > 0 else 0.0


def calc_overall_tp(stats: Dict[str, Dict[str, float]], all_tasks: List[str]) -> float:
    tok = 0.0
    t = 0.0
    for tt in all_tasks:
        if tt in stats:
            tok += stats[tt]["tokens"]
            t += stats[tt]["time"]
    return tok / t if t > 0 else 0.0


# =========================
# Main
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, nargs="+", default=["all"])
    p.add_argument("--samples-per-task", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=2048)

    p.add_argument("--out-dir", type=str, default="result")
    p.add_argument("--script-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))

    # Sampling control (accept both --do-sample and --do_sample)
    p.add_argument("--do-sample", "--do_sample", action="store_true", help="Enable sampling mode")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=0)  # default 0 to match your "future experiments: top_k=0"

    # model for run_pld/run_semantic
    p.add_argument("--model-path", type=str, default=None)

    # semantic mapping
    p.add_argument("--sem-layer-mapping", type=str, default="best_layer_mapping.json")

    # rerun control
    p.add_argument("--force-rerun", action="store_true")

    # config override
    p.add_argument("--pld-ngram", type=int, default=PLD_CONFIG["ngram"])
    p.add_argument("--pld-K", type=int, default=PLD_CONFIG["K"])
    p.add_argument("--sem-K", type=int, default=SEM_CONFIG["K"])
    p.add_argument("--sem-threshold", type=float, default=SEM_CONFIG["threshold"])
    p.add_argument("--sem-retrieval-topk", type=int, default=SEM_CONFIG["retrieval_topk"])

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline_name = "hf_sampling" if args.do_sample else "hf_greedy"
    mode_label = "Sampling" if args.do_sample else "Greedy"
    print(f"\n{'='*60}")
    print(f"Benchmark Mode: {mode_label}")
    if args.do_sample:
        print(f"  temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"{'='*60}\n")

    # coarse selection
    coarse_tasks = COARSE_ORDER if "all" in args.tasks else args.tasks

    # expand to fine tasks
    fine_tasks: List[str] = []
    for gid in coarse_tasks:
        if gid not in COARSE_GROUPS:
            raise ValueError(f"Unknown coarse group '{gid}', expect {list(COARSE_GROUPS.keys())}")
        fine_tasks.extend(COARSE_GROUPS[gid])
    fine_tasks = sorted(set(fine_tasks), key=lambda x: FINE_TASKS.index(x))

    sem_best_layer = load_sem_layer_mapping(args.sem_layer_mapping)

    # 1) Autoregressive baseline (ngram=0,K=0)
    stats_baseline = run_algo_per_task(
        algo_name=baseline_name,
        script_name="run_pld.py",
        arg_builder=lambda t: ["--n-gram", "0", "--K", "0"],
        fine_tasks=fine_tasks,
        args=args,
    )

    # 2) PLD
    stats_pld = run_algo_per_task(
        algo_name="pld",
        script_name="run_pld.py",
        arg_builder=lambda t: ["--n-gram", str(args.pld_ngram), "--K", str(args.pld_K)],
        fine_tasks=fine_tasks,
        args=args,
    )

    # 3) Semantic SD
    stats_sem = run_algo_per_task(
        algo_name="semantic",
        script_name="run_semantic.py",
        arg_builder=lambda t, m=sem_best_layer: [
            "--layer-idx", str(m.get(t, 1)),
            "--K", str(args.sem_K),
            "--sim-threshold", str(args.sem_threshold),
            "--retrieval-topk", str(args.sem_retrieval_topk),
        ],
        fine_tasks=fine_tasks,
        args=args,
    )

    # -------------------------
    # Speedup computation
    # -------------------------
    per_group: Dict[str, Any] = {}
    print(f"\n{'='*60}")
    print(f"Per-group Throughput & Speedup (vs {baseline_name})")
    print(f"{'='*60}")

    for gid in coarse_tasks:
        label = COARSE_LABELS.get(gid, gid)
        tasks_in_group = COARSE_GROUPS[gid]

        base_tp = calc_group_tp(stats_baseline, tasks_in_group)
        pld_tp = calc_group_tp(stats_pld, tasks_in_group)
        sem_tp = calc_group_tp(stats_sem, tasks_in_group)

        pld_speedup = (pld_tp / base_tp) if (base_tp > 0 and pld_tp > 0) else 0.0
        sem_speedup = (sem_tp / base_tp) if (base_tp > 0 and sem_tp > 0) else 0.0

        per_group[gid] = {
            "label": label,
            f"{baseline_name}_tok_s": base_tp,
            "pld_tok_s": pld_tp,
            "semantic_tok_s": sem_tp,
            "pld_speedup": pld_speedup,
            "semantic_speedup": sem_speedup,
            "fine_tasks": tasks_in_group,
        }

        print(
            f"[{label}] BASE={base_tp:.2f} | "
            f"PLD={pld_tp:.2f} ({pld_speedup:.2f}x) | "
            f"SEM={sem_tp:.2f} ({sem_speedup:.2f}x)"
        )

    overall = {
        f"{baseline_name}_tok_s": calc_overall_tp(stats_baseline, fine_tasks),
        "pld_tok_s": calc_overall_tp(stats_pld, fine_tasks),
        "semantic_tok_s": calc_overall_tp(stats_sem, fine_tasks),
    }
    base_tp = overall[f"{baseline_name}_tok_s"]
    overall["pld_speedup"] = (overall["pld_tok_s"] / base_tp) if base_tp > 0 else 0.0
    overall["semantic_speedup"] = (overall["semantic_tok_s"] / base_tp) if base_tp > 0 else 0.0

    out = {
        "note": f"Speedups computed on (N-1) tokens / decode_sec aggregated over selected tasks. "
                f"PLD/Semantic speedup vs {baseline_name} baseline.",
        "mode": mode_label,
        "baseline": baseline_name,
        "config": {
            "samples_per_task": args.samples_per_task,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "model_path": args.model_path,
            "pld": {"ngram": args.pld_ngram, "K": args.pld_K},
            "semantic": {"K": args.sem_K, "threshold": args.sem_threshold, "retrieval_topk": args.sem_retrieval_topk},
            "coarse_tasks": coarse_tasks,
            "fine_tasks": fine_tasks,
        },
        "per_group": per_group,
        "overall": overall,
    }

    suffix = "_sampling" if args.do_sample else "_greedy"
    speedup_path = os.path.join(args.out_dir, f"speedup_stats{suffix}.json")
    with open(speedup_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[Save] speedup_stats -> {speedup_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
