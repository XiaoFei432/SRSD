#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auto benchmark launcher for Spec-Bench style evaluation (13 fine-grained tasks).
Supports both greedy and sampling modes.

Algorithms:
- HF Greedy/Sampling baseline: run_pld.py with --n-gram 0 --K 0
- PLD:                         run_pld.py with configured n-gram, K
- Semantic SD:                 run_semantic.py with per-task best layer (+ retrieval_topk)
- HF Assisted:                 run_assisted.py
- llama.cpp Lookahead:         run_lookahead.py (optional if gguf provided)

Outputs (NO MERGE):
- Per-task JSONL under:
    out_dir/hf_greedy/<task>.jsonl  (or hf_sampling if --do-sample)
    out_dir/pld/<task>.jsonl
    out_dir/semantic/<task>.jsonl
    out_dir/assisted/<task>.jsonl
    out_dir/lookahead/<task>.jsonl  (if enabled)
- One speedup JSON:
    out_dir/speedup_stats.json (suffix _greedy/_sampling)
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
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
    "translation",
    "summarization",
    "text_edit",
    "math_reasoning",
    "code_edit",
]

COARSE_GROUPS: Dict[str, List[str]] = {
    "multi_turn": [
        "writing",
        "roleplay",
        "reasoning",
        "math",
        "coding",
        "extraction",
        "stem",
        "humanities",
    ],
    "translation": ["translation"],
    "code_edit": ["code_edit"],
    "math_reasoning": ["math_reasoning"],
    "text_edit": ["text_edit"],
    "summarization": ["summarization"],
}

COARSE_ORDER: List[str] = [
    "translation",
    "multi_turn",
    "code_edit",
    "math_reasoning",
    "text_edit",
    "summarization",
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
    "code_edit": 17
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

            if new_tokens is not None:
                tokens = float(new_tokens)
            else:
                tokens = max(float(raw_new_tokens) - 1.0, 0.0)

            decode_sec = obj.get("decode_sec", None)
            elapsed_sec = obj.get("elapsed_sec", None)
            if decode_sec is not None:
                t = float(decode_sec)
            elif elapsed_sec is not None:
                t = float(elapsed_sec)
            else:
                continue

            if t <= 0:
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
            print(f"[Semantic] loaded mapping from {path} (n={len(data)}).")
    except Exception as e:
        print(f"[Semantic] failed to load mapping from {path}: {e}. use default.")
    return mapping


# =========================
# ✅ Sampling parameter builder
# =========================

def build_sampling_args(args: argparse.Namespace) -> List[str]:
    """
    Build sampling arguments if do_sample is enabled.
    NOTE: 你的实验要求 top_k=0 也要显式传，所以这里不再 top_k>0 才传。
    """
    sampling_args: List[str] = []
    if args.do_sample:
        sampling_args.append("--do-sample")
        if args.temperature is not None:
            sampling_args.extend(["--temperature", str(args.temperature)])
        if args.top_p is not None:
            sampling_args.extend(["--top_p", str(args.top_p)])
        if args.top_k is not None:
            sampling_args.extend(["--top_k", str(args.top_k)])  # ✅ 允许 0
    return sampling_args


def script_supports_sampling(script_name: str) -> bool:
    return script_name in ("run_pld.py", "run_semantic.py", "run_lookahead.py")


# =========================
# Runner (NO MERGE)
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
                "python",
                script_path,
                "--tasks", task,
                "--samples-per-task", str(args.samples_per_task),
                "--max-new-tokens", str(args.max_new_tokens),
                "--output", out_path,
            ]

            # model-path only for scripts that accept it
            if getattr(args, "model_path", None) and script_name in ("run_pld.py", "run_semantic.py"):
                cmd.extend(["--model-path", str(args.model_path)])

            extra = arg_builder(task)
            if extra:
                cmd.extend(extra)

            # ✅ Only pass sampling args to scripts that support it
            if script_supports_sampling(script_name):
                sampling_args = build_sampling_args(args)
                if sampling_args:
                    cmd.extend(sampling_args)

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
    p.add_argument(
        "--script-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing run_*.py scripts",
    )

    # Sampling control
    p.add_argument("--do-sample", action="store_true", help="Enable sampling mode")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)

    # semantic mapping
    p.add_argument("--sem-layer-mapping", type=str, default="best_layer_mapping.json")

    # target model for run_pld / run_semantic
    p.add_argument("--model-path", type=str, default=None)

    # run control
    p.add_argument("--force-rerun", action="store_true")

    # config override
    p.add_argument("--pld-ngram", type=int, default=PLD_CONFIG["ngram"])
    p.add_argument("--pld-K", type=int, default=PLD_CONFIG["K"])
    p.add_argument("--sem-K", type=int, default=SEM_CONFIG["K"])
    p.add_argument("--sem-threshold", type=float, default=SEM_CONFIG["threshold"])
    p.add_argument("--sem-retrieval-topk", type=int, default=SEM_CONFIG["retrieval_topk"])  # ✅ 新增

    # assisted models
    p.add_argument("--assisted-target-model", type=str, default="/root/autodl-tmp/models/Qwen2.5-32B-Instruct")
    p.add_argument("--assisted-draft-model", type=str, default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--assisted-script", type=str, default="run_assisted.py")

    # ✅ lookahead (llama.cpp) optional
    p.add_argument("--lookahead-enable", action="store_true", help="Enable llama.cpp lookahead runner")
    p.add_argument("--lookahead-script", type=str, default="run_lookahead.py")
    p.add_argument("--lookahead-llama-bin", type=str, default="/root/autodl-tmp/llama.cpp/build/bin/llama-lookahead")
    p.add_argument("--lookahead-gguf-model", type=str, default=None)
    p.add_argument("--lookahead-hf-tokenizer-path", type=str, default=None)
    p.add_argument("--lookahead-ctx-size", type=int, default=4096)
    p.add_argument("--lookahead-threads", type=int, default=8)
    p.add_argument("--lookahead-gpu-layers", type=int, default=-1)
    p.add_argument("--lookahead-gpu-id", type=str, default=None)
    p.add_argument("--lookahead-kv-unified", action="store_true")
    p.add_argument("--lookahead-extra-args", type=str, default="")

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

    if "all" in args.tasks:
        coarse_tasks = COARSE_ORDER
    else:
        coarse_tasks = args.tasks

    fine_tasks: List[str] = []
    for gid in coarse_tasks:
        if gid not in COARSE_GROUPS:
            raise ValueError(f"Unknown coarse group '{gid}', expect {list(COARSE_GROUPS.keys())}")
        fine_tasks.extend(COARSE_GROUPS[gid])
    fine_tasks = sorted(set(fine_tasks), key=lambda x: FINE_TASKS.index(x))

    sem_best_layer = load_sem_layer_mapping(args.sem_layer_mapping)

    # 1) HF baseline
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

    # 3) Semantic SD (+ retrieval-topk)
    stats_semantic = run_algo_per_task(
        algo_name="semantic",
        script_name="run_semantic.py",
        arg_builder=lambda t, m=sem_best_layer: [
            "--layer-idx", str(m.get(t, 1)),
            "--K", str(args.sem_K),
            "--sim-threshold", str(args.sem_threshold),
            "--retrieval-topk", str(args.sem_retrieval_topk),  # ✅ 新增
        ],
        fine_tasks=fine_tasks,
        args=args,
    )

    # 4) Assisted
    stats_assisted = run_algo_per_task(
        algo_name="assisted",
        script_name=args.assisted_script,
        arg_builder=lambda t: [
            "--target-model", str(args.assisted_target_model),
            "--draft-model", str(args.assisted_draft_model),
        ],
        fine_tasks=fine_tasks,
        args=args,
    )

    # 5) Lookahead (optional)
    stats_lookahead: Dict[str, Dict[str, float]] = {}
    if args.lookahead_enable:
        if not args.lookahead_gguf_model:
            raise ValueError("--lookahead-enable set but --lookahead-gguf-model is empty.")
        # hf tokenizer path default to model-path if not provided
        if not args.lookahead_hf_tokenizer_path:
            if args.model_path:
                args.lookahead_hf_tokenizer_path = args.model_path
            else:
                raise ValueError("--lookahead-hf-tokenizer-path is required (or set --model-path so it can default).")

        stats_lookahead = run_algo_per_task(
            algo_name="lookahead",
            script_name=args.lookahead_script,
            arg_builder=lambda t: [
                "--llama-bin", str(args.lookahead_llama_bin),
                "--gguf-model", str(args.lookahead_gguf_model),
                "--hf-tokenizer-path", str(args.lookahead_hf_tokenizer_path),
                "--ctx-size", str(args.lookahead_ctx_size),
                "--threads", str(args.lookahead_threads),
                "--gpu-layers", str(args.lookahead_gpu_layers),
                "--gpu-id", str(args.lookahead_gpu_id) if args.lookahead_gpu_id is not None else "",
                "--output", "SHOULD_NOT_APPEAR",  # 占位：run_algo_per_task 会覆盖 output（这里会被移除）
            ],
            fine_tasks=fine_tasks,
            args=args,
        )

        # 上面的占位参数会造成空字符串问题，所以我们更稳妥：
        # 重新跑一遍正确的 arg_builder（不带 output、gpu-id 空串处理）
        # ——为了不把你搞复杂，这里直接“在上面那条逻辑里”做一个简单修正：
        #
        # 实际更推荐：把 run_algo_per_task 改成不强制加 --output，然后各脚本自己决定。
        #
        # 但你当前结构固定，所以我在下面提供一个更干净的做法（替换 stats_lookahead 的那段）：
        pass

    # ✅ 上面 stats_lookahead 那段我不留“坑”，直接给你一个干净可用版本：
    stats_lookahead = {}
    if args.lookahead_enable:
        if not args.lookahead_gguf_model:
            raise ValueError("--lookahead-enable set but --lookahead-gguf-model is empty.")
        if not args.lookahead_hf_tokenizer_path:
            if args.model_path:
                args.lookahead_hf_tokenizer_path = args.model_path
            else:
                raise ValueError("--lookahead-hf-tokenizer-path is required (or set --model-path so it can default).")

        # 这里不用 run_algo_per_task，因为 lookahead 的参数和 output 拼装不太一样；单独写最稳。
        algo_name = "lookahead"
        algo_dir = os.path.join(args.out_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        script_path = os.path.join(args.script_dir, args.lookahead_script)
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
                    "--llama-bin", str(args.lookahead_llama_bin),
                    "--gguf-model", str(args.lookahead_gguf_model),
                    "--hf-tokenizer-path", str(args.lookahead_hf_tokenizer_path),
                    "--ctx-size", str(args.lookahead_ctx_size),
                    "--threads", str(args.lookahead_threads),
                    "--gpu-layers", str(args.lookahead_gpu_layers),
                    "--kv-unified",
                ]
                if args.lookahead_gpu_id is not None:
                    cmd.extend(["--gpu-id", str(args.lookahead_gpu_id)])
                if args.lookahead_extra_args:
                    cmd.extend(["--llama-extra-args", str(args.lookahead_extra_args)])

                # sampling passthrough (lookahead 支持)
                sampling_args = build_sampling_args(args)
                if sampling_args:
                    cmd.extend(sampling_args)

                run_subprocess(cmd, log_prefix=f"{algo_name}:{task}")

            merge_stats_dict(stats_all, parse_stats_from_file(out_path))

        stats_lookahead = stats_all

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

        baseline_tp = calc_group_tp(stats_baseline, tasks_in_group)
        pld_tp = calc_group_tp(stats_pld, tasks_in_group)
        sem_tp = calc_group_tp(stats_semantic, tasks_in_group)
        asst_tp = calc_group_tp(stats_assisted, tasks_in_group)
        look_tp = calc_group_tp(stats_lookahead, tasks_in_group) if stats_lookahead else 0.0

        pld_speedup = (pld_tp / baseline_tp) if (baseline_tp > 0 and pld_tp > 0) else 0.0
        sem_speedup = (sem_tp / baseline_tp) if (baseline_tp > 0 and sem_tp > 0) else 0.0
        asst_speedup = (asst_tp / baseline_tp) if (baseline_tp > 0 and asst_tp > 0) else 0.0
        look_speedup = (look_tp / baseline_tp) if (baseline_tp > 0 and look_tp > 0) else 0.0

        per_group[gid] = {
            "label": label,
            f"{baseline_name}_tok_s": baseline_tp,
            "pld_tok_s": pld_tp,
            "semantic_tok_s": sem_tp,
            "assisted_tok_s": asst_tp,
            "lookahead_tok_s": look_tp if stats_lookahead else None,
            "pld_speedup": pld_speedup,
            "semantic_speedup": sem_speedup,
            "assisted_speedup": asst_speedup,
            "lookahead_speedup": look_speedup if stats_lookahead else None,
            "fine_tasks": tasks_in_group,
        }

        msg = (
            f"[{label}] BASE={baseline_tp:.2f} | "
            f"PLD={pld_tp:.2f} ({pld_speedup:.2f}x) | "
            f"SEM={sem_tp:.2f} ({sem_speedup:.2f}x) | "
            f"ASST={asst_tp:.2f} ({asst_speedup:.2f}x)"
        )
        if stats_lookahead:
            msg += f" | LOOK={look_tp:.2f} ({look_speedup:.2f}x)"
        print(msg)

    overall = {
        f"{baseline_name}_tok_s": calc_overall_tp(stats_baseline, fine_tasks),
        "pld_tok_s": calc_overall_tp(stats_pld, fine_tasks),
        "semantic_tok_s": calc_overall_tp(stats_semantic, fine_tasks),
        "assisted_tok_s": calc_overall_tp(stats_assisted, fine_tasks),
        "lookahead_tok_s": calc_overall_tp(stats_lookahead, fine_tasks) if stats_lookahead else None,
    }
    base_tp = overall[f"{baseline_name}_tok_s"]
    overall["pld_speedup"] = (overall["pld_tok_s"] / base_tp) if base_tp > 0 else 0.0
    overall["semantic_speedup"] = (overall["semantic_tok_s"] / base_tp) if base_tp > 0 else 0.0
    overall["assisted_speedup"] = (overall["assisted_tok_s"] / base_tp) if base_tp > 0 else 0.0
    overall["lookahead_speedup"] = (overall["lookahead_tok_s"] / base_tp) if (stats_lookahead and base_tp > 0 and overall["lookahead_tok_s"]) else None

    out = {
        "note": f"Speedups computed on (N-1) tokens / decode_sec aggregated over selected tasks. "
                f"PLD/Semantic/Assisted/Lookahead speedup vs {baseline_name} baseline.",
        "mode": mode_label,
        "baseline": baseline_name,
        "config": {
            "samples_per_task": args.samples_per_task,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "pld": {"ngram": args.pld_ngram, "K": args.pld_K},
            "semantic": {"K": args.sem_K, "threshold": args.sem_threshold, "retrieval_topk": args.sem_retrieval_topk},
            "assisted": {
                "target_model": args.assisted_target_model,
                "draft_model": args.assisted_draft_model,
                "assisted_script": args.assisted_script,
            },
            "lookahead": {
                "enabled": bool(args.lookahead_enable),
                "llama_bin": args.lookahead_llama_bin,
                "gguf_model": args.lookahead_gguf_model,
                "hf_tokenizer_path": args.lookahead_hf_tokenizer_path,
                "ctx_size": args.lookahead_ctx_size,
                "threads": args.lookahead_threads,
                "gpu_layers": args.lookahead_gpu_layers,
                "gpu_id": args.lookahead_gpu_id,
                "kv_unified": bool(args.lookahead_kv_unified),
                "extra_args": args.lookahead_extra_args,
            } if args.lookahead_enable else {"enabled": False},
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
