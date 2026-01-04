#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mini_bench_10x2.py

Run 2 methods (PLD + Semantic) on each dataset(task) with:
- 10 samples per task
- greedy + sampling
Output 4 merged JSONL files:
  out_dir/pld_greedy.jsonl
  out_dir/pld_sampling.jsonl
  out_dir/semantic_greedy.jsonl
  out_dir/semantic_sampling.jsonl

Semantic supports auto-loading per-task layer mapping:
- .json  : {"task_name": 12, "code_edit": 18, ...}
- .jsonl : each line {"task": "...", "layer_idx": 12} or {"task": "...", "best_layer": 12}

It will group tasks by layer and run run_semantic.py multiple times, then merge.

Usage example:
python run.py \
  --model-path /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --tasks all \
  --out-dir result \
  --max-new-tokens 2048 \
  --pld-n-gram 0 --pld-K 0 \
  --sem-K 16 --sem-sim-threshold 0.0 --sem-retrieval-topk 10 \
  --sem-layer-mapping /root/autodl-tmp/strict/best_layer_La_greedy.json \
  --do-sample --temperature 0.8 --top_p 0.9 --top_k 0
"""

import argparse
import json
import os
import sys
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def run_cmd(cmd: List[str]) -> None:
    print("\n[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def merge_jsonl(files: List[str], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for fp in files:
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    fout.write(line + "\n")
                    n += 1
    print(f"[Merge] {len(files)} files -> {out_path} (lines={n})")


def expand_tasks(tasks: List[str]) -> List[str]:
    """
    Convert ["all"] to explicit task list using data_utils.TASK_CHOICES.
    If data_utils is unavailable, keep ["all"] and let underlying scripts handle it,
    but note: semantic mapping mode needs explicit tasks to group by layer.
    """
    if len(tasks) == 1 and tasks[0] == "all":
        try:
            # assumes mini_bench_10x2.py is in same folder as data_utils.py
            from data_utils import TASK_CHOICES  # type: ignore
            return list(TASK_CHOICES)
        except Exception:
            return ["all"]
    return tasks


def load_layer_mapping(path: str) -> Dict[str, int]:
    """
    Supports:
      - JSON dict: {"task": layer, ...}
      - JSONL lines: {"task":"...", "layer_idx": 12} or {"task":"...", "best_layer": 12}
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping file not found: {path}")

    mapping: Dict[str, int] = {}

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("JSON mapping must be a dict: {task: layer_idx}")
        for k, v in obj.items():
            if isinstance(v, (int, float)):
                mapping[str(k)] = int(v)
        return mapping

    # JSONL fallback
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if not isinstance(o, dict):
                continue
            task = o.get("task") or o.get("dataset") or o.get("name")
            if task is None:
                continue
            layer = o.get("layer_idx")
            if layer is None:
                layer = o.get("best_layer")
            if layer is None:
                continue
            if isinstance(layer, (int, float)):
                mapping[str(task)] = int(layer)

    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="+", required=True, help="task list or 'all'")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--samples-per-task", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-id", type=int, default=0)

    # ---- PLD params ----
    ap.add_argument("--pld-n-gram", type=int, default=4)
    ap.add_argument("--pld-K", type=int, default=16)

    # ---- Semantic params ----
    ap.add_argument("--sem-K", type=int, default=16)
    ap.add_argument("--sem-sim-threshold", type=float, default=0.0)
    ap.add_argument("--sem-retrieval-topk", type=int, default=10)
    ap.add_argument("--sem-default-layer", type=int, default=3,
                    help="used when a task is missing from mapping or mapping not provided")
    ap.add_argument("--sem-layer-mapping", type=str, default="",
                    help="path to mapping json/jsonl: task->layer_idx")

    # ---- Sampling params (used by BOTH methods when enabled) ----
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)  # your future default

    ap.add_argument("--run-pld", action="store_true", default=True)
    ap.add_argument("--run-semantic", action="store_true", default=True)

    args = ap.parse_args()

    # scripts are expected to be alongside this launcher; adjust if needed
    this_dir = os.path.dirname(os.path.abspath(__file__))
    run_pld_py = os.path.join(this_dir, "run_pld.py")
    run_sem_py = os.path.join(this_dir, "run_semantic.py")

    if not os.path.exists(run_pld_py):
        raise FileNotFoundError(f"run_pld.py not found next to this script: {run_pld_py}")
    if not os.path.exists(run_sem_py):
        raise FileNotFoundError(f"run_semantic.py not found next to this script: {run_sem_py}")

    ensure_dir(args.out_dir)
    tmp_dir = os.path.join(args.out_dir, "tmp")
    ensure_dir(tmp_dir)

    # Expand tasks
    tasks_expanded = expand_tasks(args.tasks)
    mapping: Dict[str, int] = load_layer_mapping(args.sem_layer_mapping) if args.sem_layer_mapping else {}

    # If mapping mode but tasks are still ["all"], we cannot group.
    if args.sem_layer_mapping and tasks_expanded == ["all"]:
        raise RuntimeError(
            "You passed --sem-layer-mapping but tasks could not be expanded from 'all'. "
            "Place this script next to data_utils.py so it can import TASK_CHOICES, or pass explicit tasks."
        )

    # -------------------------
    # 1) PLD: greedy + sampling
    # -------------------------
    if args.run_pld:
        pld_greedy_out = os.path.join(args.out_dir, "pld_greedy.jsonl")
        pld_sampling_out = os.path.join(args.out_dir, "pld_sampling.jsonl")

        # Greedy
        cmd = [
            sys.executable, run_pld_py,
            "--model-path", args.model_path,
            "--tasks", *args.tasks,  # let run_pld handle 'all'
            "--samples-per-task", str(args.samples_per_task),
            "--max-new-tokens", str(args.max_new_tokens),
            "--n-gram", str(args.pld_n_gram),
            "--K", str(args.pld_K),
            "--seed", str(args.seed),
            "--output", pld_greedy_out,
        ]
        run_cmd(cmd)

        # Sampling
        cmd = [
            sys.executable, run_pld_py,
            "--model-path", args.model_path,
            "--tasks", *args.tasks,
            "--samples-per-task", str(args.samples_per_task),
            "--max-new-tokens", str(args.max_new_tokens),
            "--n-gram", str(args.pld_n_gram),
            "--K", str(args.pld_K),
            "--seed", str(args.seed),
            "--output", pld_sampling_out,
            "--do-sample",
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--top_k", str(args.top_k),
        ]
        run_cmd(cmd)

        print(f"[OK] PLD outputs:\n  - {pld_greedy_out}\n  - {pld_sampling_out}")

    # ------------------------------------
    # 2) Semantic: greedy + sampling (with mapping)
    # ------------------------------------
    if args.run_semantic:
        sem_greedy_final = os.path.join(args.out_dir, "semantic_greedy.jsonl")
        sem_sampling_final = os.path.join(args.out_dir, "semantic_sampling.jsonl")

        # If no mapping: one run per mode
        if not mapping:
            # Greedy
            cmd = [
                sys.executable, run_sem_py,
                "--model-path", args.model_path,
                "--gpu-id", str(args.gpu_id),
                "--tasks", *args.tasks,
                "--samples-per-task", str(args.samples_per_task),
                "--max-new-tokens", str(args.max_new_tokens),
                "--layer-idx", str(args.sem_default_layer),
                "--K", str(args.sem_K),
                "--sim-threshold", str(args.sem_sim_threshold),
                "--retrieval-topk", str(args.sem_retrieval_topk),
                "--seed", str(args.seed),
                "--output", sem_greedy_final,
            ]
            run_cmd(cmd)

            # Sampling
            cmd = [
                sys.executable, run_sem_py,
                "--model-path", args.model_path,
                "--gpu-id", str(args.gpu_id),
                "--tasks", *args.tasks,
                "--samples-per-task", str(args.samples_per_task),
                "--max-new-tokens", str(args.max_new_tokens),
                "--layer-idx", str(args.sem_default_layer),
                "--K", str(args.sem_K),
                "--sim-threshold", str(args.sem_sim_threshold),
                "--retrieval-topk", str(args.sem_retrieval_topk),
                "--seed", str(args.seed),
                "--output", sem_sampling_final,
                "--do-sample",
                "--temperature", str(args.temperature),
                "--top_p", str(args.top_p),
                "--top_k", str(args.top_k),
            ]
            run_cmd(cmd)

            print(f"[OK] Semantic outputs:\n  - {sem_greedy_final}\n  - {sem_sampling_final}")
            return

        # Mapping mode: group tasks by layer to reduce reloads
        # tasks_expanded is explicit list (not ["all"])
        layer2tasks: Dict[int, List[str]] = defaultdict(list)
        missing: List[str] = []
        for t in tasks_expanded:
            if t in mapping:
                layer2tasks[int(mapping[t])].append(t)
            else:
                missing.append(t)
                layer2tasks[int(args.sem_default_layer)].append(t)

        if missing:
            print(f"[Warn] {len(missing)} tasks missing in mapping; using sem-default-layer={args.sem_default_layer}: {missing}")

        # ---- Greedy grouped runs ----
        greedy_parts: List[str] = []
        for layer, tlist in sorted(layer2tasks.items(), key=lambda x: x[0]):
            part_out = os.path.join(tmp_dir, f"semantic_greedy_layer{layer}.jsonl")
            cmd = [
                sys.executable, run_sem_py,
                "--model-path", args.model_path,
                "--gpu-id", str(args.gpu_id),
                "--tasks", *tlist,
                "--samples-per-task", str(args.samples_per_task),
                "--max-new-tokens", str(args.max_new_tokens),
                "--layer-idx", str(layer),
                "--K", str(args.sem_K),
                "--sim-threshold", str(args.sem_sim_threshold),
                "--retrieval-topk", str(args.sem_retrieval_topk),
                "--seed", str(args.seed),
                "--output", part_out,
            ]
            run_cmd(cmd)
            greedy_parts.append(part_out)

        merge_jsonl(greedy_parts, sem_greedy_final)

        # ---- Sampling grouped runs ----
        sampling_parts: List[str] = []
        for layer, tlist in sorted(layer2tasks.items(), key=lambda x: x[0]):
            part_out = os.path.join(tmp_dir, f"semantic_sampling_layer{layer}.jsonl")
            cmd = [
                sys.executable, run_sem_py,
                "--model-path", args.model_path,
                "--gpu-id", str(args.gpu_id),
                "--tasks", *tlist,
                "--samples-per-task", str(args.samples_per_task),
                "--max-new-tokens", str(args.max_new_tokens),
                "--layer-idx", str(layer),
                "--K", str(args.sem_K),
                "--sim-threshold", str(args.sem_sim_threshold),
                "--retrieval-topk", str(args.sem_retrieval_topk),
                "--seed", str(args.seed),
                "--output", part_out,
                "--do-sample",
                "--temperature", str(args.temperature),
                "--top_p", str(args.top_p),
                "--top_k", str(args.top_k),
            ]
            run_cmd(cmd)
            sampling_parts.append(part_out)

        merge_jsonl(sampling_parts, sem_sampling_final)

        print(f"[OK] Semantic outputs:\n  - {sem_greedy_final}\n  - {sem_sampling_final}")


if __name__ == "__main__":
    main()
