#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lookahead Decoding baseline via llama.cpp (GPU) on Spec-Bench.

- 数据：spec_bench/question.jsonl
- 引擎：llama.cpp/build/bin/llama-lookahead
- Greedy：--temp 0
- Sampling：--temp T --top-p P --top-k K
- errors='replace' 防止编码崩溃
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Tuple, Optional, Any

from transformers import AutoTokenizer
from data_utils import TASK_CHOICES, get_dataset_entries

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


# =============== 工具函数 ===============

def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def extract_assistant_reply(text: str) -> str:
    """
    从 llama.cpp 的 stdout 中，只截取 assistant 段落。
    (Lookahead 二进制通常默认回显 Prompt，所以需要这个函数)
    """
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    idx = text.find(marker)
    if idx == -1:
        return text.lstrip("\n\r ")
    after = text[idx + len(marker):]
    return after.lstrip("\n\r ")


# =============== Prompt 构造 ===============

def build_full_prompt_from_entry(tokenizer: AutoTokenizer, obj: Dict[str, Any]) -> Tuple[str, str]:
    turns = obj.get("turns", [])
    user_content = "\n".join(turns) if isinstance(turns, list) else str(turns)
    messages = [{"role": "user", "content": user_content}]

    # 尽量走 chat template；没有就退化
    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = user_content + "\n"
    return full_prompt, user_content


# =============== 从 STDERR 解析 decode 统计 ===============

def parse_decode_from_stderr(stderr: str) -> Tuple[Optional[float], Optional[int]]:
    """
    解析 llama.cpp 的性能日志 (Decode 阶段)。
    """
    cand = []
    for line in stderr.splitlines():
        m = re.search(r"decoded\s+([0-9]+)\s+tokens\s+in\s+([0-9.]+)\s+(seconds|s)\b", line, re.IGNORECASE)
        if m:
            cand.append((float(m.group(2)), int(m.group(1))))
    if cand:
        # 取 token 数最大的那条
        return max(cand, key=lambda x: x[1])

    cand2 = []
    for line in stderr.splitlines():
        if "common_perf_print" in line.lower() and "eval time" in line.lower() and "prompt" not in line.lower():
            m = re.search(r"eval time\s*=\s*([0-9.+-eE]+)\s*ms\s*/\s*([0-9]+)\s*runs", line, re.IGNORECASE)
            if m:
                cand2.append((float(m.group(1)) / 1000.0, int(m.group(2))))
    if cand2:
        return max(cand2, key=lambda x: x[1])

    return None, None


# =============== 调用 llama-lookahead 一次 ===============

def run_llama_lookahead_once(args: argparse.Namespace, prompt: str) -> Tuple[str, float, int]:
    cmd: List[str] = [
        args.llama_bin,
        "-m", args.gguf_model,
        "-n", str(args.max_new_tokens),
        "-c", str(args.ctx_size),
        "-t", str(args.threads),
        "-p", prompt,
    ]

    # GPU offload
    if args.gpu_layers >= 0:
        cmd.extend(["-ngl", str(args.gpu_layers)])
    if args.kv_unified:
        cmd.append("-kvu")

    # ✅ Greedy vs Sampling
    if args.do_sample:
        # llama.cpp 参数名：--temp --top-p --top-k
        cmd.extend(["--temp", str(args.temperature)])
        cmd.extend(["--top-p", str(args.top_p)])
        cmd.extend(["--top-k", str(args.top_k)])  # 允许 0
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
    else:
        cmd.extend(["--temp", "0"])  # 强制 greedy 对齐 baseline

    # Extra args passthrough
    if args.llama_extra_args:
        import shlex
        cmd.extend(shlex.split(args.llama_extra_args))

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    t_wall_start = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        errors="replace",
    )
    t_wall_end = time.time()

    stdout = strip_ansi(proc.stdout)
    stderr = strip_ansi(proc.stderr)

    if proc.returncode != 0:
        print(f"[Warn] llama-lookahead exited with code {proc.returncode}", file=sys.stderr)

    decode_sec, eval_tokens = parse_decode_from_stderr(stderr)
    if decode_sec is None or decode_sec <= 0.0:
        decode_sec = t_wall_end - t_wall_start
        eval_tokens = 0
    if eval_tokens is None:
        eval_tokens = 0

    output_text_clean = extract_assistant_reply(stdout)
    return output_text_clean, float(decode_sec), int(eval_tokens)


# =============== 主流程 ===============

def run_lookahead_llamacpp(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = TASK_CHOICES if "all" in args.tasks else args.tasks

    total_raw_new = 0
    total_eff = 0
    total_time = 0.0
    total_eval = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for task in tasks:
            print(f"\n=== [llama.cpp Lookahead] Task: {task} ===")
            entries = get_dataset_entries(task, args.samples_per_task)

            for idx, obj in enumerate(entries, start=1):
                full_prompt, user_plain = build_full_prompt_from_entry(tokenizer, obj)
                print(f"  - [{task}] sample {idx}/{len(entries)}")

                output, decode_sec, eval_tokens = run_llama_lookahead_once(args, full_prompt)

                # 计算 token（用 HF tokenizer 做一致统计）
                prompt_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(full_prompt + output, add_special_tokens=False)["input_ids"]

                raw_new = max(len(full_ids) - len(prompt_ids), 0)
                eff_new = max(raw_new - 1, 0)  # N-1

                tp = eff_new / decode_sec if decode_sec > 0 and eff_new > 0 else 0.0
                # 你之前这里写的 ideal = eval_tokens/raw_new（保留）
                ideal = float(eval_tokens) / float(raw_new) if raw_new > 0 else 0.0

                total_raw_new += raw_new
                total_eff += eff_new
                total_time += decode_sec
                total_eval += eval_tokens

                rec = {
                    "task": task,
                    "sample_idx": idx,
                    "prompt": user_plain,
                    "output": output,
                    "prompt_tokens": len(prompt_ids),
                    "new_tokens": int(eff_new),
                    "raw_new_tokens": int(raw_new),
                    "eval_tokens": int(eval_tokens),
                    "decode_sec": float(decode_sec),
                    "throughput": float(tp),
                    "ideal_speedup": float(ideal),
                    "engine": "llama.cpp_lookahead",
                    "gpu_id": str(args.gpu_id) if args.gpu_id is not None else "0",
                    "do_sample": bool(args.do_sample),
                    "temperature": float(args.temperature) if args.do_sample else None,
                    "top_p": float(args.top_p) if args.do_sample else None,
                    "top_k": int(args.top_k) if args.do_sample else None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(
                    f"    Decode: {decode_sec:.4f}s | "
                    f"EffTok(N-1): {eff_new} | "
                    f"TP: {tp:.2f} tok/s | "
                    f"Ideal: {ideal:.3f}"
                )

    overall_tp = total_eff / total_time if total_time > 0 else 0.0
    overall_ideal = total_eval / total_raw_new if total_raw_new > 0 else 0.0

    print("\n========== llama.cpp Lookahead Summary ==========")
    print(f"Sampling Mode         : {args.do_sample}")
    if args.do_sample:
        print(f"  temperature         : {args.temperature}")
        print(f"  top_p               : {args.top_p}")
        print(f"  top_k               : {args.top_k}")
    print(f"Total Eff Tokens (N-1): {total_eff}")
    print(f"Total Decode Time     : {total_time:.3f} s")
    print(f"Overall Throughput    : {overall_tp:.2f} tok/s")
    print(f"Overall Ideal Speedup : {overall_ideal:.3f}")
    print("=================================================")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    parser.add_argument("--llama-bin", default="/root/autodl-tmp/llama.cpp/build/bin/llama-lookahead")
    parser.add_argument("--gguf-model", required=True)
    parser.add_argument("--hf-tokenizer-path", required=True)

    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--kv-unified", action="store_true")
    parser.add_argument("--llama-extra-args", type=str, default="")
    parser.add_argument("--output", required=True)

    # ✅ Sampling
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    run_lookahead_llamacpp(parse_args())
