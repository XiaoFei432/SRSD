#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared data utilities for Spec-Bench question.jsonl.

负责：
- SPEC_BENCH_QUESTION_PATH
- TASK_CHOICES（13 个细粒度任务）
- get_dataset_entries：返回原始 JSON 条目
- get_dataset_prompts：从 JSON 中抽取 user 侧对话，拼成纯文本 prompt
"""

import json
from typing import List, Dict, Any, Optional

# 路径与之前脚本保持一致
SPEC_BENCH_QUESTION_PATH: str = "../spec_bench/long_text.jsonl"

# 13 个细粒度任务，与各个 run_*.py 中定义保持一致
TASK_CHOICES: List[str] = [
    "project",
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


def get_dataset_entries(
    task: str,
    samples_per_task: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    读取 Spec-Bench 的原始 JSON 条目，用于需要保留 metadata 的场景（例如 llama.cpp lookahead）。

    Args:
        task: 任务名，必须在 TASK_CHOICES 中。
        samples_per_task: 该任务最多读取多少条；None 或 <=0 表示读取所有。

    Returns:
        一个列表，每个元素是从 question.jsonl 读取的一条 JSON 对象。
    """
    assert task in TASK_CHOICES, f"Unknown task: {task}"
    entries: List[Dict[str, Any]] = []

    try:
        with open(SPEC_BENCH_QUESTION_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("category") != task:
                    continue
                entries.append(obj)
    except FileNotFoundError:
        print(f"[Error] Not found: {SPEC_BENCH_QUESTION_PATH}")
        return []

    if not entries:
        print(f"[Warn] No samples for task={task}")
        return []

    if samples_per_task is not None and samples_per_task > 0:
        entries = entries[: samples_per_task]

    print(f"[Data] Task={task}: loaded {len(entries)} entries")
    return entries


def get_dataset_prompts(
    task: str,
    samples_per_task: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """
    读取 Spec-Bench 的 user turns，并拼成纯文本 prompt。

    设计成兼容原来所有调用：
      - semantic: get_dataset_prompts(task, samples_per_task, seed=42)
      - pld / assisted / eagle3: get_dataset_prompts(task, samples_per_task)

    这里不做 shuffle，保持原来“按顺序取前 N 条”的行为。
    """
    # 兼容性：保留 assert
    assert task in TASK_CHOICES, f"Unknown task: {task}"

    prompts: List[str] = []
    try:
        with open(SPEC_BENCH_QUESTION_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("category") != task:
                    continue

                turns = obj.get("turns", [])
                if isinstance(turns, list):
                    prompt = "\n".join(turns)
                else:
                    prompt = str(turns)
                prompts.append(prompt)
    except FileNotFoundError:
        print(f"[Error] Dataset file not found at {SPEC_BENCH_QUESTION_PATH}")
        return []

    if not prompts:
        print(f"[Warn] No samples found for task={task} in {SPEC_BENCH_QUESTION_PATH}")
        return []

    if samples_per_task is not None and samples_per_task > 0:
        prompts = prompts[: min(samples_per_task, len(prompts))]

    print(f"[Data] Task={task}: loaded {len(prompts)} prompts (ordered, no shuffle)")
    return prompts
