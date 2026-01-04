#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 检查 speculative 生成文件和 greedy 生成文件的一致性（增强版）
# python check.py --greedy result/pld_greedy.jsonl --target result/semantic_greedy.jsonl --show-diff --strict

import json
import argparse
import sys
from difflib import SequenceMatcher

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def normalize_text(s: str) -> str:
    # 可选：统一换行，去除行尾空格
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s

def load_data(filepath, do_normalize: bool):
    """读取 JSONL 文件并建立 (task, sample_idx) -> output 的映射，同时检测重复 key"""
    data = {}
    dup_keys = []
    total_lines = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                total_lines += 1
                obj = json.loads(line)

                task = obj.get('task')
                sample_idx = obj.get('sample_idx')
                key = (task, sample_idx)

                out = obj.get('output', "")
                if out is None:
                    out = ""

                if do_normalize:
                    out = normalize_text(out)

                if key in data:
                    dup_keys.append(key)
                data[key] = out

    except FileNotFoundError:
        print(f"{Colors.RED}[Error] File not found: {filepath}{Colors.RESET}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}[Error] JSON decode error in {filepath}: {e}{Colors.RESET}")
        sys.exit(1)

    return data, dup_keys, total_lines

def find_first_diff(str1, str2):
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            return i
    if len(str1) != len(str2):
        return min_len
    return -1

def safe_sort_key(k):
    task, idx = k
    task = "" if task is None else str(task)
    idx = -1 if idx is None else int(idx)
    return (task, idx)

def main():
    parser = argparse.ArgumentParser(description="Check consistency between Greedy and Speculative outputs.")
    parser.add_argument("--greedy", required=True, help="Path to the greedy (baseline) jsonl file")
    parser.add_argument("--target", required=True, help="Path to the speculative (PLD/Semantic) jsonl file")
    parser.add_argument("--show-diff", action="store_true", help="Show details of mismatched samples")
    parser.add_argument("--limit", type=int, default=5, help="Max number of diffs to show")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any missing samples or duplicate keys exist")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize outputs (CRLF->LF, strip trailing spaces) before comparing")
    args = parser.parse_args()

    print(f"Loading Baseline : {args.greedy}")
    print(f"Loading Target   : {args.target}")

    greedy_map, greedy_dups, greedy_lines = load_data(args.greedy, args.normalize)
    target_map, target_dups, target_lines = load_data(args.target, args.normalize)

    # 报告基本信息
    print(f"Baseline lines: {greedy_lines}, unique keys: {len(greedy_map)}, dup keys: {len(greedy_dups)}")
    print(f"Target   lines: {target_lines}, unique keys: {len(target_map)}, dup keys: {len(target_dups)}")
    if greedy_dups:
        print(f"{Colors.YELLOW}[Warn] Baseline has duplicate keys (show up to 5): {greedy_dups[:5]}{Colors.RESET}")
    if target_dups:
        print(f"{Colors.YELLOW}[Warn] Target has duplicate keys (show up to 5): {target_dups[:5]}{Colors.RESET}")

    greedy_keys = set(greedy_map.keys())
    target_keys = set(target_map.keys())

    common_keys = greedy_keys & target_keys
    only_in_greedy = greedy_keys - target_keys
    only_in_target = target_keys - greedy_keys

    print(f"Common keys      : {len(common_keys)}")
    print(f"Only in baseline : {len(only_in_greedy)}")
    print(f"Only in target   : {len(only_in_target)}")

    if args.strict and (only_in_greedy or only_in_target or greedy_dups or target_dups):
        print(f"{Colors.RED}[Strict Fail] Missing samples or duplicate keys detected.{Colors.RESET}")
        sys.exit(2)

    if not common_keys:
        print(f"{Colors.RED}[Error] No common (task, sample_idx) found between files.{Colors.RESET}")
        sys.exit(1)

    sorted_keys = sorted(list(common_keys), key=safe_sort_key)
    print(f"\nComparing {len(sorted_keys)} common samples...\n")

    total_samples = 0
    exact_matches = 0
    total_similarity = 0.0
    mismatched_samples = []

    for key in sorted_keys:
        total_samples += 1
        g_text = greedy_map[key]
        t_text = target_map[key]

        if g_text == t_text:
            exact_matches += 1
            total_similarity += 1.0
        else:
            matcher = SequenceMatcher(None, g_text, t_text)
            sim = matcher.ratio()
            total_similarity += sim
            mismatched_samples.append({"key": key, "g_text": g_text, "t_text": t_text, "sim": sim})

    avg_sim = total_similarity / total_samples if total_samples else 0
    exact_rate = exact_matches / total_samples if total_samples else 0

    print("=" * 50)
    print(f"{Colors.BOLD}Consistency Report{Colors.RESET}")
    print("=" * 50)
    print(f"Samples Evaluated : {total_samples}")

    color_exact = Colors.GREEN if exact_rate > 0.99 else (Colors.YELLOW if exact_rate > 0.90 else Colors.RED)
    print(f"Exact Match Rate  : {color_exact}{exact_rate:.2%}{Colors.RESET}")

    color_sim = Colors.GREEN if avg_sim > 0.99 else (Colors.YELLOW if avg_sim > 0.95 else Colors.RED)
    print(f"Avg Character Sim : {color_sim}{avg_sim:.4f}{Colors.RESET}")
    print("=" * 50)

    if exact_rate == 1.0:
        print(f"{Colors.GREEN}Perfect Match! The algorithm is strictly lossless (string-level).{Colors.RESET}")
    elif avg_sim > 0.99:
        print(f"{Colors.YELLOW}High Similarity (>99%). Check decoding/whitespace/EOS issues.{Colors.RESET}")
    else:
        print(f"{Colors.RED}Significant Deviation. Potential bug in speculation acceptance / cache logic.{Colors.RESET}")

    if mismatched_samples and args.show_diff:
        print(f"\nShowing top {min(args.limit, len(mismatched_samples))} mismatches (lowest similarity first):")
        mismatched_samples.sort(key=lambda x: x['sim'])
        for item in mismatched_samples[:args.limit]:
            key = item['key']
            g_txt = item['g_text']
            t_txt = item['t_text']
            diff_idx = find_first_diff(g_txt, t_txt)

            start = max(0, diff_idx - 20)
            end_g = min(len(g_txt), diff_idx + 50)
            end_t = min(len(t_txt), diff_idx + 50)

            snippet_g = g_txt[start:end_g].replace("\n", "\\n")
            snippet_t = t_txt[start:end_t].replace("\n", "\\n")

            print("-" * 30)
            print(f"Task: {key[0]} | ID: {key[1]} | Sim: {item['sim']:.4f}")
            print(f"Greedy: ...{snippet_g}...")
            print(f"Target: ...{snippet_t}...")
            print(f"{Colors.RED}^ Diverges at char {diff_idx}{Colors.RESET}")

if __name__ == "__main__":
    main()
