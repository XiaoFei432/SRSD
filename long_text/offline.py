#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bench_retrieval_overhead_offline.py

Offline microbenchmark (Top-R=10 only):
- retrieval overhead = cosine mv + topk selection time
- repeat 10 times and average
- BOXPLOT style visualization with real sample data
- FIXED: sorted=True to match actual semantic.py behavior


python offline.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --text-path longtext.txt \
  --lengths 4k:32k:4k \
  --window-sizes 0 \
  --layer-idx 3 \
  --samples-per-L 8 \
  --repeat 50 \
  --num-runs 10 \
  --output-json overhead_topr10.json \
  --output-plot overhead_topr10.png
"""

import argparse
import json
import random
import time
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------
# Utils
# ----------------------
retrieval_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype() -> torch.dtype:
    """获取计算精度，不指定特定GPU"""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        return torch.float32


def get_model_device(model) -> torch.device:
    """获取模型所在的设备（多卡时返回第一个参数所在的设备）"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _parse_one_len(tok: str) -> int:
    tok = tok.strip().lower()
    if tok.endswith("k"):
        return int(float(tok[:-1]) * 1024)
    return int(tok)


def parse_lengths(s: str) -> List[int]:
    s = s.strip()
    if ":" in s and "," not in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Bad --lengths range format: {s}. Expect start:end:step")
        start = _parse_one_len(parts[0])
        end = _parse_one_len(parts[1])
        step = _parse_one_len(parts[2])
        if step <= 0:
            raise ValueError("step must be > 0")
        if end < start:
            raise ValueError("end must be >= start")

        out = []
        x = start
        while x <= end:
            out.append(int(x))
            x += step
        return out

    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(_parse_one_len(part))
    return out


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_transformer_layers(model):
    for base_attr in ["model", "transformer"]:
        base = getattr(model, base_attr, None)
        if base is None:
            continue
        for layers_attr in ["layers", "h", "blocks"]:
            layers = getattr(base, layers_attr, None)
            if layers is not None:
                return layers
    layers = getattr(model, "layers", None)
    if layers is not None:
        return layers
    raise RuntimeError("Cannot locate transformer layers for hook.")


class Capture:
    def __init__(self):
        self.hs = None
        self.handle = None


def install_hook(model, layer_idx: int) -> Capture:
    layers = get_transformer_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"layer_idx={layer_idx} out of range (num_layers={len(layers)})")

    cap = Capture()

    def hook(_m, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        cap.hs = hs.detach()

    cap.handle = layers[layer_idx].register_forward_hook(hook)
    return cap


def remove_hook(cap: Capture):
    if cap and cap.handle:
        cap.handle.remove()
        cap.handle = None


def print_gpu_info():
    """打印 GPU 信息"""
    if not torch.cuda.is_available():
        print("[GPU] No CUDA available, using CPU")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"[GPU] Found {num_gpus} GPU(s):")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  [{i}] {props.name} ({mem_gb:.1f} GB)")
    print(f"{'='*60}\n")


# ----------------------
# Timing kernels
# ----------------------

def _warmup(fn, n: int = 5):
    for _ in range(n):
        fn()


def time_mv_topk_ms(keys: torch.Tensor, q: torch.Tensor, top_r: int, repeat: int) -> float:
    """
    cosine mv + topk selection (Top-R)
    
    FIXED: sorted=True to match run_semantic.py behavior
    """
    top_r = int(top_r)
    device = keys.device

    def fn():
        sims = torch.mv(keys, q)
        k = min(top_r, sims.numel())
        _ = torch.topk(sims, k=k, largest=True, sorted=True)

    _warmup(fn)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(repeat):
            fn()
        e.record()
        torch.cuda.synchronize(device)
        return float(s.elapsed_time(e) / max(repeat, 1))

    t0 = time.time()
    for _ in range(repeat):
        fn()
    return float((time.time() - t0) * 1000.0 / max(repeat, 1))


@torch.no_grad()
def prefill_one_layer_hidden(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    cap = install_hook(model, layer_idx)
    _ = model(input_ids=input_ids, use_cache=False, output_hidden_states=False, return_dict=True)
    hs = cap.hs
    remove_hook(cap)
    if hs is None:
        raise RuntimeError("Hook capture failed.")
    return hs  # [1, L, D]


# ----------------------
# Enhanced Boxplot Visualization
# ----------------------

def _save_boxplot(results: Dict[str, Any], windows: List[int], out_path: str):
    """
    Enhanced boxplot visualization using REAL sample data
    ✅ 调整后的样式：
    - 移除标题
    - 字体整体放大 1.5 倍
    - 横纵坐标标签放大 2 倍
    """
    import os
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    # ✅ 调整后的样式设置
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,              # 原来 12 → 现在 18 (1.5倍)
        "axes.titlesize": 21,         # 原来 14 → 现在 21 (1.5倍) [标题会移除，但保留设置]
        "axes.labelsize": 26,         # 原来 13 → 现在 26 (2倍)
        "legend.fontsize": 16.5,      # 原来 11 → 现在 16.5 (1.5倍)
        "xtick.labelsize": 22,        # 原来 11 → 现在 22 (2倍)
        "ytick.labelsize": 22,        # 原来 11 → 现在 22 (2倍)
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
    })

    base, ext = os.path.splitext(out_path)
    if ext == "":
        ext = ".png"

    for w in windows:
        w_key = str(w)
        
        Ls = results[w_key]["L"]
        box_data = results[w_key]["overhead_ms_raw"]
        means = np.array(results[w_key]["overhead_ms_mean"])
        
        # ✅ 图片尺寸也相应放大
        fig, ax = plt.subplots(figsize=(max(14, len(Ls)*1.2), 8))
        
        positions = list(range(1, len(Ls) + 1))
        
        # 创建箱线图
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor="#3498db", alpha=0.8, linewidth=2.2, edgecolor='#1a5490'),  # 线条也加粗
            whiskerprops=dict(linewidth=2.2, color='#555555'),
            capprops=dict(linewidth=2.2, color='#555555'),
            medianprops=dict(color="#e74c3c", linewidth=3.5),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, alpha=0.5, markeredgecolor='none')
        )
        
        # 添加均值标记（标记也放大）
        for i, pos in enumerate(positions):
            ax.plot(pos, means[i], marker="D", color="#2ecc71", 
                   markersize=10, zorder=3, markeredgecolor="white", markeredgewidth=2)
        
        # 背景网格
        ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=1.2, color='gray')
        ax.set_axisbelow(True)
        
        # X轴标签
        x_labels = [f"{L//1024}K" if L >= 1024 else str(L) for L in Ls]
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=0 if len(Ls) <= 10 else 45,
                          ha="center" if len(Ls) <= 10 else "right")
        ax.set_xlabel("Context Length (tokens)", fontweight='bold')
        ax.set_ylabel("Retrieval Overhead (ms)", fontweight='bold')
        ax.set_ylim(bottom=0)
        
        # ✅ 移除标题
        # title_w = "Global Retrieval" if w <= 0 else f"Windowed Retrieval (window={w})"
        # ax.set_title(f"...", fontweight='bold', pad=20)
        
        # 图例（标记尺寸也放大）
        handles = [
            mpatches.Patch(facecolor="#3498db", alpha=0.8, edgecolor='#1a5490',
                          linewidth=2.2, label="Distribution (box=IQR)"),
            mlines.Line2D([], [], color="#e74c3c", linewidth=3.5, label="Median"),
            mlines.Line2D([], [], color="#2ecc71", marker="D", markersize=10,
                         linestyle="", markeredgecolor="white", markeredgewidth=2,
                         label="Mean"),
            mlines.Line2D([], [], color="gray", marker="o", markersize=6,
                         linestyle="", alpha=0.5, label="Outliers"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=16.5,  # ✅ 图例字体 1.5倍
                 frameon=True, shadow=False, framealpha=0.95)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_edgecolor('#cccccc')
        
        plt.tight_layout()
        
        save_path = f"{base}_w{w}{ext}" if len(windows) > 1 else f"{base}{ext}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[Save] {save_path}")


# ----------------------
# Model Loading
# ----------------------

def load_model_and_tokenizer(model_path: str):
    """
    加载模型和分词器，自动分配到多张GPU
    """
    print(f"[Model] Loading {model_path} ...")
    
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = get_dtype()
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Model] Using {num_gpus} GPU(s) with device_map='auto'")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        print("[Model] No GPU available, loading on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    model.eval()
    
    if hasattr(model, 'hf_device_map'):
        print(f"[Model] Device map: {model.hf_device_map}")
    
    return tok, model


# ----------------------
# Main
# ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--text-path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lengths", type=str, default="1k,2k,4k,8k,16k")
    ap.add_argument("--window-sizes", type=str, default="0")
    ap.add_argument("--layer-idx", type=int, default=3)
    ap.add_argument("--samples-per-L", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--num-runs", type=int, default=10,
                   help="Number of complete runs to average over")

    ap.add_argument("--output-json", type=str, default="overhead_topr10.json")
    ap.add_argument("--output-plot", type=str, default="overhead_topr10.png")
    
    ap.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated GPU IDs (e.g., '0,1,2'). If not set, use all available GPUs.")
    
    args = ap.parse_args()

    set_seed(args.seed)
    
    print_gpu_info()
    
    if args.gpu_ids is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"[GPU] Limiting to GPUs: {args.gpu_ids}")

    tok, model = load_model_and_tokenizer(args.model_path)
    
    model_device = get_model_device(model)
    print(f"[Model] Primary device: {model_device}")

    with open(args.text_path, "r", encoding="utf-8") as f:
        text = f.read()

    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()

    lengths = parse_lengths(args.lengths)
    windows = parse_int_list(args.window_sizes)
    top_r = 10

    need = max(lengths) + 4
    if len(ids) < need:
        raise RuntimeError(f"text too short: need >= {need}, got {len(ids)}")

    # ✅ 存储所有 runs 的原始样本数据
    all_runs_results = []

    print(f"\n{'='*60}")
    print(f"Running {args.num_runs} complete experiments...")
    print(f"Model: {args.model_path}")
    print(f"Retrieval: Top-{top_r} with sorted=True")
    print(f"{'='*60}\n")

    for run_idx in range(args.num_runs):
        print(f"\n{'*'*60}")
        print(f"RUN {run_idx + 1}/{args.num_runs}")
        print(f"{'*'*60}")

        # ✅ 修改数据结构：保存原始样本
        run_results: Dict[str, Dict[str, List]] = {}
        for w in windows:
            run_results[str(w)] = {
                "L": [], 
                "overhead_ms_mean": [], 
                "overhead_ms_std": [],
                "overhead_ms_raw": []  # ✅ 新增：保存原始样本
            }

        for L in lengths:
            print(f"\n==== L={L} ====")
            per_w: Dict[int, List[float]] = {w: [] for w in windows}

            for _ in range(args.samples_per_L):
                start = random.randint(0, len(ids) - (L + 2))
                ctx = ids[start:start + L]
                
                input_ids = torch.tensor([ctx], device=model_device, dtype=torch.long)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                hs = prefill_one_layer_hidden(model, input_ids, args.layer_idx)[0]

                keys = F.normalize(hs[:-1].to(dtype=retrieval_dtype), dim=-1)
                q    = F.normalize(hs[-1].to(dtype=retrieval_dtype),  dim=-1)

                hi = L - 1
                for w in windows:
                    lo = 0 if w <= 0 else max(0, hi - w)
                    ks = keys[lo:hi].contiguous()

                    ms = time_mv_topk_ms(ks, q, top_r=top_r, repeat=args.repeat)
                    per_w[w].append(ms)

                del hs, keys, q, input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ✅ 保存原始样本数据
            for w in windows:
                arr = np.array(per_w[w], dtype=np.float64)
                run_results[str(w)]["L"].append(int(L))
                run_results[str(w)]["overhead_ms_mean"].append(float(arr.mean()))
                run_results[str(w)]["overhead_ms_std"].append(float(arr.std()))
                run_results[str(w)]["overhead_ms_raw"].append(per_w[w])  # ✅ 保存原始样本
                print(f"[w={w:5d}] overhead(ms)={arr.mean():.3f} ± {arr.std():.3f}")

        all_runs_results.append(run_results)

    # ✅ 聚合多次运行的数据
    print(f"\n{'='*60}")
    print("Aggregating data from all runs...")
    print(f"{'='*60}\n")

    final_results: Dict[str, Dict[str, List]] = {}
    for w in windows:
        w_key = str(w)
        final_results[w_key] = {
            "L": all_runs_results[0][w_key]["L"],
            "overhead_ms_mean": [],
            "overhead_ms_std": [],
            "overhead_ms_raw": []  # ✅ 聚合所有 runs 的样本
        }

        num_Ls = len(final_results[w_key]["L"])
        
        for i in range(num_Ls):
            # 收集所有 runs 在这个 L 的样本
            all_samples = []
            for run in range(args.num_runs):
                all_samples.extend(all_runs_results[run][w_key]["overhead_ms_raw"][i])
            
            # 计算聚合统计
            all_samples_arr = np.array(all_samples)
            avg_mean = float(all_samples_arr.mean())
            avg_std = float(all_samples_arr.std())
            
            final_results[w_key]["overhead_ms_mean"].append(avg_mean)
            final_results[w_key]["overhead_ms_std"].append(avg_std)
            final_results[w_key]["overhead_ms_raw"].append(all_samples)  # ✅ 保存所有样本用于箱线图
            
            L = final_results[w_key]["L"][i]
            print(f"L={L:5d} | w={w:5d} | "
                  f"mean={avg_mean:.3f}ms | std={avg_std:.3f}ms | "
                  f"n_samples={len(all_samples)}")

    # 获取 GPU 信息
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "id": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / (1024**3), 2)
            })

    # ✅ 保存 JSON（不包含原始样本，太大）
    payload = {
        "model_path": args.model_path,
        "layer_idx": args.layer_idx,
        "lengths": lengths,
        "windows": windows,
        "top_r": top_r,
        "sorted": True,
        "samples_per_L": args.samples_per_L,
        "repeat": args.repeat,
        "num_runs": args.num_runs,
        "gpu_info": gpu_info,
        "device_map": "auto",
        "retrieval": "cosine mv + topk (Top-10, sorted=True) (offline microbench, boxplot from all runs)",
        "results": {
            w_key: {
                "L": final_results[w_key]["L"],
                "overhead_ms_mean": final_results[w_key]["overhead_ms_mean"],
                "overhead_ms_std": final_results[w_key]["overhead_ms_std"],
                # ✅ 可选：如果 JSON 不会太大，可以保存原始样本
                # "overhead_ms_raw": [list(map(float, samples)) for samples in final_results[w_key]["overhead_ms_raw"]]
            }
            for w_key in final_results
        },
    }
    
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] {args.output_json}")

    # ✅ 使用真实样本数据绘制箱线图
    _save_boxplot(final_results, windows, args.output_plot)


if __name__ == "__main__":
    main()
