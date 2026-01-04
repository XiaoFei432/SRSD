# SRSD (Anonymous Repo)

This repository contains a reference implementation and evaluation scripts for **Semantic Retrieval Speculative Decoding (SRSD)**, a **training-free** decoding acceleration method that reuses a target LLM’s intermediate-layer hidden states as an in-context semantic index.

> **Anonymous submission note:** This repo is provided for double-blind review. Please avoid adding any identifying information (names/emails/affiliations) in commits, issues, logs, or file metadata.

## Third-party dependencies (not included in this repo)

This repository does **not** vendor third-party code. To reproduce our setup, please fetch the following dependencies at the pinned commits:

### EAGLE
- Source: `https://github.com/SafeAILab/EAGLE.git`
- Commit: `791597abcf8d61245ea0784d94c518acc4a5814b` (`791597a`)

```bash
mkdir -p third_party
git clone https://github.com/SafeAILab/EAGLE.git third_party/EAGLE
cd third_party/EAGLE
git checkout 791597abcf8d61245ea0784d94c518acc4a5814b
```

### llama.cpp
- Source: `https://github.com/ggerganov/llama.cpp.git`
- Commit: `22577583a38ec0d236e6b4d45357c5e79021da07` (`b7312`)

```bash
mkdir -p third_party
git clone https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
cd third_party/llama.cpp
git checkout 22577583a38ec0d236e6b4d45357c5e79021da07
```

---

## Quickstart (recommended)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Sanity check (fast)
bash eval.sh --quick --which baselines --tasks code_edit

# 3) Run a paper-scale setting (example: Llama-3.1 greedy)
bash eval.sh --paper --which llama --decode greedy

---



