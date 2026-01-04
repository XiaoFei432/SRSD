#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Defaults (edit if needed)
# -----------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs}"
MODE="${MODE:-quick}"          # quick | paper
WHICH="${WHICH:-all}"          # all | baselines | llama | qwen7b | qwen14b | mistral
DECODE="${DECODE:-greedy}"     # greedy | sample
TASKS="${TASKS:-all}"          # all | code_edit | ...
SAMPLES="${SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"

# Model paths (edit these)
QWEN7B_PATH="${QWEN7B_PATH:-models/Qwen2.5-7B-Instruct}"
QWEN14B_PATH="${QWEN14B_PATH:-models/Qwen2.5-14B-Instruct}"
LLAMA31_8B_PATH="${LLAMA31_8B_PATH:-models/LLM-Research/Meta-Llama-3.1-8B-Instruct}"
LLAMA32_1B_PATH="${LLAMA32_1B_PATH:-models/LLM-Research/Meta-Llama-3.2-1B-Instruct}"
MISTRAL24B_PATH="${MISTRAL24B_PATH:-models/Mistral-Small-3.1-24B-Instruct-2503}"

# Extra assets (edit if you need)
GGUF_PATH="${GGUF_PATH:-models/gguf/model-q4_k_m.gguf}"

# Layer maps (edit if you need)
MAP_LA_GREEDY="${MAP_LA_GREEDY:-Qwen/map/best_layer_La_greedy.json}"
MAP_LA_SAMPLE="${MAP_LA_SAMPLE:-Qwen/map/best_layer_La_sample.json}"
MAP_QW_7B_GREEDY="${MAP_QW_7B_GREEDY:-Qwen/map/best_layer_Qw_7B_greedy.json}"
MAP_QW_7B_SAMPLE="${MAP_QW_7B_SAMPLE:-Qwen/map/best_layer_Qw_7B_sample.json}"
MAP_QW_14B_GREEDY="${MAP_QW_14B_GREEDY:-Qwen/map/best_layer_Qw_14B_greedy.json}"
MAP_QW_14B_SAMPLE="${MAP_QW_14B_SAMPLE:-Qwen/map/best_layer_Qw_14B_sample.json}"
MAP_MIS_GREEDY="${MAP_MIS_GREEDY:-Mistral/map/best_layer_Mis_greedy.json}"
MAP_MIS_SAMPLE="${MAP_MIS_SAMPLE:-Mistral/map/best_layer_Mis_sample.json}"

# Retrieval settings
RETR_TOPK="${RETR_TOPK:-10}"

usage () {
  cat <<EOF
Usage:
  bash eval.sh [--quick|--paper] [--which all|baselines|llama|qwen7b|qwen14b|mistral] [--decode greedy|sample] [--tasks all|code_edit] [--samples N]

Examples:
  bash eval.sh --quick --which baselines --tasks code_edit
  bash eval.sh --paper --which llama --decode greedy
  bash eval.sh --paper --which all
EOF
}

# -----------------------
# Parse args
# -----------------------
TASKS_FROM_ARGS=0
SAMPLES_FROM_ARGS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; shift ;;
    --paper) MODE="paper"; shift ;;
    --which) WHICH="$2"; shift 2 ;;
    --decode) DECODE="$2"; shift 2 ;;
    --tasks) TASKS="$2"; TASKS_FROM_ARGS=1; shift 2 ;;
    --samples) SAMPLES="$2"; SAMPLES_FROM_ARGS=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/eval_${MODE}_${WHICH}_${DECODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "[eval] mode=${MODE} which=${WHICH} decode=${DECODE} tasks=${TASKS} samples=${SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}"
echo "[eval] out_root=${OUT_ROOT}"
echo "[eval] log=${LOG}"

# -----------------------
# Helpers
# -----------------------
need_file () {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[error] missing: $p"
    exit 1
  fi
}

run_baselines_qwen7b_code_edit () {
  local outdir="${OUT_ROOT}/baselines_qwen7b_code_edit"
  mkdir -p "${outdir}"

  echo "[run] AR (PLD with n-gram=0,K=0)"
  python Qwen/run_pld.py \
    --model-path "${QWEN7B_PATH}" \
    --tasks code_edit \
    --samples-per-task 80 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-gram 0 \
    --K 0 \
    --output "${outdir}/AR_greedy.jsonl"

  echo "[run] PLD"
  python Qwen/run_pld.py \
    --model-path "${QWEN7B_PATH}" \
    --tasks code_edit \
    --samples-per-task 80 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --n-gram 4 \
    --K 16 \
    --output "${outdir}/pld_greedy.jsonl"

  echo "[run] Semantic"
  python Qwen/run_semantic.py \
    --model-path "${QWEN7B_PATH}" \
    --tasks code_edit \
    --samples-per-task 90 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --layer-idx 23 \
    --K 16 \
    --sim-threshold 0.0 \
    --retrieval-topk "${RETR_TOPK}" \
    --output "${outdir}/semantic.jsonl"

  echo "[done] baselines -> ${outdir}"
}

run_llama31 () {
  local tag="La_${DECODE}"
  local outdir="${OUT_ROOT}/${tag}"
  mkdir -p "${outdir}"

  if [[ "${DECODE}" == "greedy" ]]; then
    need_file "${MAP_LA_GREEDY}"
    need_file "${GGUF_PATH}"

    python Qwen/auto_bench.py \
      --tasks "${TASKS}" \
      --samples-per-task "${SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --lookahead-enable \
      --out-dir "${outdir}/result" \
      --model-path "${LLAMA31_8B_PATH}" \
      --lookahead-gguf-model "${GGUF_PATH}" \
      --sem-layer-mapping "${MAP_LA_GREEDY}" \
      --sem-retrieval-topk "${RETR_TOPK}" \
      --assisted-target-model "${LLAMA31_8B_PATH}" \
      --assisted-draft-model "${LLAMA32_1B_PATH}"

  else
    need_file "${MAP_LA_SAMPLE}"

    python Qwen/auto_bench.py \
      --tasks "${TASKS}" \
      --samples-per-task "${SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --out-dir "${outdir}/result" \
      --model-path "${LLAMA31_8B_PATH}" \
      --sem-layer-mapping "${MAP_LA_SAMPLE}" \
      --sem-retrieval-topk "${RETR_TOPK}" \
      --assisted-target-model "${LLAMA31_8B_PATH}" \
      --assisted-draft-model "${LLAMA32_1B_PATH}" \
      --do-sample --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}"
  fi

  echo "[done] llama31 -> ${outdir}"
}

run_qwen7b () {
  local outdir="${OUT_ROOT}/Qw_7B_${DECODE}"
  local map="${MAP_QW_7B_GREEDY}"
  local sampling_args=()
  mkdir -p "${outdir}"
  if [[ "${DECODE}" == "sample" ]]; then
    map="${MAP_QW_7B_SAMPLE}"
    sampling_args=(--do-sample --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}")
  fi
  need_file "${map}"

  # 你这里原命令用的是 auto_bench1.py 且只给 greedy
  python Qwen/auto_bench1.py \
    --tasks "${TASKS}" \
    --samples-per-task "${SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out-dir "${outdir}/result" \
    --model-path "${QWEN7B_PATH}" \
    --sem-layer-mapping "${map}" \
    --sem-retrieval-topk "${RETR_TOPK}" \
    "${sampling_args[@]}"

  echo "[done] qwen7b -> ${outdir}"
}

run_qwen14b () {
  local outdir="${OUT_ROOT}/Qw_14B_${DECODE}"
  local map="${MAP_QW_14B_GREEDY}"
  local sampling_args=()
  mkdir -p "${outdir}"
  if [[ "${DECODE}" == "sample" ]]; then
    map="${MAP_QW_14B_SAMPLE}"
    sampling_args=(--do-sample --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}")
  fi
  need_file "${map}"

  python Qwen/auto_bench.py \
    --tasks "${TASKS}" \
    --samples-per-task "${SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out-dir "${outdir}/result" \
    --model-path "${QWEN14B_PATH}" \
    --sem-layer-mapping "${map}" \
    --sem-retrieval-topk "${RETR_TOPK}" \
    --assisted-target-model "${QWEN14B_PATH}" \
    --assisted-draft-model models/Qwen2.5-0.5B-Instruct \
    "${sampling_args[@]}"

  echo "[done] qwen14b -> ${outdir}"
}

run_mistral () {
  local outdir="${OUT_ROOT}/Mis_${DECODE}"
  local map="${MAP_MIS_GREEDY}"
  local sampling_args=()
  mkdir -p "${outdir}"
  if [[ "${DECODE}" == "sample" ]]; then
    if [[ -e "${MAP_MIS_SAMPLE}" ]]; then
      map="${MAP_MIS_SAMPLE}"
    else
      echo "[warn] missing sample map: ${MAP_MIS_SAMPLE}; falling back to greedy map"
    fi
    sampling_args=(--do-sample --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}")
  fi
  need_file "${map}"

  python Mistral/auto_bench.py \
    --tasks "${TASKS}" \
    --samples-per-task "${SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out-dir "${outdir}/result" \
    --model-path "${MISTRAL24B_PATH}" \
    --sem-layer-mapping "${map}" \
    --sem-retrieval-topk "${RETR_TOPK}" \
    "${sampling_args[@]}"

  echo "[done] mistral -> ${outdir}"
}

# -----------------------
# Quick mode overrides
# -----------------------
if [[ "${MODE}" == "quick" ]]; then
  # quick：少样本、少任务，保证审稿人能跑通
  SAMPLES=5
  if [[ "${TASKS_FROM_ARGS}" -eq 0 ]]; then
    TASKS="code_edit"
  fi
  if [[ "${SAMPLES_FROM_ARGS}" -eq 0 ]]; then
    SAMPLES=5
  fi
  MAX_NEW_TOKENS=256
  echo "[eval] quick overrides: samples=${SAMPLES} tasks=${TASKS} max_new_tokens=${MAX_NEW_TOKENS}"
fi

# -----------------------
# Dispatch
# -----------------------
case "${WHICH}" in
  baselines)
    run_baselines_qwen7b_code_edit
    ;;
  llama)
    run_llama31
    ;;
  qwen7b)
    run_qwen7b
    ;;
  qwen14b)
    run_qwen14b
    ;;
  mistral)
    run_mistral
    ;;
  all)
    # 你可以按需删减：all 可能很慢
    run_llama31
    run_qwen7b
    run_qwen14b
    run_mistral
    ;;
  *)
    echo "[error] unknown --which ${WHICH}"
    exit 1
    ;;
esac

echo "[all done] outputs in ${OUT_ROOT}"
