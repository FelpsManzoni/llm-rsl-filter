#!/usr/bin/env bash
set -euo pipefail

MODEL_1="${MODEL_1:-}"
MODEL_2="${MODEL_2:-}"
MODEL_1_FALLBACK="${MODEL_1_FALLBACK:-}"
MODEL_2_FALLBACK="${MODEL_2_FALLBACK:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/models-cache}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_PRECISION_POLICY="${VLLM_PRECISION_POLICY:-auto}"
VLLM_FALLBACK_QUANTIZATION="${VLLM_FALLBACK_QUANTIZATION:-awq}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"

INPUT_CSV="${INPUT_CSV:-}"
RULES_JSON="${RULES_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
RESUME_FLAG=""
if [[ "${RESUME:-false}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="dummy"
fi

if [[ -z "${MODEL_1}" || -z "${MODEL_2}" ]]; then
  echo "MODEL_1 and MODEL_2 must be set." >&2
  exit 1
fi

if [[ -z "${INPUT_CSV}" || -z "${RULES_JSON}" ]]; then
  echo "INPUT_CSV and RULES_JSON must be set." >&2
  exit 1
fi

mkdir -p /work /work/data /work/output "${MODEL_CACHE_DIR}" "${OUTPUT_DIR}"

VLLM_PID=""

wait_for_vllm() {
  local timeout_seconds="$1"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if python - "$VLLM_PORT" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        data = json.loads(response.read().decode("utf-8"))
        if isinstance(data, dict):
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
    then
      return 0
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_seconds )); then
      return 1
    fi

    sleep 2
  done
}

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
  VLLM_PID=""
}

start_vllm() {
  local model_id="$1"
  local quantization="$2"

  local cmd=(
    vllm serve "${model_id}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --download-dir "${MODEL_CACHE_DIR}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --dtype "${VLLM_DTYPE}"
  )

  if [[ -n "${quantization}" ]]; then
    cmd+=(--quantization "${quantization}")
  fi

  echo "Starting vLLM server for model ${model_id}."
  "${cmd[@]}" >/tmp/vllm.log 2>&1 &
  VLLM_PID="$!"

  if wait_for_vllm "${VLLM_STARTUP_TIMEOUT}"; then
    echo "vLLM server is ready for model ${model_id}."
    return 0
  fi

  echo "vLLM server failed to become ready for model ${model_id}." >&2
  tail -n 80 /tmp/vllm.log || true
  stop_vllm
  return 1
}

run_model_round() {
  local run_mode="$1"
  local primary_model="$2"
  local fallback_model="$3"

  if ! start_vllm "${primary_model}" ""; then
    if [[ "${VLLM_PRECISION_POLICY}" == "auto" ]] && [[ -n "${fallback_model}" ]]; then
      echo "Retrying with fallback model ${fallback_model} and quantization ${VLLM_FALLBACK_QUANTIZATION}."
      start_vllm "${fallback_model}" "${VLLM_FALLBACK_QUANTIZATION}"
    else
      echo "No fallback configured for model ${primary_model}; aborting." >&2
      return 1
    fi
  fi

  VLLM_BASE_URL="${VLLM_BASE_URL}" python -m src.main --run-mode "${run_mode}" ${RESUME_FLAG}
  stop_vllm
}

trap stop_vllm EXIT

run_model_round "model1" "${MODEL_1}" "${MODEL_1_FALLBACK}"
run_model_round "model2" "${MODEL_2}" "${MODEL_2_FALLBACK}"
python -m src.main --run-mode merge-only

echo "Sequential dual-model pipeline completed successfully."
