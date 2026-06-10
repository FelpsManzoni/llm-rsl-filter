#!/usr/bin/env bash
set -euo pipefail

MODEL_1="${MODEL_1:-}"
MODEL_2="${MODEL_2:-}"
MODEL_1_FALLBACK="${MODEL_1_FALLBACK:-}"
MODEL_2_FALLBACK="${MODEL_2_FALLBACK:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/models-cache}"
MODEL_STORE_DIR="${MODEL_STORE_DIR:-/models}"
MODEL_1_PATH="${MODEL_1_PATH:-}"
MODEL_2_PATH="${MODEL_2_PATH:-}"
MODEL_1_FALLBACK_PATH="${MODEL_1_FALLBACK_PATH:-}"
MODEL_2_FALLBACK_PATH="${MODEL_2_FALLBACK_PATH:-}"
LOCAL_ONLY="${LOCAL_ONLY:-true}"
CONTAINER_TASK="${CONTAINER_TASK:-evaluate}"
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

mkdir -p /work /work/data /work/output "${MODEL_CACHE_DIR}" "${MODEL_STORE_DIR}" "${OUTPUT_DIR}"

if [[ "${CONTAINER_TASK}" == "prefetch" ]]; then
  python3 /app/docker/prefetch_models.py
  exit 0
fi

if [[ "${CONTAINER_TASK}" != "evaluate" ]]; then
  echo "Unsupported CONTAINER_TASK=${CONTAINER_TASK}. Expected 'evaluate' or 'prefetch'." >&2
  exit 1
fi

if [[ "${LOCAL_ONLY}" == "true" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

VLLM_PID=""

safe_model_dir_name() {
  local model_id="$1"
  echo "${model_id//\//__}"
}

resolve_model_path() {
  local model_id="$1"
  local explicit_path="$2"

  if [[ -n "${explicit_path}" ]]; then
    echo "${explicit_path}"
    return 0
  fi

  if [[ "${model_id}" == /* || "${model_id}" == ./* || "${model_id}" == ../* ]]; then
    echo "${model_id}"
    return 0
  fi

  echo "${MODEL_STORE_DIR}/$(safe_model_dir_name "${model_id}")"
}

validate_local_model_path() {
  local model_id="$1"
  local model_path="$2"

  if [[ "${LOCAL_ONLY}" != "true" ]]; then
    return 0
  fi

  if [[ ! -d "${model_path}" ]]; then
    echo "Local-only mode is enabled, but model files for ${model_id} were not found at ${model_path}." >&2
    echo "Run: docker compose run --rm -e CONTAINER_TASK=prefetch -e LOCAL_ONLY=false llm-rsl-filter" >&2
    return 1
  fi

  if [[ ! -f "${model_path}/config.json" ]]; then
    echo "Local-only mode is enabled, but ${model_path} does not look like a complete Hugging Face model directory." >&2
    echo "Missing required file: ${model_path}/config.json" >&2
    return 1
  fi
}

wait_for_vllm() {
  local timeout_seconds="$1"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if python3 - "$VLLM_PORT" <<'PY'
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
  local model_path="$3"
  local serve_target="${model_id}"

  if [[ "${LOCAL_ONLY}" == "true" ]]; then
    validate_local_model_path "${model_id}" "${model_path}"
    serve_target="${model_path}"
  fi

  local cmd=(
    vllm serve "${serve_target}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --download-dir "${MODEL_CACHE_DIR}"
    --served-model-name "${model_id}"
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
  local primary_model_path="$4"
  local fallback_model_path="$5"
  local active_model="${primary_model}"
  local active_model_path
  local model_arg_name

  active_model_path="$(resolve_model_path "${primary_model}" "${primary_model_path}")"

  if ! start_vllm "${primary_model}" "" "${active_model_path}"; then
    if [[ "${VLLM_PRECISION_POLICY}" == "auto" ]] && [[ -n "${fallback_model}" ]]; then
      echo "Retrying with fallback model ${fallback_model} and quantization ${VLLM_FALLBACK_QUANTIZATION}."
      active_model="${fallback_model}"
      active_model_path="$(resolve_model_path "${fallback_model}" "${fallback_model_path}")"
      start_vllm "${fallback_model}" "${VLLM_FALLBACK_QUANTIZATION}" "${active_model_path}"
    else
      echo "No fallback configured for model ${primary_model}; aborting." >&2
      return 1
    fi
  fi

  if [[ "${run_mode}" == "model1" ]]; then
    model_arg_name="--model-1"
  else
    model_arg_name="--model-2"
  fi

  VLLM_BASE_URL="${VLLM_BASE_URL}" python3 -m src.main --run-mode "${run_mode}" "${model_arg_name}" "${active_model}" ${RESUME_FLAG}
  stop_vllm
}

trap stop_vllm EXIT

run_model_round "model1" "${MODEL_1}" "${MODEL_1_FALLBACK}" "${MODEL_1_PATH}" "${MODEL_1_FALLBACK_PATH}"
run_model_round "model2" "${MODEL_2}" "${MODEL_2_FALLBACK}" "${MODEL_2_PATH}" "${MODEL_2_FALLBACK_PATH}"
python3 -m src.main --run-mode merge-only

echo "Sequential dual-model pipeline completed successfully."
