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
VLLM_STREAM_LOGS="${VLLM_STREAM_LOGS:-true}"

INPUT_CSV="${INPUT_CSV:-}"
RULES_JSON="${RULES_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
RESUME_FLAG=""
if [[ "${RESUME:-false}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

section() {
  printf '\n[%s] ==== %s ====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

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
  section "Prefetch Models"
  python3 /app/docker/prefetch_models.py
  log "Prefetch task completed."
  exit 0
fi

if [[ "${CONTAINER_TASK}" != "evaluate" ]]; then
  echo "Unsupported CONTAINER_TASK=${CONTAINER_TASK}. Expected 'evaluate' or 'prefetch'." >&2
  exit 1
fi

if [[ "${LOCAL_ONLY}" == "true" ]]; then
  log "Local-only mode enabled; Hugging Face and Transformers offline flags are set."
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
else
  log "Local-only mode disabled; remote model resolution/downloads are allowed."
fi

VLLM_PID=""
VLLM_LOG_TAIL_PID=""

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
  local next_log_after
  start_ts="$(date +%s)"
  next_log_after=0

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

    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      log "vLLM process exited before the readiness endpoint became available."
      return 1
    fi

    local now_ts
    now_ts="$(date +%s)"
    local elapsed=$((now_ts - start_ts))
    if (( elapsed >= next_log_after )); then
      log "Waiting for vLLM readiness on port ${VLLM_PORT}: ${elapsed}s/${timeout_seconds}s elapsed."
      next_log_after=$((elapsed + 30))
    fi
    if (( now_ts - start_ts >= timeout_seconds )); then
      return 1
    fi

    sleep 2
  done
}

stop_vllm() {
  log "Stopping vLLM server."
  if [[ -n "${VLLM_LOG_TAIL_PID}" ]] && kill -0 "${VLLM_LOG_TAIL_PID}" 2>/dev/null; then
    kill "${VLLM_LOG_TAIL_PID}" || true
    wait "${VLLM_LOG_TAIL_PID}" || true
  fi
  VLLM_LOG_TAIL_PID=""

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

  section "Start vLLM"
  log "Model: ${model_id}"
  log "Serve target: ${serve_target}"
  log "Startup timeout: ${VLLM_STARTUP_TIMEOUT}s"
  log "GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
  log "Max model length: ${VLLM_MAX_MODEL_LEN}"
  log "dtype: ${VLLM_DTYPE}"
  : > /tmp/vllm.log
  "${cmd[@]}" >/tmp/vllm.log 2>&1 &
  VLLM_PID="$!"

  if [[ "${VLLM_STREAM_LOGS}" == "true" ]]; then
    tail -n +1 -F /tmp/vllm.log &
    VLLM_LOG_TAIL_PID="$!"
  fi

  if wait_for_vllm "${VLLM_STARTUP_TIMEOUT}"; then
    if [[ -n "${VLLM_LOG_TAIL_PID}" ]] && kill -0 "${VLLM_LOG_TAIL_PID}" 2>/dev/null; then
      kill "${VLLM_LOG_TAIL_PID}" || true
      wait "${VLLM_LOG_TAIL_PID}" || true
      VLLM_LOG_TAIL_PID=""
    fi
    log "vLLM server is ready for model ${model_id}."
    return 0
  fi

  log "vLLM server failed to become ready for model ${model_id}." >&2
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
  section "Run ${run_mode}"
  log "Primary model: ${primary_model}"
  log "Resolved local path: ${active_model_path}"

  if ! start_vllm "${primary_model}" "" "${active_model_path}"; then
    if [[ "${VLLM_PRECISION_POLICY}" == "auto" ]] && [[ -n "${fallback_model}" ]]; then
      log "Retrying with fallback model ${fallback_model} and quantization ${VLLM_FALLBACK_QUANTIZATION}."
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

  log "Starting paper evaluation for ${run_mode} with active model ${active_model}."
  VLLM_BASE_URL="${VLLM_BASE_URL}" python3 -m src.main --run-mode "${run_mode}" "${model_arg_name}" "${active_model}" ${RESUME_FLAG}
  log "Paper evaluation completed for ${run_mode}."
  stop_vllm
}

trap stop_vllm EXIT

section "Evaluation Pipeline"
log "Input CSV: ${INPUT_CSV}"
log "Rules JSON: ${RULES_JSON}"
log "Output dir: ${OUTPUT_DIR}"
log "Batch size: ${BATCH_SIZE:-20}"
run_model_round "model1" "${MODEL_1}" "${MODEL_1_FALLBACK}" "${MODEL_1_PATH}" "${MODEL_1_FALLBACK_PATH}"
run_model_round "model2" "${MODEL_2}" "${MODEL_2_FALLBACK}" "${MODEL_2_PATH}" "${MODEL_2_FALLBACK_PATH}"
section "Merge Results"
python3 -m src.main --run-mode merge-only

log "Sequential dual-model pipeline completed successfully."
