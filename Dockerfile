FROM vllm/vllm-openai:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_DIR=/app
ENV WORK_DIR=/work
ENV MODEL_CACHE_DIR=/models-cache
ENV MODEL_STORE_DIR=/models

ARG MODEL_1=Qwen/Qwen2.5-14B-Instruct
ARG MODEL_2=mistralai/Mistral-Nemo-Instruct-2407
ARG MODEL_1_FALLBACK=
ARG MODEL_2_FALLBACK=
ARG HF_TOKEN=
ARG PRELOAD_MODELS=true

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md
COPY .env.example /app/.env.example
COPY docker /app/docker

RUN chmod +x /app/docker/entrypoint.sh

RUN mkdir -p /work /work/data /work/output "${MODEL_CACHE_DIR}" "${MODEL_STORE_DIR}"

RUN if [ "${PRELOAD_MODELS}" = "true" ]; then \
			MODEL_1="${MODEL_1}" \
			MODEL_2="${MODEL_2}" \
			MODEL_1_FALLBACK="${MODEL_1_FALLBACK}" \
			MODEL_2_FALLBACK="${MODEL_2_FALLBACK}" \
			HF_TOKEN="${HF_TOKEN}" \
			MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
			MODEL_STORE_DIR="${MODEL_STORE_DIR}" \
			python3 /app/docker/prefetch_models.py; \
		else \
			echo "Skipping model prefetch during build (set PRELOAD_MODELS=true to enable)."; \
		fi

VOLUME ["/work", "/work/data", "/work/output", "/models", "/models-cache"]

WORKDIR /work
ENV PYTHONPATH=/app

ENTRYPOINT ["/app/docker/entrypoint.sh"]
