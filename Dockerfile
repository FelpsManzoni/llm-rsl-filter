FROM vllm/vllm-openai:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_DIR=/app
ENV WORK_DIR=/work
ENV MODEL_CACHE_DIR=/models-cache

ARG MODEL_1=
ARG MODEL_2=
ARG MODEL_1_FALLBACK=
ARG MODEL_2_FALLBACK=
ARG HF_TOKEN=
ARG PRELOAD_MODELS=true

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md
COPY .env.example /app/.env.example
COPY docker /app/docker

RUN chmod +x /app/docker/entrypoint.sh

RUN mkdir -p /work /work/data /work/output "${MODEL_CACHE_DIR}"

RUN if [ "${PRELOAD_MODELS}" = "true" ] && [ -n "${MODEL_1}" ] && [ -n "${MODEL_2}" ]; then \
			MODEL_1="${MODEL_1}" \
			MODEL_2="${MODEL_2}" \
			MODEL_1_FALLBACK="${MODEL_1_FALLBACK}" \
			MODEL_2_FALLBACK="${MODEL_2_FALLBACK}" \
			HF_TOKEN="${HF_TOKEN}" \
			MODEL_CACHE_DIR="${MODEL_CACHE_DIR}" \
			python /app/docker/prefetch_models.py; \
		else \
			echo "Skipping model prefetch during build (set PRELOAD_MODELS=true and pass MODEL_1/MODEL_2 build args)."; \
		fi

VOLUME ["/work", "/work/data", "/work/output", "/models-cache"]

WORKDIR /work
ENV PYTHONPATH=/app

ENTRYPOINT ["/app/docker/entrypoint.sh"]
