# llm-rsl-filter

Dual-LLM pre-evaluation pipeline for SLR papers.

This repository now supports an all-in-one Docker workflow where the same container:
- Hosts local vLLM for one model at a time.
- Runs model 1 evaluation round.
- Restarts local vLLM for model 2 evaluation round.
- Produces per-model outputs and final consensus output.

The pipeline:
- Reads papers from CSV (paper id, title, abstract, keywords)
- Evaluates papers in batches of 20 with two models in sequential rounds (never simultaneously)
- Writes one CSV per model with per-paper decision and matched rule IDs
- Writes a final consensus CSV applying your precedence rules

## Decision Rules

Per model decision:
- If any exclusion rule is matched: `exclude`
- Else if insufficient information: `manual review`
- Else if inclusion rules are matched: `include`
- Else: `manual review`

Final consensus decision:
- If any model is `exclude`: `exclude`
- Else if any model is `manual review`: `manual review`
- Else: `include`

## Input Formats

### Papers CSV

Required columns:
- `paper_id` (or `id`)
- `title`
- `abstract`
- `keywords`

Example:

```csv
paper_id,title,abstract,keywords
P001,Deep Learning for X,"This paper explores...","deep learning;nlp"
P002,Survey of Y,"A systematic survey...","survey;healthcare"
```

### Convert Scopus Export to Papers CSV

If your source file is a Scopus export, convert it to the pipeline format with:

```bash
python scripts/convert_scopus_csv.py \
	--input-csv "/home/fmanzoni-lx/Downloads/scopus_export_May 17-2026_39086f68-fa71-43ad-8ae8-86a29edf2519.csv" \
	--output-csv data/papers.csv
```

The converter keeps only these columns from Scopus:
- `Title`
- `Abstract`
- `Author Keywords`

And writes the pipeline-ready schema:
- `paper_id`
- `title`
- `abstract`
- `keywords`

### Rules JSON

Supported formats:
- A list of rules
- Or an object with `rules` key containing the list

Each rule must have:
- `rule_id`
- `type`: `inclusion` or `exclusion`
- `text`

Example:

```json
{
	"rules": [
		{"rule_id": "I1", "type": "inclusion", "text": "Focuses on machine learning methods"},
		{"rule_id": "I2", "type": "inclusion", "text": "Published in peer-reviewed venue"},
		{"rule_id": "E1", "type": "exclusion", "text": "Not written in English"},
		{"rule_id": "E2", "type": "exclusion", "text": "Not related to the target domain"}
	]
}
```

## Outputs

Generated in `--output-dir`:
- `results_model_1.csv`
- `results_model_2.csv`
- `results_final.csv`

Per-model CSV includes:
- `paper_id`
- `title`
- `decision` (`include|exclude|manual review`)
- `matched_rule_ids`
- plus audit columns (`model_name`, matched inclusion/exclusion IDs, rationale)

Final CSV includes:
- `paper_id`
- `title`
- `model_1_decision`
- `model_2_decision`
- `final_decision`

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables (or use .env with your preferred loader):

```bash
export OPENAI_API_KEY="your_key"
export MODEL_1="Qwen/Qwen2.5-14B-Instruct"
export MODEL_2="mistralai/Mistral-Nemo-Instruct-2407"
export INPUT_CSV="data/papers.csv"
export RULES_JSON="data/rules.json"
export OUTPUT_DIR="output"
export BATCH_SIZE="20"
```

3. If you already run your own endpoint, run with env-configured paths:

```bash
python -m src.main
```

4. Or override paths explicitly when needed:

```bash
python -m src.main \
	--input-csv data/papers.csv \
	--rules-json data/rules.json \
	--output-dir output \
	--batch-size 20
```

Useful flags:
- `--dry-run`: validate inputs and planned batches without model calls
- `--resume`: continue from existing per-model CSV files
- `--max-papers N`: run only first N papers

## Docker / RunPod

Recommended (no long args):

1. Copy .env.example to .env and adjust paths/model values as needed.
2. Build:

```bash
docker compose build
```

3. Run:

```bash
docker compose run --rm llm-rsl-filter
```

### Local-Only Model Workflow

The runtime is local-only by default. It expects model files under `./models`,
mounted into the container at `/models`. Use one networked prefetch step to
populate the local model store, then run evaluation offline.

Prefetch downloads model files directly into `./models` by default. Weight
files are restricted to `*.safetensors`; non-weight files required by vLLM,
such as `config.json` and tokenizer files, are also downloaded. Set
`PREFETCH_USE_HF_CACHE=true` only if you intentionally want Hugging Face cache
storage in addition to the local model directory.

Build the image without downloading model weights into image layers:

```bash
docker compose build --progress=plain
```

Prefetch configured models into `./models`:

```bash
docker compose run --rm \
	-e CONTAINER_TASK=prefetch \
	-e LOCAL_ONLY=false \
	llm-rsl-filter
```

Run evaluation using only local model files:

```bash
docker compose run --rm llm-rsl-filter
```

Progress/logging controls:
- `VLLM_STREAM_LOGS=true`: stream vLLM startup logs while waiting for readiness.
- `EVAL_PROGRESS=true`: show per-batch paper evaluation progress bars.
- `docker compose build --progress=plain`: show readable Docker build steps.

Expected local model paths:
- `./models/Qwen__Qwen2.5-14B-Instruct`
- `./models/mistralai__Mistral-Nemo-Instruct-2407`

To use a custom local model directory, set `MODEL_1_PATH` or `MODEL_2_PATH`
to the mounted container path, for example `/models/my-model`.

Optional bootstrap without creating .env first:

```bash
ENV_FILE=.env.example docker compose build
ENV_FILE=.env.example docker compose run --rm llm-rsl-filter
```

This avoids passing build args, volume mounts, and working directory flags manually.

Direct Docker (still supported):

Build image:

```bash
docker build -t llm-rsl-filter:latest .
```

Optional build args:
- HF_TOKEN: Hugging Face token for private/gated model repos.
- MODEL_1_FALLBACK, MODEL_2_FALLBACK: optional lower-memory fallback models used when VLLM_PRECISION_POLICY=auto.
- PRELOAD_MODELS=false: skip model prefetch at build time.

The image prepares /work, /work/data, /work/output, and /models-cache.

Run container:

```bash
mkdir -p data output

docker run --rm \
	--gpus all \
	--env-file .env \
	-v "$PWD/data":/work/data \
	-v "$PWD/output":/work/output \
	-v "$PWD/models-cache":/models-cache \
	-w /work \
	llm-rsl-filter:latest
```

Inside the container execution, the entrypoint performs:
1. Start local vLLM for MODEL_1.
2. Run model 1 evaluation round.
3. Stop local vLLM.
4. Start local vLLM for MODEL_2.
5. Run model 2 evaluation round.
6. Merge outputs into results_final.csv.

Outputs:
- output/results_model_1.csv
- output/results_model_2.csv
- output/results_final.csv

For RunPod pods, clone repo, build image, and run this same command in the pod.

### Notes About GPU Fit and Fallback

If MODEL_1 or MODEL_2 fail to start due to memory pressure and VLLM_PRECISION_POLICY=auto, the container tries MODEL_1_FALLBACK and MODEL_2_FALLBACK with VLLM_FALLBACK_QUANTIZATION.

Set fallback models in .env if you want automatic recovery behavior.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```
