# llm-rsl-filter

Dual-LLM pre-evaluation pipeline for SLR papers using an OpenAI-compatible endpoint (for example, vLLM on RunPod).

The pipeline:
- Reads papers from CSV (paper id, title, abstract, keywords)
- Evaluates papers in batches of 20 with two models in sequential rounds (never simultaneously)
- Writes one CSV per model with per-paper decision and matched rule IDs
- Writes a final consensus CSV applying your precedence rules

## Decision Rules

Per model decision:
- If any exclusion rule is matched: `exclude`
- Else if inclusion rules are matched and no exclusion: `include`
- Else if insufficient information: `manual review`
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

2. Set environment variables (or use `.env` with your preferred loader):

```bash
export OPENAI_API_KEY="your_key"
export VLLM_BASE_URL="http://localhost:8000/v1"
export MODEL_1="Qwen/Qwen2.5-14B-Instruct"
export MODEL_2="mistralai/Mistral-Nemo-Instruct-2407"
export INPUT_CSV="data/papers.csv"
export RULES_JSON="data/rules.json"
export OUTPUT_DIR="output"
export BATCH_SIZE="20"
```

3. Run with env-configured paths:

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

Build image:

```bash
docker build -t llm-rsl-filter:latest .
```

The image prepares `/work`, `/work/data`, and `/work/output` directories and declares them as volumes.

Run container:

```bash
docker run --rm \
	--env-file .env \
	-v "$PWD/data":/work/data \
	-v "$PWD/output":/work/output \
	-w /work \
	llm-rsl-filter:latest
```

For RunPod Pods, use the same image and command with your mounted volume paths.

### Model Download During Docker Build

This project container is a client that calls an OpenAI-compatible endpoint (`VLLM_BASE_URL`). It does not host a model server itself, so `docker build` cannot reliably download and register models for that external endpoint.

If you want model weights baked into an image, that must be done in a model-serving image (for example a dedicated vLLM server image), then this client container should point `VLLM_BASE_URL` to that server.

## Tests

```bash
pytest
```
