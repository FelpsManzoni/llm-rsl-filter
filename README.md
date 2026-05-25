# llm-rsl-filter

Dual-LLM pre-evaluation pipeline for SLR papers using an OpenAI-compatible endpoint (for example, vLLM on RunPod).

The pipeline:
- Reads papers from CSV (paper id, title, abstract, keywords)
- Evaluates papers in batches of 20 with two models
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
export MODEL_1="Qwen/qwen3.5:35b-mlx"
export MODEL_2="Mistralai/mistral-small3.2:24b"
```

3. Run:

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

Run container:

```bash
docker run --rm \
	-e OPENAI_API_KEY="$OPENAI_API_KEY" \
	-e VLLM_BASE_URL="$VLLM_BASE_URL" \
	-v "$PWD":/work \
	-w /work \
	llm-rsl-filter:latest \
	python -m src.main \
		--input-csv data/papers.csv \
		--rules-json data/rules.json \
		--output-dir output
```

For RunPod Pods, use the same image and command with your mounted volume paths.

## Tests

```bash
pytest
```
