from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    input_csv: Path
    rules_json: Path
    output_dir: Path
    base_url: str
    api_key: str | None
    model_1: str
    model_2: str
    batch_size: int
    timeout_seconds: float
    max_retries: int
    max_papers: int | None
    resume: bool
    dry_run: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Dual-LLM pre-evaluation pipeline for SLR papers."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path.")
    parser.add_argument("--rules-json", required=True, help="Rules JSON path.")
    parser.add_argument(
        "--output-dir", default="output", help="Directory for generated CSV files."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing API key.",
    )
    parser.add_argument(
        "--model-1",
        default=os.getenv("MODEL_1", "Qwen/qwen3.5:35b-mlx"),
        help="First LLM model ID.",
    )
    parser.add_argument(
        "--model-2",
        default=os.getenv("MODEL_2", "Mistralai/mistral-small3.2:24b"),
        help="Second LLM model ID.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Number of papers per batch."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries for transient API failures.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap for number of papers to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing model output CSVs in output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and planned execution without calling LLMs.",
    )

    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)

    return RunConfig(
        input_csv=Path(args.input_csv),
        rules_json=Path(args.rules_json),
        output_dir=Path(args.output_dir),
        base_url=args.base_url,
        api_key=api_key,
        model_1=args.model_1,
        model_2=args.model_2,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_papers=args.max_papers,
        resume=args.resume,
        dry_run=args.dry_run,
    )
