from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RunMode = Literal["full", "model1", "model2", "merge-only"]


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
    run_mode: RunMode


def parse_args() -> RunConfig:
    env_input_csv = os.getenv("INPUT_CSV")
    env_rules_json = os.getenv("RULES_JSON")
    env_output_dir = os.getenv("OUTPUT_DIR", "output")
    env_batch_size = os.getenv("BATCH_SIZE", "20")

    try:
        batch_size_default = int(env_batch_size)
    except ValueError as exc:
        raise ValueError("BATCH_SIZE must be an integer.") from exc

    parser = argparse.ArgumentParser(
        description="Dual-LLM pre-evaluation pipeline for SLR papers."
    )
    parser.add_argument(
        "--input-csv",
        default=env_input_csv,
        help="Input CSV path. Defaults to INPUT_CSV env var.",
    )
    parser.add_argument(
        "--rules-json",
        default=env_rules_json,
        help="Rules JSON path. Defaults to RULES_JSON env var.",
    )
    parser.add_argument(
        "--output-dir",
        default=env_output_dir,
        help="Directory for generated CSV files. Defaults to OUTPUT_DIR env var or 'output'.",
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
        default=os.getenv("MODEL_1", "Qwen/Qwen2.5-14B-Instruct"),
        help="First LLM model ID.",
    )
    parser.add_argument(
        "--model-2",
        default=os.getenv("MODEL_2", "mistralai/Mistral-Nemo-Instruct-2407"),
        help="Second LLM model ID.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size_default,
        help="Number of papers per batch. Defaults to BATCH_SIZE env var or 20.",
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
    parser.add_argument(
        "--run-mode",
        choices=["full", "model1", "model2", "merge-only"],
        default=os.getenv("RUN_MODE", "full"),
        help="Pipeline run mode: full, model1, model2, or merge-only.",
    )

    args = parser.parse_args()
    run_mode: RunMode = args.run_mode

    needs_input_files = run_mode in {"full", "model1", "model2"}
    if needs_input_files and not args.input_csv:
        parser.error("Missing input CSV path. Use --input-csv or set INPUT_CSV.")
    if needs_input_files and not args.rules_json:
        parser.error("Missing rules JSON path. Use --rules-json or set RULES_JSON.")

    api_key = os.getenv(args.api_key_env)

    return RunConfig(
        input_csv=Path(args.input_csv or ""),
        rules_json=Path(args.rules_json or ""),
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
        run_mode=run_mode,
    )
