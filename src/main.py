from __future__ import annotations

from pathlib import Path

from .config import parse_args
from .consensus import build_final_decisions
from .evaluator import evaluate_model
from .io_csv import (
    load_papers,
    load_rules,
    read_model_results,
    write_final_results,
    write_model_results,
)
from .llm_client import LLMClient, LLMClientConfig
from .models import ModelDecision, Paper


def _result_path(output_dir: Path, model_index: int) -> Path:
    return output_dir / f"results_model_{model_index}.csv"


def main() -> None:
    config = parse_args()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_1_path = _result_path(config.output_dir, 1)
    model_2_path = _result_path(config.output_dir, 2)

    if config.run_mode == "merge-only":
        model_1_rows = read_model_results(model_1_path)
        model_2_rows = read_model_results(model_2_path)
        _write_final_results(config.output_dir, model_1_rows, model_2_rows)
        print(f"Model 1 CSV: {model_1_path}")
        print(f"Model 2 CSV: {model_2_path}")
        print(f"Final CSV: {config.output_dir / 'results_final.csv'}")
        return

    papers = load_papers(config.input_csv, max_papers=config.max_papers)
    rules = load_rules(config.rules_json)

    batches = (len(papers) + max(1, config.batch_size) - 1) // max(1, config.batch_size)

    print("Input validation complete.")
    print(f"- Papers: {len(papers)}")
    print(f"- Rules: {len(rules)}")
    print(f"- Batch size: {max(1, config.batch_size)}")
    print(f"- Total batches/model: {batches}")
    print("- Execution mode: sequential (model 1 round, then model 2 round)")
    print(f"- Run mode: {config.run_mode}")

    if config.dry_run:
        print("Dry run completed. No model calls were made.")
        return

    client = LLMClient(
        LLMClientConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    )

    model_1_rows: list[ModelDecision] | None = None
    model_2_rows: list[ModelDecision] | None = None

    if config.run_mode in {"full", "model1"}:
        print("Starting round 1/2 with model 1.")
        model_1_rows = _run_one_model(
            model_name=config.model_1,
            papers=papers,
            rules=rules,
            client=client,
            output_path=model_1_path,
            resume=config.resume,
            batch_size=config.batch_size,
        )

    if config.run_mode in {"full", "model2"}:
        print("Starting round 2/2 with model 2.")
        model_2_rows = _run_one_model(
            model_name=config.model_2,
            papers=papers,
            rules=rules,
            client=client,
            output_path=model_2_path,
            resume=config.resume,
            batch_size=config.batch_size,
        )

    if config.run_mode == "full":
        model_1_rows = model_1_rows if model_1_rows is not None else read_model_results(model_1_path)
        model_2_rows = model_2_rows if model_2_rows is not None else read_model_results(model_2_path)
        _write_final_results(config.output_dir, model_1_rows, model_2_rows)

    print(f"Model 1 CSV: {model_1_path}")
    print(f"Model 2 CSV: {model_2_path}")
    if config.run_mode in {"full", "merge-only"}:
        print(f"Final CSV: {config.output_dir / 'results_final.csv'}")


def _write_final_results(
    output_dir: Path,
    model_1_rows: list[ModelDecision],
    model_2_rows: list[ModelDecision],
) -> None:
    final_rows = build_final_decisions(model_1_rows, model_2_rows)
    final_path = output_dir / "results_final.csv"
    write_final_results(final_path, final_rows)


def _run_one_model(
    model_name: str,
    papers: list[Paper],
    rules,
    client: LLMClient,
    output_path: Path,
    resume: bool,
    batch_size: int,
) -> list[ModelDecision]:
    existing_rows = read_model_results(output_path) if resume else []
    done_ids = {row.paper_id for row in existing_rows}

    if done_ids:
        print(f"[{model_name}] Resume mode: skipping {len(done_ids)} already processed paper(s).")

    pending_papers = [paper for paper in papers if paper.paper_id not in done_ids]

    if pending_papers:
        new_rows = evaluate_model(
            papers=pending_papers,
            rules=rules,
            client=client,
            model_name=model_name,
            batch_size=batch_size,
        )
    else:
        new_rows = []

    merged = _merge_rows(existing_rows, new_rows)
    write_model_results(output_path, merged)
    return merged


def _merge_rows(existing: list[ModelDecision], new_rows: list[ModelDecision]) -> list[ModelDecision]:
    by_id = {row.paper_id: row for row in existing}
    for row in new_rows:
        by_id[row.paper_id] = row
    return [by_id[paper_id] for paper_id in sorted(by_id)]


if __name__ == "__main__":
    main()
