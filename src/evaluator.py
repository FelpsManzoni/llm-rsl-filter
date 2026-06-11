from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

from .llm_client import LLMClient
from .models import Decision, LLMRawDecision, ModelDecision, Paper, Rule
from .prompts import SYSTEM_PROMPT, build_user_prompt

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - dependency is optional for library use.
    tqdm = None

T = TypeVar("T")


def evaluate_model(
    papers: list[Paper],
    rules: list[Rule],
    client: LLMClient,
    model_name: str,
    batch_size: int = 20,
) -> list[ModelDecision]:
    out: list[ModelDecision] = []
    progress_enabled = os.getenv("EVAL_PROGRESS", "true").lower() == "true"
    normalized_batch_size = max(1, batch_size)
    total_batches = (len(papers) + normalized_batch_size - 1) // normalized_batch_size

    for batch_start in range(0, len(papers), normalized_batch_size):
        batch = papers[batch_start : batch_start + normalized_batch_size]
        batch_number = batch_start // normalized_batch_size + 1
        _progress_log(
            f"[{model_name}] Batch {batch_number}/{total_batches}: "
            f"{len(batch)} paper(s)."
        )

        for paper in _progress_iter(
            batch,
            enabled=progress_enabled,
            desc=f"{model_name} batch {batch_number}/{total_batches}",
            unit="paper",
        ):
            user_prompt = build_user_prompt(paper, rules)
            try:
                response = client.evaluate(
                    model_name=model_name,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                raw = _parse_raw_decision(response)
                decision = derive_decision(raw)
                out.append(_to_model_decision(paper, model_name, raw, decision))
            except Exception as exc:
                # Parsing/API failures should not break the full run; mark for human validation.
                out.append(
                    ModelDecision(
                        paper_id=paper.paper_id,
                        title=paper.title,
                        model_name=model_name,
                        decision="manual review",
                        matched_rule_ids=[],
                        matched_inclusion_rule_ids=[],
                        matched_exclusion_rule_ids=[],
                        rationale=f"LLM error: {exc}",
                    )
                )

    return out


def _progress_iter(
    values: Iterable[T],
    enabled: bool,
    desc: str,
    unit: str,
) -> Iterator[T]:
    if enabled and tqdm is not None:
        yield from tqdm(values, desc=desc, unit=unit, leave=False)
        return
    yield from values


def _progress_log(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def derive_decision(raw: LLMRawDecision) -> Decision:
    if raw.matched_exclusion_rule_ids:
        return "exclude"
    if raw.insufficient_information:
        return "manual review"
    if raw.matched_inclusion_rule_ids:
        return "include"
    return "manual review"


def _parse_raw_decision(payload: dict) -> LLMRawDecision:
    inclusion = _as_string_list(payload.get("matched_inclusion_rule_ids", []))
    exclusion = _as_string_list(payload.get("matched_exclusion_rule_ids", []))
    insufficient_information = bool(payload.get("insufficient_information", False))
    rationale = str(payload.get("rationale", "")).strip()

    return LLMRawDecision(
        matched_inclusion_rule_ids=inclusion,
        matched_exclusion_rule_ids=exclusion,
        insufficient_information=insufficient_information,
        rationale=rationale,
    )


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _to_model_decision(
    paper: Paper,
    model_name: str,
    raw: LLMRawDecision,
    decision: Decision,
) -> ModelDecision:
    matched_rule_ids = []
    seen: set[str] = set()
    for rule_id in raw.matched_inclusion_rule_ids + raw.matched_exclusion_rule_ids:
        if rule_id not in seen:
            seen.add(rule_id)
            matched_rule_ids.append(rule_id)

    return ModelDecision(
        paper_id=paper.paper_id,
        title=paper.title,
        model_name=model_name,
        decision=decision,
        matched_rule_ids=matched_rule_ids,
        matched_inclusion_rule_ids=raw.matched_inclusion_rule_ids,
        matched_exclusion_rule_ids=raw.matched_exclusion_rule_ids,
        rationale=raw.rationale,
    )
