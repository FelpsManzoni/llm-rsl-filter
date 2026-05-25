from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import FinalDecision, ModelDecision, Paper, Rule


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def load_papers(path: Path, max_papers: int | None = None) -> list[Paper]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")

        header_map = {_normalize_key(field): field for field in reader.fieldnames}

        id_field = header_map.get("paper_id") or header_map.get("id")
        title_field = header_map.get("title")
        abstract_field = header_map.get("abstract")
        keywords_field = header_map.get("keywords")

        missing = [
            name
            for name, value in [
                ("paper_id/id", id_field),
                ("title", title_field),
                ("abstract", abstract_field),
                ("keywords", keywords_field),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )

        papers: list[Paper] = []
        for idx, row in enumerate(reader, start=2):
            paper_id = (row.get(id_field, "") or "").strip()
            title = (row.get(title_field, "") or "").strip()
            abstract = (row.get(abstract_field, "") or "").strip()
            keywords = (row.get(keywords_field, "") or "").strip()

            if not paper_id:
                raise ValueError(f"Row {idx} has an empty paper id.")
            if not title:
                raise ValueError(f"Row {idx} has an empty title for paper {paper_id}.")

            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    keywords=keywords,
                )
            )

            if max_papers is not None and len(papers) >= max_papers:
                break

    return papers


def load_rules(path: Path) -> list[Rule]:
    if not path.exists():
        raise FileNotFoundError(f"Rules JSON not found: {path}")

    with path.open("r", encoding="utf-8") as rules_file:
        data = json.load(rules_file)

    rule_list = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(rule_list, list):
        raise ValueError("Rules JSON must be a list or an object with a 'rules' list.")

    rules: list[Rule] = []
    seen_ids: set[str] = set()

    for idx, entry in enumerate(rule_list, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Rule #{idx} is not a JSON object.")

        rule_id = str(entry.get("rule_id", "")).strip()
        rule_type = str(entry.get("type", "")).strip().lower()
        text = str(entry.get("text", "")).strip()

        if not rule_id or not rule_type or not text:
            raise ValueError(
                f"Rule #{idx} must contain non-empty rule_id, type, and text."
            )
        if rule_id in seen_ids:
            raise ValueError(f"Duplicated rule_id detected: {rule_id}")
        if rule_type not in {"inclusion", "exclusion"}:
            raise ValueError(
                f"Rule {rule_id} has invalid type '{rule_type}'. Expected inclusion or exclusion."
            )

        seen_ids.add(rule_id)
        rules.append(Rule(rule_id=rule_id, rule_type=rule_type, text=text))

    return rules


def write_model_results(path: Path, decisions: list[ModelDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "paper_id",
                "title",
                "decision",
                "matched_rule_ids",
                "model_name",
                "matched_inclusion_rule_ids",
                "matched_exclusion_rule_ids",
                "rationale",
            ],
        )
        writer.writeheader()

        for row in decisions:
            writer.writerow(
                {
                    "paper_id": row.paper_id,
                    "title": row.title,
                    "decision": row.decision,
                    "matched_rule_ids": ";".join(row.matched_rule_ids),
                    "model_name": row.model_name,
                    "matched_inclusion_rule_ids": ";".join(
                        row.matched_inclusion_rule_ids
                    ),
                    "matched_exclusion_rule_ids": ";".join(
                        row.matched_exclusion_rule_ids
                    ),
                    "rationale": row.rationale,
                }
            )


def read_model_results(path: Path) -> list[ModelDecision]:
    if not path.exists():
        return []

    out: list[ModelDecision] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            out.append(
                ModelDecision(
                    paper_id=(row.get("paper_id") or "").strip(),
                    title=(row.get("title") or "").strip(),
                    model_name=(row.get("model_name") or "").strip(),
                    decision=(row.get("decision") or "manual review").strip(),
                    matched_rule_ids=_split_csv_list(row.get("matched_rule_ids", "")),
                    matched_inclusion_rule_ids=_split_csv_list(
                        row.get("matched_inclusion_rule_ids", "")
                    ),
                    matched_exclusion_rule_ids=_split_csv_list(
                        row.get("matched_exclusion_rule_ids", "")
                    ),
                    rationale=(row.get("rationale") or "").strip(),
                )
            )

    return out


def write_final_results(path: Path, rows: list[FinalDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "paper_id",
                "title",
                "model_1_decision",
                "model_2_decision",
                "final_decision",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "paper_id": row.paper_id,
                    "title": row.title,
                    "model_1_decision": row.model_1_decision,
                    "model_2_decision": row.model_2_decision,
                    "final_decision": row.final_decision,
                }
            )


def _split_csv_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [item for item in (token.strip() for token in raw_value.split(";")) if item]
