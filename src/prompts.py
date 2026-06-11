from __future__ import annotations

from .models import Paper, Rule

SYSTEM_PROMPT = (
    "You are a strict SLR pre-evaluation assistant. "
    "You receive one paper and a list of inclusion/exclusion rules. "
    "Return only a JSON object with this exact schema: "
    '{"matched_inclusion_rule_ids": string[], "matched_exclusion_rule_ids": string[], '
    '"insufficient_information": boolean, "rationale": string}. '
    "Never include markdown and never add extra keys. "
    "Rule IDs must come only from the rules provided in the prompt."
)


def build_user_prompt(paper: Paper, rules: list[Rule]) -> str:
    inclusion_lines = [
        f"- {rule.rule_id}: {rule.text}" for rule in rules if rule.rule_type == "inclusion"
    ]
    exclusion_lines = [
        f"- {rule.rule_id}: {rule.text}" for rule in rules if rule.rule_type == "exclusion"
    ]

    inclusion_block = "\n".join(inclusion_lines) if inclusion_lines else "- (none)"
    exclusion_block = "\n".join(exclusion_lines) if exclusion_lines else "- (none)"

    return (
        "Evaluate the following paper using the rules list.\n\n"
        "Paper:\n"
        f"- paper_id: {paper.paper_id}\n"
        f"- title: {paper.title}\n"
        f"- abstract: {paper.abstract}\n"
        f"- keywords: {paper.keywords}\n\n"
        "Inclusion Rules:\n"
        f"{inclusion_block}\n\n"
        "Exclusion Rules:\n"
        f"{exclusion_block}\n\n"
        "Decision semantics for matched rules:\n"
        "1) If any exclusion rule is matched, the paper is excluded.\n"
        "2) Otherwise, if one or more inclusion rules are matched, the paper is included.\n"
        "3) If information is insufficient to evaluate the rules, set insufficient_information=true.\n"
        "4) Do not output a final decision string; only output matched rule IDs and insufficient_information."
    )
