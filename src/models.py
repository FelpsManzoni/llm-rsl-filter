from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["include", "exclude", "manual review"]
RuleType = Literal["inclusion", "exclusion"]


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    keywords: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    rule_type: RuleType
    text: str


@dataclass(frozen=True)
class LLMRawDecision:
    matched_inclusion_rule_ids: list[str]
    matched_exclusion_rule_ids: list[str]
    insufficient_information: bool
    rationale: str


@dataclass(frozen=True)
class ModelDecision:
    paper_id: str
    title: str
    model_name: str
    decision: Decision
    matched_rule_ids: list[str]
    matched_inclusion_rule_ids: list[str]
    matched_exclusion_rule_ids: list[str]
    rationale: str


@dataclass(frozen=True)
class FinalDecision:
    paper_id: str
    title: str
    model_1_decision: Decision
    model_2_decision: Decision
    final_decision: Decision
