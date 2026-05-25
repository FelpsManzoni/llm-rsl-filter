from src.evaluator import derive_decision
from src.models import LLMRawDecision


def test_derive_decision_exclusion_overrides() -> None:
    raw = LLMRawDecision(
        matched_inclusion_rule_ids=["I1"],
        matched_exclusion_rule_ids=["E1"],
        insufficient_information=False,
        rationale="",
    )
    assert derive_decision(raw) == "exclude"


def test_derive_decision_manual_when_insufficient_information() -> None:
    raw = LLMRawDecision(
        matched_inclusion_rule_ids=["I1"],
        matched_exclusion_rule_ids=[],
        insufficient_information=True,
        rationale="",
    )
    assert derive_decision(raw) == "manual review"


def test_derive_decision_include_when_only_inclusion() -> None:
    raw = LLMRawDecision(
        matched_inclusion_rule_ids=["I1", "I2"],
        matched_exclusion_rule_ids=[],
        insufficient_information=False,
        rationale="",
    )
    assert derive_decision(raw) == "include"


def test_derive_decision_manual_when_no_matches() -> None:
    raw = LLMRawDecision(
        matched_inclusion_rule_ids=[],
        matched_exclusion_rule_ids=[],
        insufficient_information=False,
        rationale="",
    )
    assert derive_decision(raw) == "manual review"
