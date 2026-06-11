from src.consensus import build_final_decisions, combine_decisions
from src.models import ModelDecision


def test_combine_decisions_precedence() -> None:
    assert combine_decisions("include", "include") == "include"
    assert combine_decisions("manual review", "include") == "manual review"
    assert combine_decisions("include", "exclude") == "exclude"
    assert combine_decisions("manual review", "exclude") == "exclude"


def test_build_final_decisions_union_and_precedence() -> None:
    model_1 = [
        ModelDecision(
            paper_id="P1",
            title="Paper 1",
            model_name="m1",
            decision="include",
            matched_rule_ids=["I1"],
            matched_inclusion_rule_ids=["I1"],
            matched_exclusion_rule_ids=[],
            rationale="",
        ),
        ModelDecision(
            paper_id="P2",
            title="Paper 2",
            model_name="m1",
            decision="manual review",
            matched_rule_ids=[],
            matched_inclusion_rule_ids=[],
            matched_exclusion_rule_ids=[],
            rationale="",
        ),
    ]

    model_2 = [
        ModelDecision(
            paper_id="P1",
            title="Paper 1",
            model_name="m2",
            decision="exclude",
            matched_rule_ids=["E3"],
            matched_inclusion_rule_ids=[],
            matched_exclusion_rule_ids=["E3"],
            rationale="",
        ),
        ModelDecision(
            paper_id="P3",
            title="Paper 3",
            model_name="m2",
            decision="include",
            matched_rule_ids=["I7"],
            matched_inclusion_rule_ids=["I7"],
            matched_exclusion_rule_ids=[],
            rationale="",
        ),
    ]

    final_rows = build_final_decisions(model_1, model_2)
    by_id = {row.paper_id: row for row in final_rows}

    assert by_id["P1"].final_decision == "exclude"
    assert by_id["P2"].final_decision == "manual review"
    assert by_id["P3"].final_decision == "manual review"
