from __future__ import annotations

from .models import Decision, FinalDecision, ModelDecision


def combine_decisions(decision_a: Decision, decision_b: Decision) -> Decision:
    if "exclude" in {decision_a, decision_b}:
        return "exclude"
    if "manual review" in {decision_a, decision_b}:
        return "manual review"
    return "include"


def build_final_decisions(
    model_1_rows: list[ModelDecision],
    model_2_rows: list[ModelDecision],
) -> list[FinalDecision]:
    by_id_1 = {row.paper_id: row for row in model_1_rows}
    by_id_2 = {row.paper_id: row for row in model_2_rows}

    all_ids = sorted(set(by_id_1) | set(by_id_2))
    final_rows: list[FinalDecision] = []

    for paper_id in all_ids:
        row_1 = by_id_1.get(paper_id)
        row_2 = by_id_2.get(paper_id)

        decision_1: Decision = row_1.decision if row_1 else "manual review"
        decision_2: Decision = row_2.decision if row_2 else "manual review"
        final_decision = combine_decisions(decision_1, decision_2)

        title = row_1.title if row_1 else (row_2.title if row_2 else "")

        final_rows.append(
            FinalDecision(
                paper_id=paper_id,
                title=title,
                model_1_decision=decision_1,
                model_2_decision=decision_2,
                final_decision=final_decision,
            )
        )

    return final_rows
