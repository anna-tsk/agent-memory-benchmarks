"""Evaluation placeholders for belief revision trajectories.

The eventual evaluator should compare model answers against expected answers
while respecting the query's `belief_type`, `query_time`, and `belief_time`.

TODO:
- Define exact-match and alias-aware scoring.
- Add historical-belief checks that evaluate facts at `belief_time`.
- Add current-belief checks that evaluate facts at `query_time`.
- Decide how to score corrections, expirations, and scoped facts.
"""

from typing import Any


def evaluate_answer(query: dict[str, Any], model_answer: Any) -> dict[str, Any]:
    """Evaluate one model answer against one benchmark query.

    TODO: Implement scoring once answer normalization rules are defined.
    """
    raise NotImplementedError("Answer evaluation is not implemented yet.")
