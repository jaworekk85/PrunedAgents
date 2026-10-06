from __future__ import annotations

from role_pruning.data.base import BaseTask
from role_pruning.data.gsm8k import normalize_number


def build_role_candidate_map(
    items: list[BaseTask],
    roles: list[str],
    policy: dict | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    policy = policy or {}
    mapping: dict[tuple[str, str], dict[str, object]] = {}
    for item in items:
        for role in roles:
            mapping[(item.task_id, role)] = make_role_candidate(item, role, policy)
    return mapping


def make_role_candidate(item: BaseTask, role: str, policy: dict) -> dict[str, object]:
    mode_by_role = policy.get("mode_by_role", {})
    mode = mode_by_role.get(role, policy.get("default_mode", "gold"))

    if role == "planner":
        return {
            "candidate": "",
            "candidate_answer": "",
            "candidate_is_correct": None,
            "candidate_mode": "unused",
        }

    if role == "solver":
        if mode == "no_answer_plan":
            return {
                "candidate": "Plan: identify the quantities, choose the arithmetic operation, compute carefully, and report only the final answer.",
                "candidate_answer": "",
                "candidate_is_correct": None,
                "candidate_mode": mode,
            }
        return {
            "candidate": f"Plan: solve step by step. Reference answer for calibration: {item.answer}.",
            "candidate_answer": item.answer,
            "candidate_is_correct": True,
            "candidate_mode": mode,
        }

    if mode in {"wrong_answer_only", "wrong_final_only"}:
        wrong_answer = perturb_numeric_answer(item.answer, offset=int(policy.get("wrong_offset", 1)))
        return {
            "candidate": f"CANDIDATE_FINAL: {wrong_answer}",
            "candidate_answer": wrong_answer,
            "candidate_is_correct": False,
            "candidate_mode": mode,
        }

    if mode in {"wrong_plus_one", "wrong"}:
        wrong_answer = perturb_numeric_answer(item.answer, offset=int(policy.get("wrong_offset", 1)))
        return {
            "candidate": (
                "Candidate reasoning: solve the arithmetic quickly. "
                f"Candidate final answer: {wrong_answer}."
            ),
            "candidate_answer": wrong_answer,
            "candidate_is_correct": False,
            "candidate_mode": mode,
        }

    return {
        "candidate": f"Candidate reasoning: solve the arithmetic carefully. Candidate final answer: {item.answer}.",
        "candidate_answer": item.answer,
        "candidate_is_correct": True,
        "candidate_mode": mode,
    }


def perturb_numeric_answer(answer: str, offset: int) -> str:
    value = normalize_number(answer)
    try:
        if "." in value:
            return normalize_number(str(float(value) + offset))
        return str(int(value) + offset)
    except ValueError:
        return f"{value} plus {offset}"
