from __future__ import annotations


ROLE_ORDER = ["planner", "solver", "critic", "verifier"]


def run_toy_orchestrator(task: dict[str, str], masks_by_role: dict[str, dict]) -> dict[str, str]:
    """Run a deterministic four-role local workflow for smoke validation."""
    messages: dict[str, str] = {}
    plan_quality = _mask_density(masks_by_role.get("planner", {}))
    messages["planner"] = f"Plan with retained-unit fraction {plan_quality:.2f}."
    messages["solver"] = f"Use arithmetic carefully. Candidate final answer: {task['answer']}."
    messages["critic"] = "No arithmetic issue found in the candidate."
    messages["verifier"] = f"Verified final answer: {task['answer']}."
    messages["final"] = task["answer"]
    return messages


def _mask_density(mask: dict) -> float:
    bools = []
    for layer in mask.get("layers", {}).values():
        bools.extend(layer.get("bool_mask", []))
    if not bools:
        return 1.0
    return sum(1 for item in bools if item) / len(bools)

