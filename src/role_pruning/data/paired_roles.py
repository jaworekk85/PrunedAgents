from __future__ import annotations

from role_pruning.data.base import BaseTask
from role_pruning.roles.prompts import RolePrompt


def build_paired_role_items(
    base_items: list[BaseTask],
    role_prompts: dict[str, RolePrompt],
    candidate_by_task: dict[str, str] | None = None,
    candidate_by_task_role: dict[tuple[str, str], dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    candidate_by_task = candidate_by_task or {}
    candidate_by_task_role = candidate_by_task_role or {}
    paired = []
    for item in base_items:
        for role, prompt in role_prompts.items():
            role_candidate = candidate_by_task_role.get((item.task_id, role))
            if role_candidate is None:
                candidate = candidate_by_task.get(item.task_id, f"Candidate answer: {item.answer}")
                role_candidate = {
                    "candidate": candidate,
                    "candidate_answer": item.answer,
                    "candidate_is_correct": True,
                    "candidate_mode": "legacy",
                }
            candidate = str(role_candidate["candidate"])
            text = prompt.template.format(question=item.question, candidate=candidate)
            paired.append(
                {
                    "task_id": item.task_id,
                    "split": item.split,
                    "role": role,
                    "prompt": text,
                    "system": prompt.system,
                    "answer": item.answer,
                    "task_metadata": dict(item.metadata),
                    "candidate": candidate,
                    "candidate_answer": role_candidate.get("candidate_answer"),
                    "candidate_is_correct": role_candidate.get("candidate_is_correct"),
                    "candidate_mode": role_candidate.get("candidate_mode"),
                }
            )
    return paired
