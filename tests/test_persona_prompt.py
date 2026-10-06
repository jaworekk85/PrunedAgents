from role_pruning.models.hf_causal_lm import _messages, _raw_with_system


def test_raw_prompt_unchanged_without_system():
    assert _raw_with_system("Task:\n2+2", None) == "Task:\n2+2"


def test_raw_prompt_prepends_system():
    assert _raw_with_system("Task:\n2+2", "You are the solver.") == "You are the solver.\n\nTask:\n2+2"


def test_chat_messages_add_system_turn():
    assert _messages("Task", None) == [{"role": "user", "content": "Task"}]
    assert _messages("Task", "You are the verifier.") == [
        {"role": "system", "content": "You are the verifier."},
        {"role": "user", "content": "Task"},
    ]
