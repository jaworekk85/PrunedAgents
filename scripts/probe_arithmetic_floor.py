from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from role_pruning.data.controlled_arithmetic import ControlledArithmeticAdapter
from role_pruning.evaluation.team_pipeline import build_solver_prompt
from role_pruning.models.hf_causal_lm import HFCausalLMBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="Unmasked quality-floor probe for controlled arithmetic.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--quantization", default="none")
    args = parser.parse_args()

    adapter = ControlledArithmeticAdapter(
        calibration_size=24, dev_size=8, test_size=48, seed=42, split_strategy="template_disjoint"
    )
    test_items = adapter.load_split("test")[: args.samples]

    backend = HFCausalLMBackend(
        model_id=args.model_id,
        revision=None,
        device="auto",
        dtype="auto",
        max_new_tokens=args.max_new_tokens,
        quantization=args.quantization,
        use_chat_template=args.use_chat_template,
    )

    print(f"model={args.model_id}")
    started = time.perf_counter()
    correct = 0
    examples = []
    for item in test_items:
        prompt = build_solver_prompt(
            item.question,
            instruction="Solve the arithmetic independently.",
            answer_kind="number",
        )
        generation = backend.generate(prompt, role="solver", task={"answer": item.answer})
        score = adapter.score_answer(generation.text, item)["exact_match"]
        correct += int(score)
        if len(examples) < 5:
            examples.append((item.question, item.answer, generation.text[:100], bool(score)))
    accuracy = correct / len(test_items)
    print(f"accuracy={accuracy:.3f} ({correct}/{len(test_items)})")
    for question, gold, raw, ok in examples:
        print(f"  [{'OK' if ok else 'WRONG'}] Q={question!r} gold={gold!r} raw={raw!r}")
    print(f"total seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
