from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from role_pruning.models.hf_causal_lm import HFCausalLMBackend


def build_prompt(passage: str, question: str) -> str:
    return (
        f"{passage.strip()}\n\n"
        f"Question: {question}\n\n"
        "Answer with exactly one line: FINAL: <word>"
    )


_STOPWORDS = {"the", "a", "an", "is", "in", "was", "there", "to"}


def extract_answer(text: str) -> str:
    match = re.search(r"FINAL\s*:\s*(.+)", text, re.IGNORECASE)
    line = match.group(1) if match else text
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", line) if w.lower() not in _STOPWORDS]
    return words[-1] if words else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Unmasked quality-floor probe for bAbI tasks.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--samples-per-task", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--quantization", default="none")
    args = parser.parse_args()

    dataset = load_dataset("Muennighoff/babi", split="test")
    by_task: dict[int, list[dict]] = {}
    for row in dataset:
        by_task.setdefault(row["task"], []).append(row)

    rng = random.Random(args.seed)
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
    for task in args.tasks:
        rows = by_task.get(task, [])
        if not rows:
            print(f"task {task}: no rows found, skipping")
            continue
        sample = rng.sample(rows, min(args.samples_per_task, len(rows)))
        correct = 0
        examples = []
        for row in sample:
            prompt = build_prompt(row["passage"], row["question"])
            generation = backend.generate(prompt, role="solver", task={"answer": row["answer"]})
            predicted = extract_answer(generation.text)
            gold = row["answer"].strip().lower()
            is_correct = predicted == gold
            correct += int(is_correct)
            if len(examples) < 3:
                examples.append((row["question"], gold, predicted, generation.text[:120]))
        accuracy = correct / len(sample)
        print(
            f"task {task}: accuracy={accuracy:.3f} ({correct}/{len(sample)}), "
            f"passage_len_chars~{len(sample[0]['passage'])}"
        )
        for question, gold, predicted, raw in examples:
            status = "OK" if predicted == gold else "WRONG"
            print(f"  [{status}] Q={question!r} gold={gold!r} pred={predicted!r} raw={raw!r}")
    print(f"total seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
