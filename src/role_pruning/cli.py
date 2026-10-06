from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from role_pruning.activations.collect import collect_toy_activation_stats
from role_pruning.activations.hf_swiglu import build_response_mask, collect_swiglu_response_sums
from role_pruning.analysis.overlap import pairwise_jaccard_matrix
from role_pruning.analysis.plots import write_matrix_csv
from role_pruning.analysis.statistics import diagonal_advantage, summarize_role_mask_comparisons
from role_pruning.config import load_config, read_json, save_json
from role_pruning.data.candidates import build_role_candidate_map
from role_pruning.data.gsm8k import make_adapter, normalize_number
from role_pruning.data.paired_roles import build_paired_role_items
from role_pruning.data.splits import assert_disjoint_splits, save_splits
from role_pruning.evaluation.cross_role import build_cross_role_matrix, mean_score_matrix
from role_pruning.evaluation.mas_eval import evaluate_mas_conditions
from role_pruning.evaluation.team_pipeline import (
    build_chain_prompt,
    build_checker_prompt,
    build_solver_prompt,
    checker_decision_is_correct,
    summarize_team_bootstrap,
    summarize_team_metrics,
)
from role_pruning.hardware import write_system_info
from role_pruning.models.base import ToyRoleModel
from role_pruning.models.hf_causal_lm import HFCausalLMBackend
from role_pruning.pruning.hf_functional import apply_hf_swiglu_mask
from role_pruning.pruning.masks import average_scores, build_random_mask, build_topk_mask
from role_pruning.roles.prompts import load_role_prompts
from role_pruning.scoring.magnitude import collect_swiglu_weight_magnitude_scores, magnitude_score
from role_pruning.scoring.role_specific import role_specific_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="role_pruning")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("system-info")
    hf_test = sub.add_parser("hf-generation-test")
    hf_test.add_argument("--config", required=True)
    hf_test.add_argument(
        "--prompt",
        default="Solve briefly: Mia has 3 red pens and buys 4 blue pens. How many pens does she have?",
    )
    hf_activations = sub.add_parser("hf-collect-activations-tiny")
    hf_activations.add_argument("--config", required=True)
    hf_magnitudes = sub.add_parser("hf-collect-weight-magnitudes")
    hf_magnitudes.add_argument("--config", required=True)
    hf_masked_eval = sub.add_parser("hf-eval-masked-cross-role-tiny")
    hf_masked_eval.add_argument("--config", required=True)
    hf_team_eval = sub.add_parser("hf-eval-team-pipeline")
    hf_team_eval.add_argument("--config", required=True)
    hf_team_chain = sub.add_parser("hf-eval-team-chain")
    hf_team_chain.add_argument("--config", required=True)
    for name in [
        "prepare-data",
        "cache-responses",
        "collect-activations",
        "build-masks",
        "eval-cross-role",
        "eval-mas",
        "analyze",
        "smoke",
        "run",
    ]:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    if args.command == "system-info":
        info = write_system_info()
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    config = load_config(args.config)
    if args.command == "hf-generation-test":
        hf_generation_test(config, args.prompt)
    elif args.command == "hf-collect-activations-tiny":
        hf_collect_activations_tiny(config)
    elif args.command == "hf-collect-weight-magnitudes":
        hf_collect_weight_magnitudes(config)
    elif args.command == "hf-eval-masked-cross-role-tiny":
        hf_eval_masked_cross_role_tiny(config)
    elif args.command == "hf-eval-team-pipeline":
        hf_eval_team_pipeline(config)
    elif args.command == "hf-eval-team-chain":
        hf_eval_team_chain(config)
    elif args.command == "prepare-data":
        prepare_data(config)
    elif args.command == "cache-responses":
        cache_responses(config)
    elif args.command == "collect-activations":
        collect_activations(config)
    elif args.command == "build-masks":
        build_masks(config)
    elif args.command == "eval-cross-role":
        eval_cross_role(config)
    elif args.command == "eval-mas":
        eval_mas(config)
    elif args.command == "analyze":
        analyze(config)
    elif args.command in {"smoke", "run"}:
        smoke(config)
    return 0


def _build_candidate_map(
    config: dict[str, Any],
    adapter: Any,
    base_items: list[Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    if config["benchmark"].get("name") == "babi":
        from role_pruning.data.babi import build_babi_role_candidate_map

        return build_babi_role_candidate_map(
            base_items,
            config["roles"]["names"],
            adapter.vocabulary,
            seed=int(config.get("seed", 42)),
        )
    return build_role_candidate_map(base_items, config["roles"]["names"], config.get("candidate_policy"))


def _build_hf_backend(
    config: dict[str, Any],
    *,
    max_new_tokens: int | None = None,
) -> HFCausalLMBackend:
    model_config = config["model"]
    return HFCausalLMBackend(
        model_id=model_config["id"],
        revision=model_config.get("revision"),
        device=model_config.get("device", "auto"),
        dtype=model_config.get("dtype", "auto"),
        max_new_tokens=(
            max_new_tokens
            if max_new_tokens is not None
            else int(model_config.get("max_new_tokens", 64))
        ),
        quantization=str(model_config.get("quantization", "none")),
        use_chat_template=bool(model_config.get("use_chat_template", False)),
    )


def prepare_data(config: dict[str, Any]) -> None:
    adapter = make_adapter(config)
    splits = {split: adapter.load_split(split) for split in ["calibration", "dev", "test"]}
    assert_disjoint_splits(splits)
    save_splits(splits, "artifacts/splits/splits.json")
    prompts = load_role_prompts(config["roles"]["prompt_file"], config["roles"]["names"])
    for split, items in splits.items():
        candidate_map = build_role_candidate_map(
            items,
            config["roles"]["names"],
            config.get("candidate_policy"),
        )
        paired = build_paired_role_items(items, prompts, candidate_by_task_role=candidate_map)
        save_json(f"artifacts/splits/{split}_paired.json", paired)


def cache_responses(config: dict[str, Any]) -> None:
    if config["model"].get("backend") != "toy":
        raise RuntimeError("Current implementation validates caching with backend='toy'.")
    rows = read_json("artifacts/splits/calibration_paired.json")
    model = ToyRoleModel()
    output_path = Path("artifacts/cached_responses/calibration.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            generation = model.generate(row["prompt"], row["role"], row)
            payload = {
                **row,
                "generated_text": generation.text,
                "generated_token_count": generation.generated_token_count,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def collect_activations(config: dict[str, Any]) -> None:
    cached_rows = _read_jsonl("artifacts/cached_responses/calibration.jsonl")
    stats = collect_toy_activation_stats(
        cached_rows,
        roles=config["roles"]["names"],
        layer_count=int(config["model"].get("toy_layers", 4)),
        intermediate_size=int(config["model"].get("toy_intermediate_size", 10)),
    )
    stats["metadata"] = {
        "roles": config["roles"]["names"],
        "benchmark": config["benchmark"]["name"],
        "split": "calibration",
        "model_id": config["model"]["id"],
        "dtype": config["model"]["dtype"],
        "seed": config["seed"],
    }
    save_json("artifacts/activation_stats/calibration_stats.json", stats)


def build_masks(config: dict[str, Any]) -> None:
    mask_config = config.get("mask_building", {})
    stats_path = mask_config.get(
        "input_stats_path",
        config.get("activation_collection", {}).get(
            "output_path",
            "artifacts/activation_stats/calibration_stats.json",
        ),
    )
    stats = read_json(stats_path)
    roles = config["roles"]["names"]
    role_scores = role_specific_scores(
        stats,
        roles,
        alpha=float(config["importance"]["alpha"]),
        beta=float(config["importance"]["beta"]),
        epsilon=float(config["importance"]["epsilon"]),
    )
    layer_count = int(stats["layer_count"])
    width = int(stats["intermediate_size"])
    if stats.get("metadata", {}).get("backend") == "hf":
        magnitude_scores = stats.get("weight_magnitude_scores")
        if magnitude_scores is None:
            raise ValueError(
                "HF activation stats do not contain real weight_magnitude_scores; "
                "rerun activation collection before building the magnitude baseline."
            )
        magnitude_source = "swiglu_squared_group_l2"
    else:
        magnitude_scores = magnitude_score(layer_count, width)
        magnitude_source = "toy_deterministic_placeholder"
    for sparsity in config["pruning"]["sparsities"]:
        metadata = {
            "source": mask_config.get("source", "toy_smoke"),
            "input_stats_path": stats_path,
            "sparsity": sparsity,
            "unit": "mlp_intermediate",
        }
        masks: dict[str, dict] = {}
        for role in roles:
            masks[role] = build_topk_mask(
                name=f"{role}_mask_s{sparsity}",
                scores=role_scores[role],
                sparsity=float(sparsity),
                metadata={**metadata, "role": role},
            ).to_dict()
        masks["generic"] = build_topk_mask(
            name=f"generic_mask_s{sparsity}",
            scores=average_scores(role_scores),
            sparsity=float(sparsity),
            metadata={**metadata, "role": "generic"},
        ).to_dict()
        masks["magnitude"] = build_topk_mask(
            name=f"magnitude_mask_s{sparsity}",
            scores=magnitude_scores,
            sparsity=float(sparsity),
            metadata={
                **metadata,
                "role": "magnitude",
                "magnitude_source": magnitude_source,
            },
        ).to_dict()
        for seed in config["pruning"].get("random_seeds", [1, 2, 3]):
            masks[f"random_seed_{seed}"] = build_random_mask(
                name=f"random_mask_seed_{seed}_s{sparsity}",
                layer_count=layer_count,
                intermediate_size=width,
                sparsity=float(sparsity),
                seed=int(seed),
                metadata={**metadata, "role": "random", "random_seed": seed},
            ).to_dict()
        output_template = mask_config.get("output_template", "artifacts/masks/sparsity_{sparsity}.json")
        save_json(output_template.format(sparsity=sparsity, run_name=config.get("run_name", "run")), masks)


def eval_cross_role(config: dict[str, Any]) -> None:
    sparsity = config["pruning"]["sparsities"][0]
    masks = read_json(f"artifacts/masks/sparsity_{sparsity}.json")
    roles = config["roles"]["names"]
    role_masks = {role: masks[role] for role in roles}
    matrix = build_cross_role_matrix(role_masks, roles)
    payload = {
        "matrix": matrix,
        "diagonal_advantage": diagonal_advantage(matrix, roles),
        "sparsity": sparsity,
    }
    save_json("results/latest_cross_role_metrics.json", payload)
    write_matrix_csv(matrix, "results/latest_cross_role_matrix.csv")


def eval_mas(config: dict[str, Any]) -> None:
    sparsity = config["pruning"]["sparsities"][0]
    masks = read_json(f"artifacts/masks/sparsity_{sparsity}.json")
    roles = config["roles"]["names"]
    role_masks = {role: masks[role] for role in roles}
    metrics = evaluate_mas_conditions(role_masks, masks["generic"], roles)
    save_json("results/latest_mas_metrics.json", {"sparsity": sparsity, "metrics": metrics})


def analyze(config: dict[str, Any]) -> None:
    sparsity = config["pruning"]["sparsities"][0]
    masks = read_json(f"artifacts/masks/sparsity_{sparsity}.json")
    roles = config["roles"]["names"]
    role_masks = {role: masks[role] for role in roles}
    jaccard = pairwise_jaccard_matrix(role_masks, roles)
    save_json("results/latest_overlap_metrics.json", {"jaccard": jaccard})
    write_matrix_csv(jaccard, "results/figure1_role_mask_jaccard.csv")
    cross_role = read_json("results/latest_cross_role_metrics.json")
    write_matrix_csv(cross_role["matrix"], "results/figure2_cross_role_performance.csv")


def smoke(config: dict[str, Any]) -> None:
    write_system_info()
    prepare_data(config)
    cache_responses(config)
    collect_activations(config)
    build_masks(config)
    eval_cross_role(config)
    eval_mas(config)
    analyze(config)
    _update_status_after_smoke(config)


def hf_generation_test(config: dict[str, Any], prompt: str) -> None:
    model_config = config["model"]
    if model_config.get("backend") != "hf":
        raise RuntimeError("hf-generation-test requires a config with model.backend='hf'.")

    started = time.perf_counter()
    backend = _build_hf_backend(config)
    load_seconds = time.perf_counter() - started

    gen_started = time.perf_counter()
    generation = backend.generate(prompt, role="solver", task={"answer": ""})
    generation_seconds = time.perf_counter() - gen_started

    payload = {
        "model_id": model_config["id"],
        "revision": model_config.get("revision"),
        "device": backend.device,
        "dtype": backend.dtype,
        "parameter_count": backend.parameter_count,
        "max_new_tokens": int(model_config.get("max_new_tokens", 64)),
        "prompt": prompt,
        "generated_text": generation.text,
        "generated_token_count": generation.generated_token_count,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
    }
    save_json("artifacts/hf_generation_tests/latest.json", payload)
    _update_status_after_hf_test(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def hf_collect_activations_tiny(config: dict[str, Any]) -> None:
    model_config = config["model"]
    if model_config.get("backend") != "hf":
        raise RuntimeError("hf-collect-activations-tiny requires model.backend='hf'.")

    adapter = make_adapter(config)
    split = config.get("activation_collection", {}).get("split", "calibration")
    prompts = load_role_prompts(config["roles"]["prompt_file"], config["roles"]["names"])
    base_items = adapter.load_split(split)
    resample_indices = config.get("activation_collection", {}).get("resample_indices")
    if resample_indices is not None:
        if not isinstance(resample_indices, list) or not all(
            isinstance(index, int) and 0 <= index < len(base_items) for index in resample_indices
        ):
            raise ValueError("activation_collection.resample_indices must index the chosen split")
        base_items = [base_items[index] for index in resample_indices]
    candidate_map = _build_candidate_map(config, adapter, base_items)
    paired_rows = build_paired_role_items(
        base_items,
        prompts,
        candidate_by_task_role=candidate_map,
    )
    max_pairs = int(config.get("activation_collection", {}).get("max_pairs", len(paired_rows)))
    paired_rows = paired_rows[:max_pairs]

    started = time.perf_counter()
    backend = _build_hf_backend(config)
    load_seconds = time.perf_counter() - started

    sums_by_role: dict[str, list[list[float]]] = {}
    token_count_by_role: dict[str, int] = {}
    response_rows: list[dict[str, Any]] = []

    include_system = bool(config["roles"].get("include_system_prompt", False))
    collect_started = time.perf_counter()
    for row in paired_rows:
        system = str(row["system"]) if include_system else None
        generation = backend.generate(row["prompt"], row["role"], row, system=system)
        prompt_text = row["prompt"]
        response_text = generation.text
        prompt_inputs = backend.tokenize_prompt(prompt_text, system=system)
        full_inputs = backend.tokenize_conversation(prompt_text, response_text, system=system)
        prompt_token_count = int(prompt_inputs["input_ids"].shape[1])
        total_token_count = int(full_inputs["input_ids"].shape[1])
        input_ids = full_inputs["input_ids"].to(backend.device)
        attention_mask = full_inputs["attention_mask"].to(backend.device)
        response_mask = build_response_mask(prompt_token_count, total_token_count, input_ids.device)

        layer_sums = collect_swiglu_response_sums(
            backend.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
        )
        role = row["role"]
        if role not in sums_by_role:
            sums_by_role[role] = [
                [0.0 for _ in layer.sum_abs]
                for layer in layer_sums
            ]
            token_count_by_role[role] = 0
        for layer_position, layer in enumerate(layer_sums):
            sums_by_role[role][layer_position] = [
                old_value + new_value
                for old_value, new_value in zip(sums_by_role[role][layer_position], layer.sum_abs)
            ]
        token_count_by_role[role] += layer_sums[0].token_count

        response_rows.append(
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "role": role,
                "system": system,
                "prompt": prompt_text,
                "generated_text": response_text,
                "generated_token_count": generation.generated_token_count,
                "teacher_forced_response_tokens": int(response_mask.sum().item()),
                "candidate_answer": row.get("candidate_answer"),
                "candidate_is_correct": row.get("candidate_is_correct"),
                "candidate_mode": row.get("candidate_mode"),
            }
        )

    collect_seconds = time.perf_counter() - collect_started
    mean_abs = {
        role: [
            [value / max(token_count_by_role[role], 1) for value in layer]
            for layer in layers
        ]
        for role, layers in sums_by_role.items()
    }
    first_role = next(iter(sums_by_role))
    layer_count = len(sums_by_role[first_role])
    intermediate_size = len(sums_by_role[first_role][0])
    magnitude_started = time.perf_counter()
    weight_magnitude_scores = collect_swiglu_weight_magnitude_scores(backend.model)
    magnitude_seconds = time.perf_counter() - magnitude_started
    payload = {
        "layer_count": layer_count,
        "intermediate_size": intermediate_size,
        "sum_abs": sums_by_role,
        "token_count": token_count_by_role,
        "mean_abs": mean_abs,
        "weight_magnitude_scores": weight_magnitude_scores,
        "metadata": {
            "backend": "hf",
            "model_id": model_config["id"],
            "revision": model_config.get("revision"),
            "device": backend.device,
            "dtype": backend.dtype,
            "parameter_count": backend.parameter_count,
            "benchmark": config["benchmark"]["name"],
            "split": split,
            "roles": config["roles"]["names"],
            "paired_examples": len(paired_rows),
            "load_seconds": load_seconds,
            "collection_seconds": collect_seconds,
            "magnitude_seconds": magnitude_seconds,
            "magnitude_score": "swiglu_squared_group_l2",
            "response_token_protocol": "generate_then_teacher_force_prompt_plus_response",
            "include_system_prompt": include_system,
        },
    }

    output_path = config.get("activation_collection", {}).get(
        "output_path",
        "artifacts/activation_stats/hf_tiny_stats.json",
    )
    cache_path = config.get("activation_collection", {}).get(
        "response_cache_path",
        "artifacts/cached_responses/hf_tiny.jsonl",
    )
    save_json(output_path, payload)
    _write_jsonl(cache_path, response_rows)
    summary = {
        "output_path": output_path,
        "response_cache_path": cache_path,
        "model_id": model_config["id"],
        "device": backend.device,
        "dtype": backend.dtype,
        "layer_count": layer_count,
        "intermediate_size": intermediate_size,
        "paired_examples": len(paired_rows),
        "token_count": token_count_by_role,
        "load_seconds": load_seconds,
        "collection_seconds": collect_seconds,
        "magnitude_seconds": magnitude_seconds,
    }
    save_json(str(output_path).replace(".json", ".summary.json"), summary)
    _update_status_after_hf_activation(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def hf_collect_weight_magnitudes(config: dict[str, Any]) -> None:
    """Attach real SwiGLU group-weight magnitudes to existing activation statistics."""
    model_config = config["model"]
    if model_config.get("backend") != "hf":
        raise RuntimeError("hf-collect-weight-magnitudes requires model.backend='hf'.")

    stats_path = config.get("mask_building", {}).get(
        "input_stats_path",
        config.get("activation_collection", {}).get(
            "output_path",
            "artifacts/activation_stats/hf_tiny_stats.json",
        ),
    )
    stats = read_json(stats_path)
    started = time.perf_counter()
    backend = _build_hf_backend(config)
    load_seconds = time.perf_counter() - started

    magnitude_started = time.perf_counter()
    scores = collect_swiglu_weight_magnitude_scores(backend.model)
    magnitude_seconds = time.perf_counter() - magnitude_started
    if len(scores) != int(stats["layer_count"]):
        raise ValueError("Magnitude layer count does not match activation statistics")
    if any(len(layer) != int(stats["intermediate_size"]) for layer in scores):
        raise ValueError("Magnitude intermediate size does not match activation statistics")

    stats["weight_magnitude_scores"] = scores
    stats.setdefault("metadata", {})["magnitude_score"] = "swiglu_squared_group_l2"
    save_json(stats_path, stats)
    summary = {
        "stats_path": stats_path,
        "model_id": model_config["id"],
        "device": backend.device,
        "layer_count": len(scores),
        "intermediate_size": len(scores[0]),
        "load_seconds": load_seconds,
        "magnitude_seconds": magnitude_seconds,
        "magnitude_score": "swiglu_squared_group_l2",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def hf_eval_masked_cross_role_tiny(config: dict[str, Any]) -> None:
    model_config = config["model"]
    if model_config.get("backend") != "hf":
        raise RuntimeError("hf-eval-masked-cross-role-tiny requires model.backend='hf'.")

    adapter = make_adapter(config)
    split = config.get("functional_eval", {}).get("split", "test")
    max_items = int(config.get("functional_eval", {}).get("max_items", 1))
    roles = config["roles"]["names"]
    eval_roles = config.get("functional_eval", {}).get("eval_roles", roles)
    mask_names = config.get("functional_eval", {}).get("mask_names", roles)
    masks_required = any(mask_name != "unmasked" for mask_name in mask_names)
    masks_path = config.get("functional_eval", {}).get(
        "masks_path",
        config.get("mask_building", {}).get("output_template", "").format(
            sparsity=config["pruning"]["sparsities"][0],
            run_name=config.get("run_name", "run"),
        ),
    )
    if masks_required and not masks_path:
        raise ValueError("Could not resolve functional_eval.masks_path")
    masks = read_json(masks_path) if masks_required else {}
    prompts = load_role_prompts(config["roles"]["prompt_file"], roles)
    base_items = adapter.load_split(split)[:max_items]
    candidate_map = _build_candidate_map(config, adapter, base_items)
    paired_rows = [
        row
        for row in build_paired_role_items(base_items, prompts, candidate_by_task_role=candidate_map)
        if row["role"] in eval_roles
    ]

    backend = _build_hf_backend(
        config,
        max_new_tokens=int(
            config.get("functional_eval", {}).get(
                "max_new_tokens", model_config.get("max_new_tokens", 24)
            )
        ),
    )

    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for mask_index, mask_name in enumerate(mask_names, start=1):
        print(f"Evaluating mask {mask_index}/{len(mask_names)}: {mask_name}", flush=True)
        if mask_name == "unmasked":
            mask_context = nullcontext()
        elif mask_name in masks:
            mask_context = apply_hf_swiglu_mask(backend.model, masks[mask_name])
        else:
            raise KeyError(f"Mask {mask_name!r} not found in {masks_path}")
        with mask_context:
            for row in paired_rows:
                generation = backend.generate(row["prompt"], row["role"], row)
                item = next(item for item in base_items if item.task_id == row["task_id"])
                score_payload = _score_role_prediction(row, generation.text, item)
                predictions.append(
                    {
                        "mask_name": mask_name,
                        "eval_role": row["role"],
                        "task_id": row["task_id"],
                        "task_metadata": row.get("task_metadata", {}),
                        "score": score_payload["score"],
                        "numeric_exact": score_payload["numeric_exact"],
                        "role_score_note": score_payload["note"],
                        "candidate_answer": row.get("candidate_answer"),
                        "candidate_is_correct": row.get("candidate_is_correct"),
                        "candidate_mode": row.get("candidate_mode"),
                        "prediction": generation.text,
                        "generated_token_count": generation.generated_token_count,
                    }
                )

    matrix = mean_score_matrix(predictions)
    role_diagonal_advantage = _role_diagonal_advantage(matrix, roles)
    primary_roles = config.get("functional_eval", {}).get("primary_roles", roles)
    primary_role_diagonal_advantage = _role_diagonal_advantage(matrix, primary_roles)
    bootstrap_samples = int(
        config.get("functional_eval", {}).get("bootstrap_samples", 10_000)
    )
    paired_bootstrap = summarize_role_mask_comparisons(
        predictions,
        seed=int(config.get("seed", 42)),
        samples=bootstrap_samples,
    )
    payload = {
        "masks_path": masks_path,
        "split": split,
        "max_items": max_items,
        "mask_names": mask_names,
        "roles": roles,
        "eval_roles": eval_roles,
        "primary_roles": primary_roles,
        "matrix": matrix,
        "role_diagonal_advantage": role_diagonal_advantage,
        "mean_role_diagonal_advantage": sum(role_diagonal_advantage.values())
        / max(len(role_diagonal_advantage), 1),
        "primary_role_diagonal_advantage": primary_role_diagonal_advantage,
        "mean_primary_role_diagonal_advantage": sum(primary_role_diagonal_advantage.values())
        / max(len(primary_role_diagonal_advantage), 1),
        "paired_bootstrap": paired_bootstrap,
        "prediction_count": len(predictions),
        "seconds": time.perf_counter() - started,
        "reporting_note": "Functional masking reports quality under masks only; it is not a measured speedup.",
    }
    output_path = config.get("functional_eval", {}).get(
        "output_path",
        "results/hf_tiny_masked_cross_role_metrics.json",
    )
    predictions_path = config.get("functional_eval", {}).get(
        "predictions_path",
        "results/hf_tiny_masked_cross_role_predictions.jsonl",
    )
    save_json(output_path, payload)
    _write_jsonl(predictions_path, predictions)
    _update_status_after_hf_masked_eval(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def hf_eval_team_pipeline(config: dict[str, Any]) -> None:
    model_config = config["model"]
    if model_config.get("backend") != "hf":
        raise RuntimeError("hf-eval-team-pipeline requires model.backend='hf'.")

    team_config = config.get("team_eval", {})
    conditions = team_config.get("conditions", {})
    if not conditions:
        raise ValueError("team_eval.conditions must define at least one condition")

    adapter = make_adapter(config)
    split = str(team_config.get("split", "test"))
    max_items = int(team_config.get("max_items", 48))
    base_items = adapter.load_split(split)[:max_items]
    masks_path = str(team_config.get("masks_path", ""))
    required_mask_names = {
        str(mask_name)
        for condition in conditions.values()
        for mask_name in [condition["solver_mask"], condition["checker_mask"]]
        if mask_name != "unmasked"
    }
    if required_mask_names and not masks_path:
        raise ValueError("team_eval.masks_path is required for masked conditions")
    masks = read_json(masks_path) if required_mask_names else {}
    missing_masks = sorted(required_mask_names - set(masks))
    if missing_masks:
        raise KeyError(f"Team masks missing from {masks_path}: {missing_masks}")

    backend = _build_hf_backend(config, max_new_tokens=int(team_config.get("max_new_tokens", 32)))
    checker_role = str(team_config.get("checker_role", "verifier"))
    solver_instruction = str(
        team_config.get("solver_instruction", "Solve the arithmetic independently.")
    )
    checker_instruction = str(
        team_config.get("checker_instruction", "Recompute the answer independently.")
    )
    answer_kind = str(team_config.get("answer_kind", "number"))
    solver_reply_line = team_config.get("solver_reply_line")
    checker_reply_line = team_config.get("checker_reply_line")
    solver_system: str | None = None
    checker_system: str | None = None
    include_system = bool(team_config.get("include_system_prompt", False))
    if include_system:
        personas = load_role_prompts(config["roles"]["prompt_file"], ["solver", checker_role])
        solver_system = personas["solver"].system
        checker_system = personas[checker_role].system
    predictions_path = str(
        team_config.get(
            "predictions_path",
            "results/hf_team_pipeline_predictions.jsonl",
        )
    )
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for condition_index, (condition_name, condition) in enumerate(conditions.items(), start=1):
        solver_mask_name = str(condition["solver_mask"])
        checker_mask_name = str(condition["checker_mask"])
        print(
            f"Evaluating team condition {condition_index}/{len(conditions)}: "
            f"{condition_name} ({solver_mask_name} -> {checker_mask_name})",
            flush=True,
        )

        solver_rows: list[dict[str, Any]] = []
        with _hf_mask_context(backend.model, masks, solver_mask_name, masks_path):
            for item_index, item in enumerate(base_items, start=1):
                if item_index == 1 or item_index % 8 == 0 or item_index == len(base_items):
                    print(
                        f"  solver {item_index}/{len(base_items)}",
                        flush=True,
                    )
                solver_generation = backend.generate(
                    build_solver_prompt(
                        item.question,
                        instruction=solver_instruction,
                        answer_kind=answer_kind,
                        reply_line=solver_reply_line,
                    ),
                    role="solver",
                    task={"answer": item.answer},
                    system=solver_system,
                )
                solver_score = float(
                    adapter.score_answer(solver_generation.text, item)["exact_match"]
                )
                solver_rows.append(
                    {
                        "item": item,
                        "solver_prediction": solver_generation.text,
                        "solver_generated_token_count": solver_generation.generated_token_count,
                        "solver_score": solver_score,
                    }
                )

        condition_rows: list[dict[str, Any]] = []
        with _hf_mask_context(backend.model, masks, checker_mask_name, masks_path):
            for item_index, solver_row in enumerate(solver_rows, start=1):
                if item_index == 1 or item_index % 8 == 0 or item_index == len(solver_rows):
                    print(
                        f"  checker {item_index}/{len(solver_rows)}",
                        flush=True,
                    )
                item = solver_row["item"]
                checker_generation = backend.generate(
                    build_checker_prompt(
                        item.question,
                        solver_row["solver_prediction"],
                        instruction=checker_instruction,
                        answer_kind=answer_kind,
                        reply_line=checker_reply_line,
                    ),
                    role=checker_role,
                    task={"answer": item.answer},
                    system=checker_system,
                )
                final_score = float(
                    adapter.score_answer(checker_generation.text, item)["exact_match"]
                )
                condition_rows.append(
                    {
                        "condition": condition_name,
                        "solver_mask": solver_mask_name,
                        "checker_mask": checker_mask_name,
                        "checker_role": checker_role,
                        "task_id": item.task_id,
                        "task_metadata": dict(item.metadata),
                        "answer": item.answer,
                        "solver_prediction": solver_row["solver_prediction"],
                        "solver_generated_token_count": solver_row[
                            "solver_generated_token_count"
                        ],
                        "solver_score": solver_row["solver_score"],
                        "checker_prediction": checker_generation.text,
                        "checker_generated_token_count": checker_generation.generated_token_count,
                        "checker_decision_correct": float(
                            checker_decision_is_correct(
                                checker_generation.text,
                                bool(solver_row["solver_score"]),
                            )
                        ),
                        "final_score": final_score,
                    }
                )

        predictions.extend(condition_rows)
        _write_jsonl(predictions_path, predictions)

    bootstrap_samples = int(team_config.get("bootstrap_samples", 10_000))
    reference_condition = str(team_config.get("reference_condition", "unmasked"))
    payload = {
        "split": split,
        "max_items": max_items,
        "checker_role": checker_role,
        "include_system_prompt": include_system,
        "solver_system": solver_system,
        "checker_system": checker_system,
        "masks_path": masks_path,
        "conditions": conditions,
        "metrics": summarize_team_metrics(predictions),
        "comparisons_to_reference": summarize_team_bootstrap(
            predictions,
            reference_condition=reference_condition,
            seed=int(config.get("seed", 42)),
            samples=bootstrap_samples,
        ),
        "reference_condition": reference_condition,
        "prediction_count": len(predictions),
        "seconds": time.perf_counter() - started,
        "reporting_note": "Functional masks measure team quality, not realized speedup.",
    }
    if "role_specific" in conditions:
        payload["comparisons_to_role_specific"] = summarize_team_bootstrap(
            predictions,
            reference_condition="role_specific",
            seed=int(config.get("seed", 42)),
            samples=bootstrap_samples,
        )
    output_path = str(
        team_config.get(
            "output_path",
            "results/hf_team_pipeline_metrics.json",
        )
    )
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def hf_eval_team_chain(config: dict[str, Any]) -> None:
    """N-stage agent chain (e.g. planner -> solver -> critic -> verifier), generalizing the
    two-stage team pipeline. Each stage receives the previous stage's raw generated text as
    context (never a synthetic candidate), mirroring the two-stage pipeline's convention. Only
    stages marked `scored: true` in `stage_config` are compared against gold; the first scored
    stage's score is reported as `solver_score` and the last as `final_score` so the existing
    `summarize_team_metrics`/`summarize_team_bootstrap` helpers work unchanged.
    """
    model_config = config["model"]
    if model_config.get("backend") != "hf":
        raise RuntimeError("hf-eval-team-chain requires model.backend='hf'.")

    chain_config = config.get("team_chain", {})
    stages = list(chain_config.get("stages", []))
    if not stages:
        raise ValueError("team_chain.stages must list at least one stage")
    stage_config = chain_config.get("stage_config", {})
    conditions = chain_config.get("conditions", {})
    if not conditions:
        raise ValueError("team_chain.conditions must define at least one condition")

    adapter = make_adapter(config)
    split = str(chain_config.get("split", "test"))
    max_items = int(chain_config.get("max_items", 48))
    base_items = adapter.load_split(split)[:max_items]
    masks_path = str(chain_config.get("masks_path", ""))
    required_mask_names = {
        str(mask_name)
        for condition in conditions.values()
        for mask_name in condition.values()
        if mask_name != "unmasked"
    }
    if required_mask_names and not masks_path:
        raise ValueError("team_chain.masks_path is required for masked conditions")
    masks = read_json(masks_path) if required_mask_names else {}
    missing_masks = sorted(required_mask_names - set(masks))
    if missing_masks:
        raise KeyError(f"Chain masks missing from {masks_path}: {missing_masks}")

    backend = _build_hf_backend(config, max_new_tokens=int(chain_config.get("max_new_tokens", 48)))
    answer_kind = str(chain_config.get("answer_kind", "number"))
    predictions_path = str(
        chain_config.get("predictions_path", "results/hf_team_chain_predictions.jsonl")
    )
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for condition_index, (condition_name, stage_masks) in enumerate(conditions.items(), start=1):
        mask_names = [str(stage_masks[stage]) for stage in stages]
        print(
            f"Evaluating chain condition {condition_index}/{len(conditions)}: "
            f"{condition_name} ({' -> '.join(mask_names)})",
            flush=True,
        )

        item_state: list[dict[str, Any]] = [
            {"item": item, "context": None} for item in base_items
        ]
        for stage_index, stage in enumerate(stages):
            cfg = stage_config.get(stage, {})
            scored = bool(cfg.get("scored", True))
            with _hf_mask_context(backend.model, masks, mask_names[stage_index], masks_path):
                for row_index, row in enumerate(item_state, start=1):
                    if row_index == 1 or row_index % 8 == 0 or row_index == len(item_state):
                        print(f"  {stage} {row_index}/{len(item_state)}", flush=True)
                    item = row["item"]
                    prompt = build_chain_prompt(
                        item.question,
                        context=row["context"],
                        context_label=str(cfg.get("context_label", "Candidate solution")),
                        context_intro=str(
                            cfg.get(
                                "context_intro",
                                "The candidate may be correct or incorrect."
                                if row["context"] is not None
                                else "",
                            )
                        ),
                        instruction=str(cfg.get("instruction", "")),
                        answer_kind=answer_kind,
                        reply_line=cfg.get("reply_line"),
                    )
                    generation = backend.generate(
                        prompt, role=stage, task={"answer": item.answer}
                    )
                    row[f"{stage}_prediction"] = generation.text
                    row[f"{stage}_generated_token_count"] = generation.generated_token_count
                    if scored:
                        row[f"{stage}_score"] = float(
                            adapter.score_answer(generation.text, item)["exact_match"]
                        )
                    row["context"] = generation.text

        scored_stages = [s for s in stages if bool(stage_config.get(s, {}).get("scored", True))]
        if not scored_stages:
            raise ValueError("team_chain needs at least one stage with scored: true")
        first_scored, last_scored = scored_stages[0], scored_stages[-1]

        condition_rows: list[dict[str, Any]] = []
        for row in item_state:
            item = row["item"]
            entry: dict[str, Any] = {
                "condition": condition_name,
                "task_id": item.task_id,
                "task_metadata": dict(item.metadata),
                "answer": item.answer,
                "solver_score": row[f"{first_scored}_score"],
                "final_score": row[f"{last_scored}_score"],
                "checker_decision_correct": float(
                    checker_decision_is_correct(
                        row[f"{last_scored}_prediction"],
                        bool(row[f"{first_scored}_score"]),
                    )
                ),
            }
            for stage_index, stage in enumerate(stages):
                entry[f"{stage}_mask"] = mask_names[stage_index]
                entry[f"{stage}_prediction"] = row[f"{stage}_prediction"]
                if f"{stage}_score" in row:
                    entry[f"{stage}_score"] = row[f"{stage}_score"]
            condition_rows.append(entry)

        predictions.extend(condition_rows)
        _write_jsonl(predictions_path, predictions)

    bootstrap_samples = int(chain_config.get("bootstrap_samples", 10_000))
    reference_condition = str(chain_config.get("reference_condition", "unmasked"))
    payload = {
        "split": split,
        "max_items": max_items,
        "stages": stages,
        "masks_path": masks_path,
        "conditions": conditions,
        "metrics": summarize_team_metrics(predictions),
        "comparisons_to_reference": summarize_team_bootstrap(
            predictions,
            reference_condition=reference_condition,
            seed=int(config.get("seed", 42)),
            samples=bootstrap_samples,
        ),
        "reference_condition": reference_condition,
        "prediction_count": len(predictions),
        "seconds": time.perf_counter() - started,
        "reporting_note": "Functional masks measure team quality, not realized speedup.",
    }
    if "role_specific" in conditions:
        payload["comparisons_to_role_specific"] = summarize_team_bootstrap(
            predictions,
            reference_condition="role_specific",
            seed=int(config.get("seed", 42)),
            samples=bootstrap_samples,
        )
    output_path = str(chain_config.get("output_path", "results/hf_team_chain_metrics.json"))
    save_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _hf_mask_context(
    model: Any,
    masks: dict[str, Any],
    mask_name: str,
    masks_path: str,
):
    if mask_name == "unmasked":
        return nullcontext()
    if mask_name not in masks:
        raise KeyError(f"Mask {mask_name!r} not found in {masks_path}")
    return apply_hf_swiglu_mask(model, masks[mask_name])


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _score_role_prediction(row: dict[str, Any], prediction: str, item: Any) -> dict[str, Any]:
    answer_mentioned = _answer_mentioned(prediction, str(item.answer))
    role = str(row["role"])
    text = prediction.lower()

    if role == "critic":
        candidate_is_correct = row.get("candidate_is_correct")
        if candidate_is_correct is False:
            detects_error = any(
                marker in text
                for marker in ["incorrect", "wrong", "mistake", "error", "not correct"]
            )
            return {
                "score": float(answer_mentioned and detects_error),
                "numeric_exact": float(answer_mentioned),
                "note": "critic_wrong_candidate_requires_error_detection_and_corrected_answer",
            }
        if candidate_is_correct is True:
            accepts_correct = "correct" in text and "incorrect" not in text
            return {
                "score": float(answer_mentioned and accepts_correct),
                "numeric_exact": float(answer_mentioned),
                "note": "critic_correct_candidate_requires_acceptance_and_answer",
            }

    if role == "planner":
        return {
            "score": float(answer_mentioned),
            "numeric_exact": float(answer_mentioned),
            "note": "planner_exact_answer_is_diagnostic_only_not_primary",
        }

    if role == "verifier":
        candidate_is_correct = row.get("candidate_is_correct")
        if candidate_is_correct is False:
            detects_error = any(
                marker in text
                for marker in ["incorrect", "wrong", "mistake", "error", "not correct"]
            )
            return {
                "score": float(answer_mentioned and detects_error),
                "numeric_exact": float(answer_mentioned),
                "note": "verifier_wrong_candidate_requires_error_detection_and_answer",
            }
        if candidate_is_correct is True:
            accepts_correct = "correct" in text and "incorrect" not in text
            return {
                "score": float(answer_mentioned and accepts_correct),
                "numeric_exact": float(answer_mentioned),
                "note": "verifier_correct_candidate_requires_acceptance_and_answer",
            }
        return {
            "score": float(answer_mentioned),
            "numeric_exact": float(answer_mentioned),
            "note": "verifier_numeric_exact",
        }

    return {
        "score": float(answer_mentioned),
        "numeric_exact": float(answer_mentioned),
        "note": "numeric_exact",
    }


def _answer_mentioned(prediction: str, answer: str) -> bool:
    expected = normalize_number(answer)
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", prediction)
    return any(normalize_number(number) == expected for number in numbers)


def _update_status_after_smoke(config: dict[str, Any]) -> None:
    cross_role = read_json("results/latest_cross_role_metrics.json")
    overlap = read_json("results/latest_overlap_metrics.json")
    block = [
        "## Latest Smoke Run",
        "",
        f"- Model backend: {config['model']['backend']}",
        f"- Model ID: {config['model']['id']}",
        f"- Benchmark: {config['benchmark']['name']}",
        f"- Sparsity: {config['pruning']['sparsities'][0]}",
        f"- Diagonal advantage: {cross_role['diagonal_advantage']}",
        f"- Mask Jaccard matrix: {overlap['jaccard']}",
        "",
        "Smoke interpretation: local toy backend validates repository plumbing only; it is not evidence for or against the scientific hypothesis.",
    ]
    _replace_status_section("## Latest Smoke Run", block)


def _update_status_after_hf_test(payload: dict[str, Any]) -> None:
    block = [
        "## Latest HF Generation Test",
        "",
        f"- Model ID: {payload['model_id']}",
        f"- Device: {payload['device']}",
        f"- Dtype: {payload['dtype']}",
        f"- Parameter count: {payload['parameter_count']}",
        f"- Generated tokens: {payload['generated_token_count']}",
        f"- Load seconds: {payload['load_seconds']:.2f}",
        f"- Generation seconds: {payload['generation_seconds']:.2f}",
        "",
        "Interpretation: this validates model loading and deterministic generation only; activation hooks are the next milestone.",
    ]
    _replace_status_section("## Latest HF Generation Test", block)


def _update_status_after_hf_activation(payload: dict[str, Any]) -> None:
    block = [
        "## Latest HF Activation Tiny Run",
        "",
        f"- Model ID: {payload['model_id']}",
        f"- Device: {payload['device']}",
        f"- Dtype: {payload['dtype']}",
        f"- Layers: {payload['layer_count']}",
        f"- Intermediate size: {payload['intermediate_size']}",
        f"- Paired examples: {payload['paired_examples']}",
        f"- Token counts: {payload['token_count']}",
        f"- Load seconds: {payload['load_seconds']:.2f}",
        f"- Collection seconds: {payload['collection_seconds']:.2f}",
        f"- Output: `{payload['output_path']}`",
        "",
        (
            "Interpretation: this validates response-token SwiGLU activation collection for "
            f"{payload['model_id']}."
        ),
    ]
    _replace_status_section("## Latest HF Activation Tiny Run", block)


def _update_status_after_hf_masked_eval(payload: dict[str, Any]) -> None:
    block = [
        "## Latest HF Masked Cross-Role Tiny Eval",
        "",
        f"- Masks: `{payload['masks_path']}`",
        f"- Split: {payload['split']}",
        f"- Max items: {payload['max_items']}",
        f"- Predictions: {payload['prediction_count']}",
        f"- Seconds: {payload['seconds']:.2f}",
        f"- Role diagonal advantage: {payload.get('role_diagonal_advantage', {})}",
        f"- Matrix: {payload['matrix']}",
        "",
        "Interpretation: this is the first real functional masking evaluation path, still tiny and not paper-grade.",
    ]
    _replace_status_section("## Latest HF Masked Cross-Role Tiny Eval", block)


def _role_diagonal_advantage(matrix: dict[str, dict[str, float]], roles: list[str]) -> dict[str, float]:
    advantages: dict[str, float] = {}
    for role in roles:
        if role not in matrix or role not in matrix[role]:
            continue
        wrong_scores = [
            matrix[mask_role][role]
            for mask_role in roles
            if mask_role != role and mask_role in matrix and role in matrix[mask_role]
        ]
        if wrong_scores:
            advantages[role] = matrix[role][role] - (sum(wrong_scores) / len(wrong_scores))
    return advantages


def _replace_status_section(section_title: str, block: list[str]) -> None:
    status = Path("STATUS.md")
    lines = status.read_text(encoding="utf-8").splitlines() if status.exists() else ["# STATUS", ""]
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index] == section_title:
            index += 1
            while index < len(lines) and not lines[index].startswith("## "):
                index += 1
            continue
        kept.append(lines[index])
        index += 1

    while kept and kept[-1] == "":
        kept.pop()
    new_text = "\n".join(kept + [""] + block) + "\n"
    status.write_text(new_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
