"""Non-overwriting summaries of measured paired routing and task records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .statistics import binary_auc, choice_accuracy, choice_margin, cluster_bootstrap_mean


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input JSONL is empty")
    return rows


def summarize_task(rows: list[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    """Paired intervention-minus-baseline accuracy and correct-choice margin."""
    seen = set()
    accuracy, margin, clusters = [], [], []
    for row in rows:
        item_id = str(row["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate paired item: {item_id}")
        seen.add(item_id)
        baseline = row["baseline_choice_scores"]
        intervention = row["intervention_choice_scores"]
        gold = int(row["gold_index"])
        if len(baseline) != len(intervention):
            raise ValueError(f"choice count differs for {item_id}")
        accuracy.append(choice_accuracy(intervention, gold) - choice_accuracy(baseline, gold))
        margin.append(choice_margin(intervention, gold) - choice_margin(baseline, gold))
        clusters.append(str(row.get("cluster_id", item_id)))
    return {
        "accuracy_gain_fraction": cluster_bootstrap_mean(accuracy, clusters, repetitions=repetitions, seed=seed),
        "correct_choice_margin_gain": cluster_bootstrap_mean(margin, clusters, repetitions=repetitions, seed=seed + 1),
        "baseline_correct": sum(choice_accuracy(row["baseline_choice_scores"], int(row["gold_index"])) for row in rows),
        "intervention_correct": sum(choice_accuracy(row["intervention_choice_scores"], int(row["gold_index"])) for row in rows),
        "changed_predictions": sum(
            max(range(len(row["baseline_choice_scores"])), key=row["baseline_choice_scores"].__getitem__)
            != max(range(len(row["intervention_choice_scores"])), key=row["intervention_choice_scores"].__getitem__)
            for row in rows
        ),
    }


def summarize_routes(rows: list[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    """Summarize precomputed 2x2 crossed-route sets for aligned events."""
    labels = ("source_source", "merged_source", "source_merged", "merged_merged")
    changed, representation_only, gate_only, both, neither, clusters = [], [], [], [], [], []
    seen = set()
    for row in rows:
        event_id = str(row["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate event: {event_id}")
        seen.add(event_id)
        routes = [set(map(int, row[label])) for label in labels]
        k = len(row[labels[0]])
        if k < 1 or any(len(route) != k for route in routes):
            raise ValueError(f"invalid top-k route at {event_id}")
        actual = routes[0] != routes[3]
        rep = routes[0] != routes[1]
        gate = routes[0] != routes[2]
        changed.append(int(actual))
        representation_only.append(int(actual and rep and not gate))
        gate_only.append(int(actual and gate and not rep))
        both.append(int(actual and gate and rep))
        neither.append(int(actual and not gate and not rep))
        clusters.append(str(row["cluster_id"]))
    results = {
        "changed_fraction": cluster_bootstrap_mean(changed, clusters, repetitions=repetitions, seed=seed),
        "changed_events": sum(changed),
    }
    # Conditional fractions use a ratio of prompt-cluster totals. Keep the
    # numerator and denominator at event resolution in each bootstrap draw.
    for offset, (name, values) in enumerate((
        ("representation_only", representation_only), ("gate_only", gate_only),
        ("both", both), ("neither", neither),
    ), start=1):
        results[name + "_among_changed"] = _cluster_ratio(values, changed, clusters, repetitions, seed + offset)
    return results


def summarize_diagnosis(rows: list[dict[str, Any]], repetitions: int, seed: int) -> dict[str, Any]:
    """AUROC of prespecified route metrics for positive token-local replay gain.

    Only changed-route events enter this estimand; a gain of zero belongs to
    the non-positive class. Cluster bootstrap samples whole prompts.
    """
    import random
    from collections import defaultdict
    from .statistics import _quantile

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    event_ids = set()
    changed = []
    metric_names = None
    for row in rows:
        event_id = str(row["event_id"])
        if event_id in event_ids:
            raise ValueError(f"duplicate event: {event_id}")
        event_ids.add(event_id)
        if not row["changed_route"]:
            continue
        names = tuple(sorted(row["metrics"]))
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise ValueError("inconsistent diagnostic metric family")
        changed.append(row)
    if not changed or not metric_names:
        raise ValueError("no changed-route events with metrics")
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in changed:
        clusters[str(row["cluster_id"])].append(row)
    keys = sorted(clusters)
    labels = [float(row["source_replay_nll_gain"]) > 0 for row in changed]
    rng = random.Random(seed)
    draws = [[clusters[keys[rng.randrange(len(keys))]] for _ in keys] for _ in range(repetitions)]
    result = {"changed_events": len(changed), "clusters": len(keys), "metrics": {}}
    for name in metric_names:
        point = binary_auc([float(row["metrics"][name]) for row in changed], labels)
        estimates = []
        for sample in draws:
            sample_rows = [row for block in sample for row in block]
            sample_labels = [float(row["source_replay_nll_gain"]) > 0 for row in sample_rows]
            if any(sample_labels) and not all(sample_labels):
                estimates.append(binary_auc(
                    [float(row["metrics"][name]) for row in sample_rows], sample_labels,
                ))
        if not estimates:
            raise ValueError("bootstrap draws lack both outcome classes")
        estimates.sort()
        result["metrics"][name] = {
            "auc": point,
            "ci_low": _quantile(estimates, 0.025),
            "ci_high": _quantile(estimates, 0.975),
            "valid_draws": len(estimates),
        }
    return result


def _cluster_ratio(numerators: list[int], denominators: list[int], clusters: list[str], repetitions: int, seed: int) -> dict[str, Any]:
    # Reuse the paired cluster bootstrap by storing each cluster's numerator
    # and denominator separately; this preserves event weighting.
    from collections import defaultdict
    import random
    from .statistics import _quantile

    grouped: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for n, d, cluster in zip(numerators, denominators, clusters):
        grouped[cluster][0] += n
        grouped[cluster][1] += d
    total = sum(denominators)
    if total == 0:
        raise ValueError("no changed routes; conditional fraction undefined")
    keys = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        samples = [grouped[keys[rng.randrange(len(keys))]] for _ in keys]
        denominator = sum(value[1] for value in samples)
        if denominator:
            draws.append(sum(value[0] for value in samples) / denominator)
    if not draws:
        raise ValueError("bootstrap had no changed-route sample")
    draws.sort()
    return {"estimate": sum(numerators) / total, "ci_low": _quantile(draws, 0.025),
            "ci_high": _quantile(draws, 0.975), "valid_draws": len(draws), "clusters": len(keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("task", "routes", "diagnosis"))
    parser.add_argument("--input", type=Path, required=True, help="measured per-item/event JSONL")
    parser.add_argument("--output", type=Path, required=True, help="new JSON summary; never overwritten")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")
    rows = _read_rows(args.input)
    summarize = {"task": summarize_task, "routes": summarize_routes,
                 "diagnosis": summarize_diagnosis}[args.analysis]
    result = summarize(rows, args.repetitions, args.seed)
    result["analysis"] = args.analysis
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["input_path"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
