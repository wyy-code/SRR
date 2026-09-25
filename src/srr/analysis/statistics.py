"""Paired item and prompt-cluster uncertainty for routing interventions."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Hashable, Sequence


def cluster_bootstrap_mean(
    differences: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    repetitions: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Percentile CI for the item-weighted paired mean, resampling clusters.

    Each draw includes all items from each sampled cluster. Repeated clusters
    retain their original item multiplicity. This differs from averaging
    cluster means when prompt lengths vary.
    """
    if len(differences) != len(clusters) or not differences or repetitions < 1:
        raise ValueError("paired nonempty values/clusters and repetitions >= 1 required")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for value, cluster in zip(differences, clusters):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite difference")
        grouped[cluster].append(float(value))
    keys = sorted(grouped, key=str)
    totals = [(sum(grouped[key]), len(grouped[key])) for key in keys]
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = [totals[rng.randrange(len(totals))] for _ in totals]
        draws.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
    draws.sort()
    return {
        "estimate": sum(differences) / len(differences),
        "ci_low": _quantile(draws, 0.025),
        "ci_high": _quantile(draws, 0.975),
        "items": len(differences),
        "clusters": len(keys),
        "repetitions": repetitions,
        "seed": seed,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def choice_margin(scores: Sequence[float], gold_index: int) -> float:
    if len(scores) < 2 or not 0 <= gold_index < len(scores):
        raise ValueError("at least two choices and a valid gold index required")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("non-finite choice score")
    return float(scores[gold_index]) - max(
        float(score) for index, score in enumerate(scores) if index != gold_index
    )


def choice_accuracy(scores: Sequence[float], gold_index: int) -> int:
    choice_margin(scores, gold_index)
    return int(max(range(len(scores)), key=lambda index: scores[index]) == gold_index)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise adjusted p-values in original order."""
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise ValueError("p-values must be in [0, 1]")
    result = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        result[index] = running
    return result


def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Tie-aware AUROC; raises when one class is absent."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be aligned and nonempty")
    positives = [float(s) for s, y in zip(scores, labels) if y]
    negatives = [float(s) for s, y in zip(scores, labels) if not y]
    if not positives or not negatives or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("AUROC requires finite scores and both classes")
    ordered = sorted((float(score), bool(label)) for score, label in zip(scores, labels))
    negative_seen = 0
    favorable_pairs = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        positives_here = sum(label for _, label in ordered[index:end])
        negatives_here = end - index - positives_here
        favorable_pairs += positives_here * (negative_seen + 0.5 * negatives_here)
        negative_seen += negatives_here
        index = end
    return favorable_pairs / (len(positives) * len(negatives))
