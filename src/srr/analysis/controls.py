"""Checkpoint-independent construction of matched routing controls."""

from __future__ import annotations

import random


def matched_random_expert_sets(
    native_selected: list[int],
    observed_selected: list[int],
    expert_count: int,
    *,
    count: int = 32,
    seed: int = 0,
) -> list[list[int]]:
    """Sample unique routes matched on size and overlap with the native route.

    The observed set itself is excluded. Apply the *same observed routing
    weight values* to each returned route when computing mixture controls;
    this function samples selections only and never normalizes weights.
    """
    native = set(native_selected)
    observed = set(observed_selected)
    k = len(observed_selected)
    if count < 1 or k < 1 or len(native) != k or len(observed) != k:
        raise ValueError("unique, equal-size nonempty routes and positive count required")
    if min(native | observed) < 0 or max(native | observed) >= expert_count:
        raise ValueError("expert index out of range")
    overlap = len(native & observed)
    outside = [expert for expert in range(expert_count) if expert not in native]
    if len(outside) < k - overlap:
        raise ValueError("not enough experts outside the native route")
    rng = random.Random(seed)
    controls: set[tuple[int, ...]] = set()
    attempts = 0
    while len(controls) < count and attempts < max(1000, count * 200):
        attempts += 1
        proposal = tuple(sorted(rng.sample(sorted(native), overlap)
                                + rng.sample(outside, k - overlap)))
        if set(proposal) != observed:
            controls.add(proposal)
    if len(controls) != count:
        raise ValueError("insufficient distinct matched routes for requested count")
    return [list(route) for route in sorted(controls)]
