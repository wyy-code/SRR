"""Crossed linear-router diagnostics on aligned token representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CrossedRoutes:
    """Top-k sets for (source/merged input) x (source/merged gate)."""

    source_source: torch.Tensor
    merged_source: torch.Tensor
    source_merged: torch.Tensor
    merged_merged: torch.Tensor


def _selected(hidden: torch.Tensor, weight: torch.Tensor, k: int) -> torch.Tensor:
    if hidden.shape[-1] != weight.shape[-1] or weight.ndim != 2:
        raise ValueError("hidden and rank-2 gate weight dimensions differ")
    if not 0 < k <= weight.shape[0]:
        raise ValueError("top-k must be within the expert count")
    logits = F.linear(hidden.float(), weight.float())
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite gate logits")
    return logits.topk(k, dim=-1, sorted=True).indices


def crossed_routes(
    source_hidden: torch.Tensor,
    merged_hidden: torch.Tensor,
    source_gate_weight: torch.Tensor,
    merged_gate_weight: torch.Tensor,
    top_k: int,
) -> CrossedRoutes:
    """Cross inputs and gate weights at *aligned* token-layer positions.

    This is a structural attribution diagnostic, not an intervention on the
    model's downstream task behavior. It assumes an un-biased linear gate;
    use architecture-native hooks for actual routed forward passes.
    """
    if source_hidden.shape != merged_hidden.shape:
        raise ValueError("source and merged hidden states must be aligned")
    if source_gate_weight.shape != merged_gate_weight.shape:
        raise ValueError("source and merged gate weights must match")
    return CrossedRoutes(
        source_source=_selected(source_hidden, source_gate_weight, top_k),
        merged_source=_selected(merged_hidden, source_gate_weight, top_k),
        source_merged=_selected(source_hidden, merged_gate_weight, top_k),
        merged_merged=_selected(merged_hidden, merged_gate_weight, top_k),
    )


def same_expert_set(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("route shapes differ")
    return (left.sort(dim=-1).values == right.sort(dim=-1).values).all(dim=-1)


def route_origin_masks(routes: CrossedRoutes) -> dict[str, torch.Tensor]:
    """Return nonexclusive component effects and exclusive changed-route labels.

    Exclusive `representation_only` means the representation swap changes the
    source route while the gate-weight swap alone does not; `gate_only` is the
    converse. `both` and `neither` complete the changed-route partition.
    """
    ss = routes.source_source
    changed = ~same_expert_set(ss, routes.merged_merged)
    representation_effect = ~same_expert_set(ss, routes.merged_source)
    gate_effect = ~same_expert_set(ss, routes.source_merged)
    return {
        "changed": changed,
        "representation_effect": representation_effect,
        "gate_effect": gate_effect,
        "representation_only": changed & representation_effect & ~gate_effect,
        "gate_only": changed & ~representation_effect & gate_effect,
        "both": changed & representation_effect & gate_effect,
        "neither": changed & ~representation_effect & ~gate_effect,
    }


def full_distribution_js(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence in nats, over all experts (not only top-k)."""
    if left_logits.shape != right_logits.shape:
        raise ValueError("logit shapes differ")
    p = left_logits.float().softmax(-1)
    q = right_logits.float().softmax(-1)
    m = (p + q) / 2
    return ((p * (p.clamp_min(1e-20).log() - m.clamp_min(1e-20).log())).sum(-1)
            + (q * (q.clamp_min(1e-20).log() - m.clamp_min(1e-20).log())).sum(-1)) / 2


def topk_set_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """1 - intersection/k for equal-size expert selections."""
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("route shapes differ")
    overlap = (left.unsqueeze(-1) == right.unsqueeze(-2)).any(-1).sum(-1)
    return 1 - overlap.float() / left.shape[-1]
