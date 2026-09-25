"""Fixed-hidden-state mixture comparisons; no downstream causal claim."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def routed_mixture(
    expert_outputs: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted routed output with shape `[..., experts, hidden]`.

    Supply the *native* routing weights, including any architecture-specific
    mass. The function deliberately does not renormalize them.
    """
    if expert_outputs.ndim < 2 or selected.shape != weights.shape:
        raise ValueError("invalid mixture shapes")
    if expert_outputs.shape[:-2] != selected.shape[:-1]:
        raise ValueError("batch dimensions differ")
    if selected.numel() and (selected.min() < 0 or selected.max() >= expert_outputs.shape[-2]):
        raise ValueError("expert index out of range")
    expanded = selected.long().unsqueeze(-1).expand(*selected.shape, expert_outputs.shape[-1])
    outputs = expert_outputs.gather(-2, expanded)
    return (outputs.float() * weights.float().unsqueeze(-1)).sum(-2)


def output_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
    if left.shape != right.shape:
        raise ValueError("output shapes differ")
    lhs, rhs = left.float(), right.float()
    return {
        "cosine": F.cosine_similarity(lhs, rhs, dim=-1),
        "relative_l2": (lhs - rhs).norm(dim=-1) / lhs.norm(dim=-1).clamp_min(1e-20),
    }


def max_entering_leaving_cosine(
    expert_outputs: torch.Tensor, before: torch.Tensor, after: torch.Tensor
) -> torch.Tensor:
    """Maximum expert-output cosine across entering/leaving experts per event.

    `expert_outputs` is evaluated at one fixed hidden state. An event without
    both an entering and a leaving expert has no defined comparison.
    """
    if expert_outputs.ndim != 2 or before.ndim != 1 or after.ndim != 1:
        raise ValueError("expected one event with [experts, hidden] outputs")
    leaving = sorted(set(before.tolist()) - set(after.tolist()))
    entering = sorted(set(after.tolist()) - set(before.tolist()))
    if not leaving or not entering:
        raise ValueError("event has no entering/leaving expert pair")
    lhs = F.normalize(expert_outputs[leaving].float(), dim=-1)
    rhs = F.normalize(expert_outputs[entering].float(), dim=-1)
    return (lhs @ rhs.T).max()
