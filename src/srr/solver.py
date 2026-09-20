#!/usr/bin/env python3
"""FP64 weighted ridge solver used by Selective Router Repair.

The solver fits an expert-relative router-logit field in each merge parent's
own router activation geometry. It never consumes benchmark labels or hidden
state distances; hidden activations are only the exact design matrix of the
linear router.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PullbackResult:
    delta_weight: torch.Tensor
    audit: dict[str, Any]


def centered_log_probability_lift(
    source_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    max_row_norm: float | None = None,
) -> torch.Tensor:
    """Return the gauge-free source-minus-base log-routing field."""
    if source_logits.shape != base_logits.shape or source_logits.ndim != 2:
        raise ValueError("source/base logits must be matching [sample, expert] tensors")
    lift = torch.log_softmax(source_logits.float(), dim=-1) - torch.log_softmax(
        base_logits.float(), dim=-1
    )
    lift = lift - lift.mean(dim=-1, keepdim=True)
    if max_row_norm is not None:
        if not math.isfinite(max_row_norm) or max_row_norm <= 0:
            raise ValueError("max_row_norm must be finite and positive")
        norms = torch.linalg.vector_norm(lift, dim=-1, keepdim=True)
        lift = lift * torch.clamp(max_row_norm / norms.clamp_min(1e-30), max=1.0)
    if not torch.isfinite(lift).all():
        raise RuntimeError("non-finite expert-relative routing field")
    return lift


def diagonal_categorical_fisher(
    parent_logits: torch.Tensor,
    *,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Return diagonal categorical Fisher weights p(1-p)."""
    if parent_logits.ndim != 2:
        raise ValueError("parent_logits must be [sample, expert]")
    if not math.isfinite(floor) or not 0 < floor < 1:
        raise ValueError("floor must lie in (0, 1)")
    probabilities = torch.softmax(parent_logits.float(), dim=-1)
    weights = (probabilities * (1.0 - probabilities)).clamp_min(floor)
    if not torch.isfinite(weights).all():
        raise RuntimeError("non-finite categorical Fisher weights")
    return weights


def _validate_pullback_inputs(
    target_hidden: torch.Tensor,
    target_field: torch.Tensor,
    target_weights: torch.Tensor,
    preservation_hidden: torch.Tensor | None,
    preservation_weights: torch.Tensor | None,
    owned_rows: torch.Tensor,
) -> tuple[int, int]:
    if target_hidden.ndim != 2 or target_field.ndim != 2:
        raise ValueError("target tensors must be matrices")
    samples, hidden = target_hidden.shape
    if target_field.shape[0] != samples:
        raise ValueError("target sample counts differ")
    experts = target_field.shape[1]
    if target_weights.shape != target_field.shape:
        raise ValueError("target_weights must match target_field")
    if owned_rows.shape != (experts,) or owned_rows.dtype != torch.bool:
        raise ValueError("owned_rows must be a boolean expert-row mask")
    if preservation_hidden is None:
        if preservation_weights is not None:
            raise ValueError("preservation weights require preservation hidden states")
    else:
        if preservation_hidden.ndim != 2 or preservation_hidden.shape[1] != hidden:
            raise ValueError("preservation hidden shape mismatch")
        expected = (preservation_hidden.shape[0], experts)
        if preservation_weights is None or preservation_weights.shape != expected:
            raise ValueError("preservation_weights shape mismatch")
    for tensor in (
        target_hidden,
        target_field,
        target_weights,
        preservation_hidden,
        preservation_weights,
    ):
        if tensor is not None and not torch.isfinite(tensor).all():
            raise RuntimeError("pullback inputs contain non-finite values")
    if bool((target_weights < 0).any()):
        raise ValueError("target weights must be nonnegative")
    if preservation_weights is not None and bool((preservation_weights < 0).any()):
        raise ValueError("preservation weights must be nonnegative")
    return experts, hidden


def weighted_functional_pullback(
    *,
    target_hidden: torch.Tensor,
    target_field: torch.Tensor,
    target_weights: torch.Tensor,
    owned_rows: torch.Tensor,
    preservation_hidden: torch.Tensor | None = None,
    preservation_weights: torch.Tensor | None = None,
    preservation_weight: float = 1.0,
    ridge_relative: float = 1e-3,
) -> PullbackResult:
    """Solve row-wise weighted ridge in sample space using Woodbury."""
    experts, hidden = _validate_pullback_inputs(
        target_hidden,
        target_field,
        target_weights,
        preservation_hidden,
        preservation_weights,
        owned_rows,
    )
    if not math.isfinite(preservation_weight) or preservation_weight < 0:
        raise ValueError("preservation_weight must be finite and nonnegative")
    if not math.isfinite(ridge_relative) or ridge_relative <= 0:
        raise ValueError("ridge_relative must be finite and positive")

    original_dtype = target_hidden.dtype
    target_hidden64 = target_hidden.double()
    target_field64 = target_field.double()
    target_weights64 = target_weights.double()
    preservation_hidden64 = (
        None if preservation_hidden is None else preservation_hidden.double()
    )
    preservation_weights64 = (
        None if preservation_weights is None else preservation_weights.double()
    )
    output = torch.zeros((experts, hidden), dtype=torch.float64)
    row_audits = []

    for expert in owned_rows.nonzero(as_tuple=False).flatten().tolist():
        target_sqrt = target_weights64[:, expert].sqrt()
        design_parts = [target_sqrt[:, None] * target_hidden64]
        response_parts = [target_sqrt * target_field64[:, expert]]
        if preservation_hidden64 is not None and preservation_weight > 0:
            preserve_sqrt = (
                preservation_weight * preservation_weights64[:, expert]
            ).sqrt()
            design_parts.append(preserve_sqrt[:, None] * preservation_hidden64)
            response_parts.append(
                torch.zeros(
                    preservation_hidden64.shape[0],
                    dtype=torch.float64,
                )
            )
        design = torch.cat(design_parts, dim=0)
        response = torch.cat(response_parts, dim=0)
        gram = design @ design.T
        ridge_scale = float(torch.trace(gram)) / max(1, design.shape[0])
        ridge = ridge_relative * max(ridge_scale, 1e-12)
        system = gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64)
        dual = torch.linalg.solve(system, response)
        row = design.T @ dual
        output[expert] = row

        target_prediction = target_hidden64 @ row
        target_error = target_prediction - target_field64[:, expert]
        if preservation_hidden64 is None:
            preservation_rms = 0.0
        else:
            preservation_rms = float(
                torch.mean((preservation_hidden64 @ row).pow(2)).sqrt()
            )
        row_audits.append(
            {
                "expert": expert,
                "ridge": ridge,
                "row_l2": float(torch.linalg.vector_norm(row)),
                "target_weighted_rms": float(
                    torch.mean(
                        target_weights64[:, expert] * target_error.pow(2)
                    ).sqrt()
                ),
                "preservation_rms": preservation_rms,
                "signed_positive_elements": int((row > 0).sum()),
                "signed_negative_elements": int((row < 0).sum()),
            }
        )

    delta = output.to(dtype=original_dtype)
    predicted = target_hidden.float() @ delta.T.float()
    target_error = predicted - target_field.float()
    owned = owned_rows.to(device=target_error.device)
    if bool(owned.any()):
        weighted_error = target_weights.float()[:, owned] * target_error[:, owned].pow(2)
        target_weighted_rms = float(weighted_error.mean().sqrt())
    else:
        target_weighted_rms = 0.0
    if preservation_hidden is None:
        preservation_rms = 0.0
    else:
        preservation_rms = float(
            torch.mean((preservation_hidden.float() @ delta.T.float()).pow(2)).sqrt()
        )
    audit = {
        "owned_rows": int(owned_rows.sum()),
        "total_rows": experts,
        "target_samples": target_hidden.shape[0],
        "preservation_samples": (
            0 if preservation_hidden is None else preservation_hidden.shape[0]
        ),
        "target_weighted_rms": target_weighted_rms,
        "preservation_rms": preservation_rms,
        "delta_l2": float(torch.linalg.vector_norm(delta.float())),
        "unowned_rows_exact_zero": bool((delta[~owned_rows] == 0).all()),
        "signed_positive_elements": int((delta > 0).sum()),
        "signed_negative_elements": int((delta < 0).sum()),
        "rows": row_audits,
    }
    if not audit["unowned_rows_exact_zero"]:
        raise RuntimeError("functional pullback escaped owned expert rows")
    if not torch.isfinite(delta).all():
        raise RuntimeError("functional pullback produced non-finite update")
    return PullbackResult(delta_weight=delta, audit=audit)


def batched_weighted_functional_pullback(
    *,
    hidden: torch.Tensor,
    response: torch.Tensor,
    weights: torch.Tensor,
    owned_rows: torch.Tensor,
    ridge_relative: float = 1e-3,
    max_iterations: int = 1024,
    tolerance: float = 1e-6,
    residual_refresh_interval: int = 1024,
) -> PullbackResult:
    """Solve all owned router rows with independently converged PCG iterates.

    SRR passes positive weights only for matching-domain tokens with a
    positive source-over-parent likelihood advantage; all other weights
    are zero. No nonmatching-domain preservation target is fitted.

    Each expert row is an independent SPD system. Work is performed in FP64.
    A row is frozen only after its true residual meets the tolerance, without
    restarting still-active rows and destroying their conjugate directions.
    """
    if hidden.ndim != 2 or response.ndim != 2:
        raise ValueError("hidden and response must be matrices")
    samples, hidden_size = hidden.shape
    if response.shape[0] != samples or weights.shape != response.shape:
        raise ValueError("response/weight shapes must match hidden samples")
    experts = response.shape[1]
    if owned_rows.shape != (experts,) or owned_rows.dtype != torch.bool:
        raise ValueError("owned_rows must be a boolean expert-row mask")
    if not math.isfinite(ridge_relative) or ridge_relative <= 0:
        raise ValueError("ridge_relative must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if residual_refresh_interval <= 0:
        raise ValueError("residual_refresh_interval must be positive")
    for tensor in (hidden, response, weights):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("batched pullback inputs contain non-finite values")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative")

    indices = owned_rows.nonzero(as_tuple=False).flatten()
    output = torch.zeros(
        (experts, hidden_size),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    if indices.numel() == 0:
        return PullbackResult(
            delta_weight=output,
            audit={
                "owned_rows": 0,
                "total_rows": experts,
                "samples": samples,
                "iterations": 0,
                "converged_rows": 0,
                "max_relative_residual": 0.0,
                "delta_l2": 0.0,
                "unowned_rows_exact_zero": True,
                "signed_positive_elements": 0,
                "signed_negative_elements": 0,
                "rows": [],
            },
        )

    work_dtype = torch.float64
    design = hidden.to(dtype=work_dtype)
    selected_response = response[:, indices].to(dtype=work_dtype)
    selected_weights = weights[:, indices].to(dtype=work_dtype)
    design_sq = design.square()
    row_energy = design_sq.sum(dim=1, keepdim=True)
    ridge_scale = (selected_weights * row_energy).sum(dim=0) / max(1, samples)
    ridge = ridge_relative * ridge_scale.clamp_min(1e-12)
    right_hand_side = (selected_weights * selected_response).T @ design
    diagonal = selected_weights.T @ design_sq + ridge[:, None]
    diagonal = diagonal.clamp_min(torch.finfo(work_dtype).eps)

    def matvec(value: torch.Tensor) -> torch.Tensor:
        prediction = design @ value.T
        return (selected_weights * prediction).T @ design + ridge[:, None] * value

    def matvec_rows(
        value: torch.Tensor,
        local_indices: torch.Tensor,
    ) -> torch.Tensor:
        prediction = design @ value.T
        local_weights = selected_weights[:, local_indices]
        return (
            (local_weights * prediction).T @ design
            + ridge[local_indices, None] * value
        )

    rhs_norm = torch.linalg.vector_norm(right_hand_side, dim=1)
    denominator_floor = torch.finfo(work_dtype).eps
    zero_rhs = rhs_norm <= denominator_floor
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    residual_preconditioned = (residual * preconditioned).sum(dim=1)
    relative = torch.linalg.vector_norm(residual, dim=1) / rhs_norm.clamp_min(
        denominator_floor
    )
    relative = torch.where(zero_rhs, torch.zeros_like(relative), relative)
    active = (~zero_rhs) & (relative > tolerance)
    failed = torch.zeros_like(active)
    row_iterations = torch.zeros_like(indices, dtype=torch.int64)
    breakdown_counts = torch.zeros_like(indices, dtype=torch.int64)
    true_residual_checks = torch.zeros_like(indices, dtype=torch.int64)
    residual_refreshes = 0
    iterations = 0

    for iteration in range(max_iterations):
        if not bool(active.any()):
            break
        row_iterations[active] = iteration + 1
        direction = torch.where(active[:, None], direction, torch.zeros_like(direction))
        active_indices = active.nonzero(as_tuple=False).flatten()
        product = torch.zeros_like(direction)
        product[active_indices] = matvec_rows(
            direction[active_indices],
            active_indices,
        )
        denominator = (direction * product).sum(dim=1)
        valid = (
            active
            & torch.isfinite(denominator)
            & torch.isfinite(residual_preconditioned)
            & (denominator > 0)
            & (residual_preconditioned > 0)
        )
        restart = active & ~valid
        if bool(restart.any()):
            breakdown_counts[restart] += 1
            direction[restart] = preconditioned[restart]
            restart_indices = restart.nonzero(as_tuple=False).flatten()
            product[restart_indices] = matvec_rows(
                direction[restart_indices],
                restart_indices,
            )
            denominator = (direction * product).sum(dim=1)
            valid = (
                active
                & torch.isfinite(denominator)
                & torch.isfinite(residual_preconditioned)
                & (denominator > 0)
                & (residual_preconditioned > 0)
            )
            irrecoverable = active & ~valid
            failed |= irrecoverable
            active &= ~irrecoverable
        alpha = torch.zeros_like(denominator)
        alpha[active] = residual_preconditioned[active] / denominator[active]
        solution = solution + alpha[:, None] * direction
        residual = residual - alpha[:, None] * product
        relative = torch.linalg.vector_norm(residual, dim=1) / rhs_norm.clamp_min(
            denominator_floor
        )
        relative = torch.where(zero_rhs, torch.zeros_like(relative), relative)
        iterations = iteration + 1

        candidate = active & (relative <= tolerance)
        periodic_refresh = iterations % residual_refresh_interval == 0
        refresh_rows = active if periodic_refresh else candidate
        if bool(refresh_rows.any()):
            refresh_indices = refresh_rows.nonzero(as_tuple=False).flatten()
            residual[refresh_indices] = (
                right_hand_side[refresh_indices]
                - matvec_rows(solution[refresh_indices], refresh_indices)
            )
            relative[refresh_indices] = (
                torch.linalg.vector_norm(residual[refresh_indices], dim=1)
                / rhs_norm[refresh_indices].clamp_min(denominator_floor)
            )
            true_residual_checks[refresh_indices] += 1
            residual_refreshes += 1
        active = (~zero_rhs) & (~failed) & (relative > tolerance)
        if not bool(active.any()):
            break

        next_preconditioned = residual / diagonal
        next_residual_preconditioned = (residual * next_preconditioned).sum(dim=1)
        beta = torch.zeros_like(next_residual_preconditioned)
        stable = (
            active
            & (~refresh_rows)
            & torch.isfinite(next_residual_preconditioned)
            & (next_residual_preconditioned > 0)
            & (residual_preconditioned > 0)
        )
        beta[stable] = (
            next_residual_preconditioned[stable]
            / residual_preconditioned[stable]
        )
        direction = next_preconditioned + beta[:, None] * direction
        direction = torch.where(active[:, None], direction, torch.zeros_like(direction))
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned

    true_residual = right_hand_side - matvec(solution)
    final_relative = torch.linalg.vector_norm(true_residual, dim=1) / rhs_norm.clamp_min(
        denominator_floor
    )
    final_relative = torch.where(zero_rhs, torch.zeros_like(final_relative), final_relative)
    output[indices] = solution.to(dtype=hidden.dtype)
    prediction = design @ solution.T
    weighted_error = selected_weights * (prediction - selected_response).square()
    row_audits = []
    for local_index, expert in enumerate(indices.tolist()):
        row_audits.append(
            {
                "expert": expert,
                "ridge": float(ridge[local_index]),
                "row_l2": float(torch.linalg.vector_norm(solution[local_index])),
                "weighted_rms": float(weighted_error[:, local_index].mean().sqrt()),
                "relative_residual": float(final_relative[local_index]),
                "iterations": int(row_iterations[local_index]),
                "breakdown_restarts": int(breakdown_counts[local_index]),
                "true_residual_checks": int(true_residual_checks[local_index]),
                "converged": bool(
                    not failed[local_index]
                    and (
                        final_relative[local_index] <= tolerance
                        or zero_rhs[local_index]
                    )
                ),
                "signed_positive_elements": int((solution[local_index] > 0).sum()),
                "signed_negative_elements": int((solution[local_index] < 0).sum()),
            }
        )
    audit = {
        "owned_rows": int(indices.numel()),
        "total_rows": experts,
        "samples": samples,
        "iterations": iterations,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "residual_refresh_interval": residual_refresh_interval,
        "residual_refreshes": residual_refreshes,
        "work_dtype": str(work_dtype),
        "failed_rows": int(failed.sum()),
        "converged_rows": sum(row["converged"] for row in row_audits),
        "max_relative_residual": max(
            (row["relative_residual"] for row in row_audits),
            default=0.0,
        ),
        "weighted_rms": float(weighted_error.mean().sqrt()),
        "delta_l2": float(torch.linalg.vector_norm(output.float())),
        "unowned_rows_exact_zero": bool((output[~owned_rows] == 0).all()),
        "signed_positive_elements": int((output > 0).sum()),
        "signed_negative_elements": int((output < 0).sum()),
        "rows": row_audits,
    }
    if not audit["unowned_rows_exact_zero"]:
        raise RuntimeError("batched functional pullback escaped owned expert rows")
    if not torch.isfinite(output).all():
        raise RuntimeError("batched functional pullback produced non-finite update")
    return PullbackResult(delta_weight=output, audit=audit)


def categorical_fisher_inner(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    hidden: torch.Tensor,
    parent_logits: torch.Tensor,
) -> torch.Tensor:
    """Inner product of two router updates under the full categorical Fisher."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("updates must be matching [expert, hidden] matrices")
    if hidden.ndim != 2 or hidden.shape[1] != left.shape[1]:
        raise ValueError("hidden shape mismatch")
    if parent_logits.shape != (hidden.shape[0], left.shape[0]):
        raise ValueError("parent_logits shape mismatch")
    probabilities = torch.softmax(parent_logits.double(), dim=-1)
    left_logits = hidden.double() @ left.double().T
    right_logits = hidden.double() @ right.double().T
    first = (probabilities * left_logits * right_logits).sum(dim=-1)
    left_mean = (probabilities * left_logits).sum(dim=-1)
    right_mean = (probabilities * right_logits).sum(dim=-1)
    return (first - left_mean * right_mean).sum()


def project_domain_conflicts(
    updates: dict[str, torch.Tensor],
    *,
    hidden: torch.Tensor,
    parent_logits: torch.Tensor,
    tolerance: float = 1e-10,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Sequentially remove negative pairwise Fisher components."""
    if not updates:
        raise ValueError("updates cannot be empty")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    projected: dict[str, torch.Tensor] = {}
    events = []
    for domain, raw in updates.items():
        current = raw.clone()
        for previous_domain, previous in projected.items():
            before = categorical_fisher_inner(
                current,
                previous,
                hidden=hidden,
                parent_logits=parent_logits,
            )
            previous_norm = categorical_fisher_inner(
                previous,
                previous,
                hidden=hidden,
                parent_logits=parent_logits,
            )
            applied = bool(before < -tolerance and previous_norm > tolerance)
            coefficient = torch.zeros((), dtype=torch.float64)
            if applied:
                coefficient = before / previous_norm
                current = current - coefficient.to(current.dtype) * previous
            after = categorical_fisher_inner(
                current,
                previous,
                hidden=hidden,
                parent_logits=parent_logits,
            )
            events.append(
                {
                    "domain": domain,
                    "against": previous_domain,
                    "pre_inner": float(before),
                    "post_inner": float(after),
                    "projection_coefficient": float(coefficient),
                    "applied": applied,
                }
            )
        projected[domain] = current
    audit = {
        "events": events,
        "conflicts_before": sum(row["pre_inner"] < -tolerance for row in events),
        "conflicts_after": sum(row["post_inner"] < -tolerance for row in events),
        "passed": all(row["post_inner"] >= -10 * tolerance for row in events),
    }
    if not audit["passed"]:
        raise RuntimeError("categorical Fisher conflict projection failed")
    return projected, audit
