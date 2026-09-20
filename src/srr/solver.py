"""FP64 weighted ridge solver for Selective Router Repair (SRR)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SRRFitResult:
    delta_weight: torch.Tensor
    audit: dict[str, Any]


def solve_srr_weighted_ridge(
    *,
    hidden: torch.Tensor,
    response: torch.Tensor,
    weights: torch.Tensor,
    owned_rows: torch.Tensor,
    ridge_relative: float = 1e-3,
    max_iterations: int = 1024,
    tolerance: float = 1e-6,
    residual_refresh_interval: int = 1024,
) -> SRRFitResult:
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
            raise RuntimeError("SRR ridge inputs contain non-finite values")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative")

    indices = owned_rows.nonzero(as_tuple=False).flatten()
    output = torch.zeros(
        (experts, hidden_size),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    if indices.numel() == 0:
        return SRRFitResult(
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
        raise RuntimeError("SRR ridge update escaped selected router rows")
    if not torch.isfinite(output).all():
        raise RuntimeError("SRR ridge produced a non-finite update")
    return SRRFitResult(delta_weight=output, audit=audit)
