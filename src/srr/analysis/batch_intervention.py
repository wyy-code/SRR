#!/usr/bin/env python3
"""Batched, row-isolated architecture-native MoE route interventions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .native_intervention import (
    NativeMoERouteIntervention,
    RouteState,
    aligned_alpha,
    js_divergence,
    route_from_logits,
    route_from_selected_weights,
    route_jaccard,
)


class NativeMoERouteBatchIntervention(NativeMoERouteIntervention):
    """Patch one token-layer per batch row in a shared forward pass.

    Each batch row is an independent copy of one prompt.  This preserves the
    single-intervention estimand while amortizing model execution across the
    eight frozen token-layer records selected for that prompt.
    """

    def set_many(self, contexts: list[dict[str, Any]] | None) -> None:
        self.context = None if contexts is None else {"contexts": contexts}
        self.records.clear()
        self.native_gate_outputs.clear()

    def _hook(self, layer_id: int):
        def hook(block: Any, inputs: Any, output: Any) -> Any:
            payload = self.context
            if payload is None:
                return output
            contexts = [
                row for row in payload["contexts"] if int(row["layer_id"]) == layer_id
            ]
            if not contexts:
                return output

            hidden_all = inputs[0]
            if hidden_all.ndim != 3:
                raise RuntimeError(f"expected [batch, sequence, hidden], got {hidden_all.shape}")
            batch_size, sequence_length = hidden_all.shape[:2]
            flat_hidden = hidden_all.reshape(-1, hidden_all.shape[-1])
            native_logits_all = F.linear(
                flat_hidden, block.gate.weight, getattr(block.gate, "bias", None)
            )
            top_k = self._top_k(block)
            norm_topk_prob = self._norm_topk_prob(block)
            gate_output = self.native_gate_outputs.get(layer_id)
            if gate_output is None:
                raise RuntimeError(f"missing native gate output for layer {layer_id}")
            if self.case == "deepseek":
                native_indices, native_weights = gate_output[0], gate_output[1]
                native_all = route_from_selected_weights(native_indices, native_weights)
            else:
                native_all = route_from_logits(
                    gate_output.reshape(-1, gate_output.shape[-1]),
                    top_k,
                    norm_topk_prob,
                    softmax_in_float=True,
                    weight_dtype=flat_hidden.dtype,
                )

            target_selected = native_all.selected.clone()
            target_weights = native_all.weights.clone()
            pending: list[tuple[dict[str, Any], int, RouteState, RouteState]] = []
            seen_batch_rows: set[int] = set()
            for context in contexts:
                batch_index = int(context["batch_index"])
                token_position = int(context["position"])
                if not 0 <= batch_index < batch_size:
                    raise IndexError(f"batch_index={batch_index} outside batch={batch_size}")
                if not 0 <= token_position < sequence_length:
                    raise IndexError(
                        f"position={token_position} outside sequence={sequence_length}"
                    )
                if batch_index in seen_batch_rows:
                    raise RuntimeError(
                        f"multiple interventions in batch row {batch_index} at layer {layer_id}"
                    )
                seen_batch_rows.add(batch_index)
                flat_position = batch_index * sequence_length + token_position
                hidden = flat_hidden[flat_position]
                native = RouteState(
                    selected=native_all.selected[flat_position],
                    weights=native_all.weights[flat_position],
                    mass=native_all.mass[flat_position],
                    alpha=native_all.alpha[flat_position],
                )
                target = self._target_route(
                    block, hidden, native_logits_all[flat_position], native, context
                )
                if target.selected.numel() != top_k or target.weights.numel() != top_k:
                    raise RuntimeError("target route cardinality mismatch")
                if not torch.isfinite(target.weights).all() or bool((target.weights < 0).any()):
                    raise RuntimeError("invalid target route weights")
                expected_mass = context.get("assert_mass")
                if expected_mass is not None:
                    expected = (
                        float(native.mass.detach())
                        if expected_mass == "native"
                        else float(expected_mass)
                    )
                    actual_mass = float(target.mass.detach())
                    if abs(actual_mass - expected) > float(context.get("mass_atol", 1e-6)):
                        raise RuntimeError(
                            f"route mass assertion failed: {actual_mass} != {expected}"
                        )
                target_selected[flat_position] = target.selected.to(target_selected.dtype)
                target_weights[flat_position] = target.weights.to(target_weights.dtype)
                pending.append((context, flat_position, native, target))

            native_routed_all = self._all_expert_mixture(
                block, flat_hidden, native_all.selected, native_all.weights
            )
            target_routed_all = self._all_expert_mixture(
                block, flat_hidden, target_selected, target_weights
            )
            if self.case == "deepseek" and getattr(block, "shared_experts", None) is not None:
                shared_all = block.shared_experts(hidden_all).reshape(-1, hidden_all.shape[-1])
            else:
                shared_all = None

            for target_index, (context, flat_position, native, target) in enumerate(pending):
                native_routed = native_routed_all[flat_position]
                target_routed = target_routed_all[flat_position]
                difference = target_routed.float() - native_routed.float()
                denominator = float(native_routed.float().norm().clamp_min(1e-20).detach())
                if self.case == "deepseek":
                    actual_reference = output.reshape(-1, output.shape[-1])[flat_position].float()
                    if shared_all is None:
                        recompute_reference = native_routed.float()
                    else:
                        recompute_reference = (
                            native_routed + shared_all[flat_position]
                        ).to(output.dtype).float()
                else:
                    actual_reference = output[0].reshape(-1, output[0].shape[-1])[
                        flat_position
                    ].float()
                    recompute_reference = native_routed.float()
                recompute_error = float(
                    (
                        (recompute_reference - actual_reference).norm()
                        / actual_reference.norm().clamp_min(1e-20)
                    ).detach()
                )
                rtol = context.get("assert_native_recompute_rtol")
                if rtol is not None and recompute_error > float(rtol):
                    raise RuntimeError(
                        f"native routed output recomputation failed: {recompute_error} > {rtol}"
                    )
                universe = torch.unique(
                    torch.cat([native.selected.reshape(-1), target.selected.reshape(-1)])
                ).cpu()
                native_alpha = aligned_alpha(native, universe)
                target_alpha = aligned_alpha(target, universe)
                self.records.append(
                    {
                        "batch_index": int(context["batch_index"]),
                        "sample_id": str(context["sample_id"]),
                        "layer_id": layer_id,
                        "position": int(context["position"]),
                        "condition": str(context["condition"]),
                        "norm_topk_prob": norm_topk_prob,
                        "topk_before": [int(value) for value in native.selected.tolist()],
                        "topk_after": [int(value) for value in target.selected.tolist()],
                        "q_before": float(native.mass.detach()),
                        "q_after": float(target.mass.detach()),
                        "delta_q": float((target.mass - native.mass).detach()),
                        "alpha_before": [float(value) for value in native.alpha.tolist()],
                        "alpha_after": [float(value) for value in target.alpha.tolist()],
                        "topk_jaccard": route_jaccard(native.selected, target.selected),
                        "alpha_js_union": js_divergence(native_alpha, target_alpha),
                        "routed_output_norm_before": denominator,
                        "routed_output_norm_after": float(target_routed.float().norm().detach()),
                        "routed_output_relative_l2": float(difference.norm().detach()) / denominator,
                        "native_routed_recompute_relative_error": recompute_error,
                        "routed_output_cosine": float(
                            F.cosine_similarity(
                                native_routed.float(), target_routed.float(), dim=0
                            ).detach()
                        ),
                    }
                )

            if self.case == "deepseek":
                patched = output.clone().reshape(-1, output.shape[-1])
                for target_index, (_context, flat_position, _native, _target) in enumerate(pending):
                    delta = (
                        target_routed_all[flat_position].float()
                        - native_routed_all[flat_position].float()
                    )
                    patched[flat_position] = (
                        patched[flat_position].float() + delta
                    ).to(patched.dtype)
                return patched.reshape_as(output)

            routed, logits = output
            patched_routed = routed.clone().reshape(-1, routed.shape[-1])
            for target_index, (_context, flat_position, _native, _target) in enumerate(pending):
                patched_routed[flat_position] = target_routed_all[flat_position].to(
                    patched_routed.dtype
                )
            return patched_routed.reshape_as(routed), logits

        return hook
