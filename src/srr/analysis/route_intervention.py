"""Architecture-native MoE route interventions with explicit S/alpha/q accounting."""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import itertools
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RouteState:
    selected: torch.Tensor
    weights: torch.Tensor
    mass: torch.Tensor
    alpha: torch.Tensor


def route_from_logits(
    logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    *,
    softmax_in_float: bool = True,
    weight_dtype: torch.dtype | None = None,
) -> RouteState:
    probabilities = logits.float().softmax(dim=-1) if softmax_in_float else logits.softmax(dim=-1)
    # Qwen3's native forward sorts before normalization and expert dispatch.
    weights, selected = torch.topk(probabilities, k=top_k, dim=-1, sorted=True)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    if weight_dtype is not None:
        weights = weights.to(weight_dtype)
    mass = weights.float().sum(dim=-1)
    alpha = weights.float() / mass.unsqueeze(-1).clamp_min(1e-20)
    return RouteState(selected=selected, weights=weights, mass=mass, alpha=alpha)


def quantize_with_exact_mass(
    weights: torch.Tensor,
    target_mass: torch.Tensor,
    dtype: torch.dtype,
    *,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Quantize weights while preserving a representable native mass."""

    if weights.ndim != 1:
        raise ValueError("exact-mass quantization expects one route")
    device = weights.device
    proposal = weights.detach().float().cpu()
    target = float(target_mass.detach().float().reshape(()).cpu())
    initial = proposal.to(dtype)
    if abs(float(initial.float().sum()) - target) <= atol:
        return initial.to(device)

    def neighbors(value: torch.Tensor, radius: int) -> list[float]:
        center = value.to(dtype).reshape(())
        down, up = center.clone(), center.clone()
        values = {float(center.float())}
        for _ in range(radius):
            down = torch.nextafter(down, torch.full_like(down, float("-inf")))
            up = torch.nextafter(up, torch.full_like(up, float("inf")))
            if float(down) > 0:
                values.add(float(down.float()))
            values.add(float(up.float()))
        return sorted(values)

    split = weights.numel() // 2
    for radius in (2, 4, 8, 16, 32):
        candidates = [neighbors(value, radius) for value in proposal]
        left_rows = []
        for values in itertools.product(*candidates[:split]):
            cost = sum((value - float(proposal[index])) ** 2 for index, value in enumerate(values))
            left_rows.append((sum(values), cost, values))
        left_rows.sort(key=lambda row: row[0])
        left_sums = [row[0] for row in left_rows]
        best = None
        for right_values in itertools.product(*candidates[split:]):
            right_sum = sum(right_values)
            position = bisect.bisect_left(left_sums, target - right_sum)
            right_cost = sum(
                (value - float(proposal[split + index])) ** 2
                for index, value in enumerate(right_values)
            )
            for left_index in (position - 1, position):
                if 0 <= left_index < len(left_rows):
                    left_sum, left_cost, left_values = left_rows[left_index]
                    residual = abs(target - left_sum - right_sum)
                    candidate = (residual, left_cost + right_cost, left_values + right_values)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
        if best is not None and best[0] <= atol:
            result = torch.tensor(best[2], dtype=dtype, device=device)
            if abs(float(result.float().sum()) - target) <= atol:
                return result
    residual = None if best is None else best[0]
    raise RuntimeError(f"cannot preserve representable routed mass after quantization: residual={residual}")


def route_with_selected(
    logits: torch.Tensor,
    selected: torch.Tensor,
    *,
    mass: torch.Tensor | float | None,
    norm_topk_prob: bool,
    softmax_in_float: bool = True,
    weight_dtype: torch.dtype | None = None,
) -> RouteState:
    probabilities = logits.float().softmax(dim=-1) if softmax_in_float else logits.softmax(dim=-1)
    selected = selected.to(device=logits.device, dtype=torch.long)
    weights = probabilities.gather(-1, selected)
    native_selected_mass = weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    alpha = weights / native_selected_mass
    if mass is None:
        if not norm_topk_prob:
            if weight_dtype is not None:
                weights = weights.to(weight_dtype)
            mass_tensor = weights.float().sum(dim=-1)
            alpha = weights.float() / mass_tensor.unsqueeze(-1).clamp_min(1e-20)
            return RouteState(selected=selected, weights=weights, mass=mass_tensor, alpha=alpha)
        target_mass = torch.ones_like(native_selected_mass)
    else:
        target_mass = torch.as_tensor(mass, device=logits.device, dtype=torch.float32)
        while target_mass.ndim < weights.ndim:
            target_mass = target_mass.unsqueeze(-1)
    weights = alpha.float() * target_mass
    if weight_dtype is not None:
        weights = (
            quantize_with_exact_mass(weights, target_mass.reshape(()), weight_dtype)
            if mass is not None and weights.ndim == 1
            else weights.to(weight_dtype)
        )
    mass_tensor = weights.float().sum(dim=-1)
    alpha = weights.float() / mass_tensor.unsqueeze(-1).clamp_min(1e-20)
    return RouteState(selected=selected, weights=weights, mass=mass_tensor, alpha=alpha)


def route_jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    left_set = set(int(value) for value in left.reshape(-1).tolist())
    right_set = set(int(value) for value in right.reshape(-1).tolist())
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() / left.float().sum().clamp_min(1e-20)
    right = right.float() / right.float().sum().clamp_min(1e-20)
    midpoint = 0.5 * (left + right)
    value = 0.5 * (
        (left * (left.clamp_min(1e-20).log() - midpoint.clamp_min(1e-20).log())).sum()
        + (right * (right.clamp_min(1e-20).log() - midpoint.clamp_min(1e-20).log())).sum()
    )
    return float(value.detach())


def route_from_selected_weights(selected: torch.Tensor, weights: torch.Tensor) -> RouteState:
    mass = weights.float().sum(dim=-1)
    alpha = weights.float() / mass.unsqueeze(-1).clamp_min(1e-20)
    return RouteState(selected=selected, weights=weights, mass=mass, alpha=alpha)


def aligned_alpha(state: RouteState, universe: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(universe.numel(), dtype=torch.float32, device=state.weights.device)
    lookup = {int(expert): slot for slot, expert in enumerate(universe.tolist())}
    for expert, value in zip(state.selected.reshape(-1).tolist(), state.alpha.reshape(-1)):
        output[lookup[int(expert)]] = value.float()
    return output


class NativeRouteIntervention:
    """Replace one routed-expert mixture while preserving the surrounding MoE block.

    DeepSeek returns routed+shared output, so the hook applies target-native routed
    delta. OLMoE returns routed output separately, so the selected row is replaced.
    All target weights are explicit and never implicitly renormalized.
    """

    def __init__(self, model: Any, case: str, layer_ids: list[int]) -> None:
        if case not in {"deepseek", "olmoe"}:
            raise ValueError(case)
        self.case = case
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.native_gate_outputs: dict[int, Any] = {}
        self.handles = []
        for layer_id in layer_ids:
            block = model.model.layers[layer_id].mlp
            self.handles.append(block.gate.register_forward_hook(self._gate_hook(layer_id)))
            self.handles.append(block.register_forward_hook(self._hook(layer_id)))

    def set(self, context: dict[str, Any] | None) -> None:
        self.context = context
        self.records.clear()
        self.native_gate_outputs.clear()

    def _gate_hook(self, layer_id: int):
        def hook(_gate: Any, _inputs: Any, output: Any) -> Any:
            self.native_gate_outputs[layer_id] = output
            return output

        return hook

    def _norm_topk_prob(self, block: Any) -> bool:
        owner = block.gate if self.case == "deepseek" else block
        return bool(owner.norm_topk_prob)

    def _top_k(self, block: Any) -> int:
        return int(block.gate.top_k if self.case == "deepseek" else block.top_k)

    def _all_expert_mixture(
        self,
        block: Any,
        flat_hidden: torch.Tensor,
        selected: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = weights.to(flat_hidden.dtype)
        if self.case == "deepseek":
            return block.moe_infer(
                flat_hidden,
                selected.reshape(-1),
                weights.reshape(-1, 1),
            )
        output = torch.zeros_like(flat_hidden)
        expert_mask = F.one_hot(selected, num_classes=len(block.experts)).permute(2, 1, 0)
        for expert_id, expert in enumerate(block.experts):
            slots, rows = torch.where(expert_mask[expert_id])
            current = flat_hidden[None, rows].reshape(-1, flat_hidden.shape[-1])
            expert_output = expert(current) * weights[rows, slots, None]
            output.index_add_(0, rows, expert_output.to(flat_hidden.dtype))
        return output

    def _target_route(
        self,
        block: Any,
        hidden: torch.Tensor,
        native_logits: torch.Tensor,
        native: RouteState,
        context: dict[str, Any],
    ) -> RouteState:
        condition = str(context["condition"])
        norm_topk_prob = self._norm_topk_prob(block)
        top_k = self._top_k(block)
        native_kwargs = {
            "softmax_in_float": self.case == "olmoe",
            "weight_dtype": hidden.dtype,
        }
        if condition == "native_noop":
            return native
        if condition in {"source_set_native_mass", "source_set_mass_matched"}:
            selected = torch.as_tensor(context["source_topk"], device=hidden.device, dtype=torch.long)
            mass = native.mass if condition == "source_set_mass_matched" else None
            return route_with_selected(
                native_logits,
                selected,
                mass=mass,
                norm_topk_prob=norm_topk_prob,
                **native_kwargs,
            )
        if condition in {"gate_hidden", "full_source_policy"}:
            target_hidden = context["source_hidden"].to(device=hidden.device, dtype=block.gate.weight.dtype)
            weight = block.gate.weight
            target_logits = F.linear(target_hidden, weight, getattr(block.gate, "bias", None))
            if condition == "full_source_policy":
                weight = context["source_weight"].to(device=hidden.device, dtype=hidden.dtype)
                target_hidden = target_hidden.to(weight.dtype)
                target_logits = F.linear(target_hidden, weight)
            return route_from_logits(target_logits, top_k, norm_topk_prob, **native_kwargs)
        if condition == "source_router":
            source_weight = context["source_weight"].to(device=hidden.device, dtype=hidden.dtype)
            target_logits = F.linear(hidden.to(source_weight.dtype), source_weight)
            return route_from_logits(target_logits, top_k, norm_topk_prob, **native_kwargs)
        raise ValueError(f"unsupported native route condition: {condition}")

    def _hook(self, layer_id: int):
        def hook(block: Any, inputs: Any, output: Any) -> Any:
            context = self.context
            if context is None or layer_id != int(context["layer_id"]):
                return output
            hidden_all = inputs[0]
            flat_hidden = hidden_all.reshape(-1, hidden_all.shape[-1])
            position = int(context["position"])
            hidden = flat_hidden[position]
            native_logits_all = F.linear(flat_hidden, block.gate.weight, getattr(block.gate, "bias", None))
            native_logits = native_logits_all[position]
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
            native = RouteState(
                selected=native_all.selected[position],
                weights=native_all.weights[position],
                mass=native_all.mass[position],
                alpha=native_all.alpha[position],
            )
            target = self._target_route(block, hidden, native_logits, native, context)
            if target.selected.numel() != top_k or target.weights.numel() != top_k:
                raise RuntimeError("target route cardinality mismatch")
            if not torch.isfinite(target.weights).all() or bool((target.weights < 0).any()):
                raise RuntimeError("invalid target route weights")
            expected_mass = context.get("assert_mass")
            if expected_mass is not None:
                expected = float(native.mass.detach()) if expected_mass == "native" else float(expected_mass)
                actual_mass = float(target.mass.detach())
                if abs(actual_mass - expected) > float(context.get("mass_atol", 1e-6)):
                    raise RuntimeError(f"route mass assertion failed: {actual_mass} != {expected}")

            target_selected = native_all.selected.clone()
            target_weights = native_all.weights.clone()
            target_selected[position] = target.selected.to(target_selected.dtype)
            target_weights[position] = target.weights.to(target_weights.dtype)
            native_routed_all = self._all_expert_mixture(
                block, flat_hidden, native_all.selected, native_all.weights
            )
            target_routed_all = self._all_expert_mixture(
                block, flat_hidden, target_selected, target_weights
            )
            native_routed = native_routed_all[position]
            target_routed = target_routed_all[position]
            difference = target_routed.float() - native_routed.float()
            denominator = float(native_routed.float().norm().clamp_min(1e-20).detach())
            if self.case == "deepseek":
                actual_total = output.reshape(-1, output.shape[-1])[position].float()
                if getattr(block, "shared_experts", None) is None:
                    recomposed_total = native_routed
                    target_total = target_routed
                else:
                    shared_all = block.shared_experts(hidden_all).reshape(-1, hidden_all.shape[-1])
                    recomposed_total = (native_routed_all + shared_all).to(output.dtype).reshape_as(output)
                    recomposed_total = recomposed_total.reshape(-1, output.shape[-1])[position].float()
                    target_total = (target_routed + shared_all[position]).to(output.dtype)
                actual_reference = actual_total
                recompute_reference = recomposed_total.float()
            else:
                actual_reference = output[0].reshape(-1, output[0].shape[-1])[position].float()
                recompute_reference = native_routed.float()
            native_recompute_relative_error = float(
                (
                    (recompute_reference - actual_reference).norm()
                    / actual_reference.norm().clamp_min(1e-20)
                ).detach()
            )
            recompute_rtol = context.get("assert_native_recompute_rtol")
            if recompute_rtol is not None and native_recompute_relative_error > float(recompute_rtol):
                raise RuntimeError(
                    "native routed output recomputation failed: "
                    f"{native_recompute_relative_error} > {float(recompute_rtol)}"
                )
            universe = torch.unique(torch.cat([native.selected.reshape(-1), target.selected.reshape(-1)])).cpu()
            native_alpha = aligned_alpha(native, universe)
            target_alpha = aligned_alpha(target, universe)
            self.records.append(
                {
                    "layer_id": layer_id,
                    "position": position,
                    "condition": str(context["condition"]),
                    "norm_topk_prob": norm_topk_prob,
                    "topk_before": [int(value) for value in native.selected.reshape(-1).tolist()],
                    "topk_after": [int(value) for value in target.selected.reshape(-1).tolist()],
                    "q_before": float(native.mass.detach()),
                    "q_after": float(target.mass.detach()),
                    "delta_q": float((target.mass - native.mass).detach()),
                    "alpha_before": [float(value) for value in native.alpha.reshape(-1).tolist()],
                    "alpha_after": [float(value) for value in target.alpha.reshape(-1).tolist()],
                    "topk_jaccard": route_jaccard(native.selected, target.selected),
                    "alpha_js_union": js_divergence(native_alpha, target_alpha),
                    "routed_output_norm_before": denominator,
                    "routed_output_norm_after": float(target_routed.float().norm().detach()),
                    "routed_output_relative_l2": float(difference.norm().detach()) / denominator,
                    "native_routed_recompute_relative_error": native_recompute_relative_error,
                    "routed_output_cosine": float(
                        F.cosine_similarity(native_routed.float(), target_routed.float(), dim=0).detach()
                    ),
                }
            )

            if self.case == "deepseek":
                patched = output.clone().reshape(-1, output.shape[-1])
                patched[position] = target_total.to(patched.dtype)
                return patched.reshape_as(output)
            routed, logits = output
            patched_routed = routed.clone().reshape(-1, routed.shape[-1])
            patched_routed[position] = target_routed.to(patched_routed.dtype)
            return patched_routed.reshape_as(routed), logits

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
