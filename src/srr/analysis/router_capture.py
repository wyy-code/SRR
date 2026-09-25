"""Architecture-compatible capture of MoE router logits and gate inputs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class RouterCapture:
    """Capture full pre-softmax router logits across MoE implementations.

    Some implementations return full gate logits. Others return only top-k
    indices and weights; for a linear gate, recomputing ``hidden @ weight.T``
    is exactly the pre-softmax operation and retains the same Jacobian.
    """

    def __init__(self, model: Any, layer_ids: list[int]):
        self.layer_ids = list(layer_ids)
        self.outputs: dict[int, torch.Tensor] = {}
        self.capture_modes: dict[int, str] = {}
        self.handles = []
        for layer_id in self.layer_ids:
            gate = model.model.layers[layer_id].mlp.gate
            self.handles.append(gate.register_forward_hook(self._hook(layer_id)))

    @staticmethod
    def full_logits(
        module: Any,
        inputs: Any,
        output: Any,
    ) -> tuple[torch.Tensor, str]:
        if isinstance(output, torch.Tensor):
            if output.ndim < 2:
                raise RuntimeError("router tensor output must have rank >= 2")
            return output, "native_full_logits"
        if not isinstance(output, (tuple, list)):
            raise RuntimeError(f"unsupported router output: {type(output).__name__}")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("router hook received no tensor hidden states")
        hidden = inputs[0]
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError("tuple-returning router has no rank-2 gate weight")
        if hidden.shape[-1] != weight.shape[1]:
            raise RuntimeError("router hidden dimension does not match gate weight")
        bias = getattr(module, "bias", None)
        logits = F.linear(hidden, weight, bias)
        return logits, "recomputed_linear_pre_softmax"

    def _hook(self, layer_id: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            logits, mode = self.full_logits(module, inputs, output)
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and logits.shape[-1] != weight.shape[0]:
                raise RuntimeError(f"router expert dimension mismatch at layer {layer_id}")
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite router logits at layer {layer_id}")
            self.outputs[layer_id] = logits
            self.capture_modes[layer_id] = mode

        return hook

    def clear(self) -> None:
        self.outputs.clear()
        self.capture_modes.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "RouterCapture":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class RouterJacobianCapture(RouterCapture):
    """Capture exact linear-gate inputs together with full router logits."""

    def __init__(self, model: Any, layer_ids: list[int]):
        self.inputs: dict[int, torch.Tensor] = {}
        super().__init__(model, layer_ids)

    def _hook(self, layer_id: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("router gate hook received no tensor input")
            logits, mode = self.full_logits(module, inputs, output)
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite router logits at layer {layer_id}")
            self.inputs[layer_id] = inputs[0]
            self.outputs[layer_id] = logits
            self.capture_modes[layer_id] = mode

        return hook

    def clear(self) -> None:
        self.inputs.clear()
        super().clear()
