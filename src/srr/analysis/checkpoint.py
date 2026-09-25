"""Read-only checkpoint smoke test for native MoE routing interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--case", choices=("deepseek", "olmoe", "qwen3"), required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token-ids", default="1,2,3", help="comma-separated local token IDs")
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--architecture-code-root", type=Path)
    parser.add_argument("--attention-implementation", choices=("flash_attention_2", "sdpa", "eager"))
    parser.add_argument("--roundtrip-rtol", type=float, default=2e-3)
    parser.add_argument("--logit-atol", type=float, default=1e-2)
    args = parser.parse_args()

    import torch
    from srr.modeling import load_model
    from .native_intervention import NativeMoERouteIntervention
    from .router_capture import RouterJacobianCapture

    tokens = [int(value) for value in args.token_ids.split(",")]
    if not tokens or any(value < 0 for value in tokens) or not 0 <= args.position < len(tokens):
        parser.error("token IDs must be nonnegative and position must be within the sequence")
    if args.layer < 0 or args.roundtrip_rtol < 0 or args.logit_atol < 0:
        parser.error("layer and tolerances must be nonnegative")

    model = load_model(args.model, args.architecture_code_root, args.attention_implementation)
    ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
    case = "olmoe" if args.case == "qwen3" else args.case
    with torch.inference_mode():
        with RouterJacobianCapture(model, [args.layer]) as capture:
            baseline = model(input_ids=ids, use_cache=False).logits.float()
            if args.layer not in capture.inputs or args.layer not in capture.outputs:
                raise RuntimeError("router hook did not capture the requested layer")
            if not torch.isfinite(capture.outputs[args.layer]).all():
                raise RuntimeError("non-finite captured router logits")
            capture_mode = capture.capture_modes[args.layer]
            gate_shape = list(capture.outputs[args.layer].shape)

        operator = NativeMoERouteIntervention(model, case, [args.layer])
        try:
            operator.set({
                "layer_id": args.layer,
                "position": args.position,
                "condition": "native_noop",
                "assert_native_recompute_rtol": args.roundtrip_rtol,
            })
            no_op = model(input_ids=ids, use_cache=False).logits.float()
            if len(operator.records) != 1:
                raise RuntimeError("expected exactly one native no-op intervention record")
            max_logit_delta = float((baseline - no_op).abs().max())
            if max_logit_delta > args.logit_atol:
                raise RuntimeError(f"native no-op changed output logits by {max_logit_delta}")
            recompute_error = operator.records[0]["native_routed_recompute_relative_error"]
        finally:
            operator.close()

    print(json.dumps({
        "status": "PASS",
        "model": str(args.model.resolve()),
        "case": args.case,
        "layer": args.layer,
        "capture_mode": capture_mode,
        "gate_shape": gate_shape,
        "max_logit_delta": max_logit_delta,
        "native_recompute_relative_error": recompute_error,
        "attention_implementation": args.attention_implementation or "SRR_ATTN_IMPLEMENTATION/default",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
