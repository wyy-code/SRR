"""Materialize an SRR router overlay on its merged parent checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .build import sha256, verify_manifest, write_json_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--architecture-code-root", type=Path)
    parser.add_argument("--legacy-deepseek-cache-compat", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from . import modeling

    candidate = args.candidate.resolve()
    verify_manifest(candidate)
    record = json.loads((candidate / "candidate.json").read_text())
    if record["method"] != "srr" or record["status"] != "completed":
        raise RuntimeError("not a completed SRR candidate")
    parent = Path(record["parent"]).resolve()
    output = args.output.resolve()
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {building}")

    if args.legacy_deepseek_cache_compat:
        modeling.install_legacy_dynamic_cache_compat()
    model = modeling.load_model(parent, args.architecture_code_root)
    overlay = load_file(str(candidate / "router_master_fp32.safetensors"))
    gates = modeling.gate_modules(model)
    if set(overlay) != set(gates):
        raise RuntimeError("overlay router inventory differs from merged parent")
    changed = []
    with torch.no_grad():
        for key, module in gates.items():
            original = module.weight.detach()
            target = overlay[key].to(device=original.device, dtype=original.dtype)
            if target.shape != original.shape:
                raise RuntimeError(f"router shape mismatch: {key}")
            if not torch.equal(original, target):
                changed.append(key)
            original.copy_(target)
    if set(changed) != set(record["changed_router_keys_bf16"]):
        raise RuntimeError("BF16 changed-router scope differs from candidate audit")

    building.mkdir(parents=True)
    model.save_pretrained(building, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(parent, trust_remote_code=True, local_files_only=True)
    tokenizer.save_pretrained(building)
    write_json_x(building / "SRR_OVERLAY.json", {
        "method": "srr",
        "parent": str(parent),
        "candidate_manifest_sha256": sha256(candidate / "MANIFEST.sha256"),
        "changed_router_keys_bf16": changed,
        "all_other_parameters_untouched_in_memory": True,
    })
    building.rename(output)
    print(json.dumps({"event": "srr_overlay_complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
