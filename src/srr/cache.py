"""Generate the fixed parent continuations consumed by SRR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from .build import finish, read_jsonl, sha256, write_json_exclusive


def generate(model, tokenizer, rows: list[dict], max_new_tokens: int, batch_size: int) -> list[dict]:
    import torch

    tokenizer.padding_side = "left"
    result = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [row["prompt_ids"] for row in batch]
        maximum = max(map(len, prompts))
        ids = torch.full((len(batch), maximum), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, prompt in enumerate(prompts):
            ids[index, -len(prompt):] = torch.tensor(prompt)
            mask[index, -len(prompt):] = 1
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids.to("cuda:0"), attention_mask=mask.to("cuda:0"),
                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        continuations = output[:, maximum:].cpu().tolist()
        for row, continuation in zip(batch, continuations):
            if tokenizer.eos_token_id in continuation:
                continuation = continuation[: continuation.index(tokenizer.eos_token_id) + 1]
            if not continuation:
                continuation = [tokenizer.eos_token_id]
            result.append({**row, "continuation_ids": list(map(int, continuation))})
        print(json.dumps({"event": "trajectory_progress", "rows": len(result), "total": len(rows)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from transformers import AutoTokenizer
    from . import modeling

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise RuntimeError("SRR protocol must be frozen")
    config = protocol["configurations"][args.configuration]
    output = Path(config["parent_trajectory_cache"])
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {building}")

    rows = read_jsonl(args.prompts)
    required = {"uuid", "role", "split", "prompt_ids"}
    if not rows or any(not required <= row.keys() for row in rows):
        raise ValueError("prompt rows require uuid, role, split, and prompt_ids")
    if len({row["uuid"] for row in rows}) != len(rows):
        raise ValueError("prompt UUIDs must be unique")
    if {row["split"] for row in rows} != {"train", "validation"}:
        raise ValueError("both train and validation prompts are required")
    if set(row["role"] for row in rows) != set(protocol["roles"]):
        raise ValueError("prompt roles do not match the protocol")
    if any(not row["prompt_ids"] for row in rows):
        raise ValueError("prompt_ids must be nonempty")

    if protocol.get("legacy_deepseek_cache_compat", False):
        modeling.enable_deepseek_cache_compatibility()
    tokenizer = AutoTokenizer.from_pretrained(
        Path(protocol["base_root"]), trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    architecture_code_root = protocol.get("architecture_code_root")
    model = modeling.load_model(
        Path(config["parent"]), Path(architecture_code_root) if architecture_code_root else None,
    )
    trajectories = generate(
        model, tokenizer, rows,
        max_new_tokens=int(protocol["trajectory"]["max_new_tokens"]),
        batch_size=int(protocol["trajectory"]["batch_size"]),
    )
    del model
    modeling.clear_cuda()

    building.mkdir(parents=True)
    write_json_exclusive(building / "source.json", {
        "protocol_sha256": sha256(protocol_path),
        "prompt_file_sha256": sha256(args.prompts),
        "parent": config["parent"],
        "generation": "greedy, fixed parent continuations",
        "rows": len(trajectories),
    })
    with (building / "round0_trajectories.jsonl").open("x", encoding="utf-8") as handle:
        for row in trajectories:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    finish(building)
    shutil.move(str(building), str(output))
    print(json.dumps({"event": "srr_cache_complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
