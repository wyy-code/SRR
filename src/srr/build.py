#!/usr/bin/env python3
"""Build a Selective Router Repair (SRR) candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_manifest(root: Path) -> None:
    if not (root / ".completed").is_file():
        raise RuntimeError(f"incomplete artifact: {root}")
    if not (root / "MANIFEST.sha256").is_file():
        raise RuntimeError(f"missing manifest: {root}")
    for row in (root / "MANIFEST.sha256").read_text().splitlines():
        if not row.strip():
            continue
        expected, relative = row.split("  ", 1)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"manifest mismatch: {path}")


def finish(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256", ".completed"}:
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")
    (root / ".completed").write_text("completed\n")


def layer_number(key: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", key)
    if not match:
        raise RuntimeError(f"cannot parse router layer: {key}")
    return int(match.group(1))


def final_five(keys: list[str]) -> list[str]:
    return sorted(keys, key=layer_number)[-5:]


class RouterCapture:
    def __init__(self, modules: dict[str, Any], capture_inputs: bool):
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self.output_hook(name, capture_inputs)))

    def output_hook(self, name: str, capture_inputs: bool):
        def capture(module: Any, inputs: tuple[Any, ...], _output: Any) -> None:
            from .modeling import router_logits_from_hidden

            hidden = inputs[0]
            if capture_inputs:
                self.inputs[name] = hidden
            self.outputs[name] = router_logits_from_hidden(module, hidden)

        return capture

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def centered_lift(first_logits: Any, second_logits: Any) -> Any:
    import torch

    lift = torch.log_softmax(first_logits.float(), dim=-1) - torch.log_softmax(second_logits.float(), dim=-1)
    return lift - lift.mean(dim=-1, keepdim=True)


def token_nll(logits: Any, row: dict, start: int, stop: int) -> Any:
    import torch
    import torch.nn.functional as F

    labels = torch.tensor(row["continuation_ids"], dtype=torch.long, device=logits.device)
    predictive = logits.reshape(-1, logits.shape[-1])[start:stop].float()
    if predictive.shape[0] != labels.numel():
        raise RuntimeError("continuation-token NLL alignment failed")
    return F.cross_entropy(predictive, labels, reduction="none")


def capture_rows(common: Any, model: Any, rows: list[dict], keys: list[str], capture_inputs: bool, label: str) -> dict[str, dict]:
    import torch

    modules = common.gate_modules(model)
    if not set(keys) <= set(modules):
        raise RuntimeError(f"router inventory mismatch: {label}")
    capture = RouterCapture({key: modules[key] for key in keys}, capture_inputs)
    traces: dict[str, dict] = {}
    try:
        for number, row in enumerate(rows, start=1):
            ids, start, stop = common.prepare_continuation_inputs(row)
            capture.clear()
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            route_logits = {
                key: capture.outputs[key].reshape(-1, capture.outputs[key].shape[-1])[start:stop].float().cpu().contiguous()
                for key in keys
            }
            trace = {
                "nll": token_nll(output.logits, row, start, stop).float().cpu().contiguous(),
                "route_logits": route_logits,
            }
            if capture_inputs:
                trace["hidden"] = {
                    key: capture.inputs[key].reshape(-1, capture.inputs[key].shape[-1])[start:stop].float().cpu().contiguous()
                    for key in keys
                }
            traces[row["uuid"]] = trace
            print(json.dumps({"event": "lc_trace", "label": label, "done": number, "total": len(rows)}), flush=True)
            del output
    finally:
        capture.close()
    return traces


def prompt_profile(source: dict, base: dict, key: str, temperature: float) -> tuple[Any, Any]:
    import torch

    source_logits = source["route_logits"][key]
    base_logits = base["route_logits"][key]
    reliability = torch.sigmoid((base["nll"].float() - source["nll"].float()) / temperature)
    source_probability = torch.softmax(source_logits.float(), dim=-1)
    base_probability = torch.softmax(base_logits.float(), dim=-1)
    denominator = reliability.sum().clamp_min(1e-8)
    profile = (centered_lift(source_logits, base_logits) * reliability[:, None]).sum(0) / denominator
    activity = (torch.maximum(source_probability, base_probability) * reliability[:, None]).sum(0) / denominator
    return profile, activity


def split_summary(profiles: list[tuple[Any, Any]]) -> dict[str, Any]:
    import torch

    lifts = torch.stack([row[0] for row in profiles])
    activities = torch.stack([row[1] for row in profiles])
    return {
        "mean": lifts.mean(0),
        "positive_fraction": (lifts > 0).float().mean(0),
        "negative_fraction": (lifts < 0).float().mean(0),
        "activity": activities.mean(0),
    }


def candidate_edges(train: dict[str, Any], validation: dict[str, Any], consistency: float, min_activity: float) -> list[dict]:
    experts = int(train["mean"].numel())
    promoted = []
    demoted = []
    for expert in range(experts):
        activity = min(float(train["activity"][expert]), float(validation["activity"][expert]))
        if activity < min_activity:
            continue
        if (
            float(train["mean"][expert]) > 0
            and float(validation["mean"][expert]) > 0
            and float(train["positive_fraction"][expert]) >= consistency
            and float(validation["positive_fraction"][expert]) >= consistency
        ):
            promoted.append(expert)
        if (
            float(train["mean"][expert]) < 0
            and float(validation["mean"][expert]) < 0
            and float(train["negative_fraction"][expert]) >= consistency
            and float(validation["negative_fraction"][expert]) >= consistency
        ):
            demoted.append(expert)
    edges = []
    for promote in promoted:
        for demote in demoted:
            if promote == demote:
                continue
            train_margin = float(train["mean"][promote] - train["mean"][demote])
            validation_margin = float(validation["mean"][promote] - validation["mean"][demote])
            sign_floor = min(
                float(train["positive_fraction"][promote]),
                float(validation["positive_fraction"][promote]),
                float(train["negative_fraction"][demote]),
                float(validation["negative_fraction"][demote]),
            )
            activity_floor = min(
                float(train["activity"][promote]),
                float(validation["activity"][promote]),
                float(train["activity"][demote]),
                float(validation["activity"][demote]),
            )
            score = min(train_margin, validation_margin) * sign_floor * math.sqrt(activity_floor)
            edges.append(
                {
                    "promote": promote,
                    "demote": demote,
                    "score": score,
                    "train_margin": train_margin,
                    "validation_margin": validation_margin,
                    "sign_consistency_floor": sign_floor,
                    "route_activity_floor": activity_floor,
                }
            )
    return sorted(edges, key=lambda edge: (-edge["score"], edge["promote"], edge["demote"]))


def select_edges(candidates: dict[str, list[dict]], roles: list[str], edges_per_role: int) -> tuple[dict[str, list[dict]], set[int]]:
    selected = {role: [] for role in roles}
    used: set[int] = set()
    for _ in range(edges_per_role):
        for role in roles:
            for edge in candidates[role]:
                endpoints = {int(edge["promote"]), int(edge["demote"])}
                if endpoints & used:
                    continue
                selected[role].append(edge)
                used |= endpoints
                break
    return selected, used


def build_layer_system(
    rows: list[dict],
    key: str,
    roles: list[str],
    edges: dict[str, list[dict]],
    base_traces: dict[str, dict],
    source_traces: dict[str, dict],
    parent_traces: dict[str, dict],
    temperature: float,
) -> dict[str, Any]:
    import torch

    hidden = torch.cat([parent_traces[row["uuid"]]["hidden"][key] for row in rows], dim=0)
    parent_logits = torch.cat([parent_traces[row["uuid"]]["route_logits"][key] for row in rows], dim=0)
    token_roles = []
    source_parent = []
    source_base = []
    reliabilities = []
    for row in rows:
        uuid = row["uuid"]
        count = len(row["continuation_ids"])
        token_roles.extend([row["role"]] * count)
        source_parent.append(centered_lift(source_traces[uuid]["route_logits"][key], parent_traces[uuid]["route_logits"][key]))
        source_base.append(centered_lift(source_traces[uuid]["route_logits"][key], base_traces[uuid]["route_logits"][key]))
        advantage = parent_traces[uuid]["nll"].float() - source_traces[uuid]["nll"].float()
        reliabilities.append(torch.where(advantage > 0, torch.sigmoid(advantage / temperature), torch.zeros_like(advantage)))
    source_parent_tensor = torch.cat(source_parent, dim=0)
    source_base_tensor = torch.cat(source_base, dim=0)
    reliability = torch.cat(reliabilities, dim=0)
    flat_edges = [{"role": role, **edge} for role in roles for edge in edges[role]]
    response = torch.zeros((hidden.shape[0], len(flat_edges)), dtype=torch.float32)
    weights = torch.zeros_like(response)
    clipped_count = 0
    target_count = 0
    for index, edge in enumerate(flat_edges):
        matching = torch.tensor([role == edge["role"] for role in token_roles], dtype=torch.bool)
        target = source_parent_tensor[:, edge["promote"]] - source_parent_tensor[:, edge["demote"]]
        bound = (source_base_tensor[:, edge["promote"]] - source_base_tensor[:, edge["demote"]]).abs()
        clipped = target.sign() * torch.minimum(target.abs(), bound)
        clipped_count += int((clipped[matching].abs() + 1e-12 < target[matching].abs()).sum())
        target_count += int(matching.sum())
        response[matching, index] = clipped[matching]
        weights[matching, index] = reliability[matching]
    if bool((weights.sum(0) <= 0).any()):
        raise RuntimeError(f"SRR pair lacks positive source-over-parent NLL support: {key}")
    return {
        "hidden": hidden,
        "parent_logits": parent_logits,
        "response": response,
        "weights": weights,
        "flat_edges": flat_edges,
        "positive_reliability_fraction": float((reliability > 0).float().mean()),
        "clipped_count": clipped_count,
        "target_count": target_count,
    }


def pair_row_update(vectors: Any, flat_edges: list[dict], shape: tuple[int, int], scale: float) -> Any:
    import torch

    experts, hidden = shape
    raw = torch.zeros((experts, hidden), dtype=torch.float32)
    endpoints = set()
    for index, edge in enumerate(flat_edges):
        promote = int(edge["promote"])
        demote = int(edge["demote"])
        endpoints |= {promote, demote}
        raw[promote].add_(0.5 * vectors[index])
        raw[demote].sub_(0.5 * vectors[index])
    mask = torch.zeros(experts, dtype=torch.bool)
    mask[list(endpoints)] = True
    if not bool((raw[~mask] == 0).all()):
        raise RuntimeError("SRR escaped sparse endpoints")
    for edge in flat_edges:
        if not torch.equal(raw[edge["promote"]] + raw[edge["demote"]], torch.zeros(hidden)):
            raise RuntimeError("SRR pair gauge failed")
    return raw, (scale * raw).contiguous(), endpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from safetensors.torch import load_file, save_file
    from . import modeling as common
    from . import solver

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise RuntimeError("SRR protocol must be frozen before candidate construction")
    if protocol.get("legacy_deepseek_cache_compat", False):
        common.enable_deepseek_cache_compatibility()

    config = protocol["configurations"][args.configuration]
    source_run = Path(config["parent_trajectory_cache"])
    verify_manifest(source_run)
    output = Path(config["srr_candidate_root"])
    building = output.with_name(output.name + ".building")
    if output.exists() or building.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {building}")
    building.mkdir(parents=True)
    write_json_exclusive(
        building / "launch.json",
        {
            "pid": os.getpid(),
            "gpu": args.gpu,
            "configuration": args.configuration,
            "protocol_sha256": sha256(protocol_path),
            "benchmark_scores_read": False,
        },
    )

    settings = protocol["srr"]
    torch.manual_seed(int(settings["seed"]))
    torch.cuda.manual_seed_all(int(settings["seed"]))
    torch.set_num_threads(int(settings["cpu_threads"]))
    rows = read_jsonl(source_run / "round0_trajectories.jsonl")
    roles = list(protocol["roles"])
    if sorted({row["role"] for row in rows}) != sorted(roles):
        raise RuntimeError("trajectory roles differ from frozen teachers")
    if len({row["uuid"] for row in rows}) != len(rows):
        raise RuntimeError("trajectory UUIDs are not unique")
    if {row["split"] for row in rows} != {"train", "validation"}:
        raise RuntimeError("both train and validation trajectories are required")

    parent = Path(config["parent"])
    architecture_code_root = protocol.get("architecture_code_root")
    architecture_code_root = Path(architecture_code_root) if architecture_code_root else None
    model = common.load_model(parent, architecture_code_root)
    bases_all = common.router_weights(model)
    keys = final_five(list(bases_all))
    del model
    common.clear_cuda()

    base_model = common.load_model(Path(protocol["base_root"]), architecture_code_root)
    base_traces = capture_rows(common, base_model, rows, keys, False, "base")
    del base_model
    common.clear_cuda()

    source_traces: dict[str, dict] = {}
    for role in roles:
        role_rows = [row for row in rows if row["role"] == role]
        source_model = common.load_model(Path(protocol["sources"][role]), architecture_code_root)
        source_traces.update(capture_rows(common, source_model, role_rows, keys, False, f"source:{role}"))
        del source_model
        common.clear_cuda()

    lc = settings
    edge_audit = []
    selected_by_key: dict[str, dict[str, list[dict]]] = {}
    for key in keys:
        candidates_by_role = {}
        split_audit = {}
        for role in roles:
            summaries = {}
            for split in ("train", "validation"):
                profiles = [
                    prompt_profile(source_traces[row["uuid"]], base_traces[row["uuid"]], key, float(lc["reliability_temperature"]))
                    for row in rows
                    if row["role"] == role and row["split"] == split
                ]
                if not profiles:
                    raise RuntimeError(f"missing {role}:{split} profiles")
                summaries[split] = split_summary(profiles)
            candidates_by_role[role] = candidate_edges(
                summaries["train"],
                summaries["validation"],
                float(lc["prompt_sign_consistency"]),
                float(lc["min_route_activity"]),
            )
            split_audit[role] = {
                "train_prompts": sum(row["role"] == role and row["split"] == "train" for row in rows),
                "validation_prompts": sum(row["role"] == role and row["split"] == "validation" for row in rows),
                "candidate_edges": len(candidates_by_role[role]),
            }
        selected, used = select_edges(candidates_by_role, roles, int(lc["edges_per_role"]))
        if any(len(selected[role]) != int(lc["edges_per_role"]) for role in roles):
            raise RuntimeError(f"insufficient split-stable disjoint expert pairs: {key}")
        selected_by_key[key] = selected
        edge_audit.append(
            {
                "router_key": key,
                "layer": layer_number(key),
                "selected_edges": selected,
                "selected_endpoint_count": len(used),
                "endpoints_disjoint": len(used) == 2 * int(lc["edges_per_role"]) * len(roles),
                "splits": split_audit,
            }
        )
    write_json_exclusive(building / "stable_pair_audit.json", {"status": "passed", "layers": edge_audit})

    parent_model = common.load_model(parent, architecture_code_root)
    parent_traces = capture_rows(common, parent_model, rows, keys, True, "parent")
    del parent_model
    common.clear_cuda()

    train_rows = [row for row in rows if row["split"] == "train"]
    scaled_deltas = {}
    layer_audits = []
    scale = float(lc["scale"])
    for key in keys:
        system = build_layer_system(
            train_rows,
            key,
            roles,
            selected_by_key[key],
            base_traces,
            source_traces,
            parent_traces,
            float(lc["reliability_temperature"]),
        )
        result = solver.solve_srr_weighted_ridge(
            hidden=system["hidden"].to("cuda:0"),
            response=system["response"].to("cuda:0"),
            weights=system["weights"].to("cuda:0"),
            owned_rows=torch.ones(system["response"].shape[1], dtype=torch.bool, device="cuda:0"),
            ridge_relative=float(lc["ridge_relative"]),
            max_iterations=int(lc["cg_max_iterations"]),
            tolerance=float(lc["cg_tolerance"]),
            residual_refresh_interval=int(lc["cg_residual_refresh_interval"]),
        )
        if result.audit["converged_rows"] != result.audit["owned_rows"]:
            raise RuntimeError(f"SRR weighted ridge solve did not converge: {key}")
        vectors = result.delta_weight.float().cpu()
        raw, scaled_deltas[key], endpoints = pair_row_update(
            vectors, system["flat_edges"], tuple(bases_all[key].shape), scale
        )
        layer_audits.append(
            {
                "router_key": key,
                "layer": layer_number(key),
                "train_tokens": int(system["hidden"].shape[0]),
                "positive_parent_reliability_fraction": system["positive_reliability_fraction"],
                "clipped_fraction": system["clipped_count"] / max(system["target_count"], 1),
                "selected_endpoint_count": len(endpoints),
                "unselected_rows_exact_zero": True,
                "pair_endpoint_sum_exact_zero": True,
                "solver": result.audit,
                "raw_delta_l2": float(torch.linalg.vector_norm(raw)),
                "scaled_delta_l2": float(torch.linalg.vector_norm(scaled_deltas[key])),
            }
        )
        common.clear_cuda()

    candidate = {name: value.clone() for name, value in bases_all.items()}
    for key in keys:
        candidate[key] = (candidate[key].float() + scaled_deltas[key]).contiguous()
    changed = [name for name in bases_all if not torch.equal(candidate[name].to(torch.bfloat16), bases_all[name].to(torch.bfloat16))]
    if not changed or set(changed) != set(keys):
        raise RuntimeError(f"SRR BF16 change scope mismatch: {changed}")

    validation_rows = [row for row in rows if row["split"] == "validation"]
    validation_tokens = sum(len(row["continuation_ids"]) for row in validation_rows)
    if validation_tokens == 0:
        raise RuntimeError("empty validation trajectory set")
    dz_squared = 0.0
    for key in keys:
        delta = candidate[key].float() - bases_all[key].float()
        covariance = torch.zeros((delta.shape[1], delta.shape[1]), dtype=torch.float32)
        for row in validation_rows:
            hidden = parent_traces[row["uuid"]]["hidden"][key]
            covariance.add_(hidden.T @ hidden)
        covariance.div_(validation_tokens)
        dz_squared += float(((delta.matmul(covariance)) * delta).double().sum())
    dz = math.sqrt(max(dz_squared, 0.0))
    covariance_path = source_run / "parent_covariances.safetensors"
    if covariance_path.is_file():
        cached = load_file(str(covariance_path))
        for key in keys:
            direct = torch.zeros_like(cached[key])
            for row in validation_rows:
                hidden = parent_traces[row["uuid"]]["hidden"][key]
                direct.add_(hidden.T @ hidden)
            direct.div_(validation_tokens)
            if not torch.allclose(direct, cached[key], rtol=1e-3, atol=1e-4):
                raise RuntimeError(f"cached parent covariance mismatch: {key}")
    budget_path = source_run / "budget.json"
    current_bp = float(json.loads(budget_path.read_text())["B_p"]) if budget_path.is_file() else None
    checkpoint = building / "router_master_fp32.safetensors"
    save_file({name: value.float().cpu().contiguous() for name, value in candidate.items()}, str(checkpoint))
    write_json_exclusive(
        building / "candidate.json",
        {
            "status": "completed",
            "configuration": args.configuration,
            "method": "srr",
            "long_name": "Selective Router Repair",
            "variant": "parent_residual_parent_reliable_clipped",
            "active_router_keys": keys,
            "changed_router_keys_bf16": changed,
            "parent": str(parent),
            "base": protocol["base_root"],
            "sources": protocol["sources"],
            "parent_trajectory_cache": str(source_run),
            "parent_trajectory_manifest_sha256": sha256(source_run / "MANIFEST.sha256"),
            "scale": scale,
            "current_parent_covariance_Dz": dz,
            "current_functional_repair_B_p_for_reference_only": current_bp,
            "layer_audits": layer_audits,
            "single_seed": True,
            "benchmark_scores_read_during_construction": False,
            "trajectory_source": "fixed parent-generated continuations",
            "protocol_sha256": sha256(protocol_path),
        },
    )
    finish(building)
    shutil.move(str(building), str(output))
    print(json.dumps({"event": "srr_candidate_complete", "configuration": args.configuration, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
