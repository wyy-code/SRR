#!/usr/bin/env python3
"""Build Average, Task Arithmetic, TIES, and DARE merged parents.

The implementation is tensor-streaming. TIES obtains the exact global top-K
threshold for each task vector with two 16-bit radix passes, avoiding a dense
30B-parameter flattened allocation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


METHODS = ("average", "task_arithmetic", "ties", "dare")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def eligible(name: str) -> bool:
    lower = name.lower()
    return "embed" not in lower and "lm_head" not in lower


class ModelStore:
    def __init__(self, root: Path, stack: ExitStack):
        self.root = root
        self.stack = stack
        self.mapping: dict[str, str] = {}
        self.safe_handles: dict[str, Any] = {}
        self.bin_states: dict[str, dict[str, torch.Tensor]] = {}
        safe_index = root / "model.safetensors.index.json"
        bin_index = root / "pytorch_model.bin.index.json"
        if safe_index.is_file():
            self.index_payload = json.loads(safe_index.read_text())
            self.mapping = {str(k): str(v) for k, v in self.index_payload["weight_map"].items()}
            self.output_mapping = dict(self.mapping)
        elif (root / "model.safetensors").is_file():
            filename = "model.safetensors"
            handle = self._safe(filename)
            self.mapping = {name: filename for name in handle.keys()}
            self.index_payload = {"metadata": {}, "weight_map": dict(self.mapping)}
            self.output_mapping = dict(self.mapping)
        elif bin_index.is_file():
            self.index_payload = json.loads(bin_index.read_text())
            self.mapping = {str(k): str(v) for k, v in self.index_payload["weight_map"].items()}
            self.output_mapping = dict(self.mapping)
        elif (root / "pytorch_model.bin").is_file():
            filename = "pytorch_model.bin"
            state = self._bin(filename)
            self.mapping = {name: filename for name in state}
            self.index_payload = {"metadata": {}, "weight_map": dict(self.mapping)}
            self.output_mapping = dict(self.mapping)
        else:
            raise FileNotFoundError(f"no supported model weights in {root}")

    def _safe(self, filename: str):
        if filename not in self.safe_handles:
            self.safe_handles[filename] = self.stack.enter_context(
                safe_open(str(self.root / filename), framework="pt", device="cpu")
            )
        return self.safe_handles[filename]

    def _bin(self, filename: str) -> dict[str, torch.Tensor]:
        if filename not in self.bin_states:
            loaded = torch.load(
                self.root / filename, map_location="cpu", weights_only=True, mmap=True
            )
            if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
                loaded = loaded["state_dict"]
            if not isinstance(loaded, dict) or not loaded:
                raise RuntimeError(f"unsupported bin state in {self.root / filename}")
            if not all(torch.is_tensor(value) for value in loaded.values()):
                raise RuntimeError(f"non-tensor bin state in {self.root / filename}")
            self.bin_states[filename] = loaded
        return self.bin_states[filename]

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.mapping[name]
        if filename.endswith(".safetensors"):
            return self._safe(filename).get_tensor(name)
        return self._bin(filename)[name]

    def shape(self, name: str) -> tuple[int, ...]:
        filename = self.mapping[name]
        if filename.endswith(".safetensors"):
            return tuple(self._safe(filename).get_slice(name).get_shape())
        return tuple(self._bin(filename)[name].shape)


def normalized_source_tensor(source: ModelStore, name: str, base: torch.Tensor) -> torch.Tensor:
    value = source.tensor(name)
    if value.shape != base.shape:
        raise RuntimeError(f"shape mismatch for {name}: {value.shape} != {base.shape}")
    if not value.is_floating_point() or not base.is_floating_point():
        if value.dtype != base.dtype or not torch.equal(value, base):
            raise RuntimeError(f"non-floating tensor mismatch for {name}")
        return value
    # MergeBench loads every checkpoint at bfloat16 before constructing task vectors.
    return value.to(dtype=base.dtype).float()


def delta(source: ModelStore, name: str, base: torch.Tensor) -> torch.Tensor:
    return normalized_source_tensor(source, name, base) - base.float()


def bit_parts(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous = values.abs().float().contiguous()
    if not torch.isfinite(contiguous).all():
        raise RuntimeError("non-finite task-vector magnitude")
    bits = contiguous.view(torch.int32).to(torch.int64).bitwise_and_(0xFFFFFFFF)
    return torch.bitwise_right_shift(bits, 16), torch.bitwise_and(bits, 0xFFFF)


def choose_bucket(counts: torch.Tensor, rank: int) -> tuple[int, int]:
    cumulative = torch.cumsum(counts, dim=0)
    matches = torch.nonzero(cumulative >= rank, as_tuple=False)
    if not len(matches):
        raise RuntimeError(f"rank {rank} exceeds histogram population")
    bucket = int(matches[0].item())
    before = int(cumulative[bucket - 1].item()) if bucket else 0
    return bucket, rank - before


def exact_ties_thresholds(
    base: ModelStore,
    sources: list[ModelStore],
    names: list[str],
    keep_fraction: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    high_histograms = [torch.zeros(65536, dtype=torch.int64) for _ in sources]
    populations = [0 for _ in sources]
    for number, name in enumerate(names, start=1):
        base_tensor = base.tensor(name)
        if not eligible(name) or not base_tensor.is_floating_point():
            continue
        for index, source in enumerate(sources):
            high, _ = bit_parts(delta(source, name, base_tensor))
            high_histograms[index] += torch.bincount(high.reshape(-1), minlength=65536)
            populations[index] += high.numel()
        print(json.dumps({"event": "ties_high_radix", "tensors": number, "total": len(names)}), flush=True)

    high_buckets = []
    residual_ranks = []
    ranks = []
    for histogram, population in zip(high_histograms, populations):
        if population <= 0:
            raise RuntimeError("empty TIES population")
        keep = int(population * keep_fraction)
        rank = population - keep
        if rank <= 0 or rank > population:
            raise RuntimeError(f"invalid TIES rank {rank}/{population}")
        bucket, residual = choose_bucket(histogram, rank)
        high_buckets.append(bucket)
        residual_ranks.append(residual)
        ranks.append(rank)

    low_histograms = [torch.zeros(65536, dtype=torch.int64) for _ in sources]
    for number, name in enumerate(names, start=1):
        base_tensor = base.tensor(name)
        if not eligible(name) or not base_tensor.is_floating_point():
            continue
        for index, source in enumerate(sources):
            high, low = bit_parts(delta(source, name, base_tensor))
            chosen = low[high == high_buckets[index]]
            if chosen.numel():
                low_histograms[index] += torch.bincount(chosen.reshape(-1), minlength=65536)
        print(json.dumps({"event": "ties_low_radix", "tensors": number, "total": len(names)}), flush=True)

    thresholds = []
    records = []
    for population, rank, high, residual, histogram in zip(
        populations, ranks, high_buckets, residual_ranks, low_histograms
    ):
        low, _ = choose_bucket(histogram, residual)
        bits = (high << 16) | low
        threshold = struct.unpack("!f", struct.pack("!I", bits))[0]
        thresholds.append(threshold)
        records.append({
            "eligible_elements": population,
            "keep_fraction": keep_fraction,
            "kth_rank_one_indexed": rank,
            "threshold": threshold,
            "threshold_float32_bits_hex": f"0x{bits:08x}",
        })
    return thresholds, records


def ties_majority_sign(
    base: ModelStore,
    sources: list[ModelStore],
    names: list[str],
    thresholds: list[float],
) -> tuple[int, int]:
    signed_sum = 0
    zero_coordinates = 0
    for number, name in enumerate(names, start=1):
        base_tensor = base.tensor(name)
        if not eligible(name) or not base_tensor.is_floating_point():
            continue
        trimmed = []
        for source, threshold in zip(sources, thresholds):
            value = delta(source, name, base_tensor)
            trimmed.append(value * (value.abs() >= threshold))
        signs = torch.sign(sum(trimmed, torch.zeros_like(trimmed[0])))
        signed_sum += int(signs.sum().item())
        zero_coordinates += int((signs == 0).sum().item())
        print(json.dumps({"event": "ties_sign_election", "tensors": number, "total": len(names)}), flush=True)
    majority = 1 if signed_sum > 0 else -1 if signed_sum < 0 else 0
    return majority, zero_coordinates


def dare_seed(seed: int, role: str, name: str) -> int:
    payload = f"{seed}\0{role}\0{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def merge_values(
    name: str,
    base_tensor: torch.Tensor,
    sources: list[ModelStore],
    roles: list[str],
    thresholds: list[float],
    majority_sign: int,
    protocol: dict,
) -> dict[str, torch.Tensor]:
    if not base_tensor.is_floating_point():
        return {method: base_tensor.contiguous() for method in METHODS}
    source_values = [normalized_source_tensor(source, name, base_tensor) for source in sources]
    base_float = base_tensor.float()
    deltas = [value - base_float for value in source_values]
    average = sum(source_values, torch.zeros_like(base_float)).div_(len(sources))
    if eligible(name):
        ta = base_float + float(protocol["methods"]["task_arithmetic"]["scaling_coefficient"]) * (
            sum(deltas, torch.zeros_like(base_float)) / len(sources)
        )
        aggregate = []
        for value, threshold in zip(deltas, thresholds):
            aggregate.append(value * (value.abs() >= threshold))
        signs = torch.sign(sum(aggregate, torch.zeros_like(base_float)))
        if majority_sign:
            signs[signs == 0] = majority_sign
        ties = torch.zeros_like(base_float)
        for value in aggregate:
            ties.add_(value * torch.where(signs > 0, value > 0, value < 0))
        ties.add_(base_float)
        keep_probability = 1.0 - float(protocol["methods"]["dare"]["drop_probability"])
        dare = torch.zeros_like(base_float)
        for role, value in zip(roles, deltas):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(dare_seed(int(protocol["methods"]["dare"]["seed"]), role, name))
            mask = torch.bernoulli(torch.full(value.shape, keep_probability), generator=generator)
            dare.add_(value * mask / keep_probability)
        dare.mul_(float(protocol["methods"]["dare"]["scaling_coefficient"])).add_(base_float)
    else:
        ta = base_float
        ties = base_float
        dare = base_float
    dtype = base_tensor.dtype
    return {
        "average": average.to(dtype).contiguous(),
        "task_arithmetic": ta.to(dtype).contiguous(),
        "ties": ties.to(dtype).contiguous(),
        "dare": dare.to(dtype).contiguous(),
    }


def copy_metadata(base: Path, destinations: list[Path]) -> None:
    ignored = {
        "model.safetensors.index.json", "pytorch_model.bin.index.json", "model.safetensors",
        "pytorch_model.bin", "MANIFEST.sha256", ".completed", "DOWNLOAD_RECORD.json",
    }
    for destination in destinations:
        destination.mkdir(parents=True)
        for path in sorted(base.iterdir()):
            if path.name in ignored or path.name.endswith((".safetensors", ".bin")) or path.name == ".cache":
                continue
            target = destination / path.name
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)


def finish(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256", ".completed"}:
            rows.append(f"{sha256(path)}  {path.relative_to(root)}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")
    (root / ".completed").write_text("completed\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise RuntimeError("merge protocol must be frozen")
    family = protocol["families"][args.family]
    base_root = Path(family["base_root"])
    roles = list(family["role_order"])
    source_roots = [Path(family["sources"][role]) for role in roles]
    output_root = Path(family["output_root"])
    building_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or building_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root} or {building_root}")
    building_root.mkdir(parents=True)
    method_roots = {method: building_root / method for method in METHODS}

    with ExitStack() as stack:
        base = ModelStore(base_root, stack)
        sources = [ModelStore(root, stack) for root in source_roots]
        names = sorted(base.mapping)
        for role, source in zip(roles, sources):
            if set(source.mapping) != set(names):
                difference = sorted(set(source.mapping).symmetric_difference(names))[:20]
                raise RuntimeError(f"tensor inventory mismatch for {role}: {difference}")
            for name in names:
                if source.shape(name) != base.shape(name):
                    raise RuntimeError(f"tensor shape mismatch for {role}:{name}")

        thresholds, threshold_records = exact_ties_thresholds(
            base, sources, names, float(protocol["methods"]["ties"]["K"])
        )
        majority_sign, zero_sign_coordinates = ties_majority_sign(
            base, sources, names, thresholds
        )
        copy_metadata(base_root, list(method_roots.values()))
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, shard in base.output_mapping.items():
            by_shard[shard].append(name)
        for shard_number, (base_shard, shard_names) in enumerate(sorted(by_shard.items()), start=1):
            payloads = {method: {} for method in METHODS}
            output_name = base_shard.replace(".bin", ".safetensors")
            for name in sorted(shard_names):
                merged = merge_values(
                    name, base.tensor(name), sources, roles, thresholds, majority_sign, protocol
                )
                for method in METHODS:
                    payloads[method][name] = merged[method]
            for method in METHODS:
                temporary = method_roots[method] / f".{output_name}.partial"
                save_file(payloads[method], str(temporary))
                os.rename(temporary, method_roots[method] / output_name)
            print(json.dumps({
                "event": "external_merge_shard_complete", "family": args.family,
                "shard": shard_number, "total": len(by_shard), "source_shard": base_shard,
            }, sort_keys=True), flush=True)

        output_mapping = {
            name: shard.replace(".bin", ".safetensors") for name, shard in base.output_mapping.items()
        }
        index_payload = dict(base.index_payload)
        index_payload["weight_map"] = output_mapping
        for method in METHODS:
            (method_roots[method] / "model.safetensors.index.json").write_text(
                json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
            )
            write_json(method_roots[method] / "merge.json", {
                "schema_version": 1, "status": "completed", "completed_at": now(),
                "family": args.family, "method": method, "parameters": protocol["methods"][method],
                "base_root": str(base_root.resolve()),
                "source_roots": [str(path.resolve()) for path in source_roots],
                "role_order": roles, "merged_tensor_scope": "full model for Average; non-embedding/non-lm-head for delta methods",
                "source_load_dtype": "base dtype, matching MergeBench bfloat16 loading",
                "ties_thresholds": dict(zip(roles, threshold_records)) if method == "ties" else None,
                "ties_zero_sign_majority": majority_sign if method == "ties" else None,
                "ties_zero_sign_coordinates": zero_sign_coordinates if method == "ties" else None,
                "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
                "benchmark_scores_read": False,
            })
            finish(method_roots[method])
        write_json(building_root / "FAMILY_MERGE_SUMMARY.json", {
            "status": "completed", "family": args.family, "methods": list(METHODS),
            "completed_at": now(), "benchmark_scores_read": False,
        })
    os.rename(building_root, output_root)
    print(json.dumps({"event": "external_family_merges_complete", "family": args.family}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
