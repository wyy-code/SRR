"""Minimal model I/O and native MoE-router access for SRR."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any


def install_legacy_dynamic_cache_compat() -> None:
    """Support pinned DeepSeek remote code expecting older DynamicCache accessors."""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda cache: cache.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda _cache: None
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(cache: Any, new_seq_length: int, layer_idx: int = 0) -> int:
            previous = int(cache.get_seq_length(layer_idx))
            maximum = cache.get_max_length()
            if maximum is not None and previous + int(new_seq_length) > int(maximum):
                return int(maximum) - int(new_seq_length)
            return previous

        DynamicCache.get_usable_length = get_usable_length


def load_model(path: Path, architecture_code_root: Path | None = None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if architecture_code_root is None:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
        )
    else:
        architecture_code_root = architecture_code_root.resolve()
        config_class = get_class_from_dynamic_module(
            "configuration_deepseek.DeepseekV2Config",
            str(architecture_code_root), local_files_only=True,
        )
        config = config_class.from_pretrained(path, local_files_only=True)
        model_class = get_class_from_dynamic_module(
            "modeling_deepseek.DeepseekV2ForCausalLM",
            str(architecture_code_root), local_files_only=True,
        )
        model = model_class.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
        )
    model = model.to("cuda:0").eval()
    model.config.use_cache = False
    return model


def gate_modules(model: Any) -> dict[str, Any]:
    modules = dict(model.named_modules())
    result = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".mlp.gate.weight") and parameter.ndim == 2:
            result[name] = modules[name[: -len(".weight")]]
    if not result:
        raise RuntimeError("no MoE router gate weights found")
    return dict(sorted(result.items()))


def router_weights(model: Any) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    return {
        name: parameters[name].detach().float().cpu().contiguous()
        for name in gate_modules(model)
    }


def router_logits_from_hidden(module: Any, hidden: Any) -> Any:
    import torch.nn.functional as F

    weight = module.weight
    bias = getattr(module, "bias", None)
    if hasattr(module, "n_routed_experts") and hasattr(module, "scoring_func"):
        return F.linear(hidden.float(), weight.float(), None)
    return F.linear(hidden, weight, bias)


def aligned(row: dict) -> tuple[Any, int, int]:
    import torch

    prompt = row["prompt_ids"]
    continuation = row["continuation_ids"]
    ids = torch.tensor([prompt + continuation], dtype=torch.long, device="cuda:0")
    start = len(prompt) - 1
    return ids, start, start + len(continuation)


def clear_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
