# Selective Router Repair (SRR)

SRR is the public name for the **LC-MERGE-RC r2** router-repair implementation used in *Routing Drift Is Not Routing Failure*. It is a router-only intervention on a supplied merged mixture-of-experts (MoE) parent. It is **not** On-Policy Functional Repair: SRR never refreshes the parent continuations or the parent router inputs during fitting.

The release contains the candidate-construction code, a fixed-parent trajectory generator, a checkpoint overlay tool, and optional Average/Task Arithmetic/TIES/DARE parent-merging code. It does **not** contain pretrained model weights, specialist checkpoints, calibration prompts, benchmark data, or empirical results. The code alone makes no benchmark-efficacy claim.

## Method

1. Generate deterministic continuations with the *unrepaired merged parent* on domain-labelled calibration prompts. Split prompts into train and validation before construction.
2. Evaluate each continuation under the base, its matching source specialist, and the merged parent. Build source-versus-base expert profiles using token-likelihood weights. Select split-stable, active positive/negative expert pairs with disjoint endpoints, round-robin across domains.
3. For each selected pair, fit the source-minus-parent router-logit gap on the **original parent hidden states**. Clip each target by the corresponding source-minus-base gap. Only matching-domain tokens where the source has lower token negative log-likelihood than the parent receive positive weight.
4. Solve the independent weighted ridge systems with FP64 preconditioned conjugate gradients. Add equal and opposite scaled corrections to the chosen expert rows in the final five MoE routers. All other model parameters remain unchanged.

The candidate builder does not read benchmark scores. The validation split is used for pair selection and a covariance-based router-logit diagnostic, not for ridge fitting. The reference budget, when present, is recorded but **not** used as a projection constraint. Implementation-to-manuscript correspondence and the frozen numeric settings are in [PROVENANCE.md](docs/PROVENANCE.md).

## Installation

Use a CUDA/PyTorch/Transformers environment compatible with the chosen MoE checkpoint; the pinned DeepSeek remote code may require its architecture-code directory and the optional legacy cache compatibility flag.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

For full runs, install FlashAttention 2 as required by the checkpoints. Do not install model or dataset artifacts into this repository.

## Inputs and run order

Start from [configs/srr.example.json](configs/srr.example.json). Replace all placeholder paths, adapt `roles` and `sources` to the architecture, and change `status` from `draft` to `frozen` **before** producing a candidate. Record the frozen JSON's SHA-256. The OLMoE example uses `math`, `commonsense`, and `reasoning`; the original DeepSeek setup used `math`, `code`, and `commonsense`, and Qwen3-MoE used `math`, `science`, and `math_code`.

Supply a JSONL prompt file outside the repository. Each row must have a unique `uuid`, a domain `role`, `split` equal to `train` or `validation`, and nonempty tokenizer-specific `prompt_ids`. These token IDs must match the base/parent/source tokenizer family. For example:

```json
{"uuid":"sample-001","role":"math","split":"train","prompt_ids":[1,42,73]}
```

```bash
srr-cache --protocol configs/srr.example.json --configuration average \
  --prompts /path/to/calibration-prompts.jsonl --gpu 0
srr-build --protocol configs/srr.example.json --configuration average --gpu 0
srr-overlay --candidate /path/to/output/average-srr \
  --output /path/to/output/average-srr-model --gpu 0
```

The commands above require the example JSON to be customized and frozen first. Outputs are non-overwriting; an existing final or `.building` directory is treated as an error. The candidate contains `router_master_fp32.safetensors`, a pair-selection audit, a numerical-solver audit, and `MANIFEST.sha256`. `srr-overlay` materializes the merged parent with SRR router rows for use by an evaluation runner. For pinned DeepSeek remote code, add `architecture_code_root` and `legacy_deepseek_cache_compat: true` to the protocol, and pass `--architecture-code-root` and `--legacy-deepseek-cache-compat` to the overlay command.

Optional baseline merges can be built with `srr-merge-baselines --protocol configs/merges.example.json --family olmoe` after customizing and freezing that separate JSON. This script reproduces the experimental Average, Task Arithmetic, TIES and DARE merge definitions; these are **parent construction**, not part of SRR fitting.

## Evaluation

Evaluation is deliberately not reimplemented here. Use the [RoMA evaluation code and benchmark configurations](https://github.com/tianyi-lab/RoMA) with the same model family, datasets, prompts, decoding settings, and context lengths for all compared checkpoints. A score from a different dataset or evaluation protocol is not directly comparable. Run and archive a parent endpoint and its SRR overlay under the same configuration; do not select expert pairs or tune SRR with benchmark results.

## Scope

The three demonstrated architecture families are DeepSeekMoE, OLMoE and Qwen3-MoE, provided compatible base/source/parent checkpoints and matching router modules are supplied. `srr-build` loads one model at a time, but traces and dense per-layer solves still need substantial CPU RAM and GPU memory. The package does not include a model-training pipeline or a one-command reproduction of historical private experiments.

License: Apache-2.0.
