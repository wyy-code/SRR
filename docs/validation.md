# Validation scope

The public code was checked in an isolated server clone with PyTorch 2.3 and Transformers 4.46, including a CUDA tensor smoke test. Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m srr.analysis.cli --help
PYTHONPATH=src python -m srr.cache --help
PYTHONPATH=src python -m srr.build --help
PYTHONPATH=src python -m srr.overlay --help
PYTHONPATH=src python -m srr.analysis.checkpoint --help
```

The tests cover SRR pair selection and weighted ridge writeback; crossed input/gate attribution; native routing mass; router-logit/input capture; single-event no-op and changed-route forward hooks; batch-row isolation; CUDA tensor operations; paired task summaries; matched random controls; diagnostic AUROC; and non-overwriting CLI output.

The server runtime used for these checks did not have `flash_attn`. The loader now fails early with a clear message under its unchanged FlashAttention-2 default. `SRR_ATTN_IMPLEMENTATION=sdpa` or `eager` is an explicit portability override, not a reproduction of the paper attention backend.
On the server's existing OLMoE Average checkpoint, a read-only `sdpa` smoke test captured full logits from a 64-expert gate and reported zero maximum output-logit change and zero routed-output recomputation error for `native_noop`. That environment also required `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to work around an unrelated ONNX/protobuf import conflict. No benchmark scores were read or produced.

These are **checkpoint-free contract tests**, not a claim that any paper table has been reproduced from raw checkpoints. Core SRR functions (`prompt_profile`, `split_summary`, `candidate_edges`, and `build_layer_system`) were also compared numerically with the archived builder identified by SHA-256 `05e5a53b8545450ed8bac88802efc10ca346cd6bc1db5b187a80d1dd0161e718`; all tested outputs matched exactly. The optional repository test using `SRR_REFERENCE_BUILDER` is skipped unless that archived file is supplied. Full architecture-specific reproduction still requires the exact checkpoint identities, tokenizer, frozen prompts, route caches, item-level scores, runtime implementation, and benchmark protocol. The repository intentionally does not include private model weights or benchmark outputs.
