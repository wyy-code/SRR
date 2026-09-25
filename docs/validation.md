# Validation scope

The public code was checked in an isolated server clone with PyTorch 2.3 and Transformers 4.46, including a CUDA tensor smoke test. Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m srr.analysis.cli --help
PYTHONPATH=src python -m srr.cache --help
PYTHONPATH=src python -m srr.build --help
PYTHONPATH=src python -m srr.overlay --help
```

The tests cover SRR pair selection and weighted ridge writeback; crossed input/gate attribution; native routing mass; router-logit/input capture; single-event no-op and changed-route forward hooks; batch-row isolation; CUDA tensor operations; paired task summaries; matched random controls; diagnostic AUROC; and non-overwriting CLI output.

These are **checkpoint-free contract tests**, not a claim that any paper table has been reproduced from raw checkpoints. Core SRR functions (`prompt_profile`, `split_summary`, `candidate_edges`, and `build_layer_system`) were also compared numerically with the archived builder identified by SHA-256 `05e5a53b8545450ed8bac88802efc10ca346cd6bc1db5b187a80d1dd0161e718`; all tested outputs matched exactly. The optional repository test using `SRR_REFERENCE_BUILDER` is skipped unless that archived file is supplied. Full architecture-specific reproduction still requires the exact checkpoint identities, tokenizer, frozen prompts, route caches, item-level scores, runtime implementation, and benchmark protocol. The repository intentionally does not include private model weights or benchmark outputs.
