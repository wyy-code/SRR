# SRR: Routing Analysis Toolkit and Selective Router Repair

Code for the paper's routing analysis toolkit and **Selective Router Repair (SRR)** case study. The toolkit tests what changes after MoE merging and whether a *specified* routing intervention recovers task loss. SRR constructs selective router updates; its fit objective alone does not establish downstream improvement.

## Repository Structure

| Path | Contents |
| --- | --- |
| `src/srr/analysis/` | Crossed input/gate attribution, native route interventions, mixture comparisons, paired task statistics |
| `src/srr/` | SRR calibration cache, pair selection, solver, checkpoint materialization |
| `configs/` | Example SRR and merge configuration |
| `docs/analysis.md` | Analysis contracts, commands, provenance, and interpretation |
| `tests/` | Checkpoint-free numerical and contract tests |

## Installation

Use a Python/CUDA environment compatible with your MoE checkpoint, then install:

```bash
pip install -e .
```

The checkpoint-facing commands may require FlashAttention 2 and architecture-specific model code. The JSONL analysis command is CPU-only.

For CPU-only analysis without installing the model dependencies, run `PYTHONPATH=src python -m srr.analysis.cli` in place of `srr-analyze`.

## Routing analysis

From measured, paired task-item records:

```bash
srr-analyze task --input /path/to/paired_items.jsonl --output /path/to/task_summary.json
```

For aligned 2x2 crossed-route event records:

```bash
srr-analyze routes --input /path/to/crossed_events.jsonl --output /path/to/route_summary.json
```

For measured route metrics versus token-local intervention gains:

```bash
srr-analyze diagnosis --input /path/to/diagnostic_events.jsonl --output /path/to/diagnosis.json
```

The output contains prompt-cluster bootstrap intervals and the SHA-256 of its input. Crossed routes alone describe structural changes, **not** task-level repair. Native intervention hooks and mixture metrics are Python APIs; see [analysis guide](docs/analysis.md) for exact schemas and architecture scope.

Run the checkpoint-free tests with `PYTHONPATH=src python -m unittest discover -s tests -v`. The optional archived-builder parity test additionally requires `SRR_REFERENCE_BUILDER=/path/to/frozen_builder.py`.
The scope of server-side validation and its remaining limits are recorded in [validation notes](docs/validation.md).

## SRR checkpoint construction

Set model and output paths in [`configs/srr.example.json`](configs/srr.example.json), change `status` to `frozen`, and prepare calibration prompts. Then:

```bash
srr-cache --protocol configs/srr.example.json --configuration average --prompts /path/to/prompts.jsonl --gpu 0
srr-build --protocol configs/srr.example.json --configuration average --gpu 0
srr-overlay --candidate /path/to/output/average-srr --output /path/to/output/average-srr-model --gpu 0
```

Each command refuses to overwrite an existing output. Optional Average, Task Arithmetic, TIES, and DARE parent construction uses `srr-merge-baselines` and [`configs/merges.example.json`](configs/merges.example.json).

## Data Format

Each prompt is one JSONL record with a unique `uuid`, a domain `role`, a `split` (`train` or `validation`), and tokenizer-specific `prompt_ids`:

```json
{"uuid":"sample-001","role":"math","split":"train","prompt_ids":[1,42,73]}
```

Model checkpoints, calibration data, and benchmark scores are not included.

## Evaluation

Use the [RoMA evaluation code and configurations](https://github.com/tianyi-lab/RoMA) to evaluate each merged parent and its SRR checkpoint under the same benchmark settings. Keep checkpoint identities, items, scoring rules, and intervention scope paired and explicit.
