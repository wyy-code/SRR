# Routing analysis toolkit

The analysis has three distinct levels. **Crossed routing** asks whether aligned source/merged hidden states or gate weights change expert selection. **Mixture comparison** evaluates expert outputs at one fixed hidden state. **Task-grounded intervention** compares two routed forward passes on identical items while holding non-routing parameters fixed. Neither of the first two establishes task benefit.

## 1. Capture and cross routes

`RouterJacobianCapture` in `srr.analysis.router_capture` records each gate's input and full pre-softmax logits. Run source and merged models on the **same token IDs**. Match prompt ID, token position, sparse-layer ID, tokenizer, and expert ordering before calling `crossed_routes`:

```python
from srr.analysis.routing import crossed_routes, route_origin_masks

routes = crossed_routes(
    source_hidden, merged_hidden,
    source_gate.weight, merged_gate.weight,
    top_k=8,
)
masks = route_origin_masks(routes)
```

The function assumes rank-2 **linear, unbiased** gate weights and aligned hidden-state shapes. It returns routes for source-input/source-gate (SS), merged-input/source-gate (MS), source-input/merged-gate (SM), and merged-input/merged-gate (MM). Its exclusive labels partition only changed SS-versus-MM events: `representation_only`, `gate_only`, `both`, and `neither`. Report their denominators explicitly. `full_distribution_js` operates on all expert probabilities, while `topk_set_distance` measures $1-|S_1\cap S_2|/k$; neither is a task metric.

The CPU summary expects one JSONL record per aligned event:

```json
{"event_id":"prompt-001:layer-7:token-3","cluster_id":"prompt-001","source_source":[0,1],"merged_source":[0,2],"source_merged":[0,1],"merged_merged":[0,2]}
```

`srr-analyze routes --input events.jsonl --output route_summary.json --seed 2026 --repetitions 10000` writes changed-route prevalence and prompt-cluster intervals for each conditional category. The example illustrates the **schema only** and is not paper data.

For Section 3-style metric screening, supply measured full-distribution JS, set distance, and token-local source-route NLL gains as one JSONL record per event, then run `srr-analyze diagnosis`. The `metrics` object must name the same prespecified metric family in every changed-route row:

```json
{"event_id":"prompt-001:layer-7:token-3","cluster_id":"prompt-001","changed_route":true,"source_replay_nll_gain":0.01,"metrics":{"js":0.12,"set_distance":0.5}}
```

It reports tie-aware AUROC with prompt-cluster bootstrap intervals. Positive gain means the specified replay has lower next-token NLL. This is **token-local discrimination**, not a downstream task recovery test. The values above are schema illustrations only, not paper observations. Freeze the metric family before testing; use `holm_adjust` when conducting a prespecified family of hypothesis tests.

## 2. Native interventions and mixture outputs

`NativeMoERouteIntervention` and `NativeMoERouteBatchIntervention` are the original native-forward hooks used for DeepSeekMoE and OLMoE-style MoE blocks. The Qwen3 extension used the OLMoE-style adapter after a native-routing round-trip check. The hooks retain the model's surrounding MoE computation, explicit selected weights and routed mass; `native_noop` and native-recompute tolerances should be checked **before** interpreting any patched pass. The original per-event implementation and batch-row isolation are retained in `srr.analysis.native_intervention` and `srr.analysis.batch_intervention`. They depend on compatible checkpoint implementations; they are not a universal MoE wrapper.

`routed_mixture(expert_outputs, selected, weights)` computes the weighted mixture at the **same** hidden state. `output_comparison` returns cosine and relative L2; `max_entering_leaving_cosine` compares entering/leaving experts. Supply the checkpoint's native weights, including its unnormalized mass where applicable. A high cosine means directionally aligned outputs, not equal magnitudes or unchanged downstream predictions.

`srr.analysis.controls.matched_random_expert_sets(native_selected, observed_selected, expert_count, count=32, seed=...)` supplies fixed-seed random alternatives with the same route size and overlap with the native selected set. It excludes the observed route and fails if it cannot produce enough distinct matches. Apply the same observed routing-weight vector to every control, compute each mixture at the same hidden state, and retain the individual event/control results before taking paired prompt-cluster intervals. This matching controls two structural factors; it does not turn cosine into task accuracy.

## 3. Paired task evaluation

The item-level input is JSONL with a stable item ID, gold choice, and the log-likelihood scores for all choices under the baseline and intervention:

```json
{"item_id":"arc-001","cluster_id":"arc-001","gold_index":1,"baseline_choice_scores":[-3.0,-2.0],"intervention_choice_scores":[-2.8,-1.9]}
```

Use `srr-analyze task --input items.jsonl --output task_summary.json`. The result reports intervention-minus-baseline accuracy **as a fraction**, correct-choice margin, paired percentile CIs, counts of correct answers and changed predictions. Scores must be calculated on the **same** task item/choice sequences; whether to use summed or length-normalized log-likelihood is a benchmark protocol choice and must be fixed before comparison. The CLI does not infer a positive result from a CI containing zero. The example is a schema illustration, not an empirical observation.

`cluster_bootstrap_mean` samples entire prompt clusters, preserving all their tokens/items; `holm_adjust` adjusts a prespecified family of p-values; `binary_auc` is tie-aware. The CLI refuses to overwrite an existing output and includes an input SHA-256. For multiple architectures or parents, run each frozen condition separately and retain the condition/checkpoint manifest alongside its result.

## Source provenance and scope

The SRR construction code remains in `src/srr`. The native intervention modules were taken from the local Section 3 native-order recovery (`native_moe_route_intervention_20260812.py`, SHA-256 `855ca297419db268a895d65c1149e3a7d03dd1b271349dfacede326497208723`; `native_moe_route_batch_intervention_retry02_20260813.py`, SHA-256 `043ea48b2a399e4378ca48839dea1cc9a03a9d6486abd145579c68897995a34f`), with only the package import changed in the batch module. `router_capture.py` came from the SAR-MergeCal submission package (SHA-256 `71fd5a9f99e3e2f74bd257553fb3201b157c2939fece8e085066574fe64010fa`). The reusable crossed-route, mixture, and paired-statistics APIs are factored from the same analysis contracts; they are **not** a claim that this repository contains the original private checkpoints or reproduces every paper table from raw data.

For paper-level reproduction, retain the frozen checkpoint hashes, sample lists, route-cache hashes, intervention settings, benchmark scoring protocol, and matched per-item outputs. No checkpoint, dataset, or score file is bundled here. See the paper appendix for the exact sample counts and estimands; do not equate the full-token origin sample with the final-five-layer hash-selected diagnostic sample.
