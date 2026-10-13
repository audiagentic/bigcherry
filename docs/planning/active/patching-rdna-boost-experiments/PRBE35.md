---
id: PRBE35
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:53.286868+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# AMD-STREAM-005: Protect concurrent-region join node from fusion

## Authoritative disposition — 2026-10-13, llama.cpp b11552

**Pending; standalone correctness-only; opt-in/untested; no speedup claim.** This replaces the duplicated September implementation instructions, central-runner proposal, and 1215-dependent qualification procedure. Historical evidence and earlier designs remain available in Git history and `engines/llamacpp/patches/1216_rd43_concurrent_join_fusion_guard/README.md`. **1215/RD39–42 was rejected** in the later four-session b11126 qualification; do not reuse its earlier positive measurements or reactivate its implementation. PRBE112 owns 1215's remaining diagnostic; PRBE35 owns only 1216's native concurrent-region join safety.

### Pinned implementation and dataflow

- Pinned llama.cpp b11552 = `23b0202a189c44a54625aadcb37a946dd1d6278d`. Its `ggml/src/ggml-cuda/ggml-cuda.cu` is byte-identical to upstream `master` at this audit. `ggml_cuda_graph_evaluate_and_capture` restores concurrent-region node order, forks streams through `try_launch_concurrent_event`, detects `concurrent_event->join_node`, records/waits join events, and calls `ggml_cuda_try_fuse` before `i += nodes_to_skip`. Upstream has no join-fusion cap. The two old-string anchors in 1216's `patch.py` each match b11552 exactly once (static only, no applied-patch build).
- `1216/patch.py` inserts `exec_node_idx` (tensor pointer -> execution index) after order restoration. For a live region it finds `join_idx`, saves `cgraph->n_nodes`, caps the fusion horizon at `join_idx`, calls `ggml_cuda_try_fuse`, restores the original node count, and asserts no returned fusion skip crosses the join. The cap must precede fusion: joining after a fused op has already read the shared-expert result would not repair ordering. No additional scheduler, allocator, stream, cache, or dispatch table is justified.
- This creates an `O(cgraph->n_nodes)` host map and allocation when native concurrent events are launchable. It duplicates some traversal work in the upstream restore-order block, but that earlier index map predates the restore and cannot automatically be reused after reorder. Cost is theoretical; no CPU/capture timing or inference speedup was measured.
- **Canonical package identity:** `1216/patch.toml` has `requires = []`, `state = "untested"`, and `experiment-contract = "RD43-CONCURRENT-JOIN-FUSION-GUARD"`. `tools/bigcherry/patch/registry.py::_packaged_descriptor` reads manifest `record["requires"]`; `patchset.py::_module_from_descriptor` forwards it. Legacy `1216/patch.py::REQUIRES = ("1215_...",)` is stale, non-authoritative source metadata. A future *tested* patch metadata cleanup should remove it; **do not** restore the rejected 1215 dependency.
- `1216/validation.toml` and `validation/producer.py::run` already exist. Current producer uses validated BC as control, validated BC + 1216 as subject, `GGML_CUDA_GRAPH_OPT=1` for both, `full_vocab.compare_servers(criterion=NEAR_LOSSLESS)`, subject-only trace-marker activation, and `run_paired_llama_benchmark` for 10 `tg128` control rounds. It requires standard control/subject llama-server and llama-bench binaries. The `validation/producer.toml` comment claiming a +1215 control and skipped scaffold is obsolete. Do not add a central `run_rd43_contract_qualification` or a second producer.
- `config/experiment-contracts.toml` puts `prerequisites = ["RD39-42-STREAM-MOE-OVERLAP"]` under `[contract.RD43-CONCURRENT-JOIN-FUSION-GUARD.source]`, while `experiment/contract.py::parse_contract` reads prerequisites only from the **contract top level**. Thus that nested entry does not enforce a prerequisite. Treat it as stale provenance text, not permission to re-enable 1215; reconcile via the existing contract owner with a parser/contract-hash fixture.

### Source-confirmed limitation of the activation marker

`BIGCHERRY_PATCH_HIT patch=1216_rd43 path=join_fusion_cap` is emitted when `is_concurrent_event_active && join_idx > i` and tracing is enabled. It does **not** ask whether an uncapped `ggml_cuda_try_fuse` would cross the join. Therefore the marker proves guard-path execution, **not** a prevented capture abort or a throughput gain. Synthetic no-crossing and crossing cases both emit it. A missing marker on the correctly bound model can mean no native region; it must not be silently treated as a successful safety qualification.

### Historical measurements versus current pin

- The tracked 1216 `evidence/validation.json` contains **22 records, all `base_ref=b11126`**; 8 were marked eligible on that historical pin. The 1215 evidence contains 8 b11126 records, none eligible. No b11552 1216 validation record was found.
- Earlier combined 1215+1216 gfx1100 single-GPU tests reported +2.38% (10 paired rounds, 95% CI [1.45%, 3.32%]) and 82.28% overlap of auxiliary-stream busy time. These were real **historical combined-stack** observations, not independent 1216 gains. The later b11126 four-session 1215 qualification rejected the stack: target decode -2.8..-3.3% gfx1100 and -8.0..-8.2% gfx1201, with control regressions. Do not promote 1216 or claim a speedup using the older result.
- All previous full-vocabulary parity and graph-capture results are historical b11126 evidence. A new b11552 standalone run must establish current behavior independently.

### Cheapest discriminator and bounded implementation gate

1. **Host tests before GPU:** extend existing 1216 patch/producer/contract tests for (a) a synthetic candidate crossing the join, (b) legal within-region fusion, (c) no native concurrent event, (d) stale/missing join-node index, (e) cap restoration and repeated capture, (f) rejected 1215 absent from canonical dependency closure, (g) contract TOML prerequisite scoping, and (h) marker-present but no candidate crossing. Verify current b11552 patch composition and `validation/producer.toml` comments/actual binaries. Do not add another matcher or producer.
2. **Native-event reachability gate:** on the correctly bound `tierM-qwen35b-a3b-moe-mtp` model, first prove an upstream **native** concurrent region exists without 1215 under `GGML_CUDA_GRAPH_OPT=1`. Record exact model SHA, runtime build, GPU stable identity, graph nodes/fork/join, prompt/context/ubatch and MTP configuration. If absent, close 1216 as dormant/no-op without a throughput campaign.
3. **Safety/correctness if reachable:** control=validated b11552 BC, subject=identical BC + 1216; same fixed model/seed/prompt/context and device. Require >=20 same-process graph capture/replay and multi-request/multi-ubatch runs, zero capture errors or invalid join state, full-vocabulary near-lossless comparison, fixed-draft MTP acceptance parity and no disappeared work. Check graph-opt-off and dense-model negative controls. Qualify gfx1100 and gfx1201 separately; gfx1030 is an auxiliary, model-fit-gated control, not a mixed-GPU performance extrapolation. For dual-XTX/layer-split/R9700 configurations, test only if native events actually activate and distinguish lack of GPU P2P from a join failure.
4. **Performance and promotion:** use existing paired `tg128` controls (>=10 rounds/session, >=4 independent sessions per eligible architecture), record capture/recapture time, host map overhead, GPU launches and peak memory. Enforce contract <=1% control regression and all correctness/activation gates. **No speedup threshold** for a safety guard; do not infer one from external gfx1151 results. If map cost is measurable, optimize within existing graph-index ownership only.
5. **Terminal outcomes:** (A) native region absent -> retire/close with no patch; (B) reachable and safe/non-regressing -> keep opt-in until fresh current-pin evidence qualifies lifecycle promotion; (C) abort, mismatch or regression -> default-off, localize minimal join fix. Do not queue hardware work overlapping current RAD01/Radiance activity.

### Upstream, forks, and ownership

- [AMD-Ecosystem llama.cpp PR #71](https://github.com/AMD-Ecosystem/llama.cpp/pull/71) (merged 2026-07-27): source of 1216's horizon-cap mechanism and external gfx1151 decode result. Its causal failure involved 1215-style MoE overlap, not proven current native-only events.
- [llama.cpp b11552](https://github.com/ggml-org/llama.cpp/blob/b11552/ggml/src/ggml-cuda/ggml-cuda.cu) and current upstream master: existing fork/join, no 1216 cap. Do not port an already-present guard or infer a stock regression.
- [SGLang breakable CUDA graph](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/breakable_cuda_graph.mdx): explicit fork/join tracking across graph breaks on CUDA/ROCm; **test-design precedent only**, not a llama.cpp port or BigCherry benchmark.
- PRBE112 owns rejected 1215; PVPS15 owns generic activation evidence; 1216's patch-local producer owns its tests. Current RAD01/Radiance/1340/release capabilities worked within 12h are protected and were not modified.

### Audit validation / change log

2026-10-13: 15/15 actual-source static assertions passed; 2/2 pinned patch anchors matched once; 6/6 synthetic host assertions passed (TOML nesting and marker non-causality). No repository pytest, patch application, HIP compilation, GPU execution or new performance benchmark ran. Historical details remain in Git history and patch README. This is a documentation-only reconciliation.

