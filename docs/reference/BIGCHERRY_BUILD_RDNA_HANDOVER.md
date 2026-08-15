# BigCherry Build Campaign and RDNA Planning Handover

Date: 2026-08-15
Repository: `https://github.com/audiagentic/bigcherry`
Branch: `tuning-code-rebase`
Latest pushed commit: `84185e4`

## Scope

The work covered two related scopes:

1. The reusable build/refactor campaign from `BIGCHERRY_REUSABLE_BUILD_CAMPAIGN_RUN_SHEET_REVISED.md`.
2. The planning-only RDNA enhancement audit from `BIGCHERRY_RDNA_BOOSTS_PREIMPLEMENTATION_RUN_SHEET.md`.

The RDNA run sheet explicitly forbids vendor edits, BigCherry C/C++ implementation, cherry-picking external commits, RDNA patch benchmarking, and promotion during its planning phase. Those restrictions were followed.

Governing repository references:

- `AGENTS.md` — ledger and planning workflow.
- `docs/reference/BUILD_AND_TEST.md` — Brutus build/test procedures and GPU indices.
- `docs/standards/HIP_AUTOTUNE_STANDARDS.md` — candidate, identity, precision, replay, and evidence rules.
- `recipes.toml` — current strict v2 campaign configuration.
- `docs/planning/active/reusable-build-campaign/RE03.md` through `RE15.md` — build campaign work items.
- `docs/planning/active/rdna-boost-experiments/RD01.md` through `RD23.md` — RDNA planning set.

## Repository state and commits

The branch is clean and synchronized with origin. The implementation work is at `386f959`; the planning checkpoint is `5b716e2`; the latest correction is `48b0325`; the handover is `8d501b2` with campaign-ledger synchronization at `84185e4`.

Implementation history:

- `82e6a19` — strict v2 configuration, ProjectContext, exact catalog, doctor, and out-of-tree plumbing.
- `b811a80` — BuildPlan/build identity, ArtifactStore, provenance v2, and DB schema 3.
- `89f56b5` — PipelineService, CampaignGraph, ResourceLock, CampaignRun, comparisons, and promotion.
- `9e39a85` — preserve release stage on repeated audit.
- `2df826c` — repair the coverage-counter patch anchor after the dispatch ABI change.
- `a37709e` — accept native-only ranking decisions without a schedule.
- `5149a5b` — align paired-round handling with C++ tie semantics.
- `895e99c` — accept native-retention rows without a challenger winner verdict.
- `c30dafe` — retain finite ties for median effects while excluding them from sign rounds.
- `4999729` — tolerate the documented three-decimal serialized effect precision.
- `386f959` — propagate successful tune correctness evidence into production replay manifests.

Planning/documentation history:

- `5b716e2` — create the complete RD01–RD23 RDNA planning set, incorporate linked reviews, and record the planning ledger event.
- `48b0325` — correct RD namespace/repository references and remove two accidentally captured wrapper labels from the ledger file list.

## Build/refactor work completed

Implemented foundations include:

- strict v2 recipe/config models and explicit ProjectContext paths;
- exact patch-module catalog and reviewed production patch-set resolution;
- isolated source/worktree primitives;
- content-addressed BuildPlan and effective build identity;
- immutable artifact/provenance v2;
- campaign identity fields and DB schema 3 migration;
- typed lifecycle PipelineService;
- campaign DAG, resource claims, locks, failure blocking, and resume scaffolding;
- identity-aware comparison planning and promotion validation;
- production replay manifest correctness-evidence propagation.

The implementation deliberately remains fail-closed where producer and consumer identities do not agree.

## Validation completed

Local and Brutus evidence:

- Full Python/Brutus suite: `586 passed, 9 subtests`.
- Strict patch audit: `33/33` checks passed.
- Stock, control, record, tune, and replay-slim builds completed on Brutus.
- Tune run: 29 result rows, 4,268 measurements, 178 candidates; 18 promoted rows after offline validation.
- Production replay manifests now reject non-native candidates without successful correctness evidence.
- Single-GPU record inventory completed on Brutus for gfx1100, producing 29 signatures and one hardware inventory.

Direct native llama.cpp benchmark, Qwen3.5-0.8B Q5 model, short exploratory run:

| Build | Prompt tokens/s | Generation tokens/s |
|---|---:|---:|
| upstream-stock | 16917.802 | 309.322 |
| bigcherry-native | 17526.302 | 302.772 |
| bigcherry-control | 17336.579 | 309.322 |
| bigcherry-tune | 17866.645 | 308.892 |

These are single exploratory runs, not a definitive causal performance claim.

Native server-bench harness runs were also completed, but the harness reported a Vulkan target while the server was HIP and showed prompt metadata inconsistent with the requested `pp512` configuration. Treat those results as exploratory only until the runner metadata/target selection is corrected.

## Brutus environment and preservation rules

Brutus: `ssh 10.10.100.10`
BigCherry tree: `/mnt/vault/development/bc-branch`
Model mapping: `J:\llm-models = /mnt/vault/llm-models`

GPU inventory observed:

- devices 0/1: gfx1100, RX 7900 XTX;
- device 2: gfx1201 integrated AMD GPU;
- device 3: gfx1030, RX 6900 XT.

Preserve these unrelated Brutus root changes; do not reset or clean them:

```text
M  releases/b10362.json
M  releases/index.json
?? models/
?? releases/unknown.json
```

The Brutus vendor overlay is intentionally dirty after the audit/apply/build cycle. Synchronization must preserve the named release/model changes and must not use a destructive reset.

## Remaining build blockers

The reusable build campaign is not fully closed. RE09, RE14, and RE15 remain open.

### Replay identity

The full tune measurement artifact reported manifest hash `d5e64aa5e84e51ee290303b562c00818`, while the replay-slim generated manifest reported `386aec675dac74dc81d10d4b4f043e5a`. Attempting replay-cache export correctly failed because the measurement producer provenance did not match the supplied manifest. This is a correct fail-closed result, but replay-cache export has not yet been proven end to end.

The earlier replay-full attempt also failed when non-native candidates lacked correctness evidence. That producer/consumer gap was fixed, and replay-slim generation then succeeded; replay-full still needs a clean acceptance run.

### Multi-GPU RCCL/allreduce

The initial tensor-split multi-GPU record run failed in `ggml_backend_cuda_comm_allreduce_nccl` with HIP “operation cannot be performed in present state”, while the current device was gfx1030. This is evidence of an unresolved real multi-GPU path issue, not a successful HI14/RE15 result. Do not claim RCCL acceptance until a controlled same-architecture run succeeds and records topology/provider telemetry.

### Server-bench comparison

The native llama.cpp server-bench runner needs corrected target and prompt metadata before its numbers can support a causal stock-versus-BigCherry claim. Re-run balanced comparisons only after correcting that harness boundary.

## RDNA planning completed

The RDNA plan is `rdna-boost-experiments`, with 23 pending items `RD01`–`RD23`. The planning service derived the `RD` prefix from the plan name; this is the approved equivalent of the run sheet’s suggested `RB` namespace.

RD01 contains the full external disposition ledger. The reviewed external snapshot is:

- repository: `https://github.com/stew675/llama.cpp`;
- branch locator: `rdna-boosts`;
- reviewed base: `a94d563ed801d1da1b8c2432946de07d0231bb3d`;
- reviewed head: `ed89854b2aeb0e333dd61424f14af2aedaca126e`;
- current read-only branch resolution: `ed89854b2aeb0e333dd61424f14af2aedaca126e`;
- fork baseline is materially ahead of BigCherry’s pinned llama.cpp and must be audited semantically, not imported wholesale.

Logical planning waves:

- RD01–RD03: provenance, A/B/C semantic portability, overlap/dependency/conflict/safety matrix.
- RD04–RD08: BF16 flash attention, WMMA correctness, RDNA4 configuration, Q6_K MMQ/MMVQ work.
- RD09–RD12: Q8_1 cache foundation and dependent Q8/MMVQ work.
- RD13–RD20: graph fusions and topology/correctness changes.
- RD21–RD23: deferred gfx1151/integrated-GPU work and diagnostic support.

Linked reviews incorporated and closed:

- `RV32` for HI26;
- `RV33` for HI25;
- `RV34` for EX02;
- `RV35` for EX03;
- `RV36`–`RV45` for RE03, RE04, RE05, RE07, RE08, RE09, RE10, RE11, RE12, and RE15.

Safety decisions preserved:

- HI25 remains the exact gfx1100 Q8_0 × Q8_1 objective; external WMMA work does not close or widen it.
- HI26 remains open; the external Q8 cache is a separately gated overlap, not completion evidence.
- EX02 remains pending with its exact gfx1100 Q6_K quarantine mandatory.
- EX03 remains narrow and contained; no broad q1_0/MMVQ quarantine was added.
- Unpromoted patch 1004 remains an explicit prerequisite/control for the IMRoPE/SET_ROWS extension; it was not silently promoted.

## GPT-auto review status

The persistent GPT-auto work was retained where possible. The earlier successful request `req_5a70cfc5d00a4e24` on `ses_be17d3b5c4c6457e` reviewed BC01–BC03 and advised strict v2 migration, exact catalog identity, ProjectContext authority, and a pure immutable resolver.

The later live session was `ses_edae00e123614bac`. A new review submission was attempted through `mcp__ag_agents_gateway__agent_task_submit`, but the gateway rejected both `gpt-auto` and `chatgpt` as unavailable configured agent definitions (`RES-AGD-001`). No new RDNA review response was returned, and no duplicate session was created.

## Exact next work

Resume the build campaign before implementing any RDNA item:

1. Inspect `tools/bigcherry/replay_cache.py`, `tools/bigcherry/autotune_catalog.py`, `tools/bigcherry/pipeline.py`, and the associated tests.
2. Make the replay manifest/cache export consume the exact tune producer namespace, or explicitly create a new identity-compatible replay artifact; do not weaken the mismatch check.
3. Run replay-full and replay-cache export/validation from the same producer inputs, then validate the production C++ replay reader.
4. Reproduce the RCCL failure on controlled same-architecture Brutus device pairs, verify `GGML_HIP_RCCL=ON`, and capture topology/provider/fallback telemetry.
5. Correct the native llama.cpp server-bench target/prompt metadata and rerun balanced stock/control/native/tuned/replay comparisons.
6. Update RE09, RE14, and RE15 only with evidence that satisfies their acceptance contracts. Do not close them from build success alone.
7. After the build gates are genuinely complete, begin RD01–RD03 semantic planning execution. The first implementation candidate should then be selected from the dependency/safety matrix, not from the external commit order.

## Do not do

- Do not cherry-pick the external `rdna-boosts` branch.
- Do not apply RD04–RD23 yet; the RDNA run sheet was planning-only.
- Do not reset or clean the dirty Brutus release/model files.
- Do not fabricate replay-cache identity compatibility.
- Do not treat the exploratory benchmark numbers as a confirmed end-to-end improvement.
- Do not close EX02, EX03, RE09, RE14, or RE15 without their stated evidence.
