# BigCherry Build Campaign and RDNA Planning Handover

Date: 2026-08-15
Repository: `https://github.com/audiagentic/bigcherry`
Branch: `tuning-code-rebase`
Handover content commit: `8d501b2` (later commits only synchronize ledger metadata)

## Scope

The work covered two related scopes:

1. The reusable build/refactor campaign from `BIGCHERRY_REUSABLE_BUILD_CAMPAIGN_RUN_SHEET_REVISED.md`.
2. The planning-only RDNA enhancement audit from `BIGCHERRY_RDNA_BOOSTS_PREIMPLEMENTATION_RUN_SHEET.md`.

Governing repository references:
- `docs/reference/build/BUILD.md` + `docs/reference/testing/TEST.md` — Brutus build/test procedures and GPU indices.
- `docs/standards/HIP_AUTOTUNE_STANDARDS.md` — candidate, identity, precision, replay, and evidence rules.
- `recipes.toml` — current strict v2 campaign configuration.
- `docs/planning/active/reusable-build-campaign/RE03.md` through `RE15.md` — build campaign work items.
- `docs/planning/active/rdna-boost-experiments/RD01.md` through `RD23.md` — RDNA planning set.

## Repository state and commits

The branch is clean and synchronized with origin. The implementation work is at `386f959`; the planning checkpoint is `5b716e2`; the latest correction is `48b0325`; and the handover content is at `8d501b2`.

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

Root cause found (2026-08-15): the manifest hash mismatch was not incidental drift. `manifest_hash()` is computed over `variant_set` and the candidate list, and replay-slim manifests are *always* a different `variant_set` with a narrower candidate set than the workload/full-max manifest that produced the tuning measurements — so `d5e64aa5e84e51ee290303b562c00818` (full tune) and `386aec675dac74dc81d10d4b4f043e5a` (replay-slim) could never have matched, by construction. The export gate in `replay_cache.py` was comparing the wrong identity: it required exact `manifest_hash` equality between producer and consumer alongside `source_revision`, which no correct replay-slim export could ever satisfy.

Fixed: the export check now compares `source_revision` only (the actual producer/consumer identity); candidate-set compatibility is still enforced separately when winners are resolved against the target manifest's catalog (an unknown winner is rejected there).

While tracing this, two related runtime gaps were found and fixed in `hip-autotune-replay.cpp`:

- A single stale `implementation_version` entry previously caused the whole replay cache to be discarded (`g_winners.clear()`), silently losing every other retained build's winners. Now only that one entry is skipped.
- Cross-build fallback (using a previous build's winner when the current build hasn't been re-tuned yet) did not exist at runtime — lookup required an exact `source_revision`/`manifest_hash` match and fell back straight to native otherwise. Added `GGML_HIP_DISPATCH_REPLAY_REVISION_MATCH=0` to opt into fallback to the newest available entry for a dispatch key across builds, gated by the existing per-entry `implementation_version` check for safety. Default behavior (unset, or `=1`) is unchanged: exact match required, same as before.

Validated on Brutus (2026-08-15): all three fixes were copied to `/mnt/vault/development/bc-branch`, the Python replay-cache test suite was extended with a regression test proving the actual bug (`test_manifest_hash_mismatch_alone_is_accepted_when_source_revision_matches`) and rerun (45/45 pass; full suite 447/447 excluding 6 modules that fail to import for an unrelated pre-existing reason — `pytest` is not installed in the Brutus Python environment). Replay-cache export was then run against the real, previously-failing artifacts:

```text
python3 -m bigcherry.replay_cache artifacts/campaign-winners-full.jsonl \
  --manifest artifacts/4801e3c567d5/hip-autotune-manifest.json \
  --output artifacts/dispatch-slim.cache
# 29 winner(s) across generations [0], 16 distinct candidate(s), 563 string byte(s)
# manifest 386aec675dac74dc81d10d4b4f043e5a
# wrote artifacts/dispatch-slim.cache (3229 bytes)
```

The measurements header carries producer `manifest_hash=d5e64aa5e84e51ee290303b562c00818` against a target manifest whose own `manifest_hash=386aec675dac74dc81d10d4b4f043e5a` — the exact mismatch this handover originally reported, now correctly accepted because `source_revision` (`4801e3c567d5131dd41b387df5f2d4b1370d92be`) matches on both sides. `validate_blob`/`read_cache` confirm the output is structurally sound (29 entries, correct manifest binding). **Replay-cache export is proven end to end.**

C++ runtime validated on Brutus (2026-08-15): `python3 -m bigcherry apply` propagated the fixed `hip-autotune-replay.cpp` into `vendor/llama.cpp/...` (the actual compiled tree; `src/...` is the overlay source, not what cmake builds), `build/bigcherry-replay` (`GGML_HIP_DISPATCH_REPLAY=ON`, `variant_set=replay-slim`, same `4801e3c` source revision as the cache) was rebuilt clean, and `llama-bench` was run against `dispatch-slim.cache` with `GGML_HIP_DISPATCH_MODE=replay` and `GGML_HIP_DISPATCH_COVERAGE` enabled:

```json
"replay": {"entries": 29, "misses": 0, "stale": false}
```

All 29 exported winners loaded and matched exactly (not through the new fallback switch — this binary is the same source revision the cache was tuned against, so `fresh` is true for every entry), across 1026 real dispatches with zero misses. **Replay-cache export and load are both proven end to end on real hardware.**

Both new/changed C++ behaviors were independently exercised with deliberately corrupted caches (constructed by patching raw entry bytes and recomputing the content checksum, so `validate_blob` still accepts the file):

- **Skip-not-clear**: one entry's `implementation_version` was set to an impossible value (`0xFFFF`). The rebuilt binary loaded `28` of the `29` entries (`artifacts/dispatch-slim-badimplver.cache`) instead of discarding the whole cache — confirms the fix; under the old behavior this would have been `0`.
- **Cross-build fallback**: every entry's `source_revision` digest was overwritten with a value that cannot match this binary's compiled-in revision (`artifacts/dispatch-slim-badrevision.cache`), then run twice with `GGML_HIP_DISPATCH_MISS=native-record` to count actual replay misses:
  - default (`GGML_HIP_DISPATCH_REPLAY_REVISION_MATCH` unset): `{"entries": 29, "misses": 29, "stale": true}` — every dispatch correctly falls back to native, exact-match behavior unchanged.
  - `GGML_HIP_DISPATCH_REPLAY_REVISION_MATCH=0`: `{"entries": 29, "misses": 14, "stale": true}` — roughly half the dispatch keys are now served from the revision-mismatched cache instead of falling to native, proving the opt-in fallback switch works.

RE09's replay-identity blocker (export/load path) is now closed with real evidence: export, load, and both new runtime code paths are proven on Brutus against gfx1100.

Replay-full generation was retried against the existing full-tune artifact (`artifacts/campaign-winners-full.jsonl`) and still correctly refuses, on a real and quantified gap rather than a code defect: `python3 -m bigcherry generate --variant-set replay-full --winners artifacts/campaign-winners-full.jsonl --inventory artifacts/campaign-gfx1100-inventory.json --arch gfx1100` wants correctness evidence for **101** non-native candidates; the existing tune artifact only carries evidence for **76** of them. The 25 missing are all in the `mmq` family across various tile/geometry variants (e.g. `mmq:q8_0:j16:fb1:t128:o2:i64:sram-q8_0:k256:sk0:v1`) — plausibly eliminated during tuning's screening phase before reaching the correctness-check stage, so they were never exercised for correctness even though they exist in the candidate universe replay-full wants to ship. This is not fixable by more code changes: it needs a tune run (fresh or extended) whose correctness checking covers the full replay-full candidate set, which is real GPU time — the original full tune took an unknown but non-trivial duration to produce 178 candidates / 4,268 measurements. Not attempted this session; flagging for an explicit decision on whether to spend that GPU time now.

### Multi-GPU RCCL/allreduce

The initial tensor-split multi-GPU record run failed in `ggml_backend_cuda_comm_allreduce_nccl` with HIP "operation cannot be performed in present state", while the current device was gfx1030. This is evidence of an unresolved real multi-GPU path issue, not a successful HI14/RE15 result.

**Resolved on a controlled same-architecture pair (2026-08-15).** Devices 0 and 1 are both `gfx1100` (RX 7900 XTX). Ran `ROCR_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 build/bigcherry-tune/bin/llama-bench -m .../Qwen3.5-0.8B-UD-Q5_K_XL.gguf -p 512 -n 128 -r 3 -ngl 99 -sm tensor -o json` — completed cleanly, 3/3 repetitions for both pp512 (`avg_ts≈18988 t/s`) and tg128 (`avg_ts≈232 t/s`), no crash. Reran with `GGML_HIP_REDUCE_TELEMETRY=artifacts/rccl-telemetry.jsonl` to capture per-reduce provider/topology/fallback evidence directly (not `GGML_HIP_RCCL`, which is a compile-time CMake option already `ON` in this build, not a runtime env var):

```json
{"effective_provider": "rccl", "fallback_depth": 0, "topology_key": "n2:peer1001", "peer_access": "partial", "device_count": 2, "devices": [0, 1]}
```

All 18,672 recorded reduce operations used `effective_provider=rccl` with `fallback_depth=0` — no silent fallback to a slower path. This is a genuine, evidenced RCCL acceptance on matched architecture, confirming the earlier failure was specifically a cross-architecture issue (gfx1030 mixed with gfx1100), not a general RCCL defect. **Multi-GPU RCCL/allreduce blocker for RE15 is closed for the 2×gfx1100 case.** Not yet attempted: wider device counts (all 3 gfx1100/gfx1201/gfx1030 devices together, which remains a genuinely mixed-architecture case and should still be expected to fail or require explicit exclusion) and any other same-architecture pairing.

### Server-bench comparison

The native llama.cpp server-bench runner needs corrected target and prompt metadata before its numbers can support a causal stock-versus-BigCherry claim.

**Root cause found and worked around (2026-08-15).** In `/mnt/vault/development/llmhosts/llamacpp/bench/run_bench.py`, when `--target` is passed without an explicit `--rig`, the harness silently defaults `rig_name = "vulkan"` — regardless of what backend the server actually runs. `"hip"` is a valid, already-configured rig id in that harness's own `harness-definitions.json` (used elsewhere in its config); the CLI just never selects it automatically for a HIP server. This is why the earlier run reported a Vulkan target while the server was HIP.

This harness lives outside the bigcherry repo (a shared benchmarking tool, not something owned by this campaign), so rather than patching its default-selection logic, the workaround is to always pass `--rig hip` explicitly for BigCherry/HIP server-bench runs. Verified against `upstream-stock` on gfx1100:

```text
Using target: RX 7900 XTX (GPU1)
  Rig:      hip
Config Args:  pp512 (p=512, n=128); tg128 (p=1, n=128)
  Metrics:   pp512=9521.5, tg128=289.1 t/s
```

`Rig: hip` is now correct, and the `pp512` config args (`p=512, n=128`) match the requested configuration — the earlier prompt-metadata inconsistency is also gone. **Server-bench harness boundary is fixed** (by explicit `--rig hip`, not a harness code change). Re-run balanced stock/control/native/tuned/replay comparisons is still open — this session only validated the harness metadata itself against `upstream-stock`, not the full comparison matrix.

### MTP draft-decoding exploratory matrix (Qwen3.8-27B)

Exploratory, not a build-campaign gate. Run on 2026-08-15 to exercise the now-fixed server-bench path (`llama-server` from `build/bigcherry-tune`, benched only via `run_bench.py --server-url` pointed at that already-running server — the harness never spawned its own server) across MTP speculative-decoding depth and KV cache quant, on dual `gfx1100` (tensor split `1,1`, `-c 65536`). 18/18 combinations completed cleanly; every server process was confirmed killed and VRAM returned to baseline (~28 MiB/GPU, driver overhead only) between runs, so no combination could be contaminated by a prior one's leftover state.

| Model | Cache | spec-draft-n-max | pp512 t/s | tg128 t/s |
|---|---|---:|---:|---:|
| Q4_K_M | f16 | 4 | 886.0 | 66.6 |
| Q4_K_M | f16 | 5 | 872.2 | 61.4 |
| Q4_K_M | f16 | 6 | 842.0 | 66.5 |
| Q4_K_M | q8_0 | 4 | 887.1 | 70.3 |
| Q4_K_M | q8_0 | 5 | 883.6 | 58.6 |
| Q4_K_M | q8_0 | 6 | 869.5 | 60.4 |
| Q4_K_M | bf16 | 4 | 888.0 | 73.9 |
| Q4_K_M | bf16 | 5 | 876.6 | 58.0 |
| Q4_K_M | bf16 | 6 | 885.6 | 57.4 |
| Q8_0 | f16 | 4 | 963.0 | 70.5 |
| Q8_0 | f16 | 5 | 949.8 | 76.6 |
| Q8_0 | f16 | 6 | 953.0 | 71.0 |
| Q8_0 | q8_0 | 4 | 962.5 | 80.9 |
| Q8_0 | q8_0 | 5 | 958.4 | 72.1 |
| Q8_0 | q8_0 | 6 | 957.6 | 72.4 |
| Q8_0 | bf16 | 4 | 966.1 | 75.2 |
| Q8_0 | bf16 | 5 | 932.5 | 69.3 |
| Q8_0 | bf16 | 6 | 960.5 | 75.3 |

Raw results: `artifacts/mtp-matrix-results.jsonl` on Brutus. Observations, held loosely given single-repetition-per-config noise (stddevs in the 6–16 t/s range on tg128 were common):

- Q8_0 model consistently outperforms Q4_K_M on both pp512 (~955 vs ~875 t/s) and tg128 (~72 vs ~64 t/s) here — MoE-heavy model, likely compute/dispatch-bound rather than memory-bandwidth-bound at this size on dual XTX, so the larger quant's better arithmetic behavior outweighs its bandwidth cost. Not yet root-caused.
- No clear monotonic trend of tg128 with `spec-draft-n-max` (4 vs 5 vs 6) in either model — deeper MTP speculation neither reliably helped nor hurt at these settings, and no run failed or showed anomalous behavior at depth 6.
- Cache quant (f16/q8_0/bf16) shows no consistent ranking within the observed noise band.
- No conclusions about IMRoPE/SET_ROWS extension or the unpromoted patch 1004 objective are drawn from this — this matrix used stock MTP draft decoding, not any RDNA-boost candidate.

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
