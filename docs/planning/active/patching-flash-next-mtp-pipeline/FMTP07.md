---
id: FMTP07
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Qualify Flash-Next MTP verification overlap on the AMD deployment topology

## 2026-10-15 current-pin gate: independent draft physical ubatch (upstream #30459)

**Disposition:** FMTP07 owns *qualification* of the draft-context physical-ubatch/memory trade-off; no new BigCherry patch, scheduler, queue, placement selector, or default change is authorised. This is distinct from FMTP03's ahead/overlap policy, QFP16's target verification submit costs, QFP50/1359's NextN prefill pipeline, and QFP17's target/QSA ubatch. The older `patch-refactor`/`c061` baseline in this document is historical: `config/recipes.toml` currently pins llama.cpp **b11552**. No GPU performance or VRAM saving is claimed here.

### Source and lifetime trace

- **Pinned b11552:** `common/speculative.cpp::common_base_params_to_speculative` copies the target `common_params`, changes draft model/device/output settings, but does **not** override `result.n_ubatch`. `tools/server/server-context.cpp` calls that conversion before `common_speculative_init_from_params`. Its constructor calls `common_context_params_to_llama(params_dft)`, sets `ctx_type=MTP` for MTP, `n_ctx=llama_n_ctx(ctx_tgt)`, `n_rs_seq=0`, `ctx_other=ctx_tgt`, then creates `ctx_dft` with `llama_init_from_model`. Thus draft compute graph/workspace sizing currently inherits target physical ubatch; **full draft context/KV length and model weight residency are separate**. The target's `-ub` and target verification width are not altered by this draft-only knob.
- **New upstream:** [llama.cpp #30459](https://github.com/ggml-org/llama.cpp/pull/30459), merged **2026-10-14 10:58 UTC**, after b11552, adds `--spec-draft-n-ubatch`/`LLAMA_ARG_SPEC_DRAFT_N_UBATCH`, `common_params_speculative_draft::n_ubatch = 0`, and `if (params_spec.n_ubatch > 0) result.n_ubatch = params_spec.n_ubatch`. Zero **inherits** target; negative CLI values are rejected; a positive value is **not** automatically clamped below target. This is not present in b11552 and must not be described as a usable production flag until the exact deployed source/patch composition includes it. Inspect [changed code](https://github.com/ggml-org/llama.cpp/pull/30459/files), not only the release description.
- **Upstream test coverage is narrower than BigCherry's use:** the PR adds parser inheritance/override assertions (512 -> 8) and a server token-equality test with a smaller draft ubatch; it does not demonstrate Flash-Next native MTP, two-XTX target + gfx1030 draft, long-context MTP rollback, 1321/1322 ahead replay, 1348 catch-up, 1359 two-slot NextN fencing, memory reduction, or AMD speedup. Pinned `common/speculative.cpp` already uses `llama_n_ubatch(ctx_dft)` to chunk EAGLE3 encoder/extend paths; that does **not** prove every MTP input/catch-up batch is safe at arbitrarily small physical ubatch. Confirm actual effective `llama_n_ubatch(ctx_dft)`, not just the CLI request.
- **Allocation/graph mechanism (hypothesis):** reducing the draft physical batch may lower draft compute/graph workspace and peak VRAM, especially during prompt catch-up; model weights and the draft's target-length recurrent/KV state do not shrink automatically. More microbatches may increase graph submissions, Meta/HIP graph shape churn, draft H2D/NextN staging and host overhead, extending FMTP03 overhang. With a single-head depth <=7, steady decode may never approach the old ubatch, so no decode improvement is implied. No-P2P PCIe and gfx1030 residency make target/draft allocation and transfer receipts essential; neither vLLM's separate draft placement nor SGLang's chunked-prefill scheduler is a transferable implementation. See [vLLM speculative config](https://docs.vllm.ai/en/v0.22.0/features/speculative_decoding/) and [SGLang chunk-budget semantics](https://github.com/sgl-project/sglang/issues/20018).

### Bounded, implementation-ready admission

**Gate 0 (no patch, no hardware queue):** confirm the production recipe's pin, MTP mode (native single-head versus separate draft model/EAGLE), model/sidecar hashes, `-b/-ub`, physical target and draft device UUIDs, current draft `llama_n_ubatch`, allocated graph/workspace/KV bytes, maximum live prompt catch-up batch, and any existing queued experiment. Use existing `1317/1318/1319/1346` timing, allocator receipts, rocprof and server logs; do not add a second telemetry registry. If no draft memory pressure, no independent draft graph workspace, or no safe same-pin control, terminate `NO_EXPOSED_DRAFT_HEADROOM` without a port or benchmark. A source pin bump containing #30459 is the preferred adoption route; a one-off backport requires separate review, not this audit.

**Gate 1 (only after Gate 0 admission):** on **one identical composed binary**, compare `--spec-draft-n-ubatch=0` (inherits target) against draft 32/64/128 while keeping target `-ub=512`, model/quant/KV, target split, gfx1030 draft placement, MTP front depth, temperature, and all other flags fixed. Include 8 only if a direct backend-op/initialization fixture proves that every MTP catch-up/verify input fits or safely chunks; do not assume upstream's server smoke establishes it. Record requested and *effective* draft/target ubatch, graph signatures/reallocation, number of draft microbatches, allocations/peak free VRAM, NextN copy bytes, full executed work, and per-round serial draft/forced replay/target sync/overhang. Compare `ahead=0` first; only after parity add `1322` ahead on/off and `1348/1359` pipeline controls. No active FMTP or QFP hardware lane should be duplicated.

**Gate 2 (correctness/performance):** exercise 8K/24K/98K and memory-fit context, prefill + steady decode, repeated same-process requests, multi-ubatch, abort/reset/context shift, graph capture/replay and MTP rollback/reseed. Require full-vocabulary logits/KLD or deterministic greedy identity, unchanged MTP acceptance/verified work, no stale NextN/frontier, no OOB/graph allocation fault, and no >1% regression in unaffected target-only/regular MTP controls. Four independent order-balanced sessions with >=10 paired rounds/session per admitted architecture; distinguish gfx1030 draft from gfx1100/gfx1201 target measurements. For **memory-only** adoption require a reproducible OOM->fit transition or >=256 MiB *measured* draft peak-VRAM saving with no >1% E2E regression; for a performance claim require CI95-low >=3% E2E gain. A theoretical workspace-size reduction is not a measured saving. Terminal `PIN_MISSING`, `NO_EXPOSED_DRAFT_HEADROOM`, `UNSAFE_DRAFT_CHUNK`, `STATE_OR_NUMERIC_FAIL`, `REGRESSION`, `MEMORY_FIT_OPT_IN`, or `QUALIFIED_OPT_IN`. Preserve upstream default inheritance and fail closed to the known target-sized draft on any uncertainty.

### Novelty, protection and validation

New information since the last FMTP07 edit (2026-10-04) is merged post-pin #30459, not a newly measured BigCherry bottleneck. Selected `main` at independent `a1d5eeaffe5bb6e152629e7c52a85178c6972c25` (2026-10-15 02:58 UTC, RHP merge). No independent FMTP draft-ubatch implementation, plan update, submitted PR, hardware evidence or newly queued lane was found in the preceding 12 hours in inspected active branches/PRs; historical `queue-fmtp-calib*.sh` last changed 2026-10-04. Protected RHP01-06 heat/placement, MEN10 engine CLI, RAD02-06/09 mailbox/remote experts, RWQ and PGC15 precision/AllReduce are not modified or scheduled. Recheck before any future hardware queue. This audit ran **11/11 pinned/PR source assertions** and **10/10 host-only override/chunk-partition model tests**. No BigCherry repository pytest, C++ build, HIP execution, physical VRAM measurement or GPU benchmark ran.

## Description

Run correctness, scheduling and performance gates on the real BigCherry Flash-Next topology. Promotion requires target/MTP device overlap and critical-path reduction, not merely successful ahead generation.

## Steps

1. Build paired arms from one `patch-refactor` baseline:
   - control: MTP, ahead=0;
   - fixed ahead;
   - adaptive ahead.
2. Hold model/sidecar metadata, target split, KV types, batch/ubatch, MTP GPU, front depth, sampler and unrelated BigCherry toggles constant.
3. First lane: `--parallel 1`, temperature 0, fixed front depth; then 1268 adaptive front depth.
4. Test shallow ~8-10K and deep ~65-80K contexts plus long steady decode.
5. Include code, structured/repetitive text, prose and deliberately low-acceptance output.
6. Separate round classes in telemetry:
   - cold/flush round with serial fresh front + live continuation;
   - promoted-hit round with no serial fresh front + forced-front replay/continuation under target verification.
7. Capture:
   - tok/s and ms/generated token;
   - full-front and bridge-match rates;
   - promoted/flushed tokens;
   - target `llama_process` submit host time;
   - target device interval and final `llama_synchronize` wait;
   - serial fresh-front draft time;
   - promoted-front replay time;
   - continuation time;
   - total MTP overlap work and overhang;
   - authoritative `common_speculative_process()` replay/reseed time;
   - GPU utilization/power and rocprof timeline.
8. Correctness: exact greedy target IDs vs control, no stale/frontier acceptance, reset/stop/context-shift/replay stress, long-run assert/memory scan.
9. Compatibility: 1268 off/on; 1308 off/on; other Flash-Next recipe patches identical between arms.
10. Promote only from paired/ABBA evidence with activation markers and profiler proof.

## Detailed Solution & Technical Design

Primary metrics:

```text
pipeline_hit_rate = promoted_tails / ahead_attempts
bridge_match_given_full = bridge_matches / full_front_accepts
hidden_fraction = 1 - overhang_us / max(mtp_overlap_us, 1)
critical_path_delta = control_ms_per_token - subject_ms_per_token
```

For the worker-free design, verify directly:

```text
target submit
  -> target kernels continue running
  -> MTP replay/continuation kernels run on separate GPU
  -> target sync waits only for remaining target work
```

A reduced target sync wait that matches the overlapped MTP interval is stronger evidence than a reduced host-sync count.

### Promotion gates

Absolute:
- zero greedy token divergence;
- zero accepted stale/frontier-mismatched result;
- zero draft-context lifetime/rollback failure;
- ahead=0 unchanged.

Performance:
- paired CI excludes material regression;
- >=3% median gain is a useful first screen, not a universal threshold;
- target submit/sync and profiler timeline explain the observed gain;
- non-speculative controls do not regress >1% without an explained artifact.

### Failure interpretation

- `llama_process` does not leave target work in flight: reject worker-free FMTP03 before adding complexity; only then evaluate a worker fallback.
- Promoted-front replay exceeds target verify window: reduce ahead depth or reject steady-state chaining.
- High bridge mismatch: adaptive controller should turn off; never weaken bridge rule.
- Good hit rate but worse target time: diagnose PCIe/host/thermal contention.
- Win only on predictable prompts: adaptive mode may still qualify if probe/hysteresis behavior disables elsewhere.

## Files

- experiment contract/recipe evidence
- `tools/lab/flash-next/*` helpers if needed
- plan notes after hardware runs

## Validation

Before hardware:

```bash
python docs/planning/active/patching-flash-next-mtp-pipeline/mock_pipeline.py
```

Then repository-standard patch lint/rebase tests and paired hardware runner. Record BigCherry commit, llama.cpp pin, ROCm version, device topology, model/MTP identity and full server args.

## Acceptance Criteria

- Exact greedy output.
- Profiler proves concurrent target/MTP kernels on disjoint devices.
- Promoted rounds visibly use forced-front replay under the target window, not stale speculative KV.
- Telemetry explains hit/flush/overhang and throughput delta.
- Ahead=0, 1268 and 1308 compatibility lanes pass.
- Default remains off until evidence supports promotion.
