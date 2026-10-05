---
id: QFP28
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T04:10:44.361443+00:00'
breadth: ''
skill: intermediate
created-by: claude
priority: P1
work: M
---

# Qualify external Flash-Next mechanisms with corruption-resistant performance gates

## Description

Sources: https://github.com/Shali12/r9700-flash-next-notes and stew675/llama-cpp-rdna-boosts r30 (`f1082619d7b80ea26af6242b9fceecade072d5f5`) / r31 (`76a083ed492756803187c85de9ce4537a787b606`), including issue #85 and `wip/moe-mmq-overread/RESOLUTION.md`. Their main Flash-Next rig is one R9700 with host-resident experts; Brutus is 2x gfx1100 + gfx1201 in the meta tensor split with all target experts device-resident plus a gfx1030 MTP drafter. Host-expert mechanisms are therefore mechanism/correctness evidence, not direct ports.

Current BigCherry pin is `0504396140d1c882f5f6ee34466a42db7ae90114`. QFP22 currently holds 1295 QSA gather, 1332 QSA token chunking and 1007 VSX4 FA pending deterministic identity/correctness work.

### Corrected external DEVGATHER finding

r29/r30 DEVGATHER is a scheduler host-expert optimisation. `sched_input_gatherable` identifies routed expert host weights, bypasses the normal staging/copy path, and gathers active expert payload into a device destination consumed by quantized `MUL_MAT_ID` MMQ. r30 attempted a once-only head zero on the graph-allocator-owned gather destination. The expert cache also has a separate persistent device arena.

Issue #85/r31 found **two independent MMQ legal-tile over-read holes**, both requiring partial/pruned host-expert buffers:

1. **Decode-cache arena:** slots were allocated without zeroing and without a tail/head guard. MMQ can read the next slot head; empty/unfilled heads could contain NaN/stale data. r31 allocates `arena_slots * expert_bytes + min(expert_bytes, 512)` and zeros the full arena once.
2. **DEVGATHER destination:** r30's one-time zero was keyed to `(input_cpy->data, expert_bytes)`. The graph allocator reuses that region between ubatches, so the zero does not survive a multi-ubatch prefill. MMQ can then over-read stale/uninitialized head bytes. NaN contamination can suppress routed-expert work, explaining the false 2-4x throughput result. r31 disables DEVGATHER by default; the normal staging/host-copy path copies the upstream guard footprint every pass into backend-owned storage. `GGML_SCHED_DEVGATHER=1` remains an explicit A/B override.

The resolution explicitly states device-resident MoE (`-ncmoe 0`) is unaffected: direct model tensors have backend zero padding, and gather/staging/cache-arena machinery requires host weights. This exact bug therefore does **not** execute on Brutus's all-GPU target model.

BigCherry has no `GGML_SCHED_DEVGATHER` implementation. `1295_qsa_gather_decode` gathers selected KV rows; `1331_alloc_peak_live` is allocator telemetry; `1332_qsa_token_chunk` builds token-chunked QSA masks. None shares the DEVGATHER code. The transferable failure class is narrower: any partial/padded temporary buffer consumed by a kernel whose legal read footprint exceeds the logical payload must initialize/provide the whole read footprint. That class is worth checking in 1295. If 1332 differs from dense on request 1, stale cross-request state is unlikely to be its primary cause; first inspect chunk-boundary/mask/index semantics.

## Plan Review — 2026-10-05

The current remote QFP28 has 12 steps; the requested core triage is steps 1-9. Verdicts cover all current steps so the review stays valid after the intervening plan edit.

| Step | Verdict | Brutus assessment |
|---|---|---|
| 1 multi-request gate | **ADAPT** | Keep one server and one slot alive; explicitly run cache-on and cache-off lanes. Shali's `text-check.py` sends only 3 deterministic requests with `cache_prompt:false`; persistent `rig-model` sequencing, not that helper alone, exposed later-request corruption. Complement greedy-MD5; do not replace it. |
| 2 prove gate sensitivity | **ADOPT** | Required. Use a known-corrupt external build when available; on Brutus also use a deterministic response-#2 corruption fixture so comparator/wiring is proven even though host DEVGATHER cannot execute. |
| 3 transport plausibility | **ADOPT** | High-value guard for any future host movement. No current all-GPU speed benefit, but it prevents skipped-work results from being accepted. |
| 4 gate 1295/1332 first | **ADOPT** | Highest priority. 1295's run-to-run instability is compatible with race/uninitialized-padding/extent defects; 1332's dense mismatch needs first-request chunk/mask diagnosis before any stale-state theory. |
| 5 DEVGATHER lineage | **ADOPT** | Resolved: mechanism-only/no shared code. No dependency on stew DEVGATHER should be created. |
| 6 host staging-ring experiment | **REJECT** for current topology | All target experts are GPU-resident. Re-open only if MET01/MET06 deliberately moves experts to host RAM. |
| 7 Q3 for >300K | **ADAPT** | Conditional memory-fit fallback only. First candidate: Unsloth `UD-Q3_K_XL` (~90.0 GB), which the source header-inspected but did **not** download/test. ISTA GSQ-RCO `IQ3_XXS` (~75.8 GB) was measured but is a more aggressive quality trade. Keep f16 preferred/q8_0 minimum KV. |
| 8 draft depth | **ADAPT** | Useful only as a Brutus gfx1030 MTP profile experiment; external single-R9700 numbers do not transfer. Run after target correctness, and leave mechanism ownership with QFP05/FMTP. |
| 9 sampling/reasoning | **ADAPT** | `min_p=0` is profile/model-card conformance, not a kernel optimisation. Keep sampling fixed while benchmarking; assess reasoning-budget policy separately. |
| 10 tools + JSON schema | **ADAPT** | Failure is still present at pin `050439614`; isolate as a separate upstream-fix item rather than mixing with Flash-Next kernels. |
| 11 rollback log check | **ADOPT** | Cheap diagnostic. Any hit belongs to lifecycle/cache ownership after minimization. |
| 12 power-cap arm | **REJECT** by default | Only run if thermal/power variance prevents discrimination. Treat as benchmark-control/perf-per-watt work, not throughput promotion. |

### Prioritized actions

1. **P0:** implement the persistent multi-request gate; retain greedy-MD5 as the fast request-1 identity gate.
2. **P0:** rerun dense/control, 1007, 1295 and 1332 on the current pin with target-only/MTP-off first, then production MTP. Classify request-1 failures separately from later-request failures.
3. **P0:** audit 1295 and 1332 temporary-buffer extents, padding/initialization, row/chunk bounds and ownership/lifetime; use issue #85 only as the invariant model, not as code lineage.
4. **P1:** measure 245760 and 262144 context memory on the real 2:2:3 meta split with f16 and q8_0 KV; only then decide whether >300K needs lower weight quant.
5. **P1:** reconcile r30/r31 provenance and BigCherry's existing GDN chunking lineage; do not port duplicates.
6. **P1:** add an external-source snapshot/provenance entry for r30/r31 only if the project wants ongoing diff tracking; current registry snapshot is older (`c8af5361`, 2026-08-30) and does not cover r30/r31.
7. **P2:** if memory requires Q3, test `UD-Q3_K_XL` first, then IQ3_XXS only if the former does not free enough memory.
8. **P2:** create a separate upstream-fix plan for Qwen3 tools + `response_format` grammar handling.
9. **P2/conditional:** revisit expert-cache devmap/staging/lazy-load work only if model placement changes away from all-GPU.

## Steps

1. Land one reusable multi-request/multi-ubatch correctness gate under `tools/lab/flash-next/`. Keep one `llama-server` process alive with `--parallel 1`/one slot. Run two lanes without restart: (A) `cache_prompt=true` with explicit same-slot reuse where supported; (B) `cache_prompt=false` with the same process/slot to isolate backend persistent state from prompt-cache reuse. Use deterministic temperature 0/fixed seed. Sequence at minimum: `A short sentinel -> B >=2-ubatch dirtying prompt -> A -> C shared-prefix/divergent-suffix -> A -> D medium distinct prompt -> A`, then repeat the sequence. Include one deliberate abort before a repeated A. Capture response text/token IDs/hash, slot/cache metadata, status, logs, token counts and elapsed time. Fail on any control mismatch, repeated-A mismatch, invalid/empty output, nontermination, excessive repeated text, or post-abort corruption. Do not record performance for a failing arm.
2. Prove the gate is sensitive before trusting it: run against an injected response-#2 corruption fixture; additionally use a known-bad external gather build if available on a host-expert topology. A gate that cannot reject known corruption is BLOCKED, not PASS.
3. Add a bandwidth/work plausibility check for host/data-movement experiments. Record expected bytes, observed copy bytes where available, elapsed time and implied GB/s. Reject physically impossible transport unless profiling proves caching/residency removed the bytes.
4. Apply greedy-MD5 first, then the persistent gate, to dense/control, 1007, 1295 and 1332. For 1295/1332 include 24K/80K and the longest practical lane. Run MTP disabled first to isolate target state, then production MTP. Throughput cannot promote a failing arm.
5. Record DEVGATHER lineage as **mechanism-only/no shared code**. No BigCherry patch dependency is warranted. Audit QSA paths for the same *buffer-footprint invariant*, not the same implementation.
6. Do not run host-expert staging-ring/devmap experiments on the current all-GPU target. Re-open only if MET01/MET06 selects host residency; reuse an existing scheduler staging path rather than adding a QFP28-private ring.
7. Profile long-context fit at 245760 and 262144 with f16 KV and q8_0 KV on the real 2:2:3 split. If >300K does not fit with acceptable headroom, evaluate lower weight quant. Test `UD-Q3_K_XL` first; use IQ3_XXS only if additional reduction is required.
8. ABA `--spec-draft-n-max` 2/3/4/5 on the gfx1030 drafter at short/24K/80K only after target correctness is green. Rank by accepted target tokens per wall-second; mechanism ownership remains QFP05/FMTP.
9. Verify model-card/server sampling (`min_p=0` included) as profile correctness. Hold sampling/reasoning policy fixed during kernel benchmarking.
10. Reproduce tools + JSON-schema on the current pin. If present, create a separate upstream-fix item; do not mix the fix into QFP28 runtime patches.
11. Search multi-slot/server logs for `seq_rm: rollback crossed a batch boundary`; minimize any hit and hand it to lifecycle/cache ownership.
12. Run a 220 W/default power-cap arm only if benchmark variance warrants it; do not classify a power cap as a throughput optimisation.

## Detailed Solution & Technical Design

### Multi-request text gate

`text-check.py` from the external notes is useful as a deterministic comparator but is insufficient as the statefulness gate because it explicitly disables prompt caching. The BigCherry gate must exercise both cache reuse and persistent backend state.

Recommended fixed prompt classes:

- **A sentinel:** short deterministic known-answer prompt, <=64 output tokens.
- **B dirtying:** deterministic 8K-16K token prompt spanning multiple ubatches.
- **C prefix-reuse:** same long prefix as A/B-derived fixture with a divergent suffix to exercise cache reuse/removal.
- **D structured deterministic:** compact code/math task with an exact expected result; keep tools+JSON-schema out until the grammar issue is fixed.

Pass requires byte/token identity to the trusted dense/control arm for each request and identity among every repeated A. Run the cache-on and cache-off sequences twice in the same process when practical. A response-#2 mutation fixture proves that the gate itself detects a later-request corrupt arm. Keep existing greedy-MD5 because it is a cheap, high-sensitivity request-1 gate; the persistent gate catches a different class.

### QSA corruption hypothesis ordering

- **1332:** if chunked differs from dense on request 1, prioritize mask/chunk boundary, selected-index ordering, tail inclusion and shape/stride semantics. Cross-request stale state is secondary.
- **1295:** non-run-to-run stability justifies checking full physical read/write footprint, initialized padding, gather destination extents, asynchronous lifetime and deterministic ordering. No evidence currently ties it to stew DEVGATHER code.

### Long-context Q3 decision

External file facts:

- AtomicChat Q4_K_M reference in the notes: ~94.5 GB.
- Unsloth `UD-Q3_K_XL`: ~90.0 GB; metadata/header inspected only, **not downloaded or benchmarked** by the source.
- ISTA GSQ-RCO `IQ3_XXS`: ~75.8 GB; actually run by the source.

Do not treat 94.5 GB as BigCherry's exact production weight size; measure the production GGUF payload. For rough budgeting only, versus a 94.5 GB-class weight set and a 2:2:3 proportional split, Q3_K_XL frees ~4.5 GB total (~1.29/~1.29/~1.93 GB by rank) and IQ3_XXS frees ~18.7 GB total (~5.34/~5.34/~8.01 GB). Actual rank savings follow tensor placement, not file-size ratio exactly.

Quality risk at medium reasoning: Q3_K_XL is the lower-risk first fallback; IQ3_XXS is materially more aggressive and should be last-resort capacity. Cheap gate: 8-12 fixed temperature-0 prompts covering exact-answer reasoning, code, long-context retrieval and tool-call schema validity; run at the production medium-reasoning setting, reject catastrophic formatting/tool regressions or >1 exact-answer regression versus the current quant.

### r30/r31 provenance and portability

`config/external-sources.toml` currently tracks the older stew675 `rdna-boosts` v3 snapshot `c8af5361` (2026-08-30). It does not cover r30 `f108261` or r31 `76a083e`.

Portability triage:

- `GGML_CUDA_GDN_CHUNKED`: **do not port blindly**. This is GDN recurrent-state/workspace chunking, not 1332 QSA token chunking. BigCherry already has 1221 (`rd50_gdn_chunked_recurrence`) and 1253 (`gfx1100_bf16_chunked_gdn`); diff/reconcile lineage first.
- expert-cache devmap / partial expert cache: no present benefit with all experts GPU-resident; reject until placement changes.
- DEVGATHER fixes/cleanup/faster CPU gather: correctness lesson only on current topology; no runtime path to port.
- staging-ring capacity: host-residency-only; reject for current topology.
- lazy/load modes: startup/host-memory placement controls, not a demonstrated all-GPU inference win; do not port as QFP28 performance work.

### Tools + JSON-schema grammar failure

Not fixed at pin `050439614`. Upstream issue `ggml-org/llama.cpp#27114` was closed as unsupported rather than fixed. In `common/parsers/qwen3-coder.cpp`, tools and `response_format` are not composed as a single valid grammar: response-format grammar selection and tool-auto lazy-grammar behavior can combine into an invalid/empty lazy-root configuration.

Safest upstream-fix patch: detect `has_tools && has_response_format` before grammar construction and return an explicit unsupported-combination error with a focused test. True simultaneous support is a larger change: construct a union grammar for the response schema and tool-call productions, then define lazy triggers/root handling for both. Keep this separate from Flash-Next performance patches.

### Additional source relevance

- External configs use host expert placement and KV choices that do not match Brutus policy; do not copy them wholesale.
- Source power-cap results are useful for variance/perf-per-watt only, not as evidence of a Brutus throughput win.
- Record build SHA, model hash, placement, KV type, slot/cache settings and MTP state in every gate result so a model/config mismatch cannot masquerade as identity failure.

## Files

- `tools/lab/flash-next/` — reusable correctness/plausibility gate and fixtures.
- `docs/planning/active/patching-qwen-flash-next/QFP28.md` — external triage/review/results.
- `profile/flashnext.ini` — only validated profile-policy winners.
- `config/external-sources.toml` — provenance tracking when maintained; no code dependency solely because a source was reviewed.

## Validation

Correctness order: greedy-MD5 -> persistent cache-off lane -> persistent cache-on/same-slot lane -> target-only long-context -> MTP-enabled production lane. Minimum 3 correctness repetitions per arm before ABBA/5-rep performance. Any semantic failure voids throughput from that arm.

For 1295/1332, capture whether divergence occurs on request 1 or only after state reuse. For host-movement work, additionally require pinned-H2D calibration and profiler evidence of bytes/overlap on each relevant GPU/link.

## Acceptance Criteria

- Reusable multi-request/multi-ubatch gate exists and rejects an injected later-request corruption.
- Existing greedy-MD5 remains the request-1 identity gate.
- 1295/1332 are classified by request-1 vs later-request failure and cannot be promoted while failing either gate.
- DEVGATHER is documented as mechanism-only/no shared BigCherry code on the current all-GPU topology.
- Q3 is considered only after measured >300K memory need; Q3_K_XL precedes IQ3_XXS.
- r30/r31 mechanisms are reconciled against existing BigCherry GDN/scheduler work before any port.
- tools+JSON-schema is tracked as a separate upstream-fix issue if reproduced.
- No new generic gather/ring/cache mechanism is created in QFP28.

## Notes

The strongest transferable result from r31 is not a speed patch: **partial buffers must satisfy the consumer kernel's full legal read footprint, and correctness must be tested across ubatches/requests before speed is trusted.** On Brutus the exact host-expert bug is bypassed, but that invariant is directly useful for diagnosing 1295 and for preventing false wins.

## Change Log

- 2026-10-05T04:10:44.361443+00:00 (created-by): Created by claude
- 2026-10-05T04:11:07.895131+00:00 (updated-by): Updated initial external findings
- 2026-10-05: Deep-scanned r31 gather corruption/root cause; consolidated QFP28 around multi-request correctness + bandwidth-plausibility gates; added conditional staging-ring qualification and ownership boundaries.
- 2026-10-05: Plan review against Shali12 notes + stew675 r30/r31: corrected DEVGATHER mechanism/root cause, Q3 evidence, snapshot coverage and grammar status; recorded per-step verdicts and prioritized Brutus actions.
