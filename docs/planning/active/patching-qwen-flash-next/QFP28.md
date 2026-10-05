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

Source: https://github.com/Shali12/r9700-flash-next-notes plus stew675/llama-cpp-rdna-boosts r31+ investigation. Their host-expert configuration is not our normal all-GPU 2x gfx1100 + gfx1201 target, so host-expert mechanisms are controls/mechanism evidence rather than direct ports. The important transferable finding is methodological: a scheduler/data-movement optimisation can report a very large throughput gain while silently skipping real MoE work after the first ubatch/request. QFP28 therefore owns the Flash-Next performance-integrity gate and external-mechanism triage; it does not own QSA kernels (QFP04/QFP17/QFP22), persistent expert residency (MET01), aux-device execution (MET05/1328), or generic scheduler lifetime (QFP27).

Fresh 2026-10-05 review of stew675 issue #85 and wip/moe-mmq-overread/RESOLUTION.md identifies two independent host-resident MMQ tail-overread holes. Quantized MUL_MAT_ID fast MMQ reads a complete K tile; partial expert buffers must therefore preserve the upstream copy_experts guard contract (expert_size_copy + min(expert_size,512)). The cache arena lacked zeroed tail/head storage, and the gather destination's one-time zero was invalidated when the graph allocator reused the region on later ubatches. NaN contamination could then skip routed expert work, making corrupt gather runs appear 2-4x faster. r31 fixes the arena with a zeroed tail pad and defaults gather off in favour of staging. This is directly relevant to BigCherry's 1295/1332 acceptance discipline even though those patches do not share this host-expert implementation.

A later R9700 x16 measurement on the issue reports a second useful mechanism: correct staging is sensitive to ring capacity. On host-resident Flash-Next IQ3_XXS, increasing the staging ring from the default to 8 slots/4 GiB or 16 slots/8 GiB reportedly moved pp8192 at ub2048 from 1559 to 2174-2209 t/s (+39-42%), ub4096 +25%, ub8192 +21%, with byte-identical long temperature-0 outputs. Treat these as external evidence only. BigCherry should test capacity sizing only if/when MET01/MET06 leaves host experts in the production configuration; do not add a second staging implementation here.

## Steps

1. Land one reusable multi-request/multi-ubatch correctness gate under tools/lab/flash-next/. It must keep one server process alive and run, in order: short known-answer prompt; >=2-ubatch deterministic prompt; executable Python-function prompt; long summary/needle prompt; deliberate interrupted/aborted generation followed by another known-answer prompt; then repeat the short prompt. Capture response text/hash, HTTP status, server log, token counts and elapsed time. Fail on wrong answer, nontermination, >=200 repeated non-whitespace characters, non-finite logits if exposed, or post-abort corruption. Do not record performance for a failing arm.
2. Prove the gate is sensitive before trusting it: run against an injected-corruption fixture or a known-bad external gather build. A gate that cannot reject known corruption is BLOCKED, not PASS.
3. Add a bandwidth/work plausibility check for data-movement experiments. For every host-expert arm record bytes expected to cross H2D per ubatch/layer, measured copy bytes if tracing exposes them, elapsed prefill time, and implied GB/s. Reject a result if implied required transport materially exceeds the measured link ceiling unless profiling proves reuse/residency removed those bytes. This prevents NaN/skipped-work speedups from entering the ledger.
4. Apply the gate first to 1332 chunked QSA and 1295 QSA gather because both currently have text-identity concerns. 1332 remains blocked until dense-vs-chunk deterministic text/logit identity is understood; throughput alone cannot promote it. 1295 must survive multiple sequential requests, not only a single greedy-md5 lane.
5. Keep external gather lineage separate. Check whether any tracked stew675 snapshot or 1295 shares the host-expert DEVGATHER code. If no code lineage exists, document 'mechanism-only/no shared code' and do not create a patch dependency.
6. Host-expert staging-ring experiment is conditional on MET01/MET06 selecting host residency. Reuse the upstream/fork scheduler staging path. Sweep slots/capacity while holding resident VRAM equal; size candidate capacity from largest staged tensor x in-flight copies, bounded by explicit VRAM headroom. Measure copy/compute overlap, H2D GB/s, PP, TG and peak VRAM. No private QFP28 ring.
7. Q3 quant for context >300K remains a fit arm: ISTA GSQ-RCO IQ3_XXS ~75.8 GB versus ~94.5 GB AtomicChat Q4_K_M. Require medium-effort quality/eval, not load success alone. KV remains separately controlled f16/q8_0.
8. Draft depth: ABA --spec-draft-n-max 2/3/4/5 on the gfx1030 drafter at short/24K/80K. Rank by accepted target tokens per wall-second, not acceptance alone. Winner belongs in profile/flashnext.ini; QFP05/FMTP policy remains owner of draft-kernel/vocab mechanisms.
9. Sampling/reasoning: verify model-card min_p=0 against current server defaults and test reasoning-budget only as a profile policy. Do not conflate sampling changes with kernel speed measurements.
10. Reproduce the tools+JSON-schema grammar failure on BigCherry's current pin before filing/porting anything. If absent, close that sub-slice as upstream-fixed. If present, isolate to a minimal request and assign a separate upstream-fix item.
11. Search multi-slot logs for 'seq_rm: rollback crossed a batch boundary'. If present, reduce to a scheduler/cache correctness reproducer and hand ownership to the relevant lifecycle/cache item rather than adding another QFP28 mechanism.
12. Optional power-cap arm: 220 W vs default only if it reduces coefficient of variation enough to improve benchmark discrimination. Never promote a power cap as a throughput optimisation from temperature alone.

## Detailed Solution & Technical Design

### Performance-integrity invariant

A candidate result is eligible for comparison only when all three are true:

- semantic work is preserved across multiple ubatches and requests;
- observed transport/work is physically plausible for the measured hardware;
- the candidate does not change graph/runtime state in a way that invalidates the control.

For host expert movement, compute a conservative transport lower bound from the selected expert payload actually required by the algorithm. Compare `required_bytes / measured_prefill_seconds` with independently measured pinned-H2D bandwidth. A result above the physical ceiling is not a performance win until profiling proves those bytes were avoided by caching/residency. This is a diagnostic, not a rigid equality check: cache hits and overlap must be accounted explicitly.

### MMQ partial-buffer contract learned from r31

Any future BigCherry partial/pruned quantized expert buffer must provide finite readable storage for the kernel's legal tile over-read. Do not rely on allocator adjacency or one-time initialization of graph-owned memory. The reviewed r31 fix uses:

```c
head_pad   = min(expert_bytes, 512);
arena_size = arena_slots * expert_bytes + head_pad;
allocate(arena_size);
zero(arena_size);
```

This is mechanism guidance, not a direct port. If BigCherry introduces a partial expert arena, derive the actual guard size from the pinned upstream MMQ/copy_experts contract rather than hard-coding 512 without verification.

### Staging-ring qualification

If host experts become relevant, compare the existing staging path at equal model placement with capacity arms. Required instrumentation: staged tensor size, slots in flight, ring bytes, fallback count, H2D copy count/bytes, copy-engine occupancy, kernel/copy overlap and peak free VRAM. The optimisation target is service time saved per GiB, consistent with MET01; a larger ring that steals enough VRAM to evict profitable resident experts is a net regression even if staging itself is faster.

## Code Samples & Guidance

Gate output should be machine-readable JSONL, one record per request plus one summary. Suggested fields: `build_id`, `arm`, `request_index`, `prompt_class`, `prompt_tokens`, `output_tokens`, `output_sha256`, `known_answer_pass`, `repeat_run_max`, `aborted_previous`, `elapsed_ms`, `pp_tps`, `tg_tps`, `expected_h2d_bytes`, `observed_h2d_bytes`, `implied_h2d_gbps`, `link_h2d_gbps`, `verdict`.

Do not use token/s as the first assertion. Correctness and plausibility execute before benchmark aggregation.

## Files

- `tools/lab/flash-next/` — reusable correctness/plausibility gate and fixtures.
- `docs/planning/active/patching-qwen-flash-next/QFP28.md` — external triage and results.
- `profile/flashnext.ini` — only validated profile-policy winners.
- `config/external-sources.toml` — register a fork snapshot only when code is actually adopted/tracked; a research citation alone does not justify a patch dependency.

## Validation

Run the gate on control and candidate in the same server/process mode used for production. Minimum: 3 repetitions per arm for correctness, then ABBA/5-rep performance only after PASS. Include at least one prompt spanning >1 ubatch and one post-abort request. For 1295/1332 include 24K/80K and the longest practical lane where their optimisation is intended to matter. Any semantic failure voids all throughput from that arm.

Host-staging qualification additionally requires a pinned-H2D calibration on the tested PCIe topology and profiler evidence for copy bytes/overlap. Report per-GPU topology because x4/x8/x16 changes the ceiling materially.

## Effort & Risk

Gate: S/M, low implementation risk, high value. Host-staging capacity test: M, conditional. Directly porting external gather: high correctness risk and explicitly out of scope until a proven need and lineage review exist.

## Standards

One owner per mechanism. External numbers are evidence, never BigCherry validation. Correctness before speed. No promotion from a single request, single seed, or throughput-only run. No duplicate scheduler ring/cache implementation.

## Acceptance Criteria

- Reusable multi-request/multi-ubatch gate exists and rejects a known-corrupt fixture/build.
- 1295 and 1332 cannot be performance-promoted without passing it.
- Data-movement experiments include transport plausibility evidence.
- External DEVGATHER lineage is resolved as shared-code or mechanism-only.
- Any staging-ring candidate shows >=5% end-to-end PP/TTFT gain at equal effective model residency, <=2% TG regression, no correctness failure, and no unacceptable VRAM/context loss.
- Draft-depth/profile changes are promoted only on accepted-target-tokens/s plus quality/correctness controls.
- No new generic gather/ring/cache mechanism is created in QFP28.

## Notes

2026-10-05 scan: latest llama.cpp release remains b11401 (a7fb71f, 2026-10-05 00:36 UTC). New PRs #29971/#29972 are Hexagon/fuzzing work and do not supersede this slice. Relevant open work remains #29963 (host-RAM MoE pipeline), #29958 (constant graph/reallocation), #29953 (MMQ OOB), #29948 (MMQ+GLU fusion), #29927 (AMD perm), #29910 (Q2_K VGPR), #29901 (lightning indexer). QFP28 should not duplicate them.

Deep review result: prioritize the correctness/plausibility gate over another kernel patch. The r31 investigation demonstrates that a corrupt MoE path can look 2-4x faster precisely because work is skipped. This is more urgent for current 1332/1295 qualification than importing another external optimisation.

## Change Log

- 2026-10-05T04:10:44.361443+00:00 (created-by): Created by claude
- 2026-10-05T04:11:07.895131+00:00 (updated-by): Updated initial external findings
- 2026-10-05: Deep-scanned r31 gather corruption/root cause; consolidated QFP28 around multi-request correctness + bandwidth-plausibility gates; added conditional staging-ring qualification and ownership boundaries.
