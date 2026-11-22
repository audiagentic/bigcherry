# RUM14: reuse native container weight materialization

**Status: source analysis and Planning MCP registration only, 22 November 2026.** This document does not claim an implemented loader, an issued checkpoint-backed C06 self-attention operation, TP1 parity, or a fresh GPU run.

Owner: [RUM14](../../planning/active/patching-radiance-upstream/RUM14.md) — generic Radiance weight-loading reuse. Consumer: [LTX02](../../planning/active/build-radiance-ltx25-tp3/LTX02.md) — model-specific execution and numerical checks. The frozen reference is Radiance 2.2 / ABI44, source b6b8751cccb0e834a13ae9e5afca824ee9a487aa, against BigCherry main 06d5831a0 (bc-radiance 2020.1.1).

## Actual implementation and the missing seam

| Existing native source | Ownership and constraint |
| --- | --- |
| core/engine.h, Engine::start | Starts and declares the complete model, plans residency and loads weights; no observed public API accepting a previously declared test Program |
| core/engine_bringup.cpp:3923 onward, Engine::load_weights | Resolves rank-local weight destinations through native placement and mover, handles mapped and resident tiers, plans each source share |
| core/engine_bringup.cpp:3863, plan_container_weight | Validates selected source file offsets and produces LoadRun and staged-weight descriptors from the actual PlaneShare rectangles |
| core/engine_bringup.cpp:2836 and around 3200, RelayoutPool and its run method | Owns buffered staging, direct identity copy, widen/narrow conversion, kernel-selected RadKernelInfo::relayout, upload and cleanup |
| core/engine_bringup.cpp:3369, sweep_container | Performs file-ordered source reads through that staged materialization pipeline |
| core/engine_bringup.cpp:4259 onward, Engine::refill_weights | Reuses the same planning and staging machinery for restoration; a test-specific reader would diverge |
| core/format/share.h, rad::weight_share | Already shared rank/plane rectangle arithmetic, but not a complete loader |
| core/runtime/ctx.h, rad::Ctx | Issues operations against already resident Program weights; Ctx::init does not materialize checkpoint weights |

Line positions refer to the inspected frozen 2.2 composition; recheck them when replaying an upstream patch.

The existing source-backed C06 test at tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c02_lin_declare_test.cpp, verify_c06_load_contract, verifies 10 declared weights per rank and runs 24 native nonidentity IQ8R scale relayout hooks across 12 rank/modality/split cases. Its host-only temporary source-plane vectors are not resident Program weight slots. Each rank requires the QKV and output IQ8R scale layout, each with 256 extra stored bytes relative to its selected source plane. Copying the selected raw plane into an allocated stored slot would bypass the actual native kernel layout.

The same fixture's run_self_attn declares fused QKV, gate, all-head Q/K norm, RoPE, self-attention, exact output reduction and one post-reduction bias. Earlier GPU receipts show four three-rank graphs and 108 AQL instances initialized and retired in **each** physical-device order, but no complete C06 model step was issued. Separately, the existing C05 RadCtx test issued real Q/K with checkpoint BF16 gains and seeded activations on three cards, with 2,088,960 of 2,088,960 exact BF16 comparisons. Its special-case unsharded gain copy is not an IQ8R loader or captured TP1 activation comparison. Evidence: [C05 issued AQL](../2026-11-22-ltx25-ltx02-issued-aql-rms/README.md) and [C06 weight contract](../2026-11-22-ltx25-ltx02-c06-weight-contract/README.md).

## Minimal solution and code reduction

**First reuse, then extract only if necessary.** Inspect for an existing native API capable of receiving a frozen Builder/Program, a validated RadFile and placement-authorized destinations. The public Engine::start currently owns the full model and cannot simply accept the C06 test's subset Program. It is not acceptable to bypass the LTX world-size refusal or run a whole production model just to supply a test fixture.

If no usable API exists, expose the smallest **generic, core-owned batch materialization seam** that delegates to the same plan_container_weight, RelayoutPool and sweep_container used by the production load/refill paths. The production caller retains placement, identity, memory-tier and allocation authority. The native C06 host test supplies bounded, already allocated destinations. Require failure-atomic publication and ordered cleanup. Keep this internal to the source unless a reviewed ABI contract proves public exposure unavoidable. Add no per-step branches, allocations, locks or lookup.

A mere wrapper around the existing C06 test's custom plane gather is **not sufficient**: it would create a second loader. Document before/after production and test line counts, executable paths, source rectangles, tier behavior and refills. Eliminate the temporary test-only gathering once the genuine engine-owned loader is exercised. If extraction copies large production state or grows more paths than it removes, stop for review rather than shipping a shim.

## Executable acceptance ladder

1. **Test red before production changes.** Construct the existing model-backed C06 Builders and confirm a full RadCtx step cannot be accepted with unbound model weights. Introduce missing source, wrong rank, invalid plane extent, unresolved relayout owner, canceled partial staging and failed upload cases. Prove each guard is mutation-sensitive.
2. **Host generic loader contract.** Materialize all 10 C06 weights per rank for video and audio block zero, with both legal 8/8/16 and 12/12/8 head maps. Verify that the two scale weights use the *same* native kernel-selected layout as production, compare bytes, and ensure errors leave no published pointers. Exercise current production SSD, mapped, host, VRAM, refill and source replay tests.
3. **Consume from LTX02, not RUM14.** Reuse existing C05 RadCtx/AQL and C06 Attn Builders to issue actual Act::step, Lin::step, attn_core and output reduction with resident real checkpoint weights. Compare captured TP1 intermediates for both modalities and splits with an explicitly approved tolerance; test rows 1/16/64, reversed GPU ordinals and rank failure or timeout.
4. **Qualify production change.** Require exact Radiance 2.2 source-managed patch apply and idempotence, component build, existing native serving regression, pointer lifetime/fault cleanup, and net implementation simplification. If shared engine loader code changed, run one unchanged installed configuration under a short exclusive GPU lease and record source, model, build, weight-byte and start/load timing identities. Do not enable TP3 serving until LTX04 accepts it.

## Scope boundaries

RUM14 is the sole active owner of generic weight materialization and any corresponding Radiance source refactor. LTX02 owns LTX self-attention issue and numerical parity. Completed RUM13 and RUM06 continue to own native AQL collective testing and engine-generation lifetime respectively; RUM05 owns admission. Neither reimplement an already accepted collective or introduce an LTX-only loader.

This analysis was derived from source review, existing committed tests and historical receipts. No production loader files were changed or GPU tests launched in this slice.
