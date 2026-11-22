---
proposed-id: RUM13
group: patching-radiance-upstream
proposed-state: pending
priority: P1
work: M
skill: advanced
owner: Planning MCP registration required
---

# Reuse native Radiance AQL qualification across mixed GPU families

**Planning status:** Candidate for registration with Planning MCP, **not**
an authoritative item and **not** permission to create or edit
`docs/planning/` by hand. Verify uniqueness and scope against RUM02,
RUM03, RUM05, RUM06, RUM09 and the Radiance kernel coverage plans before
registering. If a live RUM item already owns these exact unimplemented
pieces, merge this candidate into that RUM item **and retire this candidate**,
not a second implementation.

## Description

Own the **remaining model-independent native GPU qualification code** that
was provisionally developed in an LTX-specific laboratory. Use native
Radiance 2.2 AQL streams, one HIP-enabled ABI44 core and existing exact
collectives for one cross-architecture GPU regression gate. Delete the
second direct-HIP launcher and duplicated HIP-off test core linkage.
Preserve its diagnostic failure receipt. The outcome must benefit any
Radiance mixed-architecture TP client, not introduce LTX-only runtime
code into the shared engine.

This RUM item owns **only a reusable model-independent validation seam**:
it does **not** take RUM06's production group rendezvous or lifecycle,
RUM05's admission/YAML/device mapping, RUM03's ABI composition,
RUM02's completed patch extraction, or RUM09's production release canary.
Those implementations and their existing regression tests retain their
current owners. Do not reopen them.

**Incoming evidence:** The historical C07 direct-HIP launcher (gfx1100
plus gfx1201) crashed at the first collective launch. C08 linking the
actual HIP-enabled Radiance 2.2 AQL core passed the existing one-shot
and two-shot exact native BF16 all-reduce across gfx1100/gfx1100/gfx1201,
**33/33 comparisons per entry** at 1, 4, 16 and 64 rows. This is a
standalone native collective proof, **not full LTX model inference**.

## Steps

1. **Inventory and reject duplication.** Inspect
   `tools/lab/radiance/ltx-tp3/mixed-ar-gate/` for the two CMake
   targets, distinct HIP-off and HIP-enabled Radiance archives, conditional
   HIP-vs-AQL launches, exported `rad_bc_shared_slot`, runner,
   evidence and tooling dispositions. Determine the actual model-neutral
   harness boundary. Record before/after source line, build-target,
   branch and external dependency counts. Do not count moving a file as
   a deletion.
2. **Replace, do not fork.** Retain only the working AQL launch path:
   `rad_dev_set`, `rad_stream_create`,
   `rad_dev_peer_launch`, `rad_stream_sync`,
   native `RadKernelInfo` declarations and exact `all_reduce` rows.
   Use the matching **HIP-enabled 2.2/ABI44 core as one whole archive**
   with the existing `bc_shared_slot` already included. Remove the
   direct-HIP executable, conditional dispatch, duplicate independently
   compiled `bc_shared_slot.cpp`, and HIP-off core/archive requirement
   together, not a compatibility option. Keep the historical C07
   negative receipt in tracked evidence.
3. **Own one native test surface.** Where reuse outside LTX is real,
   relocate the **single** resulting executable and bounded runner to
   a shared engine lab under `tools/lab/radiance/`, update all actual
   test callers, lab docs and TOOL_DISPOSITION in the **same** change,
   and delete the obsolete LTX-only standalone executable/runner.
   Keep geometry as explicit test inputs/fixtures; do not add a
   generic test framework, model loader, new config parser or a second
   collective. If relocating does not reduce duplicated callers or
   future maintenance, retain the same lab path but make the
   **ownership and entry point RUM13**; justify the simpler choice with
   numbers. LTX-specific model tolerance/evidence remains in LTX02.
4. **Preserve correctness and refusal.** Host build checks must assert
   frozen Radiance source, HIP-enabled configuration, matching SDK
   ABI/layout, exported AQL/HIP registration/launcher and shared-slot
   symbols. No fallback to direct HIP or inconsistent library mixtures.
   Under a short GPU lease rerun the gfx1100/gfx1100/gfx1201
   one-shot and two-shot host-mailbox cases, with **three distinct
   physical cards** and at least **33/33 exact BF16 comparisons per
   entry**, zero mismatches and named run receipts. For a homogeneous
   control use individual kernel rows on one R9700 **without**
   running three spinning collective ranks on the same physical
   card: that unsupported colocation stalled in the post-main probe.
   Add a bounded prelaunch refusal for duplicated physical GPU
   ordinals in a claimed independent-card TP world. Do not infer
   that production RUM06 needs another transport from this test.
   Include wrong model-independent device ordinal, missing row/library,
   foreign ABI, timeout and rank-failure/cleanup refusal where supported
   by the existing runner; do not invent a production cancellation test
   here (RUM06 owns that).
5. **Record objective reduction.** Report actual deleted code and
   targets, retained tests, source-vs-installed plugin identity and
   performance by message size including 64-row (~512 KiB) transfer
   relative to the C08 reference, distinguishing diagnostic timing
   overhead from real production hot-path cost. Leave production
   collective and engine code unchanged. Include exact reproducible
   commands/receipts in a RUM-owned evidence directory.

## Detailed Solution & Technical Design

Radiance owns the AQL dispatch backend and plugin code-object
registration. BigCherry's `rad0017` shared slot is **not redundant**:
it joins `libr3` and `libr4d` when they are dlopened separately.
RUM06 remains its sole lifecycle/poisoning/retirement owner.
Use the original native ABI and an immutable engine build fingerprint;
the lab must not introduce another runtime surface, device dispatcher,
callback lifecycle or host-mailbox implementation.

This item is a **model-independent test/code-reduction task**. In
particular, it does not own LTX `C01Slice`, `Tp3GlobalRms`,
DiT `attn_core`, model weights, conditional guidance or VAE/video
output. Those remain exclusively in LTX01–LTX05.

**Optional future kernel-gap routing, not an active implementation
subtask:** If LTX06 proves a missing **generic** gfx1030 Gemma/dense
kernel row, the correction must enter a RUM-owned/native-kernel item,
not LTX06. Check existing Radiance kernel coverage owners before
expanding this item; otherwise create a separately reviewed RUM
follow-on with actual missing-row evidence. Do not preemptively add
unproven kernel code to RUM13.

## Code Samples & Guidance

```text
Historical C07 direct-HIP failure -> C08 AQL exact BF16 control
-> one RUM-owned AQL test executable + existing native shared slot
-> C08 numerical parity/regression and build identity
-> LTX02 consumes result, issues actual model-backed RadCtx graph
```

No `hipLaunchKernel` direct escape hatch, no new `all_reduce`
implementation, no static copied rendezvous, and no extra
source/version compatibility shim. Tests run only with an acquired
GPU lease; build and raw receipts live in the configured work root.

## Files

- `tools/lab/radiance/ltx-tp3/mixed-ar-gate/{CMakeLists.txt,mixed_ar_test.cpp,run_mixed_ar.py,README.md}` — source to consolidate; **delete redundant entry point**.
- `tools/lab/radiance/` — shared RUM-owned lab surface **only if justified** by cross-model reuse.
- `docs/reference/tooling/TOOL_DISPOSITION.md` — migrate actual tooling/callers in the same change.
- `docs/evidence/2026-11-22-ltx25-c07-native-mixed-ar/README.md` and
  `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md` — frozen baseline proof.
- `docs/evidence/rum13-native-qualification/` — new stable test and deletion receipts.

## Validation

Source/SDK/ABI gate, version mismatch refusals, native CMake build
and symbol inspection; runner negative exits; repository tooling,
forbidden-term and patch/catalog guards; existing native shared-slot
unit tests; all original C08 AQL exact BF16 cases on the three-card
lease. Report baseline versus after and total LOC/target deletion,
not "no behavioral change" by inspection alone. Do not use a
synthetic-only control as evidence for real cross-card correctness.

## Effort & Risk

**P1, M.** The direct HIP failure can reappear if AQL export/link
semantics change. Mixed-library shared state and all-rank cleanup
need unchanged ABI and peer synchronization. If cross-model
genericization costs more code than it removes, prioritize
single AQL launch-path deletion over relocation.

## Standards

BigCherry worktree and Planning MCP ownership rules, native Radiance
2.2 / ABI44 exact source identity, short GPU lease, first failing
test before changing production, no legacy/shims/parallel dispatch,
net code reduction and stable evidence links. **No LTX-specific
model-source changes in this RUM item.**

## Acceptance Criteria

One native AQL GPU regression executable, one registered tooling
owner, zero active direct-HIP mixed-card collective execution modes,
no separately compiled shared-slot copy and no extra HIP-off core
linkage. Matching 2.2 SDK/source/ABI enforced; one-shot and two-shot
cross-DSO mixed-card tests remain **at least 33/33 exact BF16 each**,
with raw receipts, timeouts/refusals (including a **prelaunch
duplicate-device-ordinal refusal**), actual lines and build targets
deleted, and no production runtime change. RUM05/RUM06 implementation
ownership is unchanged. LTX01/LTX02 must only *consume* this gate.

## Notes

Proposed RUM13 is a **planning registration candidate**, not an
authoritative plan. Its number and group must be checked using the
Planning MCP before creation. This candidate intentionally replaces
the formerly proposed **LTX01/S1 shared harness-refactor
responsibility**; it is not a parallel obligation.

See `docs/evidence/ltx25-plan-registration/README.md` for the
sole RUM/LTX ownership matrix. Historical C07/C08 names remain
evidence labels, not distinct implementation plans.

**Post-main hardware review (2026-11-22):** both actual Radiance
2.2 AQL host-mailbox variants passed again on distinct
gfx1100/gfx1100/gfx1201 cards (33/33 each). An optional
three-rank/one-gfx1201 control stalled in a native kernel and
timed out after 40 s, without an observed numerical verdict.
This is **not** a mixed-card regression or a passing same-card
control. RUM13 must refuse the duplicate-device lab placement
before launch. See [post-main receipt](POST-MAIN-2026-11-22.md).
