---
id: PWR12
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-11-25T22:42:31.992985+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Independent platform AI stability screening for named GPU power profiles

## Description

Build a small engine-independent GPU arithmetic and memory-integrity qualification tool on the BigCherry platform, not bc-radiance. The external Sweet-Spot Tuner v3.5.15 (MIT, a2a31ce63cc1d40af72997de621115eaa2dcd54d) demonstrates intermittent AI arithmetic errors despite conventional benchmark stability; review docs/reference/power-tuning-evidence-2026-11.md. Do not fork the app, distribute another ROCm stack, copy its settings or add an engine startup test. One canonical power-profiles.yaml remains authored intent. Existing PWR07/08/10 and PWR01/02 own production controls, restore, profile qualification and energy/performance; they are not modified by this plan. No additional runtime profile authority or auto promotion.

## Steps

1. Platform numerical screen (implemented first slice): native HIP/hipBLASLt executable built separately from engine against installed SDK, platform hardware.ai_probe runner, explicit BDF/expected architecture/role and profile selection from existing reader; lease required, no writes. Check actual observed controls before/between/after, match stock receipt and algorithm/library/version/geometry, independent exact CPU oracle, sustained and burst modes, bounded thermal monitoring and timeout. No pass from missing library, unsupported type or wrong card.
2. Portability and broader numerical coverage: stock-qualify actual SDK algorithm/provider per gfx1030/1100/1201 and dtype. FP16, BF16, FP32, E4M3 FP8, INT8 first; gfx1030 missing hipBLASLt provider, gfx1100 BF16 first selected algorithm has nonzero CPU-oracle deviations. Diagnose on verified stock, use existing kernel replay/native dot/math provider where hipBLASLt unavailable or not exact; do not reinterpret failed stock as voltage instability or loosen the exact oracle. Add floating non-exact corpus only with independent tolerance and frozen stock byte oracle; no hand-declared ABI or new provider framework. Use existing tests for actual IQ8R/IQR/MoE/attention/GDN, memory patterns, H2D/D2H and collective transfers. For standalone VRAM coverage reuse existing copy integrity harness or a small bounded pattern probe only if missing; explicit measured free budget/headroom, no blanket near-full reservation.
3. Hot profile qualification: normal cooling, observe hotspot and memory temp where supported, bounded preconditioning to actual serving envelope, cold boost and hot bursts, stock-candidate-stock and matched actual workload. Existing power controller alone may apply/restore (PWR07/08/10); coordinate with those owners after in-flight work, no concurrent edits or second writer. Test-only candidate path requires explicit consent, lease, fresh range/readback, durable journal and verified child exit/restore; never use a synthetic pass as authorization. If owner has no independent test transaction entrypoint yet, keep screen read-only and explicitly record this blocker. No automatic undervolt search to crash, no fan/firmware override.
4. Reuse results under PWR10/PWR02: actual per-card and profile content digest, effective settings, matched runtime/driver/firmware and production replay, thermal envelope, host/device fault logs, phase tokens/J and throughput. Same current reference and actual controlled conditions; nonregression or explicit operator tradeoff. No second receipt store or default changes. Add discoverable gpu-power screen CLI alias only if useful when current power CLI settles; current platform module is directly runnable. No native engine dependency.

## Detailed Solution & Technical Design

Architecture: platform tools/bigcherry/hardware/ai_probe.py orchestrates a standalone compiled ai_probe.cpp; imports existing power_profiles, lease, power_probe and atomic power receipt writer. No engine build/packaging changes, SDK library downloads, Windows GUI, Vulkan, kernel source edits or mandatory new tests at every engine release. Profile file and role are read as intent; source digest and effective control observations recorded. Existing stock profile means observe; operator must verify factory clock/voltage/cap before --stock-confirmed. Numerical oracle computes deterministic exact signed integer dot sums and uses binary representations, not candidate self-reference. FP16 output uses exact binary16; BF16/E4M3/FP32 outputs FP32; INT8 output signed int32. Signed floating zero is numerically equivalent; no other bit tolerance. Algorithm and SDK versions pinned by receipt comparison. Native probe is a diagnostic screen, not full profile qualification and not authority to set controls. Missing provider (including gfx1030 SDK library), stock arithmetic discrepancy, thermal stop, insufficient coverage, timeout and device error remain nonpass. No production_qualified true output. Wrapper never changes sysfs or creates a recovery journal: existing owner is responsible for application and restoration. Source build documented separately, hardware leases short and serialized. Previous PWR01/02/07/08/10 remain authoritative and untouched.

## Code Samples & Guidance



## Files

tools/bigcherry/hardware/ai_probe.cpp, tools/bigcherry/hardware/ai_probe.py, tools/tests/hardware/test_ai_probe.py, docs/reference/tooling/AI_POWER_SCREEN.md; later reuse existing replay/transfer tools, no engine package additions.

## Validation

First slice: card-free oracle self-test and existing profile/lease plus new screen/refusal tests; compile against installed ROCm SDK independently. Short no-write hardware smoke on available current observed settings, not factory or stability qualification; document any stock/provider mismatch and unsupported device. No deliberate voltage changes, no tests on busy cards. Remaining: representative stock reference and candidate screen with actual existing owner transaction and exact restoration, thermal and memory coverage, real production role replay and quality, matched energy/throughput; use PWR10/PWR02 existing gates, no new test matrix.

## Effort & Risk



## Standards



## Acceptance Criteria

A separate platform tool tests one selected physical device and named profile without engine startup, reusing canonical YAML and current control readback. Exact CPU oracle, same stock-reference algorithm identity, positive completed repeats, thermal and lease bounds, nonpass for unsupported/mismatch/timeout/changed identity or settings; no automatic hardware writes or profile promotion. Native matrix and selected real production kernel/transfer tests report actual provider/type coverage for each intended hardware role; unsupported explicitly refused. Independent clock/voltage candidate qualification only after existing authorized transaction can apply/read back/reap/restore with no new writer; hot and burst workload, memory integrity, driver health, production quality and phase energy evidence under existing PWR10/PWR02, no default activation without approval. Keep pending until residual portability, hot/residency and owner integration pass; completed first slice recorded, no claims of qualification.

## Notes

Baseline main 3305d3064 (2026-11-26). PWR07/PWR08/PWR06 in_progress; no existing plan item edited. First slice native compile ROCm 7.2.4 headers/libs and CPU oracle passed, 27 targeted offline tests passed. Current observed settings read-only smoke: gfx1201 768x768, 1 s/2 ms gap FP16 307, BF16 275, FP8 290, INT8 301, FP32 277 clean iterations (first four before signed-zero-only refinement); gfx1100 FP16 248 clean after signed-zero fix (8246 signed zeros before); gfx1100 BF16->FP32 first heuristic 93503 nonzero mismatching elements including tiny deviations; no hardware instability attribution. gfx1030 no installed TensileLibrary_lazy_gfx1030.dat, unavailable. Readbacks before discovery offset zero when exposed and cap default; operator factory confirmation not independently established. No hardware setting changed, no hot soak or candidate applied. Preserve raw evidence in work root and compact results in docs/reference/tooling/AI_POWER_SCREEN.md; pending full qualification.

Final first-slice checks: 29 targeted offline tests pass including native-result coverage and lease-loss kill/reap checks; warning-free ROCm compile and CPU oracle test passed. Platform tooling map points to independent command. Matching reference safety ceiling is required. First slice remains observe-only; no automatic application, no full profile qualification. No existing plans altered.

## Change Log

- 2026-11-25T22:42:31.992985+00:00 (created-by): Created by agent
- 2026-11-25T22:44:23.614740+00:00 (updated-by): Updated: section:notes
