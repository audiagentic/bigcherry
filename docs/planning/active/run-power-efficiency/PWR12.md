---
id: PWR12
order: 0
plan: run-power-efficiency
state: in_progress
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

1. Delivered (#1627 ca5f35ad9): canonical platform gpu-ai-screen, native exact CPU oracle, lease and physical BDF binding, canonical profile intent and readbacks. No engine dependency.
2. Delivered in completion extension: explicit installed hipBLASLt or rocBLAS provider, five matrix formats where supported; no silent fallback, no weakened oracle. Real short observe-current smoke covers gfx1030/gfx1100 FP16/BF16/FP32/INT8 and gfx1201 all five across slices. Retain earlier hipBLASLt BF16 mismatch and unavailable gfx1030 library; use rocBLAS passing path. Memory dtype independently exercises explicit total VRAM allocation, live free-space headroom before and after, address/iteration-dependent patterns and H2D/D2D/D2H, complete-buffer CPU verification; 128 MiB smoke on all three architectures.
3. Delivered in completion extension: bounded per-case warmup and observed final hot-window coverage by native monotonic timestamps and sampled junction temperature; memory temperature and reported limits when available, optional explicit ceiling, nonpass if target not attained or sensor/lease lost. Current-settings diagnostics cannot qualify as stock reference. Schema-2 stock receipt validates provider, coverage, device, effective controls and recipe before candidate. Explicit temporary apply uses existing platform PowerTransaction and same gpu-power journal directory, never a new writer. Requires --apply-profile and --confirm-idle, verified current stock, preflight, safe thermal state, matching reference, named lease. Reap before restore; cancellation cooperative; lost lease/unreaped child retains unresolved journal; full readback discrepancy or restore failure nonpass. Existing platform controller does not support SCLK offset; refuse this control on independent apply, use authorized native owner for observe-only screen. Native owner per-session journals are separate; operator must verify all resolved, not infer from platform scan. Real integration tested against existing fake sysfs driver.
4. Outstanding physical qualification only: with explicit hardware-change authorization and actual verified factory clocks, select one intended candidate and actual safe operating envelope, run stock-candidate-stock, representative extended hot/resident soak, exact restoration and real production role replay/quality/driver health and phase energy under existing PWR10/PWR02 owners. No new engine wrapper or general subprocess protocol; reuse existing native replay, transfer and production qualification harnesses, no duplicate controller/store or default change. Unsupported formats/controls explicitly refused. No actual power setting was changed during implementation; do not mark physical profile qualification complete.

## Detailed Solution & Technical Design

Independent platform code: hardware/ai_probe.cpp compiled against installed HIP, hipBLASLt and rocBLAS, hardware/ai_probe.py observes and validates the named YAML recipe, cli/ai_probe_cmds.py exposes gpu-ai-screen. Native probe itself never writes controls. Exact signed dyadic CPU oracle, only signed floating zero equivalent. Explicit provider and versions, opaque hipBLASLt heuristic; stable public rocBLAS default/no-atomics route is honestly recorded, not an exact kernel ID. Memory workload explicit bounded two buffers and checked free VRAM, whole-buffer readbacks. Thermal sample coverage brackets native final measured interval; no forced target or fans. Same existing profile reader, lease, hardware discovery and atomic JSON receipt. Observe-current cannot pass as stock reference. Schema 2 reference requires confirmed stock and all requested cases; same physical device and settings outside candidate, runtime/library/recipe. Explicit application only through existing platform PowerTransaction, no change to controller or other active plans, same existing recovery journal directory; explicit confirmation of no unresolved journals from other native owners (no invented global registry). PWR10/PWR02 retain actual production replay, phase energy and promotion gates. No claim of hardware stability from synthetic smoke or simulated apply.

## Code Samples & Guidance



## Files

tools/bigcherry/hardware/ai_probe.cpp, tools/bigcherry/hardware/ai_probe.py, tools/bigcherry/cli/ai_probe_cmds.py, tools/tests/hardware/test_ai_probe.py, docs/reference/tooling/AI_POWER_SCREEN.md and TOOLING.md. No engine files or other active plan changes.

## Validation

ROCm 7.2.4 compile for gfx1030/gfx1100/gfx1201 and CPU oracle pass; 72 targeted offline tests pass including actual PowerTransaction on fake sysfs apply/readback/failure/restore, no restore under unreaped child, lost lease kill/reap, malformed stock reference refusal before transaction, memory allocation/headroom and thermal interval coverage. Real named-lease observe-current CLI: gfx1030 rocBLAS FP16/BF16/FP32/INT8 288/264/238/270 clean iterations; gfx1100 rocBLAS BF16/FP32/INT8 312/264/267; gfx1201 hipBLASLt FP8 332; earlier gfx1100 FP16 and all gfx1201 types passed. HIP memory 128 MiB passes gfx1030/1100/1201 9/14/12 full passes with 512 MiB reserve. gfx1100 FP16 40 C floor, 1 s warmup/2 s final, observed 48 C, 1584 sustained and 738 burst iterations pass; 70 C floor same bounded workload inconclusive despite 1585 clean iterations. All passing CLI readbacks unchanged; non-production-qualified. Exact logs retained in work root; compact table in docs/reference/tooling/AI_POWER_SCREEN.md. No factory assertion, full heat/residency soak, real application/restore, production replay or energy test performed.

## Effort & Risk



## Standards



## Acceptance Criteria

A separate platform tool tests one selected physical device and named profile without engine startup, reusing canonical YAML and current control readback. Exact CPU oracle, same stock-reference algorithm identity, positive completed repeats, thermal and lease bounds, nonpass for unsupported/mismatch/timeout/changed identity or settings; no automatic hardware writes or profile promotion. Native matrix and selected real production kernel/transfer tests report actual provider/type coverage for each intended hardware role; unsupported explicitly refused. Independent clock/voltage candidate qualification only after existing authorized transaction can apply/read back/reap/restore with no new writer; hot and burst workload, memory integrity, driver health, production quality and phase energy evidence under existing PWR10/PWR02, no default activation without approval. Keep pending until residual portability, hot/residency and owner integration pass; completed first slice recorded, no claims of qualification.

## Notes

Baseline main 3305d3064 (2026-11-26). PWR07/PWR08/PWR06 in_progress; no existing plan item edited. First slice native compile ROCm 7.2.4 headers/libs and CPU oracle passed, 27 targeted offline tests passed. Current observed settings read-only smoke: gfx1201 768x768, 1 s/2 ms gap FP16 307, BF16 275, FP8 290, INT8 301, FP32 277 clean iterations (first four before signed-zero-only refinement); gfx1100 FP16 248 clean after signed-zero fix (8246 signed zeros before); gfx1100 BF16->FP32 first heuristic 93503 nonzero mismatching elements including tiny deviations; no hardware instability attribution. gfx1030 no installed TensileLibrary_lazy_gfx1030.dat, unavailable. Readbacks before discovery offset zero when exposed and cap default; operator factory confirmation not independently established. No hardware setting changed, no hot soak or candidate applied. Preserve raw evidence in work root and compact results in docs/reference/tooling/AI_POWER_SCREEN.md; pending full qualification.

Final first-slice checks: 29 targeted offline tests pass including native-result coverage and lease-loss kill/reap checks; warning-free ROCm compile and CPU oracle test passed. Platform tooling map points to independent command. Matching reference safety ceiling is required. First slice remains observe-only; no automatic application, no full profile qualification. No existing plans altered.

PR #1627 initial CI failed the existing tooling-boundary test rejecting direct python3 -m bigcherry module invocation in docs. Corrected to the canonical platform CLI gpu-ai-screen with separate handler and one parser registration (existing gpu-power handlers untouched); new hardware tests added to existing card-free CI. 40 targeted tests including tooling boundary now pass. No engine/runtime/other plan changes.

Completion extension based on merged #1627 ca5f35ad9. All remaining independent tooling implementation now present; production profile qualification remains explicitly open, not transferred by falsely marking acceptance passed. Hardware-change authorization is required for real candidate test. First CLI attempt failed missing PyYAML in system Python, then anonymous lease refused; platform Python and named task-acquired lease passed. PWR07/PWR08 and all other plans untouched. Earlier negative provider evidence retained.

## Change Log

- 2026-11-25T22:42:31.992985+00:00 (created-by): Created by agent
- 2026-11-25T22:44:23.614740+00:00 (updated-by): Updated: section:notes
- 2026-11-25T22:50:51.480167+00:00 (updated-by): Updated: section:steps, section:files, section:notes
- 2026-11-25T23:06:16.646122+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:files, section:validation, section:notes
- 2026-11-25T23:06:41.278723+00:00 (state-transition): State: pending → in_progress
