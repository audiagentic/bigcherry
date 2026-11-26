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

Output reconciliation: remove caller-selected --output; use existing WSL01 runs/capture/<run-id>, subkind gpu-ai-screen, plan PWR12, engine none, standard run.manifest.json and platform Receipt v1 <run-id>.power.json with producer-specific ai_screen observations. Existing capture/v1 harvest and retention; screen diagnostic only, never synthetic inference or calibration record. Reference by completed managed stock run ID and exact receipt digest, with typed run reference protecting retention. Same run ID for existing gpu-power transaction/journal; unverified recovery holds the run. Historical 15 JSON files archived byte-for-byte in capture run 20261125T231746Z-gpu-ai-screen-history-26493383ed00 and verified evidence-hold bundle; historical diagnostics cannot serve as stock references. 73 targeted tests pass, including managed lifecycle, linked stock, failed candidate and harvest. No engine, hardware, or other plan changes. Physical qualification remains open.

2026-11-26 authorized physical research: #1634 merged; exact observed-reference research mode (factory confirmation unavailable), bounded batched native GEMM checker with CPU cross-check, in-load sampled hot-window correction, 78 targeted tests and ROCm 7.2.4 build/self-test pass. Four-card arithmetic and 512 MiB memory pass; ten short cap/clock/paired-UV applications pass, exact restoration. 30 s hot screens: both XTX 2700/0 pass, -125 mV clean but insufficient thermal floor; R9700 210 W thermal stop at 80 C; RX6900XT 272 W passes without demonstrated saving. Both XTX observed/2700 MHz 20992 MiB memory pass; R9700 observed 29184 MiB pass; RX6900XT 12800 MiB pair queued behind Gate E. 36 screen captures harvested/verified, input capture retained, full evidence table in docs/reference/tooling/AI_POWER_SCREEN.md. No defaults/profile promotion, no engine or other plan changes. Remains in_progress: pending large-residency and extended hot/burst, actual verified factory reference, production-role replay/quality, driver health and phase energy under existing PWR10/PWR02 owners; native/platform journal admission fence requires owner coordination.

2026-11-26 follow-up: the queued gfx1030 helper2 observed and 272 W candidate 12800 MiB memory cases completed, two checked whole-buffer passes each; candidate restored exact original controls. Managed run IDs 20261125T234829Z-gpu-ai-screen-97b163609ed5 and 20261125T234916Z-gpu-ai-screen-64226fcb1beb; both hash-verified by the standard run harvest. Sampled mean board power 48.36/48.44 W (not phase energy); no demonstrated saving. All four cards now have the planned observed large-residency diagnostic, both XTX clock-only arms and gfx1030 cap arm tested; R9700 candidate remains thermally stopped. No factory assertion, hot simultaneous model residency, or production qualification.

2026-11-26 extended observed-reference BF16 diagnostic: both gfx1100 helpers used the same batched N=768, 30 s warmup + 60 s final per pattern, sustained and 2 ms burst, junction floor 60 C and junction/memory ceiling 80 C. Helper0 sustained pass 669120 clean GEMMs, verified hot window 72–75 C; burst 418624 clean GEMMs but no full hot window. Helper1 sustained pass 678784 clean GEMMs, 75–78 C; burst 416928 clean GEMMs but no full hot window. Both overall inconclusive, observed controls unchanged, no candidate mutation. Harvested run IDs 20261126T003601Z-gpu-ai-screen-f853a264fda9 and 20261126T003902Z-gpu-ai-screen-1a5e1e347d28; separate initial lease refusal 20261126T003525Z-gpu-ai-screen-835df6a9a21e also harvested, no GPU launch. Do not reuse an overall inconclusive receipt as a candidate reference. Separate sustained-only matching research arms are the next gate, without weakening thermal criteria. Factory and production qualification remain outstanding.

2026-11-26 separate sustained-only matched research (same N=768 BF16 rocBLAS, batch 32, 30 s warmup/60 s final, 60 C floor/80 C junction and memory ceilings): helper0 observed reference 667328 clean GEMMs, 207.60 W sample mean, 74–76 C, run 20261126T004204Z-gpu-ai-screen-6a482ae0c172; matched 2700 MHz / 0 mV candidate 635616, 168.78 W, 66–67 C, run 20261126T004508Z-gpu-ai-screen-ca59ff2a7abe. Helper1 observed reference 679680, 242.52 W, 75–78 C, run 20261126T004335Z-gpu-ai-screen-4310f902d81b; matched candidate 635648, 177.32 W, 65–66 C, run 20261126T004640Z-gpu-ai-screen-acf831cdd992. All four passed; both candidate transactions restored exact original readback; all harvested. Sample mean board power reduced 18.7% and 26.9%, but completed GEMM counts decreased 4.8% and 6.5%; no phase-integrated energy, factory stock assertion, or production qualification. Burst thermal nonpasses remain; do not treat sustained-only results as full qualification. Post-candidate matched observed references pending.

2026-11-26 matched post-candidate observed references: helper0 pass 668864 clean GEMMs, 207.40 W mean sampled board power, 73–75 C verified junction, run 20261126T004812Z-gpu-ai-screen-e7610fe49260; helper1 pass 679584, 240.49 W, 74–77 C, run 20261126T004944Z-gpu-ai-screen-553b88ebbb82. Both receipts harvested. Pre/post request, full start and stop controls and device identity matched exactly for each card; candidate original/restored readbacks exact. Observed-candidate-observed sustained research is complete for 2700 MHz / 0 mV; mean sampled board power relative to the two observed arms' midpoint reduced 18.7%/26.6%, GEMM counts reduced 4.9%/6.5%. No stock confirmation, no hot burst pass, no phase energy or production replay, no promotion. PWR12 remains in_progress; native serving admission/replay and owner coordination still required. All task GPU leases released.

2026-11-26 authorized voltage sweep: PR #1640 merged (a8ba2396d); retain the 2700 MHz / 0 mV observed-candidate-observed comparison. Managed experimental input 20261126T005904Z-pwr12-voltage-input-cfdc66286460, linked to the original input and hash-verified by capture/v1; no canonical production profile edits. Queue independently per gfx1100 helper at fixed 2700 MHz with offsets -25, -50, -75, -100, -125 mV, ascending depth only after the previous matched BF16 sustained screen passes with exact restore. Same N=768, batch 32, 30 s warmup and 60 s final, 60 C continuous junction floor, 80 C junction/memory ceiling, separate leases and idle/recovery preflight; stop each card on any nonpass. References are the matching post-candidate observed runs, not factory stock; no new results claimed while the RSC02 acceptance owner holds the GPU. For finalists, require existing native kernel selftest/replay on actual installed plugin composition (including IQR/IQ8R, attention/GDN and MoE where used, exact output or justified existing tolerance, matched per-card observed/candidate/observed reference, no silent skips), existing native production owner admission/recovery, hot/burst and memory coverage, driver health and phase energy/quality under PWR10/PWR02. Synthetic rocBLAS/hipBLASLt correctness alone does not qualify Radiance kernels; no arbitrary subprocess path added to platform tool and no other active plans edited.

2026-11-26 first matched voltage arms completed: 2700 MHz / -25 mV sustained BF16 helper0 passed 621888 clean GEMMs, 154.52 W mean sampled board power, 62–64 C verified junction, run 20261126T010017Z-gpu-ai-screen-bb063285fb11; helper1 passed 635488, 166.08 W, 63–64 C, run 20261126T010148Z-gpu-ai-screen-9a8a4492b230. Both original/restored controls matched exactly and managed receipts were harvested and hash-verified. Relative to the prior 2700 MHz / 0 mV arms, sampled watts fell 8.5% and 6.3%; iteration counts fell 2.2% and 0.03%. Synthetic results only. Both -50 mV arms are next in the separate lease queue. No production promotion or Radiance kernel pass is implied.

2026-11-26 next matched voltage arms: 2700 MHz / -50 mV sustained BF16 helper0 passed 635680 clean GEMMs, 145.91 W mean sampled board power, 61–63 C verified junction, run 20261126T010333Z-gpu-ai-screen-7585383d5229; helper1 passed 636864, 158.10 W, 63–65 C, run 20261126T010505Z-gpu-ai-screen-7ec9b6af2691. Both restored exactly and receipts were harvested and hash-verified. Relative to 2700 MHz / 0 mV, sampled board power fell 13.6% and 10.8%, while GEMM counts changed +0.01% and +0.19%; these are synthetic screens, not serving measurements. Both -75 mV arms are next.

## Change Log

- 2026-11-25T22:42:31.992985+00:00 (created-by): Created by agent
- 2026-11-25T22:44:23.614740+00:00 (updated-by): Updated: section:notes
- 2026-11-25T22:50:51.480167+00:00 (updated-by): Updated: section:steps, section:files, section:notes
- 2026-11-25T23:06:16.646122+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:files, section:validation, section:notes
- 2026-11-25T23:06:41.278723+00:00 (state-transition): State: pending → in_progress
- 2026-11-25T23:18:17.777002+00:00 (updated-by): Updated: section:notes
- 2026-11-25T23:46:49.293197+00:00 (updated-by): Updated: section:notes
- 2026-11-26T00:36:29.997424+00:00 (updated-by): Updated: section:notes
- 2026-11-26T00:42:36.050989+00:00 (updated-by): Updated: section:notes
- 2026-11-26T00:48:59.954682+00:00 (updated-by): Updated: section:notes
- 2026-11-26T00:51:43.453278+00:00 (updated-by): Updated: section:notes
- 2026-11-26T01:00:44.850238+00:00 (updated-by): Updated: section:notes
- 2026-11-26T01:03:50.950277+00:00 (updated-by): Updated: section:notes
- 2026-11-26T01:07:03.005446+00:00 (updated-by): Updated: section:notes
