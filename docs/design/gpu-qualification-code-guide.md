# GPU qualification: code-level implementation guide

Reviewed against BigCherry 9f196ff42 (2026-11-26), after the PWR12 voltage sweep (#1641) and model-driven kernel qualification design (#1642). This is a source review and implementation specification, not a build, GPU test or approval of new power settings.

Read with [AI power screen](../reference/tooling/AI_POWER_SCREEN.md), [kernel group qualification](radiance-kernel-group-qualification.md), [native YAML launch](radiance-native-yaml-launch.md) and [bump integration](radiance-bump-integration.md). PWR12 retains its existing physical research; RCS26–28 own native inventory, harness coverage and integration; PWR07/08/10/PWR02 own control/qualification and production energy. Do not rewrite an in-flight plan or create a second profile authority. PWR13 is the separate pending independent-workload implementation slice. All identifiers and examples in this guide are proposals unless marked current.

## 1. Review verdict: fix the proof before expanding the workload

Current PWR12 is a good diagnostic: a real HIP/BLAS execution, independent exact CPU corpus, per-output GPU comparison, per-batch CPU cross-check, deliberate comparator fault, physical BDF binding, bounded thermal polling and an existing reversible transaction. It is not proof of all GPU instructions, the engine dispatch path, or error-free operation over a production lifetime.

| Severity | Source seam | Concrete failure |
| --- | --- | --- |
| P0 | hardware/power_transaction.py: apply, _write_all, _restore_touched, recover | Lease is checked at prepare and in caller, but not before each control write; a lost lease mid-transaction can allow further writes/rollback. Recovery reconstructs from saved BDF/hwmon without a strong fresh device identity check. |
| P0 | ai_probe.screen finally; PowerTransaction.restore/stale_journals | Full snapshot comparison happens after restore has already written terminal journal. If extra controls or identity disagree, receipt says restore_unverified but journal may say restored; next writer can pass journal scan. Unknown journal state is not in _ACTIVE. |
| P0 | native DiskRecords and platform stale_journals | Separate owner directories and schemas; --confirm-idle is an operator assertion, not a machine-enforced cross-owner admission fence. A crashed owner can be invisible to the other. |
| P1 | unit_harness.py: run_selftest, parse_hashes, summary | Exit code discarded; dictionary silently overwrites duplicate keys; empty hash intersection, failed reference, extra rows, skipped replay or absent measure.json can still produce PASS. |
| P1 | unit_harness.py: device_index, stage_measure | First matching architecture is not a physical card. Resource-delta-triggered replay runs nothing for unchanged binary at different voltage. |
| P1 | unit_harness.py: stage_overlay_measure | Only first round parsed for correctness; a later mismatch can be hidden. Minimum timing and intersection-only keys hide missing points. Minimal stub base proves ownership, not actual installed composition. |
| P1 | ai_probe.cpp: Corpus, main, verify_matrix | Fixed seed 7, only +/-1 or +/-1/16, square NN, beta=0, identical operands repeated. Input or arithmetic defects dependent on data, stride, tails, conversion, accumulator or transition are poorly covered. A verifier on the stressed GPU can itself malfunction. |
| P1 | ai_probe.py: run_case, hot_window, case_identity | Polls junction at 250 ms and checks memory ceiling, but only junction samples are time-stamped in receipt; power mean includes warmup/setup; no phase counters, reset baseline or actual clock/voltage trace. Library version plus heuristic/default token is not complete loaded code identity. |
| P2 | ai_probe.cpp: memory_screen | H2D/D2D/invert/D2H covers two buffers sequentially, not simultaneous GEMM + residency, multiple streams, peer links or CPU/GPU overlap. Headroom checked before/after allocation, not a general live reservation. |

These are static findings, not demonstrated hardware failures. The recent successful restoration receipts remain valid for their observed cases. Treat P0 as a prerequisite to new unattended mutation, not permission to broaden the hardware writer.

## 2. Keep one executor and explicit proof levels

| Scope | Executor/owner | Gate |
| --- | --- | --- |
| Independent GPU | Existing gpu-ai-screen / platform native HIP probe | L1 exact arithmetic and L2 instruction, memory, transition coverage |
| Actual kernels | RCS26/27 existing unit_harness.py, unit_families.toml, rad-kbench and real dispatch | L3 oracle, exact per-case repeat and role coverage |
| Physical finalist | PWR12 and PWR10 under existing owner transactions | L4 bounded hot, burst, mixed residency, driver health and verified restore |
| Serving | Existing PWR10/PWR02 and phase harness | L5 quality, non-regression and integrated energy |

The power profile selects effective settings; a workload recipe selects coverage. Do not put tests in power-profiles.yaml. Do not add engine startup stress tests, generic shell hooks, downloaded executables, new journal store or new scheduler. Test selection derives from required capability/role, not an arbitrary exhaustive Cartesian product.

Fast screen: target 30–90 s per test family. Expanded coverage: initial 5–10 min/card, split by lease. Finalist soak: initial 30–60 min accumulated across bounded runs, with separate cold/hot/transient coverage. These are starting budgets, not validated time requirements. Require positive minimum operations AND elapsed coverage, explicit incomplete coverage and a measured cost report before tightening the budgets.

## 3. Platform code map and migration

Current files: tools/bigcherry/hardware/ai_probe.cpp (native), ai_probe.py (screen/monitor), cli/ai_probe_cmds.py (validation/managed run), hardware/power_probe.py (read-only capability/sensors), power_transaction.py and gpu_power.py (existing writer), power_receipt.py (envelope); tools/tests/hardware/test_ai_probe.py (offline gates). Keep native code within this platform owner. Start in the current translation unit; extract local headers only when independently testable (corpus/oracle, HIP dispatch, verifier). Build with the existing installed ROCm SDK, explicit actual offload targets and CPU self-test. Do not link the engine. No device instructions in CPU-only tests.

Replace the flat dtype x pause recipe expansion with one typed internal case list. Existing public flags resolve to the same cases; new cases have explicit test kind, capability, dimensions, format, layout and traffic contract. Avoid another persisted profile file; keep reviewed defaults in code, allow explicit bounded selection, and serialize the fully resolved recipe into the existing receipt. Migrate native positional CLI, Python caller, validators, fixtures and docs together if protocol changes. Prefer a typed native protocol, not arbitrary command strings.

Proposed Python sketch (internal types, not new CLI/schema contract):

~~~python
@dataclass(frozen=True)
class ProbeCase:
    kind: str           # registered implementation only
    provider: str       # actual dispatch route
    geometry: tuple[int, ...]
    dtype: str
    corpus_version: int
    seeds: tuple[int, ...]
    batch: int
    duration_s: int
    warmup_s: int
    pattern: str        # sustained, burst, mixed, etc.
    pause_ms: int
    memory_bytes: int
    headroom_bytes: int
def validate_coverage(required, actual):
    # Count before indexing: duplicates must not disappear.
    if not required:
        raise ScreenError("empty required case set")
    keys = [case_key(c) for c in actual]
    if len(keys) != len(set(keys)):
        raise ScreenError("duplicate case result")
    if set(keys) != set(required):
        raise ScreenError("missing or unexpected case")
    for c in actual:
        validate_native_case(c)  # positive counts, checks, identity
        require_pass(c)          # unsupported is not a required pass
~~~

Use a closed registry of built-in implementations, explicit supported target features and a bounded per-case byte/operation budget. Resolve unsupported combinations before applying power. Optional unsupported cases are visible with rationale; required unsupported cases are nonpass. Store semantic version for corpus, comparison and native protocol; changing any invalidates that reference. Existing schema-2 screen receipts remain historical diagnostics; use a new observations schema version for incompatible case semantics (outer Receipt v1 unchanged).

## 4. Native arithmetic: deterministic diversity and exact oracles

First extend Corpus and format() in ai_probe.cpp. Generate several CPU-owned corpus seeds before timed load, using explicit unsigned PRNG arithmetic with a fixed version. Cover zeros, one-hot/identity, sparse/dense, alternating sign, cancellation, lane/address-specific tags and varied exact dyadic magnitudes. Cycle distinct buffers across launches; do not recompute an O(MNK) CPU reference in every iteration. Use bounded working sets with checked size_t products. Keep guard regions, alignment padding and output poison; make logical output and untouched guards separate checks. Check A/B readbacks periodically and at case end. No undefined signed overflow.

Use an exact int64 accumulator for integer/dyadic cases; derive maximum magnitude from K and operands before allocation, and bound representability of each intermediate/product and output. Encode expected FP16/BF16/FP32 explicitly; no GPU-derived reference. For conversion cases add independently generated IEEE rounding vectors (ties, near-boundary, +/- zero, normal/subnormal only where the documented mode guarantees it, saturation and overflow as specified). Exact dyadic cases keep only signed-zero equivalence; never accept NaN by blanket tolerance. A separate realistic floating corpus may use a frozen, justified per-op error budget against a CPU high-precision reference and repeated stock output; do not change exact tolerances.

~~~cpp
// Sketch: allocate guards outside the provider-visible logical span.
// checked_mul and oracle must be separately unit tested.
for (const auto& seed : recipe.seeds) {
    auto c = build_corpus(recipe, seed);
    require_exact_range(c, recipe);
    auto expected = cpu_reference(c, recipe);
    slots.push_back(upload_guarded(c, expected));
}
for (uint64_t i = 0; !done(); ++i) {
    auto& slot = slots[i % slots.size()];
    poison_output(slot);
    launch_registered_case(slot, recipe);
    verify_matrix_on_device(slot, recipe);
    // Host-selected index/slot independent of the GPU comparator.
    if (host_check_due(i)) verify_on_host(slot, recipe);
}
~~~

Choose host sample positions/launch selection from a recorded host-side seed; always include full readbacks at fixed cadence and before/after. To sample intermediate launches without turning every launch into a host sync, use a bounded ring of distinct outputs and copy selected outputs to pinned host memory after an event and before reuse. Compare all outputs on device, audit exact expected/seen/bad counts and input/guard checks on host; GPU verifier must not be the only oracle. Negative controls: wrong expected bit, wrong result bit, stale previous output, wrong output length, guard overwrite, skipped launch, counter corruption, repeated output and corrupted input. Reuse the existing deliberate error self-check, extend it to first/middle/last positions and per-verifier implementation. A test of the checker is not workload coverage.

## 5. Instruction and subsystem coverage

Implement small engine-independent HIP kernels only for independently defined arithmetic. Use installed compiler feature checks per explicit target and disassembly proof that intended instructions were generated (not merely C++ names). Record actual code object target and executed case. Do not hardcode FP8/WMMA/MFMA support by generation; gfx1030, gfx1100 and gfx1201 have different supported paths. Use vector/packed dot where supported, real WMMA/MFMA only where compiler and target support them; scalar emulation is a separate case, never a tensor-instruction pass. No copied Radiance trellis code in platform.

| Family | Minimal deterministic workload and oracle |
| --- | --- |
| Integer/bit operations | Packed lane extraction, xor/shift/rotate, unsigned wraparound MAD, multiply-high and signed dot; CPU unsigned reference plus edge bit patterns. |
| FP vector/scalar | Exact dyadic FMA, add/mul, conversions, mixed sign, lane dependent operands; inspect generated ISA, separate rounding vectors. |
| Matrix instructions | Native small fragment microkernel; CPU matrix oracle for every lane, accumulation and conversion; non-square and tile-edge dimensions. |
| Wave exchange | Shuffle/broadcast/ballot/reduce at actual wave width, masked active lanes with defined semantics; CPU lane permutation and mask oracle. |
| LDS/barriers | Bank-varied strided and cross-wave exchange, barriers reached by all workgroup lanes, alternate epochs and disjoint guard words; CPU mapping. |
| Cache/VRAM | Walking bits, alternating/address+epoch patterns, read-modify-write, disjoint/scattered ranges, varied alignments and strides; full host verification. |
| Async/streams | Two or more disjoint streams with explicit event dependencies; CPU expected generations, no unsynchronized races. |
| PCIe/DMA | H2D/D2H/D2D, pinned host transfers and actual peer copies only when supported; verify source/destination identities, full address+epoch patterns. |
| Mixed load | Simultaneous resident memory buffers and checked matrix/ALU workloads, separate queues and disjoint buffers, bounded live footprint; both oracles must pass. |

All device operations check launch status and async completion. Use no intentional device hang, undefined barrier, out-of-bounds access or thermal limit manipulation for a negative test. Deliberate corruption stays inside allocated valid test memory. Use offline/fake runtime fault injection for timeouts and device resets.

Memory extension: keep current memory_screen full readback, add resident allocation that remains alive during compute, checked free-memory reserve including provider workspace and all streams, then verify every retained page before/after and during bounded epochs. Preserve two independent buffers for transfer case. Do not assume logical allocation equals physically touched bytes. For multi-device tests, bind both endpoints by actual HIP BDF and require the same global GPU lease and both thermal sensors. If P2P is not available, mark it unsupported; host-staged copy is a distinct capability.

## 6. Shape and workload selection

Independent screening should cover instruction classes cheaply. Use a small selected geometry matrix, not one large square or every combination: tiny, tile boundary minus/at/plus one, aligned and non-aligned leading dimensions, rectangular tall/wide, transpose variants when supported, split/accumulate and beta nonzero with exact expected C. Examples for a target with a 16-wide tile are 15/16/17 and 63/64/65; derive relevant boundaries from actual implementation/dispatch, not fixed architecture assumptions. For BLAS, unsupported alignment is explicitly unavailable, not an error in voltage. Freeze actual algorithm selection when the public API permits; otherwise report default-dispatch scope and do not claim exact kernel coverage.

For real model kernels, selection comes from RCS26/27 inventory: units, owned rows, actual dispatch and recorded workload shapes. Use existing unit_families.toml. Include IQR computed-trellis kinds 5/6, supported B/L/scale forms, chained and bias paths, M16/M64/tiled prefill, N=5376 ragged width where supported; IQ8R decode and prefill incl. tile boundaries; actual embed/vocab overlays; attention (paged BF16/int8 KV, context splits), GDN state, MXFP4/MoE routed and grouped, speculative verify and role-specific transfer. These are required *where selected*, not a claim that every card supports every form. Missing oracle or inventory is an explicit gap, never infer from family name.

## 7. Fix native harness gates before using them for voltage

RCS27 owns changes in tools/lab/radiance/unit_harness.py. Replace run_selftest returning only text with typed result containing exit code, timeout, stdout, stderr, selected physical device and expected cases. Resolve requested physical BDF with HIP PCI identity (reuse platform discovery rather than rocminfo ordering); set visibility to exactly that card and have child prove it. Explicit release/build, same loaded plugins, actual row/layout selection, same driver/runtime and oracle. Retain every round, not only first. Parse structured rows or strict legacy lines until migrated; reject duplicate within a round, missing and unexpected keys, zero count and invalid timing. For required rows, UH_TOLERATE_ABSENT cannot waive missing cases. Explicit legitimate layout-refusal tests must verify expected RAD_E_DTYPE and no launch.

~~~python
# Sketch: a power comparison requires replay even when resources match.
required = resolve_required_cases(intent, selected_rows, shapes)
if intent == "power":
    replay_rows = required.replay_rows
else:
    replay_rows = affected_rows_from_change_or_widen(required)
for result in all_rounds:
    require_exit_zero(result)
    require_exact_case_set(required.hash_cases, result.hashes)
    require_no_failures_or_unapproved_skips(result)
require_equal_outputs(reference_rounds, candidate_rounds)
require_all_replay_points_present_and_converged(replay_rows)
~~~

Do not use compare_hashes() alone as the gate: require both reference and candidate pass, positive expected case count, exact key sets, duplicate rejection and matching outputs. For exact cases retain full host output or an existing output hash plus independent oracle (hash comparison is not itself an oracle); hash bytes with layout and shape identity. Reject nondeterministic exact reference before candidate. For nonexact math use justified existing per-op tolerance and CPU comparator; retain max error and mismatch diagnostics. For performance, replay windows must be positive/finite and converged; keep all samples including slow-mode windows, use matched ABBA and actual workload distribution, not only best median. Resource-only/skip flags remain diagnostic; summary must say incomplete, not PASS.

Overlay stage uses stub base to prove row ownership, which remains valuable for code changes; production power qualification additionally needs the installed full ordered hierarchy and actual selected provider per op/role. Use existing native selection/replay, not separate selftest calls to only the base or a fake stub. If rad-kbench timing-only has no correctness output, use independent selftest/hash or actual model output before claiming correctness. No new engine runtime wrapper. Coordinate RCS28 so profile power owner holds transaction across actual kernel children, and reaps them before restore.

## 8. Thermal, transition and telemetry contract

Preserve safety ceilings bounded by actual sensors, and stop on reaching them, missing required sensor, reset/timeout, invalid sample, lost lease or uncontrolled identity/settings. A hot minimum is a *coverage requirement*, not a safe lower operating limit. Record separate cold-start, sustained-normal, verified-hot, burst/idle-transition and mixed-resident verdicts; do not downgrade a historical hot nonpass to pass. Use a separate deliberate recipe for normal/cool operation with no hot-minimum claim. Choose burst duty patterns with alternating short/long idle intervals and compute intensity, record scheduled and actual monotonic boundaries and counts. Change workloads, not cooling, voltage or temperature limits mid-test. Longer warmup may help only inside approved maximum duration. If target not reached, mark hot incomplete; do not retry by raising ceiling or turning fans down.

Extend run_case's observe() to timestamp junction, memory, board power, effective clocks, voltage (reported sensor not requested offset), busy and available counters with units and read status. Reuse power_probe's read-only facts and existing driver-health and phase sampler; read driver event cursor before load and after reaping (if permissions unavailable, required health gate incomplete). Keep full samples local. Sample telemetry cannot prove nanosecond transient voltage. Validate monotonic timestamps, no gaps beyond recipe, in-load start/end, sample completeness, per-sensor limits and actual measured phases; retain last unsafe sample on thermal stop. Split warmup vs measured counts, do not divide whole-run iteration count by final-only power.

Energy: integrate timestamped power only over matched workload interval (trapezoidal interpolation with bounded gaps) or use verified monotonic hardware energy counter deltas with known wrap/reset; use the existing PWR01/PWR02 phase-energy owner for GPU+CPU joules/token. Unknown or missing counters are not zero watts. Compute J/verified operation only if count and interval align, and distinguish diagnostic from serving energy. Report repeat variability, not precision implied by one poll average. A measured throughput loss is not accepted merely because power falls; production owner must approve any alternative objective.

## 9. Mutation and recovery hardening (P0, before unattended expansion)

Coordinate with PWR07/08/RSC02 and existing platform transaction owner. Do not move controller to the engine or add another writer. Native and platform recovery records are distinct. Resolve all configured owner directories under the same lease and validated roots; unreadable, unknown-schema/state, missing required directory or unresolved record blocks mutation. A terminal prepared/no-write record is distinguishable from an active/uncertain record only after validating its full inventory. Bind journal to fresh physical BDF and stable identity (PCI vendor/device/subsystem, available unique ID and VBIOS, control capabilities and hwmon association); refuse recovery if identity or controls are not resolvable, never replay stale hwmon path blindly. Validate run IDs and journal paths through existing owner APIs; reject duplicate run ID. Preserve old records, recover with their owner only.

Apply checks lease and cancellation before every write/OD commit and readback boundary, pre-journal touched intent durably, and verify all coupled controls. On lost ownership do not rollback without reacquisition; retain unresolved state and require explicit recovery. If ownership is held, stop/reap all children and restore exactly. Parent/child contract must prove descendants exited, not just direct Popen termination. Use bounded process-group management within existing executor; never kill unrelated processes. Re-check ownership inside restoration. Persist complete verified identity and original state readback before marking journal terminal, not after. If persistence or full verification fails, leave unresolved durable state and hold managed evidence. Existing native transaction's receipt-first terminal ordering is a useful reference; do not duplicate native writer. Test failure after every individual write and before/after journal/receipt commits using injected FileOps and lease predicates; prove no further writes with lost lease and no restore with a live child. Ensure admission rejects a known terminal receipt with unresolved journal and vice versa. A true no-write diagnostic is separate.

~~~text
lease -> owner/recovery admission -> snapshot + matching reference
 -> durable prepare -> bounded apply/readback -> run and monitor
 -> reap/verify descendants -> restore + full original readback
 -> durable receipt -> terminal journal -> managed harvest -> release
~~~

This is required ordering for the existing owner path; design exact atomic finalization/recovery with current receipt and journal APIs rather than inventing a third format. A failed harvest must not erase recovery proof; a process crash retains recoverable managed evidence.

## 10. Evidence, reference and verdict

Reuse workspace.create_run, run_manifest.update_manifest, power_receipt.Receipt, ai_probe.screen_receipt/read_screen_receipt and capture/v1 harvest. Store per-case observations within existing ai_screen producer record, not a second database. Keep typed references to immutable source input and stock/candidate/post-stock run IDs; retain negative results and fault diagnostics. Local records hold physical identity, effective settings, installed code/build/SDK/driver, exact geometry, provider/algorithm, corpus and verifier versions, declared and actual checks, operation counts, raw telemetry, phase boundaries and recovery state. Compact portable Git records hold logical role, architecture, recipe/result, settings, managed IDs, decisions; no BDF, host path, serial, credentials or raw local logs.

Reference matching: require identical recipe and code identity, actual loaded library/dispatch facts, same physical card/role and all untouched controls, exact scope, pass of every required reference case, no write on reference, no stale/historical/observed-as-factory substitution. Compare pre/post reference controls and results, handle drift with matched repeat; explicit operator factory verification is a separate prerequisite. The profile input digest differs legitimately between candidate settings; match relevant unchanged controls and freeze the actual candidate source rather than demanding equal full profile files. Invalidate on meaningful driver/firmware/SDK/build/provider/layout/shape/corpus/recipe or control change. Do not allow screen pass to set production_qualified.

Use separate result axes: execution (completed/failed/aborted), numerical (pass/mismatch/not-run), coverage (complete/incomplete/unsupported with required flag), thermal (safe/stop/target-unreached), health (clean/fault/unknown), restoration (not-requested/verified/unresolved), performance/energy (measured/inconclusive/regression/acceptable). Keep current outward status compatibility only where semantics are unchanged, and derive overall pass strictly from required gates. Any mismatch or unexpected device error stops deeper candidate exploration. A thermal floor miss is inconclusive, not a numerical failure and not permission to deepen under this recipe. No automatic promotion or shared settings across nominally identical cards.

## 11. Verification matrix and implementation slices

A. Admission/recovery first (PWR07/08/RSC02 coordination): fake sysfs plus native owner records, unknown states, unreadable journal, stale identity, lease loss mid-write/rollback, kill/reap, power failure and post-restore mismatch. Require durable recovery state and refusal of next writer. No real mutation before owner review.

B. Platform protocol/oracle (new independent slice): CPU golden vectors, exact integer/dyadic proof, seed/shape replay, edge conversion, intentional verifier errors, overflow bounds, malformed/duplicate case results, reference version mismatch. Reuse tools/tests/hardware/test_ai_probe.py and existing card-free CI. Compile explicit supported targets, inspect ISA for each instruction case. CPU-only self-test never counts GPU coverage.

C. Instruction and traffic cases: real observe-only baseline per physical card/role under short named lease. Each case requires actual feature, positive verified operations and exact oracle; retain unsupported. Measure wall time and cut redundant cases only when coverage equivalence is demonstrated. Add mixed VRAM/compute, async streams and peer coverage after individual cases pass.

D. Native power lane (RCS27/28): offline fixtures for zero hashes, duplicates, failed reference, missing row/provider, wrong card, later-round error, resource-equal power replay, skipped/unconverged timing. Run actual selected rows on each applicable card with unchanged installed binary and matched reference/candidate/reference, full dispatch stack. No new build per voltage. If a selftest needs building, use existing harness and pinned release source, nice -n 19 outside measurements.

E. Finalist qualification (PWR12/PWR10/PWR02): matched repeated controls, cold/normal/burst/hot/resident scopes, health, exact restore; actual serving quality (deterministic output or existing quality/KLD thresholds), phase throughput/latency and energy. No promotion absent explicit owner approval.

## 12. First hardware cohort and stop conditions

Use prior 2700 MHz / 0, -25, -50 and -75 mV observations as historical diagnostic context, not as references for changed code or recipes. At the current BF16 sustained recipe the first helper passed -75 mV and missed the hot floor at -100 mV; the second passed through -125 mV. No arithmetic failure was observed: these are not failure cliffs. Start read-only new suite on both cards, establish cold/normal/burst/hot coverage independently; after admission hardening and authorization select shallowest useful candidates and repeat. Preserve R9700 thermal-stop and gfx1030 no-saving evidence; no higher thermal ceiling or unsupported voltage backend.

Deliverables: one maintained platform CLI, registered probes with CPU/device oracle tests, explicit case identity/coverage and managed receipts; native harness fixes under RCS27; production-owner integration under RCS28/PWR10. Each implementation slice gets independent review, offline gates, hardware evidence where required and a commit/PR. Source review alone does not count as implementation or qualification. Final duration, per-target supported case matrix and acceptable production performance budget remain subject to measured evidence and owner approval.

## 13. Implementation update: direct instruction slice

Following release bc-platform 1.78.0, PWR13 adds registered alu/wave/lds
cases to the existing screen. Resolved provider and batch are hip-native
and 1; n denotes workgroup count for these cases. Native protocol 3,
screen observations schema 4; old evidence is retained, not accepted as
a new matching reference. Every launch is independently host-verified;
reported lane and word counts are checked exactly against geometry and
iterations. New case runs use PWR13 managed captures. Explicit application
of power profiles for new cases is refused pending P0 owner recovery work.

Actual code-object inspection found that declaring LDS volatile selected
generic flat accesses despite shared storage. Using nonvolatile shared
storage with full-workgroup barriers produced explicit ds_read/write on
gfx1030, ds_load/store on gfx1100/gfx1201, and the target's barrier
instructions. Keep this as a build gate, not a source-level assumption.
The direct cases deliberately verify every output on the host and do not
claim peak load saturation. Next slices: varied matrix geometry and
mixed residency, further instruction coverage (including actual matrix
instructions), phase telemetry and transition recipes, then qualified
native/production owner integration. No historical hot nonpass is
reclassified and no production setting is promoted.

## 14. Implementation update: geometry and rejected results

Protocol 4/schema 5 adds explicit M,N,K,LDA,LDB,LDC column-major NN
geometry, exact logical and padding coverage and separate provider
layouts. Legacy square requests resolve to the same geometry type.
Explicit shape requests are observe-only pending recovery hardening.
The first actual cohort exposed a missing native geometry field; the
strict receipt validator correctly refused all four runs. Rejected native
results are now retained in the per-case record, including the diagnostic
(or a bounded raw tail for malformed JSON). Keep integration-level
hardware tests in addition to synthetic payload fixtures: the latter
alone could not catch the native serialization omission. Transpose,
nonzero beta, richer magnitudes and actual tensor instructions remain
independent coverage extensions.


## 15. Implementation update: mixed VRAM and compute

Protocol 5/schema 6 adds an explicit resident budget to matrix cases,
with checked full allocation headroom, bounded 4 MiB epoch transforms
on a separate nonblocking stream, exact changed-chunk readback, and
initial/final full-buffer verification. A pass requires at least one
complete compute-time sweep even when the nominal minimum duration
is exceeded. Exact coverage counters and residency parameters bind
the receipt and reference. A 129 MiB smoke exercises a partial last
chunk. These cases remain observe-only. This tests mixed issued work
and verified live residency, not measured stream overlap or a serving
model. Keep the native peer/collective, matrix ISA, transition, health
and production-owner gates separate.

Further ISA review: gfx1100 and gfx1201 require separate native WMMA operand and accumulator layouts (see libr3/r3_compat.h); gfx1030 has no WMMA and must not report an emulated tile as native coverage. Check actual emitted WMMA instructions and independently verify every output lane against an exact CPU tile oracle. Existing libr2 notes identify a historical gfx1030 fdot2 builtin miscompile; verify the emitted packed-dot form and exact output on that target instead of assuming the intrinsic is correct.


## 16. Native tensor instruction and precision review

Protocol 6/schema 7 adds vector FP, signed packed dot4, signed and
nonnegative FP16 dot2, native FP16/BF16 WMMA (signed and nonnegative),
and mixed unsigned/signed IU8 WMMA. All are independent CPU-oracle
cases with real target ISA inspection. The two gfx1100 cards reproduce
small signed cancellation differences under observed controls, in
both FP16/BF16 WMMA and FP16 dot2; nonnegative cases and mixed IU8
pass. gfx1201 passes signed cases too. gfx1030 native WMMA is
unavailable, not emulated. Keep strict signed cases nonpass: an
architecture-specific mathematical precision contract and matched
output-repeatability oracle require separate review and evidence.
Do not accept a tolerance inferred solely from these few mismatches,
or reinterpret the nonnegative coverage as a signed pass.

## 17. Managed independent qualification recipe

PWR13 now has a data-only architecture/recipe/group policy at
`config/gpu-ai-qualification.toml`. The `gpu-ai-qualify plan` command
resolves the architecture's supported matrix and direct-instruction cases,
a bounded allocation from observed VRAM, explicit headroom and thermal limits,
and a selected canonical power profile. Planning never writes GPU controls.
The managed plan binds the probe, profiles, recipe and optional native-family
plan by content digest. Native scope is imported from the RCS27 plan, with
uncovered units preserved; it is not executed or certified by PWR13.

`gpu-ai-qualify run --plan ID --group NAME` executes one group using the
existing independent screen, under a verified GPU lease and idle-process
preflight. It refuses non-stock selection, changed inputs and changed physical
control snapshots. Stock selection here means observe-current, not factory
settings. Signed precision cases are distinct required groups, even where the
baseline disagrees with the mathematical oracle. Unsupported native WMMA is
explicitly not applicable, never a software-emulated pass.

Harvest each screen receipt with the existing managed capture harvester.
`gpu-ai-qualify report --plan ID --result RUN_ID ...` checks verified bundle
identity, plan bindings, recipe, actual case counts, and unchanged physical
control snapshots. Missing and nonpassing groups remain visible. Even a complete
independent pass cannot certify native or production qualification.

The screen and extended policies are capability-driven, not a model-specific
kernel coverage claim. The existing RCS27 native-family planner remains the
authority for model/release route selection. Candidate execution remains
blocked pending cross-owner recovery and reference qualification.

## 18. Phase telemetry and driver-health observation

The independent screen now records timestamped junction and memory
temperatures, raw hwmon power/clock/voltage interfaces and GPU busy percentage,
with explicit sensor availability. A trapezoid estimate over the native
load interval requires bracketing power samples with no missing or excessive
gaps. Warmup splits use the requested duration boundary, not an observed
kernel phase; counts remain aggregate. No estimate of energy per token or
instantaneous rail power is implied.

A private read-only non-consuming kernel-log cursor observes AMDGPU fault
classes during each case. A matching reset, timeout, page fault or device
loss stops the case. Unreadable or overrun log coverage is incomplete.
Only classes/counts are retained, not raw unrelated log text. A clean
matching window is not proof of complete hardware health.

The managed helper0 screen plan completed all six groups: three passing
groups and three signed-arithmetic mismatch groups. Report
20261126T035431Z-gpu-ai-qualification-report-2b0b6674b9be correctly
refused independent qualification. This hardware report predates the
additive telemetry schema; a new hardware smoke is needed for that schema.

## 19. Explicit peer-pair qualification

Reuse the GP11/GP13 exact position-dependent source/poison oracle in
`tools/tests/fixtures/hardware/p2p_copy_correctness_test.cpp`. Its managed
mode verifies both physical BDFs/architectures, checks directional HIP access
capability and reserves VRAM independently at each endpoint. It attempts
push/pull hipMemcpyDefault, explicit hipMemcpyPeer and hipMemcpyPeerAsync
in both directions at six boundary sizes, checking every destination word
and the source afterwards. The legacy regression mode remains available.
The managed wrapper requires a GPU lease, idle preflight, both thermal
ceilings and unchanged snapshots. Only complete exact coverage can pass;
pull mismatches are not waived. API correctness alone does not prove a
direct physical transport path.

The six physical pairs in the initial allocation all reported both
directional capabilities false. All six managed captures are unavailable,
not passing; no copy was launched. Driver log coverage was also unavailable.
A peer-capable allocation is required for actual data-path qualification.


## 20. Scoped RDNA3 numerical exception and alternatives

Protocol 7/schema 8 and recipe schema 2 retain strict signed oracles
and add gfx1100-only precision-reference variants for dot2 and FP16/BF16
WMMA. The eight-epoch fixed corpus checks exact mathematical lattice,
a conservative independent finite-error budget, and full byte
repeatability. A native observation alone is a nonpass. Explicitly
reviewed, two prior verified same-physical-card managed captures and a
later byte-fingerprint match under the same plan can produce
allowed_precision, never a generic numerical pass or a production
qualification. No tolerance is added to any other case. All three
observed signatures matched across both cards and both complete
independent screen reports pass with precisely these exceptions.
No candidate writes.

Two tested alternatives retain the exact mathematical oracle: vector
FP32 FMA for packed dot, and a bounded nonnegative WMMA operand shift
with algebraic row/column correction. Both passed all 12 gfx1100
workaround cases; generated ISA proves the intended native WMMA.
This is not a general arbitrary-float rewrite. A future native
optimization may investigate sign-split four-product WMMA, compensated
accumulation or exact integer quantized paths, with resource and
performance gates under RCS27/28 and kernel-family ownership.
Keep the existing serving kernels until real-model quality warrants
an alternative. A reusable candidate precision receipt must bind
the same reviewed stock reference and exact executable/runtime
identity through existing PWR08/RSC02/PWR10 admission and restore
gates; it is not enabled by this observe-only implementation.

The remaining work is operational hot/transition runs under the
existing lease, peer-capable transfer evidence (unavailable on the
current allocation), verified driver health (both kmsg and journal
lack readable kernel records), native scope integration and matched
factory-confirmed production finalists. Do not close PWR13 as full
qualification until these gaps are resolved or explicitly scoped
out by the owner.

## 21. Performance contract for numerical alternatives

The reference exception is diagnostic policy, not a serving rewrite. The
original signed dot2/WMMA paths remain unchanged. Exact vector FMA and
biased WMMA experiments are correctness-only results. Host-verification
launch counts are not kernel throughput; the static gfx1100 code shows
packing/correction tradeoffs, not measured performance. See the performance
and adoption decision in `docs/reference/tooling/AI_POWER_SCREEN.md`.
Use actual model-selected native families and existing replay/quality/energy
owners for any future replacement. Require equivalent work, exact integer
checks or independently justified floating-point contracts, resources and
matched cold/warm timing, model quality, and no serving regression. No
blanket architecture rewrite or automatic promotion.


### Platform recovery hardening (PWR15, 2026-11-26)

The existing PowerTransaction now checks the lease before each control
write, rechecks original state and restore ranges before applying, binds
device identity and hwmon in the journal, and compares all observed
controls before terminal restoration. A later contradictory screen
readback reopens a resolved journal. Unknown or malformed journals,
reused IDs and invalid recovery inventories refuse. Legacy active
journals lacking identity require manual owner recovery; do not fabricate
their original identity. No new writer or registry.

Card-free hardware/CLI fault tests cover lease loss between OverDrive
writes, unrequested drift, identity replacement, stale or malformed
journals, zero original cap and late readback disagreement.
Cross-owner admission remains open: platform CLI uses the existing
flashnext/power/journals path, native managed starts use explicit work-root
power/journals, and one-off starts use run-local journals. Reciprocal
admission must resolve existing roots under a verified lease and refuse
unknown inventory before automatic candidates. Preserve both recovery
formats; do not change root paths by inference. No physical control write
or unattended candidate was performed by this slice.
